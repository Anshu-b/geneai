"""Deterministic cloud scheduling simulator.

The genome is a restricted JSON policy, not executable code. Train, validation,
and test are disjoint workloads drawn from one job distribution. A feasible
policy never places a job that would exceed a machine's CPU or memory.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

POLICY_FAMILIES = ("round_robin", "shortest_job", "least_loaded", "weighted_queue", "composed")
SCORE_TERMS = (
    "short_job",
    "deadline_urgency",
    "wait_time",
    "free_cpu",
    "cheap_machine",
    "rotation",
    "cpu_pressure",
    "memory_pressure",
)
TIE_BREAKS = ("duration", "arrival", "job_id", "free_cpu", "machine_cost", "machine_id", "deadline")
JOB_CLASSES = ("short", "long", "cpu_heavy", "memory_heavy")
# 18 workloads so each split contains two copies of every burst x mix cell.
WORKLOADS_PER_SPLIT = 18
TRAIN_INDEXES = tuple(range(0, WORKLOADS_PER_SPLIT))
VALIDATION_INDEXES = tuple(range(WORKLOADS_PER_SPLIT, 2 * WORKLOADS_PER_SPLIT))
TEST_INDEXES = tuple(range(2 * WORKLOADS_PER_SPLIT, 3 * WORKLOADS_PER_SPLIT))
SPLIT_INDEXES = {
    "train": TRAIN_INDEXES,
    "validation": VALIDATION_INDEXES,
    "test": TEST_INDEXES,
}

# Heterogeneous pool. The scenario in the blueprint is a virtual server fleet;
# this scaled pool keeps the same constraints and finishes in milliseconds.
MACHINES: tuple[dict[str, float | str], ...] = (
    {"id": "a0", "cpu": 4.0, "mem": 8.0, "cost": 0.8},
    {"id": "a1", "cpu": 4.0, "mem": 8.0, "cost": 0.8},
    {"id": "b0", "cpu": 8.0, "mem": 16.0, "cost": 1.4},
    {"id": "b1", "cpu": 8.0, "mem": 16.0, "cost": 1.4},
    {"id": "c0", "cpu": 8.0, "mem": 32.0, "cost": 1.8},
    {"id": "c1", "cpu": 16.0, "mem": 32.0, "cost": 2.6},
)


@dataclass(frozen=True)
class Job:
    jid: int
    arrival: float
    cpu: float
    mem: float
    duration: float
    deadline: float


@dataclass
class Placement:
    machine_id: str
    job: Job
    start: float
    end: float


def workload(seed: int, index: int) -> list[Job]:
    """One workload. Index selects a disjoint draw and a burstiness pattern."""

    rng = random.Random(f"cloud-workload-v2|{seed}|{index}")
    # index % 9 is one of 3 burst shapes x 3 short-job mixes. A split of 18
    # indexes contains each cell twice, so validation mirrors the test design.
    cell = index % 9
    burst = cell % 3
    mix = cell // 3
    short_probability = (0.40, 0.62, 0.80)[mix]
    count = 36
    jobs: list[Job] = []
    clock = 0.0
    for jid in range(count):
        # Arrivals are faster than the pool can clear, so a policy has a queue.
        if burst == 0:
            gap = rng.uniform(0.05, 0.2)
        elif burst == 1:
            gap = 0.02 if jid % 4 != 3 else rng.uniform(2.0, 4.0)
        else:
            gap = rng.uniform(0.02, 0.12) if jid < count // 2 else rng.uniform(0.6, 1.4)
        clock += gap
        short = rng.random() < short_probability
        duration = rng.uniform(2.0, 4.5) if short else rng.uniform(8.0, 14.0)
        cpu = rng.choice((2.0, 4.0, 4.0, 8.0))
        mem = rng.choice((4.0, 8.0, 8.0, 16.0))
        slack = rng.uniform(0.4, 2.5) if rng.random() < 0.72 else rng.uniform(6.0, 12.0)
        jobs.append(
            Job(
                jid=jid,
                arrival=round(clock, 4),
                cpu=cpu,
                mem=mem,
                duration=round(duration, 4),
                deadline=round(clock + duration + slack, 4),
            )
        )
    return jobs


def _used(running: list[Placement], machine_id: str) -> tuple[float, float]:
    cpu = 0.0
    mem = 0.0
    for place in running:
        if place.machine_id == machine_id:
            cpu += place.job.cpu
            mem += place.job.mem
    return cpu, mem


def _fits(machine: dict[str, Any], running: list[Placement], job: Job) -> bool:
    cpu, mem = _used(running, str(machine["id"]))
    return cpu + job.cpu <= float(machine["cpu"]) + 1e-9 and mem + job.mem <= float(machine["mem"]) + 1e-9


def _free_cpu(machine: dict[str, Any], running: list[Placement]) -> float:
    cpu, _mem = _used(running, str(machine["id"]))
    return float(machine["cpu"]) - cpu


_MIN_COST = min(float(machine["cost"]) for machine in MACHINES)
_MAX_COST = max(float(machine["cost"]) for machine in MACHINES)


def _zero_weights() -> dict[str, float]:
    return {term: 0.0 for term in SCORE_TERMS}


def _job_class_matches(job: Job, job_class: str) -> bool:
    if job_class == "short":
        return job.duration <= 5.0
    if job_class == "long":
        return job.duration >= 8.0
    if job_class == "cpu_heavy":
        return job.cpu >= 8.0
    if job_class == "memory_heavy":
        return job.mem >= 16.0
    return False


class _Policy:
    """One scorer. Families are weight settings, not separate simulators."""

    def __init__(self, payload: dict[str, Any]) -> None:
        rules = payload["rules"]
        self.max_queue = int(rules["max_queue_depth"])
        self.weights = {term: float(rules["score_weights"][term]) for term in SCORE_TERMS}
        self.tie_break = list(rules["tie_break"])
        self.overrides = list(rules.get("class_overrides") or [])
        self.cursor = 0

    def _weights_for(self, job: Job) -> dict[str, float]:
        for override in self.overrides:
            if _job_class_matches(job, str(override["job_class"])):
                return {term: float(override["score_weights"][term]) for term in SCORE_TERMS}
        return self.weights

    def _features(
        self,
        job: Job,
        machine: dict[str, Any],
        running: list[Placement],
        now: float,
    ) -> dict[str, float]:
        cpu_used, mem_used = _used(running, str(machine["id"]))
        index = next(i for i, item in enumerate(MACHINES) if item["id"] == machine["id"])
        offset = (index - self.cursor) % len(MACHINES)
        span = _MAX_COST - _MIN_COST
        cheap = 1.0 if span <= 1e-9 else (_MAX_COST - float(machine["cost"])) / span
        return {
            "short_job": 1.0 / job.duration,
            "deadline_urgency": min(4.0, 1.0 / max(0.5, job.deadline - now)),
            "wait_time": min(1.0, max(0.0, now - job.arrival) / 10.0),
            "free_cpu": _free_cpu(machine, running) / float(machine["cpu"]),
            "cheap_machine": cheap,
            "rotation": (len(MACHINES) - offset) / len(MACHINES),
            "cpu_pressure": (cpu_used + job.cpu) / float(machine["cpu"]),
            "memory_pressure": (mem_used + job.mem) / float(machine["mem"]),
        }

    def _tie_key(self, job: Job, machine: dict[str, Any], running: list[Placement]) -> tuple[Any, ...]:
        values: list[Any] = []
        for key in self.tie_break:
            if key == "duration":
                values.append(job.duration)
            elif key == "arrival":
                values.append(job.arrival)
            elif key == "job_id":
                values.append(job.jid)
            elif key == "free_cpu":
                values.append(-_free_cpu(machine, running))
            elif key == "machine_cost":
                values.append(float(machine["cost"]))
            elif key == "machine_id":
                values.append(str(machine["id"]))
            elif key == "deadline":
                values.append(job.deadline)
        return tuple(values)

    def choose(
        self,
        queue: list[Job],
        running: list[Placement],
        now: float,
    ) -> tuple[Job, dict[str, Any]] | None:
        visible = queue[: self.max_queue]
        best: tuple[float, tuple[Any, ...], Job, dict[str, Any]] | None = None
        for job in visible:
            weights = self._weights_for(job)
            for machine in MACHINES:
                if not _fits(machine, running, job):
                    continue
                features = self._features(job, machine, running, now)
                score = sum(weights[term] * features[term] for term in SCORE_TERMS)
                tie = self._tie_key(job, machine, running)
                if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and tie < best[1]):
                    best = (score, tie, job, machine)
        if best is None:
            return None
        _score, _tie, job, machine = best
        index = next(i for i, item in enumerate(MACHINES) if item["id"] == machine["id"])
        self.cursor = (index + 1) % len(MACHINES)
        return job, machine


def simulate(payload: dict[str, Any], jobs: list[Job]) -> dict[str, float]:
    policy = _Policy(payload)
    pending = sorted(jobs, key=lambda job: (job.arrival, job.jid))
    queue: list[Job] = []
    running: list[Placement] = []
    finished: list[tuple[Job, float]] = []
    violations = 0
    cursor = 0
    now = 0.0
    cost = 0.0
    rates = {str(machine["id"]): float(machine["cost"]) for machine in MACHINES}
    steps = 0
    while cursor < len(pending) or queue or running:
        steps += 1
        if steps > 20000:
            violations += 1
            break
        next_arrival = pending[cursor].arrival if cursor < len(pending) else 1e18
        next_end = min((place.end for place in running), default=1e18)
        nxt = min(next_arrival, next_end)
        span = max(0.0, nxt - now)
        for place in running:
            cost += rates[place.machine_id] * span * (place.job.cpu / 4.0)
        now = nxt
        running = [place for place in running if place.end > now + 1e-9]
        while cursor < len(pending) and pending[cursor].arrival <= now + 1e-9:
            queue.append(pending[cursor])
            cursor += 1
        assigned = 0
        while queue and assigned < len(MACHINES) + len(queue):
            choice = policy.choose(queue, running, now)
            if choice is None:
                break
            job, machine = choice
            if not _fits(machine, running, job):
                violations += 1
                queue.remove(job)
                continue
            running.append(
                Placement(
                    machine_id=str(machine["id"]),
                    job=job,
                    start=now,
                    end=round(now + job.duration, 4),
                )
            )
            after = _used(running, str(machine["id"]))
            if after[0] > float(machine["cpu"]) + 1e-6 or after[1] > float(machine["mem"]) + 1e-6:
                violations += 1
            queue.remove(job)
            finished.append((job, now + job.duration))
            assigned += 1
        if not running and cursor >= len(pending) and queue:
            violations += len(queue)
            break
    if not finished:
        return {
            "mean_completion": 1e6,
            "sla_miss_rate": 1.0,
            "cost": 1e6,
            "violations": float(violations + 1),
            "completed": 0.0,
        }
    completion = [end - job.arrival for job, end in finished]
    misses = sum(1 for job, end in finished if end > job.deadline + 1e-9)
    return {
        "mean_completion": sum(completion) / len(completion),
        "sla_miss_rate": misses / len(jobs),
        "cost": cost if violations == 0 else 1e6,
        "violations": float(violations),
        "completed": float(len(finished)),
    }


_ROUND_ROBIN: dict[tuple[int, int], dict[str, float]] = {}


def round_robin_metrics(seed: int, index: int) -> dict[str, float]:
    key = (seed, index)
    cached = _ROUND_ROBIN.get(key)
    if cached is None:
        cached = simulate(baseline_payload("round_robin"), workload(seed, index))
        _ROUND_ROBIN[key] = cached
    return cached


def reference_completion(seed: int, index: int) -> float:
    return float(round_robin_metrics(seed, index)["mean_completion"])


def _weight_vector(**values: float) -> dict[str, float]:
    weights = _zero_weights()
    for key, value in values.items():
        if key not in weights:
            raise KeyError(key)
        weights[key] = value
    return weights


def _rules(
    weights: dict[str, float],
    tie_break: list[str],
    *,
    depth: int = 36,
    overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "max_queue_depth": depth,
        "score_weights": weights,
        "tie_break": tie_break,
        "class_overrides": overrides or [],
    }


def baseline_payload(name: str) -> dict[str, Any]:
    """The four hand policies written in the score DSL, not as Python branches."""

    if name == "round_robin":
        family = "round_robin"
        rules = _rules(
            _weight_vector(rotation=1.0, wait_time=0.1),
            ["arrival", "job_id", "machine_id"],
        )
    elif name == "shortest_job":
        family = "shortest_job"
        rules = _rules(
            _weight_vector(short_job=1.0, cheap_machine=0.05),
            ["duration", "machine_cost", "arrival", "job_id"],
        )
    elif name == "least_loaded":
        family = "least_loaded"
        rules = _rules(
            _weight_vector(free_cpu=1.0, wait_time=0.05),
            ["arrival", "job_id", "machine_id"],
        )
    elif name == "deadline_weighted":
        family = "weighted_queue"
        rules = _rules(
            _weight_vector(short_job=0.5, deadline_urgency=0.85, cpu_pressure=-0.35, memory_pressure=-0.15),
            ["deadline", "job_id", "machine_id"],
        )
    else:
        raise KeyError(name)
    return {
        "policy_family": family,
        "rules": rules,
        "description": f"seeded baseline {name}",
    }


BASELINE_NAMES = ("round_robin", "shortest_job", "least_loaded", "deadline_weighted")


def _weight_map_ok(weights: Any, label: str) -> str | None:
    if not isinstance(weights, dict):
        return f"{label} must be an object"
    extra = set(weights) - set(SCORE_TERMS)
    if extra:
        return f"unknown {label} keys: {sorted(extra)}"
    for term in SCORE_TERMS:
        if term not in weights:
            return f"{label}.{term} is required"
        value = weights[term]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{label}.{term} must be a number"
        if not -1.0 <= float(value) <= 1.0:
            return f"{label}.{term} must be inside [-1, 1]"
    return None


def _rules_ok(rules: Any) -> str | None:
    if not isinstance(rules, dict):
        return "rules must be an object"
    allowed = {"max_queue_depth", "score_weights", "tie_break", "class_overrides"}
    extra = set(rules) - allowed
    if extra:
        return f"unknown rule keys: {sorted(extra)}"
    for key in allowed:
        if key not in rules:
            return f"rules.{key} is required"
    depth = rules["max_queue_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 36:
        return "max_queue_depth must be an integer from 1 to 36"
    problem = _weight_map_ok(rules["score_weights"], "score_weights")
    if problem:
        return problem
    tie = rules["tie_break"]
    if not isinstance(tie, list) or not 3 <= len(tie) <= 4:
        return "tie_break must list 3 or 4 keys"
    if len(set(tie)) != len(tie):
        return "tie_break keys must be unique"
    for key in tie:
        if key not in TIE_BREAKS:
            return f"unknown tie_break key {key}"
    overrides = rules["class_overrides"]
    if not isinstance(overrides, list) or len(overrides) > 2:
        return "class_overrides must be a list of at most 2"
    seen: set[str] = set()
    for override in overrides:
        if not isinstance(override, dict):
            return "each class override must be an object"
        if set(override) != {"job_class", "score_weights"}:
            return "class override keys must be job_class and score_weights"
        job_class = override["job_class"]
        if job_class not in JOB_CLASSES:
            return f"unknown job_class {job_class}"
        if job_class in seen:
            return f"duplicate class override {job_class}"
        seen.add(job_class)
        problem = _weight_map_ok(override["score_weights"], f"class_overrides.{job_class}")
        if problem:
            return problem
    return None


class CloudSchedulingAdapter(DomainAdapter):
    name = "cloud_scheduling"
    evaluator_version = "cloud-scheduler-v2"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def _weights_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(SCORE_TERMS),
            "properties": {
                term: {"type": "number", "minimum": -1, "maximum": 1} for term in SCORE_TERMS
            },
        }

    def genome_schema(self) -> dict[str, Any]:
        weights = self._weights_schema()
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["policy_family", "rules", "description"],
            "properties": {
                "policy_family": {"type": "string", "enum": list(POLICY_FAMILIES)},
                "rules": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["max_queue_depth", "score_weights", "tie_break", "class_overrides"],
                    "properties": {
                        "max_queue_depth": {"type": "integer", "minimum": 1, "maximum": 36},
                        "score_weights": weights,
                        "tie_break": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 4,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": list(TIE_BREAKS)},
                        },
                        "class_overrides": {
                            "type": "array",
                            "minItems": 0,
                            "maxItems": 2,
                            "items": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["job_class", "score_weights"],
                                        "properties": {
                                            "job_class": {"const": job_class},
                                            "score_weights": weights,
                                        },
                                    }
                                    for job_class in JOB_CLASSES
                                ]
                            },
                        },
                    },
                },
                "description": {"type": "string"},
            },
        }

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        self._seed = seed

        def digest(split: str) -> str:
            indexes = SPLIT_INDEXES[split]
            raw = f"{self.evaluator_version}|{split}|{seed}|{list(indexes)}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {
            "train_hash": digest("train"),
            "validation_hash": digest("validation"),
            "test_hash": digest("test"),
        }

    def family_of(self, payload: dict[str, Any]) -> str:
        rules = payload.get("rules") if isinstance(payload.get("rules"), dict) else {}
        weights = rules.get("score_weights") if isinstance(rules.get("score_weights"), dict) else {}
        overrides = rules.get("class_overrides") if isinstance(rules.get("class_overrides"), list) else []
        if weights:
            dominant = max(SCORE_TERMS, key=lambda term: (abs(float(weights.get(term) or 0.0)), term))
        else:
            dominant = str(payload.get("policy_family") or "default")
        if overrides:
            return f"composed:{dominant}"
        return str(dominant)

    def seeded_baselines(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "payload": baseline_payload(name),
                "hypothesis": f"seeded baseline {name} written in the scheduling DSL",
            }
            for name in BASELINE_NAMES
        ]

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        del objective
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        extra = set(payload) - {"policy_family", "rules", "description"}
        if extra:
            raise ValueError(f"unknown payload keys: {sorted(extra)}")
        if payload.get("policy_family") not in POLICY_FAMILIES:
            raise ValueError(f"policy_family must be one of {POLICY_FAMILIES}")
        problem = _rules_ok(payload.get("rules"))
        if problem:
            raise ValueError(problem)
        if not isinstance(payload.get("description"), str) or not payload["description"].strip():
            raise ValueError("description must be a non-empty string")

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return (
            "Propose one scheduling policy as JSON with keys payload and hypothesis. "
            "Behavior comes from rules.score_weights, rules.tie_break, and up to two "
            "rules.class_overrides. Weights are numbers in [-1, 1]. "
            "A class override replaces the weight vector for short, long, cpu_heavy, or memory_heavy jobs. "
            "policy_family is a label: round_robin, shortest_job, least_loaded, weighted_queue, or composed. "
            "Do not emit Python. The simulator rejects any assignment that would exceed CPU or memory.\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Change one weight, the tie break, or a class override. Stay inside the schema. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent: {parent.payload}\nFeedback: {feedback.feedback}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Combine the two policies. Average weights and keep at most two class overrides. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent A: {a.payload}\nParent B: {b.payload}"
        )

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        started = time.perf_counter()
        if split not in SPLIT_INDEXES:
            raise ValueError(f"unknown split {split}")
        try:
            self.validate(genome.payload, ObjectiveSpec(domain=self.name, prompt="", evaluation_version=self.evaluator_version))
        except ValueError as exc:
            return self._infeasible(genome, split, str(exc), started)
        totals = {"mean_completion": 0.0, "sla_miss_rate": 0.0, "cost": 0.0, "violations": 0.0, "ratio": 0.0}
        indexes = SPLIT_INDEXES[split]
        for index in indexes:
            jobs = workload(self._seed, index)
            metrics = simulate(genome.payload, jobs)
            if metrics["violations"] > 0 or metrics["completed"] < len(jobs):
                return self._infeasible(genome, split, "capacity violation or unfinished jobs", started)
            reference = round_robin_metrics(self._seed, index)
            totals["mean_completion"] += metrics["mean_completion"]
            totals["sla_miss_rate"] += metrics["sla_miss_rate"]
            totals["cost"] += metrics["cost"]
            totals["ratio"] += metrics["mean_completion"] / max(reference["mean_completion"], 1e-9)
            totals["reference_cost"] = totals.get("reference_cost", 0.0) + reference["cost"]
        count = float(len(indexes))
        mean_completion = totals["mean_completion"] / count
        sla = totals["sla_miss_rate"] / count
        cost = totals["cost"] / count
        ratio = totals["ratio"] / count
        reference_cost = totals.get("reference_cost", 0.0) / count
        cost_ratio = cost / max(reference_cost, 1e-9)
        fitness = -(ratio + 0.25 * sla + 0.10 * cost_ratio)
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=fitness,
            metrics={
                "mean_completion": mean_completion,
                "completion_ratio_vs_round_robin": ratio,
                "sla_miss_rate": sla,
                "cost": cost,
                "cost_ratio_vs_round_robin": cost_ratio,
                "violations": 0.0,
                "workloads": count,
            },
            feedback=(
                f"{split} mean_completion={mean_completion:.4f} "
                f"vs_round_robin={ratio:.4f} sla_miss={sla:.4f} cost_ratio={cost_ratio:.4f}. "
                "Lower completion and SLA miss is better. Capacity violations are infeasible."
            ),
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]:
        return {
            "domain": self.name,
            "genome_id": genome.id,
            "policy_family": genome.payload.get("policy_family"),
            "rules": genome.payload.get("rules"),
            "feasible": evaluation.feasible,
            "fitness": evaluation.fitness,
            "metrics": evaluation.metrics,
            "split": evaluation.data_split,
        }

    def baselines(self) -> dict[str, dict[str, Any]]:
        return {name: baseline_payload(name) for name in BASELINE_NAMES}

    def _infeasible(self, genome: Genome, split: str, reason: str, started: float) -> Evaluation:
        return Evaluation(
            genome_id=genome.id,
            feasible=False,
            fitness=None,
            metrics={"violations": 1.0},
            constraint_violations=[reason],
            feedback=f"infeasible: {reason}",
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )


def cloud_objective() -> ObjectiveSpec:
    return ObjectiveSpec(
        domain="cloud_scheduling",
        prompt=(
            "Schedule jobs onto a heterogeneous server pool. "
            "Hard constraint: never exceed a machine's CPU or memory. "
            "Reduce mean completion time and SLA misses relative to round robin."
        ),
        constraints={"machines": len(MACHINES), "capacity_violations": 0},
        evaluation_version="cloud-scheduler-v2",
    )
