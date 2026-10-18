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

POLICY_FAMILIES = ("round_robin", "shortest_job", "least_loaded", "weighted_queue")
TRAIN_INDEXES = tuple(range(0, 6))
VALIDATION_INDEXES = tuple(range(6, 9))
TEST_INDEXES = tuple(range(9, 12))
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

    rng = random.Random(f"cloud-workload-v1|{seed}|{index}")
    burst = index % 3
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
        short = rng.random() < 0.62
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


class _Policy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.family = str(payload["policy_family"])
        rules = payload.get("rules") or {}
        self.prefer_short = bool(rules.get("prefer_short_jobs"))
        self.cpu_w = float(rules.get("cpu_pressure_weight"))
        self.mem_w = float(rules.get("memory_pressure_weight"))
        self.deadline_w = float(rules.get("deadline_urgency_weight"))
        self.max_queue = int(rules.get("max_queue_depth"))
        self.cursor = 0

    def choose(
        self,
        queue: list[Job],
        running: list[Placement],
        now: float,
    ) -> tuple[Job, dict[str, Any]] | None:
        visible = queue[: self.max_queue]
        if not visible:
            return None
        if self.family == "round_robin":
            return self._round_robin(visible, running)
        if self.family == "shortest_job":
            ordered = sorted(visible, key=lambda job: (job.duration, job.arrival, job.jid))
            return self._first_fit(ordered, running)
        if self.family == "least_loaded":
            return self._least_loaded(visible, running)
        return self._weighted(visible, running, now)

    def _round_robin(self, jobs: list[Job], running: list[Placement]) -> tuple[Job, dict[str, Any]] | None:
        count = len(MACHINES)
        for step in range(count):
            machine = MACHINES[(self.cursor + step) % count]
            for job in jobs:
                if _fits(machine, running, job):
                    self.cursor = (self.cursor + step + 1) % count
                    return job, machine
        return None

    def _first_fit(self, jobs: list[Job], running: list[Placement]) -> tuple[Job, dict[str, Any]] | None:
        """Shortest job first, placed on the cheapest machine that can hold it."""

        for job in jobs:
            fitting = [machine for machine in MACHINES if _fits(machine, running, job)]
            if not fitting:
                continue
            machine = min(fitting, key=lambda item: (float(item["cost"]), str(item["id"])))
            return job, machine
        return None

    def _least_loaded(self, jobs: list[Job], running: list[Placement]) -> tuple[Job, dict[str, Any]] | None:
        machines = sorted(MACHINES, key=lambda item: (-_free_cpu(item, running), str(item["id"])))
        oldest = sorted(jobs, key=lambda job: (job.arrival, job.jid))
        for machine in machines:
            for job in oldest:
                if _fits(machine, running, job):
                    return job, machine
        return None

    def _weighted(
        self,
        jobs: list[Job],
        running: list[Placement],
        now: float,
    ) -> tuple[Job, dict[str, Any]] | None:
        best: tuple[float, Job, dict[str, Any]] | None = None
        for job in jobs:
            urgency = 1.0 / max(0.5, job.deadline - now)
            short_bonus = (1.0 / job.duration) if self.prefer_short else 0.0
            for machine in MACHINES:
                if not _fits(machine, running, job):
                    continue
                cpu_used, mem_used = _used(running, str(machine["id"]))
                cpu_pressure = (cpu_used + job.cpu) / float(machine["cpu"])
                mem_pressure = (mem_used + job.mem) / float(machine["mem"])
                score = (
                    short_bonus
                    + self.deadline_w * urgency
                    - self.cpu_w * cpu_pressure
                    - self.mem_w * mem_pressure
                )
                if best is None or score > best[0] + 1e-12:
                    best = (score, job, machine)
                elif best is not None and abs(score - best[0]) <= 1e-12 and job.jid < best[1].jid:
                    best = (score, job, machine)
        if best is None:
            return None
        return best[1], best[2]


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


def baseline_payload(name: str) -> dict[str, Any]:
    """Hand-written policies. shortest_job and deadline_weighted are the strong pair."""

    rules = {
        "prefer_short_jobs": False,
        "cpu_pressure_weight": 0.2,
        "memory_pressure_weight": 0.2,
        "deadline_urgency_weight": 0.2,
        "max_queue_depth": 12,
    }
    if name == "round_robin":
        family = "round_robin"
    elif name == "shortest_job":
        family = "shortest_job"
        rules["prefer_short_jobs"] = True
    elif name == "least_loaded":
        family = "least_loaded"
    elif name == "deadline_weighted":
        family = "weighted_queue"
        rules = {
            "prefer_short_jobs": True,
            "cpu_pressure_weight": 0.35,
            "memory_pressure_weight": 0.15,
            "deadline_urgency_weight": 0.85,
            "max_queue_depth": 16,
        }
    else:
        raise KeyError(name)
    return {"policy_family": family, "rules": rules, "description": f"baseline {name}"}


BASELINE_NAMES = ("round_robin", "shortest_job", "least_loaded", "deadline_weighted")


def _rules_ok(rules: Any) -> str | None:
    if not isinstance(rules, dict):
        return "rules must be an object"
    required = (
        "prefer_short_jobs",
        "cpu_pressure_weight",
        "memory_pressure_weight",
        "deadline_urgency_weight",
        "max_queue_depth",
    )
    extra = set(rules) - set(required)
    if extra:
        return f"unknown rule keys: {sorted(extra)}"
    for key in required:
        if key not in rules:
            return f"rules.{key} is required"
    if not isinstance(rules["prefer_short_jobs"], bool):
        return "prefer_short_jobs must be a boolean"
    for key in ("cpu_pressure_weight", "memory_pressure_weight", "deadline_urgency_weight"):
        value = rules[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key} must be a number"
        if not 0.0 <= float(value) <= 1.0:
            return f"{key} must be inside [0, 1]"
    depth = rules["max_queue_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 28:
        return "max_queue_depth must be an integer from 1 to 28"
    return None


class CloudSchedulingAdapter(DomainAdapter):
    name = "cloud_scheduling"
    evaluator_version = "cloud-scheduler-v1"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def genome_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["policy_family", "rules", "description"],
            "properties": {
                "policy_family": {"type": "string", "enum": list(POLICY_FAMILIES)},
                "rules": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "prefer_short_jobs",
                        "cpu_pressure_weight",
                        "memory_pressure_weight",
                        "deadline_urgency_weight",
                        "max_queue_depth",
                    ],
                    "properties": {
                        "prefer_short_jobs": {"type": "boolean"},
                        "cpu_pressure_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "memory_pressure_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "deadline_urgency_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_queue_depth": {"type": "integer", "minimum": 1, "maximum": 28},
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
        return str(payload.get("policy_family") or "default")

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
            "payload.policy_family is round_robin, shortest_job, least_loaded, or weighted_queue. "
            "payload.rules weights are numbers in [0, 1]. max_queue_depth is an integer from 1 to 28. "
            "Do not emit Python. The simulator rejects any assignment that would exceed CPU or memory.\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Change one rule or the policy family. Stay inside the schema. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent: {parent.payload}\nFeedback: {feedback.feedback}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Combine the two policies. Inherit one family and mix the numeric weights. "
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
        evaluation_version="cloud-scheduler-v1",
    )
