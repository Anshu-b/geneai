"""Deterministic cloud scheduling simulator.

The genome is a restricted JSON policy, not executable code. Train, validation,
and test are disjoint workloads drawn from one job distribution. A feasible
policy never places a job that would exceed a machine's CPU, memory, or GPU.

cloud-scheduler-v3 is multi-objective on purpose. The pool has cheap CPU
nodes, one high-memory node, and one GPU. Some jobs have a hard deadline.
Fitness keeps the blueprint terms and adds hard-deadline tardiness plus
energy, with fixed coefficients that were not fit to the test split:

    -(completion_ratio + 0.25*sla_miss_rate + 0.50*hard_tardiness_ratio + 0.15*energy_ratio)

Shortest-job packs short work onto whatever is free, including the GPU and
the high-memory node. A stronger policy keeps those machines for the jobs
that cannot run elsewhere and still prefers short work on cheap CPUs.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
    "energy_frugal",
    "rotation",
    "cpu_pressure",
    "memory_pressure",
    "class_affinity",
    "hard_deadline",
)
TIE_BREAKS = ("duration", "arrival", "job_id", "free_cpu", "machine_cost", "machine_id", "deadline")
JOB_CLASSES = ("gpu", "high_mem", "sla", "batch")
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
EVALUATOR_VERSION = "cloud-scheduler-v3"

# Scaled fleet: four CPU nodes, one high-memory node, one GPU. Energy is the
# full draw while the machine is busy, so a short batch job on the GPU is
# expensive even when it finishes quickly.
MACHINES: tuple[dict[str, float | str], ...] = (
    {"id": "a0", "kind": "cpu", "cpu": 4.0, "mem": 8.0, "gpu": 0.0, "energy": 0.45},
    {"id": "a1", "kind": "cpu", "cpu": 4.0, "mem": 8.0, "gpu": 0.0, "energy": 0.45},
    {"id": "b0", "kind": "cpu", "cpu": 4.0, "mem": 8.0, "gpu": 0.0, "energy": 0.70},
    {"id": "b1", "kind": "cpu", "cpu": 4.0, "mem": 8.0, "gpu": 0.0, "energy": 0.70},
    {"id": "m0", "kind": "high_mem", "cpu": 8.0, "mem": 64.0, "gpu": 0.0, "energy": 1.70},
    {"id": "g0", "kind": "gpu", "cpu": 4.0, "mem": 16.0, "gpu": 1.0, "energy": 4.20},
)


@dataclass(frozen=True)
class Job:
    jid: int
    arrival: float
    cpu: float
    mem: float
    gpu: float
    duration: float
    deadline: float
    hard: bool
    kind: str


@dataclass
class Placement:
    machine_id: str
    job: Job
    start: float
    end: float


def workload(seed: int, index: int) -> list[Job]:
    """One workload. Index selects a disjoint draw and a burstiness pattern."""

    rng = random.Random(f"cloud-workload-v3|{seed}|{index}")
    # index % 9 is one of 3 burst shapes x 3 class mixes. A split of 18
    # indexes contains each cell twice, so validation mirrors the test design.
    cell = index % 9
    burst = cell % 3
    mix = cell // 3
    gpu_p, mem_p, sla_p = (
        (0.10, 0.10, 0.20),
        (0.14, 0.14, 0.24),
        (0.18, 0.16, 0.22),
    )[mix]
    count = 36
    jobs: list[Job] = []
    clock = 0.0
    for jid in range(count):
        if burst == 0:
            gap = rng.uniform(0.04, 0.18)
        elif burst == 1:
            gap = 0.03 if jid % 5 != 4 else rng.uniform(3.0, 6.0)
        else:
            gap = rng.uniform(0.05, 0.15) if jid < count // 2 else rng.uniform(0.8, 1.6)
        clock += gap
        draw = rng.random()
        if draw < gpu_p:
            kind = "gpu"
            cpu, mem, gpu_need = 4.0, 8.0, 1.0
            duration = rng.uniform(4.0, 7.0)
            slack = rng.uniform(0.5, 1.8)
            hard = True
        elif draw < gpu_p + mem_p:
            kind = "high_mem"
            cpu, mem, gpu_need = 8.0, 40.0, 0.0
            duration = rng.uniform(4.0, 8.0)
            slack = rng.uniform(0.5, 2.0)
            hard = True
        elif draw < gpu_p + mem_p + sla_p:
            kind = "sla"
            cpu, mem, gpu_need = 2.0, 2.0, 0.0
            duration = rng.uniform(1.0, 2.5)
            slack = rng.uniform(0.8, 2.2)
            hard = True
        else:
            kind = "batch"
            cpu = rng.choice((2.0, 4.0))
            mem = rng.choice((4.0, 8.0))
            gpu_need = 0.0
            duration = rng.uniform(4.0, 10.0)
            slack = rng.uniform(12.0, 22.0)
            hard = False
        jobs.append(
            Job(
                jid=jid,
                arrival=round(clock, 4),
                cpu=cpu,
                mem=mem,
                gpu=gpu_need,
                duration=round(duration, 4),
                deadline=round(clock + duration + slack, 4),
                hard=hard,
                kind=kind,
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
    if job.gpu > float(machine["gpu"]) + 1e-9:
        return False
    cpu, mem = _used(running, str(machine["id"]))
    return cpu + job.cpu <= float(machine["cpu"]) + 1e-9 and mem + job.mem <= float(machine["mem"]) + 1e-9


def _free_cpu(machine: dict[str, Any], running: list[Placement]) -> float:
    cpu, _mem = _used(running, str(machine["id"]))
    return float(machine["cpu"]) - cpu


_MIN_ENERGY = min(float(machine["energy"]) for machine in MACHINES)
_MAX_ENERGY = max(float(machine["energy"]) for machine in MACHINES)


def _zero_weights() -> dict[str, float]:
    return {term: 0.0 for term in SCORE_TERMS}


def _job_class_matches(job: Job, job_class: str) -> bool:
    return job.kind == job_class


def _class_affinity(job: Job, machine: dict[str, Any]) -> float:
    kind = str(machine["kind"])
    if job.kind == "gpu":
        return 1.0 if kind == "gpu" else 0.0
    if job.kind == "high_mem":
        return 1.0 if kind == "high_mem" else 0.0
    if kind == "cpu":
        return 1.0
    if kind == "high_mem":
        return 0.35
    return 0.0


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
        span = _MAX_ENERGY - _MIN_ENERGY
        frugal = 1.0 if span <= 1e-9 else (_MAX_ENERGY - float(machine["energy"])) / span
        return {
            "short_job": 1.0 / job.duration,
            "deadline_urgency": min(4.0, 1.0 / max(0.5, job.deadline - now)),
            "wait_time": min(1.0, max(0.0, now - job.arrival) / 10.0),
            "free_cpu": _free_cpu(machine, running) / float(machine["cpu"]),
            "energy_frugal": frugal,
            "rotation": (len(MACHINES) - offset) / len(MACHINES),
            "cpu_pressure": (cpu_used + job.cpu) / float(machine["cpu"]),
            "memory_pressure": (mem_used + job.mem) / float(machine["mem"]),
            "class_affinity": _class_affinity(job, machine),
            "hard_deadline": 1.0 if job.hard else 0.0,
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
                values.append(float(machine["energy"]))
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


def _empty_metrics(violations: float) -> dict[str, float]:
    return {
        "mean_completion": 1e6,
        "sla_miss_rate": 1.0,
        "hard_tardiness": 1e6,
        "mean_hard_duration": 1.0,
        "energy": 1e6,
        "violations": float(violations),
        "completed": 0.0,
    }


def simulate(payload: dict[str, Any], jobs: list[Job]) -> dict[str, float]:
    policy = _Policy(payload)
    pending = sorted(jobs, key=lambda job: (job.arrival, job.jid))
    queue: list[Job] = []
    running: list[Placement] = []
    finished: list[tuple[Job, float]] = []
    violations = 0
    cursor = 0
    now = 0.0
    energy = 0.0
    gpu_foreign = 0.0
    highmem_foreign = 0.0
    rates = {str(machine["id"]): float(machine["energy"]) for machine in MACHINES}
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
        busy = {place.machine_id for place in running}
        for machine_id in busy:
            energy += rates[machine_id] * span
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
            if str(machine["kind"]) == "gpu" and job.kind != "gpu":
                gpu_foreign += job.duration
            if str(machine["kind"]) == "high_mem" and job.kind != "high_mem":
                highmem_foreign += job.duration
            assigned += 1
        if not running and cursor >= len(pending) and queue:
            violations += len(queue)
            break
    if not finished:
        return _empty_metrics(violations + 1)
    completion = [end - job.arrival for job, end in finished]
    misses = sum(1 for job, end in finished if end > job.deadline + 1e-9)
    hard = [job for job in jobs if job.hard]
    tardiness = [max(0.0, end - job.deadline) for job, end in finished if job.hard]
    return {
        "mean_completion": sum(completion) / len(completion),
        "sla_miss_rate": misses / len(jobs),
        "hard_tardiness": (sum(tardiness) / len(hard)) if hard else 0.0,
        "mean_hard_duration": (sum(job.duration for job in hard) / len(hard)) if hard else 1.0,
        "energy": energy if violations == 0 else 1e6,
        "gpu_foreign_work": gpu_foreign,
        "highmem_foreign_work": highmem_foreign,
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


# Measured by scripts/cloud_oracle_search.py (seed 42, cloud-scheduler-v3).
# Not a GeneAI champion and not recomputed during a run.
CLOUD_SEARCH_CEILING: dict[str, Any] = {
    "label": "non-GeneAI reference",
    "method": "random samples plus hill-climb on train, select on validation, test read once",
    "source": "scripts/cloud_oracle_search.py",
    "evaluator": "cloud-scheduler-v3",
    "seed": 42,
    "oracle": {
        "train": -1.4307296154304623,
        "validation": -1.5848065221180798,
        "test": -1.581589322380942,
    },
    "deadline_weighted": {
        "train": -1.5269014004674266,
        "validation": -1.6344745961198561,
        "test": -1.600963325837405,
    },
    "rounded": {
        "oracle": {"train": -1.431, "validation": -1.585, "test": -1.582},
        "deadline_weighted": {"train": -1.527, "validation": -1.634, "test": -1.601},
    },
    "note": (
        "Search ceiling from a non-GeneAI reference run. "
        "This row is not the champion of the evolutionary export it is attached to."
    ),
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
            _weight_vector(short_job=1.0, energy_frugal=0.05),
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
            _weight_vector(short_job=0.5, deadline_urgency=0.85),
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
_TIE_PAD = ("duration", "arrival", "job_id", "machine_id")
_LOCAL_SAMPLE_BIAS = {
    "short_job": 0.55,
    "deadline_urgency": 0.45,
    "class_affinity": 0.85,
    "hard_deadline": 0.5,
    "energy_frugal": 0.25,
}


def _clamp_unit(value: float) -> float:
    return round(max(-1.0, min(1.0, float(value))), 6)


def _normalize_weights(value: Any, notes: list[str], label: str) -> dict[str, float]:
    raw = value if isinstance(value, dict) else {}
    if not isinstance(value, dict):
        notes.append(f"{label} defaulted to zeros")
    cleaned: dict[str, float] = {}
    for key, item in raw.items():
        if key not in SCORE_TERMS:
            notes.append(f"dropped unknown {label} key {key}")
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            cleaned[str(key)] = 0.0
            notes.append(f"{label}.{key} defaulted to 0")
            continue
        number = float(item)
        clamped = _clamp_unit(number)
        if clamped != number:
            notes.append(f"clamped {label}.{key}")
        cleaned[str(key)] = clamped
    for term in SCORE_TERMS:
        if term not in cleaned:
            cleaned[term] = 0.0
            notes.append(f"{label}.{term} filled with 0")
    return cleaned


def _normalize_tie(value: Any, notes: list[str]) -> list[str]:
    raw = value if isinstance(value, list) else []
    if not isinstance(value, list):
        notes.append("tie_break defaulted")
    kept: list[str] = []
    for key in raw:
        text = str(key)
        if text not in TIE_BREAKS:
            notes.append(f"dropped unknown tie_break key {text}")
            continue
        if text in kept:
            notes.append(f"dropped duplicate tie_break key {text}")
            continue
        kept.append(text)
    for key in _TIE_PAD:
        if len(kept) >= 3:
            break
        if key not in kept:
            kept.append(key)
            notes.append(f"padded tie_break with {key}")
    if len(kept) > 4:
        notes.append("truncated tie_break to 4 keys")
        kept = kept[:4]
    return kept


def _normalize_overrides(value: Any, notes: list[str]) -> list[dict[str, Any]]:
    if value is None:
        notes.append("class_overrides defaulted to []")
        return []
    if not isinstance(value, list):
        notes.append("class_overrides defaulted to []")
        return []
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped_extra = False
    for item in value:
        if len(kept) >= 2:
            dropped_extra = True
            continue
        if not isinstance(item, dict):
            notes.append("dropped non-object class override")
            continue
        job_class = item.get("job_class")
        if job_class not in JOB_CLASSES:
            notes.append(f"dropped class override with job_class {job_class}")
            continue
        if job_class in seen:
            notes.append(f"dropped duplicate class override {job_class}")
            continue
        seen.add(str(job_class))
        weights = _normalize_weights(item.get("score_weights"), notes, f"class_overrides.{job_class}")
        kept.append({"job_class": job_class, "score_weights": weights})
    if dropped_extra:
        notes.append("truncated class_overrides to 2")
    return kept


def normalize_cloud_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fill the fields Veri omits and drop tie-break keys that are not in the enum."""

    notes: list[str] = []
    if not isinstance(payload, dict):
        return payload, notes
    body = dict(payload)
    inner = body.get("payload")
    if isinstance(inner, dict) and "rules" not in body and "policy_family" not in body:
        notes.append("unwrapped payload")
        body = dict(inner)
    if body.get("policy_family") not in POLICY_FAMILIES:
        body["policy_family"] = "composed"
        notes.append("policy_family defaulted to composed")
    description = body.get("description")
    if not isinstance(description, str) or not description.strip():
        body["description"] = "normalized scheduling policy"
        notes.append("description defaulted")
    rules_in = body.get("rules")
    rules = dict(rules_in) if isinstance(rules_in, dict) else {}
    if not isinstance(rules_in, dict):
        notes.append("rules defaulted to an object")
    depth = rules.get("max_queue_depth")
    if isinstance(depth, bool) or not isinstance(depth, (int, float)):
        rules["max_queue_depth"] = 36
        notes.append("max_queue_depth defaulted to 36")
    else:
        coerced = int(round(float(depth)))
        if coerced != depth:
            notes.append("max_queue_depth coerced to int")
        if coerced < 1 or coerced > 36:
            notes.append("max_queue_depth clamped to 1..36")
        rules["max_queue_depth"] = min(36, max(1, coerced))
    rules["score_weights"] = _normalize_weights(rules.get("score_weights"), notes, "score_weights")
    rules["tie_break"] = _normalize_tie(rules.get("tie_break"), notes)
    rules["class_overrides"] = _normalize_overrides(rules.get("class_overrides"), notes)
    allowed = {"max_queue_depth", "score_weights", "tie_break", "class_overrides"}
    for key in list(rules):
        if key not in allowed:
            del rules[key]
            notes.append(f"dropped rules.{key}")
    body["rules"] = rules
    for key in list(body):
        if key not in {"policy_family", "rules", "description"}:
            del body[key]
            notes.append(f"dropped {key}")
    return body, notes


def _tie_sample(rng: random.Random) -> list[str]:
    keys = list(TIE_BREAKS)
    rng.shuffle(keys)
    return keys[: rng.choice((3, 4))]


def _biased_weights(rng: random.Random, bias: dict[str, float]) -> dict[str, float]:
    weights = {term: 0.0 for term in SCORE_TERMS}
    for term in SCORE_TERMS:
        if rng.random() < 0.35:
            weights[term] = round(rng.uniform(-0.2, 1.0), 3)
    for key, value in bias.items():
        weights[key] = round(max(-1.0, min(1.0, value + rng.uniform(-0.2, 0.2))), 3)
    return weights


def objective_local_payload(rng: random.Random) -> dict[str, Any]:
    """One train-only sample biased toward the published objective, not the test split."""

    bias = dict(_LOCAL_SAMPLE_BIAS)
    overrides: list[dict[str, Any]] = []
    classes = ["gpu", "batch", "high_mem", "sla"]
    rng.shuffle(classes)
    for job_class in classes[: rng.choice((1, 1, 2))]:
        tilted = dict(bias)
        if job_class == "gpu":
            tilted.update(class_affinity=1.0, short_job=0.25, hard_deadline=0.6)
        elif job_class == "batch":
            tilted.update(energy_frugal=0.85, class_affinity=-0.3, short_job=0.7)
        elif job_class == "high_mem":
            tilted.update(class_affinity=0.9, hard_deadline=0.5)
        overrides.append({"job_class": job_class, "score_weights": _biased_weights(rng, tilted)})
    payload = {
        "policy_family": "composed",
        "rules": {
            "max_queue_depth": rng.choice((18, 24, 36)),
            "score_weights": _biased_weights(rng, bias),
            "tie_break": _tie_sample(rng),
            "class_overrides": overrides,
        },
        "description": "local objective sample",
    }
    normalized, _notes = normalize_cloud_payload(payload)
    return normalized


def _nudge_weight(value: float, rng: random.Random) -> float:
    updated = float(value) + rng.choice((-0.4, -0.15, 0.15, 0.4))
    return round(max(-1.0, min(1.0, updated)), 3)


def perturb_cloud_payload(payload: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """One local edit of a feasible policy. Scored later on train only."""

    child = copy.deepcopy(payload)
    normalized, _notes = normalize_cloud_payload(child)
    child = normalized
    child["policy_family"] = "composed"
    child["description"] = "local perturbation"
    rules = child["rules"]
    kind = rng.randrange(5)
    if kind == 0:
        term = rng.choice(SCORE_TERMS)
        rules["score_weights"][term] = _nudge_weight(float(rules["score_weights"][term]), rng)
    elif kind == 1:
        rules["tie_break"] = _tie_sample(rng)
    elif kind == 2:
        rules["max_queue_depth"] = rng.choice((12, 18, 24, 36))
    elif kind == 3 and rules["class_overrides"]:
        override = rules["class_overrides"][rng.randrange(len(rules["class_overrides"]))]
        term = rng.choice(SCORE_TERMS)
        override["score_weights"][term] = _nudge_weight(float(override["score_weights"][term]), rng)
    else:
        used = {item["job_class"] for item in rules["class_overrides"]}
        free = [name for name in JOB_CLASSES if name not in used]
        if free and len(rules["class_overrides"]) < 2:
            rules["class_overrides"].append(
                {"job_class": rng.choice(free), "score_weights": _biased_weights(rng, _LOCAL_SAMPLE_BIAS)}
            )
        elif rules["class_overrides"] and rng.random() < 0.5:
            rules["class_overrides"].pop(rng.randrange(len(rules["class_overrides"])))
        else:
            term = rng.choice(SCORE_TERMS)
            rules["score_weights"][term] = _nudge_weight(float(rules["score_weights"][term]), rng)
    normalized, _notes = normalize_cloud_payload(child)
    return normalized


def _cloud_prompt_contract() -> str:
    example = baseline_payload("deadline_weighted")
    return (
        "rules.max_queue_depth is required (integer 1 to 36). "
        "rules.score_weights is required and must include every score term, each in [-1, 1]. "
        "rules.class_overrides is required. Use [] when there is no override, or 1 or 2 objects. "
        "Each override has job_class (gpu, high_mem, sla, or batch) and a full score_weights object. "
        "rules.tie_break is required and lists 3 or 4 keys. "
        f"Allowed tie_break keys only: {', '.join(TIE_BREAKS)}. "
        "Do not put weight names (short_job, class_affinity) or job classes (gpu, batch) in tie_break. "
        f"Example genome: {json.dumps(example, separators=(',', ':'))}"
    )


MUTATOR_RETURN = (
    'Return JSON exactly as {"children":[{"genome": <one payload object>, "rationale": "one line"}]}. '
    "No markdown and no prose outside the JSON."
)


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
    evaluator_version = EVALUATOR_VERSION

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

    def prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        return normalize_cloud_payload(payload)

    def random_seed_payloads(self, rng: random.Random, count: int) -> list[dict[str, Any]]:
        return [objective_local_payload(rng) for _ in range(max(0, count))]

    def local_perturbations(self, payload: dict[str, Any], rng: random.Random, count: int) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _ in range(max(0, count) * 4):
            if len(found) >= count:
                break
            child = perturb_cloud_payload(payload, rng)
            key = json.dumps(child.get("rules"), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            found.append(child)
        return found

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
            "Propose scheduling policies as compact JSON. "
            "Each candidate is {\"payload\": <genome>, \"hypothesis\": \"one line\"}. "
            "Several candidates use {\"candidates\": [{\"payload\": <genome>, \"hypothesis\": \"one line\"}]}. "
            f"{_cloud_prompt_contract()} "
            "Use class_affinity and hard_deadline together with short_job so accelerator "
            "jobs are not stuck behind short batch work, and keep batch work off the GPU. "
            "policy_family is one of round_robin, shortest_job, least_loaded, weighted_queue, composed. "
            "Do not emit Python.\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Change one weight, one tie_break key, or one class override. "
            f"{_cloud_prompt_contract()} "
            f"{MUTATOR_RETURN}\n"
            f"Parent: {parent.payload}\nFeedback: {feedback.feedback}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Combine the two policies. Average weights. "
            "rules.class_overrides is a required array of 0, 1, or 2 objects. "
            f"{_cloud_prompt_contract()} "
            f"{MUTATOR_RETURN}\n"
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
        totals = {
            "mean_completion": 0.0,
            "sla_miss_rate": 0.0,
            "hard_tardiness": 0.0,
            "energy": 0.0,
            "completion_ratio": 0.0,
            "hard_ratio": 0.0,
            "energy_ratio": 0.0,
            "gpu_foreign_work": 0.0,
            "highmem_foreign_work": 0.0,
        }
        indexes = SPLIT_INDEXES[split]
        for index in indexes:
            jobs = workload(self._seed, index)
            metrics = simulate(genome.payload, jobs)
            if metrics["violations"] > 0 or metrics["completed"] < len(jobs):
                return self._infeasible(genome, split, "capacity violation or unfinished jobs", started)
            reference = round_robin_metrics(self._seed, index)
            totals["mean_completion"] += metrics["mean_completion"]
            totals["sla_miss_rate"] += metrics["sla_miss_rate"]
            totals["hard_tardiness"] += metrics["hard_tardiness"]
            totals["energy"] += metrics["energy"]
            totals["gpu_foreign_work"] += metrics["gpu_foreign_work"]
            totals["highmem_foreign_work"] += metrics["highmem_foreign_work"]
            totals["completion_ratio"] += metrics["mean_completion"] / max(reference["mean_completion"], 1e-9)
            ref_tardiness = reference["hard_tardiness"]
            if ref_tardiness < 1e-3:
                totals["hard_ratio"] += metrics["hard_tardiness"] / max(metrics["mean_hard_duration"], 1e-6)
            else:
                totals["hard_ratio"] += metrics["hard_tardiness"] / ref_tardiness
            totals["energy_ratio"] += metrics["energy"] / max(reference["energy"], 1e-9)
        count = float(len(indexes))
        mean_completion = totals["mean_completion"] / count
        sla = totals["sla_miss_rate"] / count
        hard_tardiness = totals["hard_tardiness"] / count
        energy = totals["energy"] / count
        completion_ratio = totals["completion_ratio"] / count
        hard_ratio = totals["hard_ratio"] / count
        energy_ratio = totals["energy_ratio"] / count
        fitness = -(completion_ratio + 0.25 * sla + 0.50 * hard_ratio + 0.15 * energy_ratio)
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=fitness,
            metrics={
                "mean_completion": mean_completion,
                "completion_ratio_vs_round_robin": completion_ratio,
                "sla_miss_rate": sla,
                "hard_tardiness": hard_tardiness,
                "hard_tardiness_ratio_vs_round_robin": hard_ratio,
                "energy": energy,
                "energy_ratio_vs_round_robin": energy_ratio,
                "gpu_foreign_work": totals["gpu_foreign_work"] / count,
                "highmem_foreign_work": totals["highmem_foreign_work"] / count,
                "violations": 0.0,
                "workloads": count,
            },
            feedback=(
                f"{split} mean_completion={mean_completion:.4f} "
                f"vs_round_robin={completion_ratio:.4f} sla_miss={sla:.4f} "
                f"hard_tardiness_ratio={hard_ratio:.4f} energy_ratio={energy_ratio:.4f}. "
                "Lower completion, hard-deadline tardiness, and energy is better. "
                "Capacity violations are infeasible."
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
            "Schedule jobs onto a heterogeneous pool: CPU nodes, one high-memory node, and one GPU. "
            "Hard constraint: never exceed a machine's CPU, memory, or GPU count. "
            "GPU jobs run only on the GPU. High-memory jobs run only where they fit. "
            "Reduce mean completion time, SLA misses, hard-deadline tardiness, and energy "
            "relative to round robin. Energy is charged at the machine's full draw while it is busy."
        ),
        constraints={"machines": len(MACHINES), "capacity_violations": 0},
        evaluation_version=EVALUATOR_VERSION,
    )
