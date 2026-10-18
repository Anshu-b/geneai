"""Deterministic toy objective used to prove the engine.

The phenotype is a point in a shifted Rastrigin basin. Train, validation, and
test use disjoint shifts derived from the run seed so selection cannot peek at
the held-out test. Families are different parametrizations of that point, not
separate engines.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

BOUNDS = (-5.12, 5.12)
FAMILIES = ("weighted_policy", "offset_policy", "scaled_policy")
COMPLEXITY = {"weighted_policy": 0.02, "offset_policy": 0.04, "scaled_policy": 0.03}
LAMBDA_COST = 0.05
LAMBDA_COMPLEXITY = 0.5


def _shift(seed: int, split: str) -> tuple[float, float]:
    # Disjoint, deterministic basins. The numbers are part of the dataset hash.
    digest = hashlib.sha256(f"toy-shift-v1|{seed}|{split}".encode("utf-8")).digest()
    first = (int.from_bytes(digest[:4], "big") / 2**32) * 1.6 - 0.8
    second = (int.from_bytes(digest[4:8], "big") / 2**32) * 1.6 - 0.8
    return (round(first, 6), round(second, 6))


def phenotype(payload: dict[str, Any]) -> tuple[float, float]:
    x = float(payload["params"]["x"])
    y = float(payload["params"]["y"])
    family = payload["family"]
    if family == "offset_policy":
        x, y = x + 0.2, y - 0.15
    elif family == "scaled_policy":
        x, y = 0.75 * x, 0.75 * y
    lo, hi = BOUNDS
    return (min(hi, max(lo, x)), min(hi, max(lo, y)))


def rastrigin(x: float, y: float, shift: tuple[float, float]) -> float:
    ax = x - shift[0]
    ay = y - shift[1]
    return 20.0 + (ax * ax - 10.0 * math.cos(2.0 * math.pi * ax)) + (
        ay * ay - 10.0 * math.cos(2.0 * math.pi * ay)
    )


class ToyOptimizationAdapter(DomainAdapter):
    name = "toy_optimization"
    evaluator_version = "toy-rastrigin-v1"

    def genome_schema(self) -> dict[str, Any]:
        lo, hi = BOUNDS
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["family", "params", "description"],
            "properties": {
                "family": {"type": "string", "enum": list(FAMILIES)},
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y"],
                    "properties": {
                        "x": {"type": "number", "minimum": lo, "maximum": hi},
                        "y": {"type": "number", "minimum": lo, "maximum": hi},
                    },
                },
                "description": {"type": "string"},
            },
        }

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        self._seed = seed

        def digest(split: str) -> str:
            shift = _shift(seed, split)
            raw = f"{self.evaluator_version}|{split}|{seed}|{shift[0]}|{shift[1]}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {
            "train_hash": digest("train"),
            "validation_hash": digest("validation"),
            "test_hash": digest("test"),
        }

    def family_of(self, payload: dict[str, Any]) -> str:
        return str(payload.get("family") or "default")

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        del objective
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        extra = set(payload) - {"family", "params", "description"}
        if extra:
            raise ValueError(f"unknown payload keys: {sorted(extra)}")
        family = payload.get("family")
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        extra_params = set(params) - {"x", "y"}
        if extra_params:
            raise ValueError(f"unknown param keys: {sorted(extra_params)}")
        for name in ("x", "y"):
            value = params.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"params.{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"params.{name} must be finite")
            if not BOUNDS[0] <= float(value) <= BOUNDS[1]:
                raise ValueError(f"params.{name} out of range {BOUNDS}")
        if not isinstance(payload.get("description"), str) or not payload["description"].strip():
            raise ValueError("description must be a non-empty string")

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return (
            "Propose a diverse seed strategy for this toy search. "
            "Return JSON only, with keys payload and hypothesis. "
            "payload.family is weighted_policy, offset_policy, or scaled_policy. "
            "payload.params.x and payload.params.y are numbers inside [-5.12, 5.12]. "
            "Spread candidates across the square; do not repeat a point.\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Mutate exactly one coherent feature of the parent (one of x, y, or a small "
            "family-preserving adjustment). Stay inside [-5.12, 5.12]. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent payload: {parent.payload}\n"
            f"Train feedback: {feedback.feedback}\n"
            f"Train fitness: {feedback.fitness}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Combine the two parents into one child. Average or inherit their x and y "
            "and choose one family. Stay inside [-5.12, 5.12]. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent A: {a.payload}\n"
            f"Parent B: {b.payload}"
        )

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        started = time.perf_counter()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {split}")
        try:
            self.validate(genome.payload, ObjectiveSpec(domain=self.name, prompt="", evaluation_version=self.evaluator_version))
        except ValueError as exc:
            return Evaluation(
                genome_id=genome.id,
                feasible=False,
                fitness=None,
                metrics={},
                constraint_violations=[str(exc)],
                feedback=f"infeasible: {exc}",
                evaluator_version=self.evaluator_version,
                data_split=split,  # type: ignore[arg-type]
                elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
            )
        shift = _shift(self._seed, split)
        x, y = phenotype(genome.payload)
        raw = rastrigin(x, y, shift)
        quality = 1.0 / (1.0 + raw)
        cost = (abs(x) + abs(y)) / 10.24
        complexity = COMPLEXITY[genome.payload["family"]]
        fitness = quality - LAMBDA_COST * cost - LAMBDA_COMPLEXITY * complexity
        metrics = {
            "quality": quality,
            "rastrigin": raw,
            "normalized_cost": cost,
            "complexity": complexity,
            "phenotype_x": x,
            "phenotype_y": y,
        }
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=fitness,
            metrics=metrics,
            constraint_violations=[],
            feedback=(
                f"{split} quality={quality:.4f} rastrigin={raw:.4f} "
                f"cost={cost:.4f} complexity={complexity:.3f}. "
                "Move the phenotype toward the basin without leaving the bounds."
            ),
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]:
        return {
            "domain": self.name,
            "genome_id": genome.id,
            "family": genome.payload.get("family"),
            "params": genome.payload.get("params"),
            "description": genome.payload.get("description"),
            "feasible": evaluation.feasible,
            "fitness": evaluation.fitness,
            "metrics": evaluation.metrics,
            "split": evaluation.data_split,
            "hypothesis": genome.hypothesis,
        }

    def bind_seed(self, seed: int) -> None:
        self._seed = seed

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
