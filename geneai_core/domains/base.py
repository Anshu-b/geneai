"""Domain adapter contract. Concrete domains live in their own modules.

The evolution controller depends only on this interface.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from typing import Any

from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec


class DomainAdapter(ABC):
    name: str
    evaluator_version: str

    @abstractmethod
    def genome_schema(self) -> dict[str, Any]:
        """JSON schema for the domain payload (not the provider wrapper)."""

    @abstractmethod
    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        """Raise ValueError when the payload is illegal. Do not score it."""

    @abstractmethod
    def seed_instructions(self, objective: ObjectiveSpec) -> str: ...

    @abstractmethod
    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str: ...

    @abstractmethod
    def crossover_instructions(self, a: Genome, b: Genome) -> str: ...

    @abstractmethod
    def evaluate(self, genome: Genome, split: str) -> Evaluation: ...

    @abstractmethod
    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]: ...

    def family_of(self, payload: dict[str, Any]) -> str:
        family = payload.get("family")
        if family is None:
            family = payload.get("kind") or payload.get("species") or "default"
        return str(family)

    def species_of(self, payload: dict[str, Any]) -> str:
        return self.family_of(payload)

    def prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Deterministic cleanup before validation. Notes are provenance."""

        return payload, []

    def realize_proposal(
        self,
        payload: dict[str, Any],
        parent_payloads: list[dict[str, Any]],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Turn a short operator edit into a domain payload. Identity by default."""

        del parent_payloads, rng
        return payload

    def random_seed_payloads(self, rng: random.Random, count: int) -> list[dict[str, Any]]:
        """Extra generation-0 payloads labeled provider=random. Empty by default."""

        del rng, count
        return []

    def local_perturbations(self, payload: dict[str, Any], rng: random.Random, count: int) -> list[dict[str, Any]]:
        """Small edits of an elite, labeled provider=random. Empty by default."""

        del payload, rng, count
        return []

    def uses_memetic_fallback(self) -> bool:
        """Algorithmic children fill every generation, and model calls use a short slice."""

        return False

    def algorithmic_offspring(
        self,
        payloads: list[dict[str, Any]],
        rng: random.Random,
        count: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Cheap domain operators. Each pair is (operator name, payload)."""

        del payloads, rng, count
        return []

    def alter_duplicate(self, payload: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        """A different behavior when a child copies an existing genome.

        None leaves the controller's duplicate rejection in place.
        """

        del payload, rng
        return None

    def seeded_baselines(self) -> list[dict[str, Any]]:
        """Generation-0 payloads inserted before model seeds.

        Each item has ``name``, ``payload``, and ``hypothesis``. The controller
        labels them as seeded baselines. Domains without hand baselines return
        an empty list.
        """

        return []

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        def digest(split: str) -> str:
            raw = f"{self.name}|{self.evaluator_version}|{split}|{seed}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {
            "train_hash": digest("train"),
            "validation_hash": digest("validation"),
            "test_hash": digest("test"),
        }
