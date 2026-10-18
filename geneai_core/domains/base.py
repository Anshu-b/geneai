"""Domain adapter contract. Concrete domains live in their own modules.

The evolution controller depends only on this interface.
"""

from __future__ import annotations

import hashlib
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

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        def digest(split: str) -> str:
            raw = f"{self.name}|{self.evaluator_version}|{split}|{seed}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {
            "train_hash": digest("train"),
            "validation_hash": digest("validation"),
            "test_hash": digest("test"),
        }
