from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from geneai_core.controller import EvolutionController
from geneai_core.domains.base import DomainAdapter
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.schemas import Evaluation, EvolutionConfig, Genome, ObjectiveSpec
from geneai_core.store import RunRepository


class _NumericAdapter(DomainAdapter):
    def __init__(self, name: str, target: float) -> None:
        self.name = name
        self.evaluator_version = f"{name}-v1"
        self._target = target

    def genome_schema(self) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "value", "label"],
            "properties": {
                "kind": {"type": "string", "enum": [self.name]},
                "value": {"type": "number", "minimum": -5, "maximum": 5},
                "label": {"type": "string"},
            },
        }

    def family_of(self, payload: dict) -> str:
        return str(payload.get("kind") or self.name)

    def validate(self, payload: dict, objective: ObjectiveSpec) -> None:
        del objective
        if payload.get("kind") != self.name:
            raise ValueError("kind mismatch")
        value = payload.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be numeric")
        if not -5 <= float(value) <= 5:
            raise ValueError("value out of range")
        if not isinstance(payload.get("label"), str) or not payload["label"]:
            raise ValueError("label required")

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return f"Seed a {self.name} candidate for: {objective.prompt}"

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return f"Mutate {self.name} parent {parent.payload}. Feedback: {feedback.feedback}"

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return f"Cross {self.name} {a.payload} with {b.payload}"

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        value = float(genome.payload["value"])
        fitness = -abs(value - self._target)
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=fitness,
            metrics={"distance": abs(value - self._target)},
            feedback=f"{split} distance {abs(value - self._target):.4f}",
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=0,
        )

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict:
        return {"name": self.name, "value": genome.payload["value"], "fitness": evaluation.fitness}


@pytest.mark.asyncio
async def test_two_fake_adapters_share_one_controller(tmp_path: Path) -> None:
    controller = EvolutionController()
    champions = {}
    for name, target in (("alpha_contract", 1.25), ("beta_contract", -2.0)):
        adapter = _NumericAdapter(name, target)
        repository = RunRepository(tmp_path / name)
        result = await controller.evolve(
            objective=ObjectiveSpec(
                domain=name,
                prompt=f"move value toward {target}",
                evaluation_version=adapter.evaluator_version,
            ),
            adapter=adapter,
            config=EvolutionConfig(seed=11, population_size=6, generations=2, elite_count=1),
            providers={"model": MockModelProvider(), "workflow": MockAgentProvider()},
            mode="mock",
            repository=repository,
        )
        assert result.status == "completed"
        assert result.champion_id
        assert result.summary["domain"] == name
        champions[name] = result.summary["champion_payload"]["kind"]
    assert champions == {"alpha_contract": "alpha_contract", "beta_contract": "beta_contract"}


def test_controller_has_no_domain_or_provider_imports() -> None:
    source = inspect.getsource(EvolutionController)
    module_source = Path(inspect.getfile(EvolutionController)).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    banned = ("toy", "shopping", "cloud", "quant", "veri", "creao", "frontier")
    for module_name in imported:
        lowered = module_name.lower()
        for word in banned:
            assert word not in lowered, module_name
    for word in ("shirt", "ticker", "lululemon"):
        assert word not in source.lower()
        assert word not in module_source.lower()
