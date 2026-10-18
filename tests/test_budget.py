from __future__ import annotations

import json
from pathlib import Path

import pytest

from geneai_core.budgets import Budget
from geneai_core.controller import EvolutionController
from geneai_core.domains.toy import ToyOptimizationAdapter
from geneai_core.experiments import toy_objective
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.schemas import EvolutionConfig
from geneai_core.store import RunRepository, verify_export


def _config(**kwargs: object) -> EvolutionConfig:
    base = dict(seed=42, population_size=6, generations=3, elite_count=1, max_wall_seconds=300)
    base.update(kwargs)
    return EvolutionConfig(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_model_call_cap_returns_partial_export(tmp_path: Path) -> None:
    model = MockModelProvider()
    workflow = MockAgentProvider()
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=_config(max_model_calls=2, max_evaluations=50),
        providers={"model": model, "workflow": workflow},
        mode="mock",
        repository=repository,
    )
    assert result.status == "completed"
    assert result.stop_reason == "max_model_calls"
    assert model.calls + workflow.calls <= 2
    assert result.summary["budget"]["model_calls"] <= 2
    export = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert verify_export(export)
    assert export["run"]["stop_reason"] == "max_model_calls"
    assert export["genomes"]


@pytest.mark.asyncio
async def test_evaluation_cap_stops_new_evaluations(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=_config(max_model_calls=30, max_evaluations=2),
        providers={"model": MockModelProvider(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    assert result.status == "completed"
    assert result.stop_reason == "max_evaluations"
    assert result.summary["budget"]["evaluations"] <= 2
    assert not any(item.data_split == "test" for item in repository.evaluations)


@pytest.mark.asyncio
async def test_spend_cap_stops_new_provider_calls(tmp_path: Path) -> None:
    model = MockModelProvider(cost_per_call=1.0)
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=_config(max_model_calls=20, max_evaluations=40, max_usd_estimate=2.0),
        providers={"model": model, "workflow": MockAgentProvider(cost_per_call=1.0)},
        mode="mock",
        repository=repository,
    )
    assert result.stop_reason == "max_usd_estimate"
    assert result.summary["budget"]["spend_usd"] <= 2.0 + 1e-9
    assert result.summary["budget"]["model_calls"] <= 2


@pytest.mark.asyncio
async def test_cancel_emits_cancelled(tmp_path: Path) -> None:
    from geneai_core.budgets import Budget

    budget = Budget(_config(max_model_calls=40, max_evaluations=40))

    class CancelModel(MockModelProvider):
        async def propose(self, instruction, schema, *, call_id):
            proposal = await super().propose(instruction, schema, call_id=call_id)
            if self.calls >= 3:
                budget.cancel()
            return proposal

    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=budget.config,
        providers={"model": CancelModel(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
        budget=budget,
    )
    assert result.status == "cancelled"
    assert result.stop_reason == "cancelled"
    assert any(event.type == "run.cancelled" for event in repository.events)
    assert not any(event.type == "run.completed" for event in repository.events)


def test_wall_clock_budget_blocks() -> None:
    budget = Budget(EvolutionConfig(max_wall_seconds=5, max_model_calls=10, max_evaluations=10))
    budget.started -= 10
    assert budget.allows_model_call() is False
    assert budget.exhausted_reason == "max_wall_seconds"
    assert budget.allows_evaluation() is False
