"""Hybrid-run failure modes, with mocked provider latency.

The live failure was serial seed calls, duplicate repairs, and a champion
frozen before validation because the wall clock had already tripped.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from geneai_core.controller import EvolutionController
from geneai_core.domains.cloud import CloudSchedulingAdapter, cloud_objective
from geneai_core.domains.toy import ToyOptimizationAdapter
from geneai_core.experiments import toy_objective
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.schemas import EvolutionConfig
from geneai_core.store import RunRepository


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _WallModel(MockModelProvider):
    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self.clock = clock

    async def propose(self, instruction, schema, *, call_id):
        self.clock.now += 20
        return await super().propose(instruction, schema, call_id=call_id)


class _DuplicateModel(MockModelProvider):
    async def propose(self, instruction, schema, *, call_id):
        proposal = await super().propose(instruction, schema, call_id=call_id)
        proposal.payload = {
            "family": "weighted_policy",
            "params": {"x": 0.1, "y": 0.1},
            "description": "same seed",
        }
        proposal.extra = {}
        return proposal


class _LatencyModel(MockModelProvider):
    async def propose(self, instruction, schema, *, call_id):
        await asyncio.sleep(3)
        return await super().propose(instruction, schema, call_id=call_id)


class _LatencyAgent(MockAgentProvider):
    def __init__(self) -> None:
        super().__init__(model=MockModelProvider())

    async def run_workflow(self, workflow_name, inputs, *, call_id):
        await asyncio.sleep(35)
        return await super().run_workflow(workflow_name, inputs, call_id=call_id)


@pytest.mark.asyncio
async def test_wall_stop_still_scores_validation(tmp_path) -> None:
    clock = _Clock()
    repository = RunRepository(tmp_path)
    result = await EvolutionController(clock=clock).evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=EvolutionConfig(
            seed=42,
            population_size=4,
            generations=3,
            elite_count=1,
            max_wall_seconds=15,
            max_model_calls=40,
            max_evaluations=80,
        ),
        providers={"model": _WallModel(clock), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    assert result.stop_reason == "max_wall_seconds"
    assert result.generations_completed == 0
    assert result.summary["champion_validation_fitness"] is not None
    assert result.champion_id
    frozen = [event for event in repository.events if event.type == "champion.frozen"]
    assert frozen and frozen[0].payload.get("split") == "validation"
    assert frozen[0].payload.get("basis") == "validation"


@pytest.mark.asyncio
async def test_duplicate_seed_does_not_schedule_a_repair(tmp_path) -> None:
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=EvolutionConfig(
            seed=42,
            population_size=4,
            generations=1,
            elite_count=1,
            max_model_calls=12,
            max_evaluations=40,
        ),
        providers={"model": _DuplicateModel(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    reasons = [event.payload.get("reason_code") for event in repository.events if event.type == "genome.rejected"]
    assert "duplicate" in reasons
    assert "repair_scheduled" not in reasons
    assert result.summary["champion_validation_fitness"] is not None


@pytest.mark.asyncio
async def test_mocked_hybrid_latency_finishes_twelve_by_five(tmp_path) -> None:
    repository = RunRepository(tmp_path)
    started = time.perf_counter()
    result = await EvolutionController().evolve(
        objective=cloud_objective(),
        adapter=CloudSchedulingAdapter(seed=42),
        config=EvolutionConfig(
            seed=42,
            population_size=12,
            generations=5,
            elite_count=2,
            strategist_calls_per_generation=2,
            max_concurrent_model_calls=8,
            max_model_calls=800,
            max_evaluations=900,
            max_wall_seconds=900,
        ),
        providers={"model": _LatencyModel(), "workflow": _LatencyAgent()},
        mode="mock",
        repository=repository,
    )
    elapsed = time.perf_counter() - started
    assert result.generations_completed == 5
    assert result.stop_reason == "generation_limit"
    assert result.summary["champion_validation_fitness"] is not None
    assert elapsed < 360
    assert result.summary["budget"]["elapsed_seconds"] < 360
