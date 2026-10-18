from __future__ import annotations

import json
from pathlib import Path

import pytest

from geneai_core.controller import EvolutionController
from geneai_core.domains.toy import (
    INSTANCES_PER_SPLIT,
    SHIFT_HALF_WIDTH,
    ToyOptimizationAdapter,
    split_shifts,
)
from geneai_core.experiments import execute_toy_run, toy_objective
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.schemas import EvolutionConfig
from geneai_core.store import RunRepository, verify_export


def test_toy_held_out_split_matches_the_train_distribution() -> None:
    groups = {name: split_shifts(42, name) for name in ("train", "validation", "test")}
    assert [len(shifts) for shifts in groups.values()] == [INSTANCES_PER_SPLIT] * 3
    pooled = [shift for shifts in groups.values() for shift in shifts]
    assert len(set(pooled)) == len(pooled)
    assert all(abs(coord) <= SHIFT_HALF_WIDTH + 1e-9 for shift in pooled for coord in shift)
    adapter = ToyOptimizationAdapter(seed=42)
    genome = type(
        "Genome",
        (),
        {
            "id": "g0001",
            "payload": {
                "family": "weighted_policy",
                "params": {"x": 0.0, "y": 0.0},
                "description": "origin",
            },
        },
    )()
    scores = {name: adapter.evaluate(genome, name).fitness for name in groups}
    assert all(score is not None and score > 0 for score in scores.values())
    assert max(scores.values()) / min(scores.values()) < 3


@pytest.mark.asyncio
async def test_five_generation_mock_run(tmp_path: Path) -> None:
    result = await execute_toy_run(
        mode="mock",
        seed=42,
        population_size=12,
        generations=5,
        root=tmp_path,
    )
    assert result.status == "completed"
    assert result.stop_reason == "generation_limit"
    assert result.generations_completed == 5
    assert result.summary["mutation_children_evaluated"] >= 1
    assert result.summary["crossover_children_evaluated"] >= 1
    assert result.summary["elite_preserved"]
    export = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert verify_export(export)
    assert export["run"]["mode"] == "mock"
    assert export["one_shot_baseline"] is None
    genomes = {item["id"]: item for item in export["genomes"]}
    operators = {item["operator"] for item in export["genomes"]}
    assert "mutation" in operators
    assert "crossover" in operators
    crossovers = [item for item in export["genomes"] if item["operator"] == "crossover"]
    assert crossovers
    assert all(len(item["parent_ids"]) == 2 for item in crossovers)
    mutations = [item for item in export["genomes"] if item["operator"] == "mutation"]
    assert all(len(item["parent_ids"]) == 1 for item in mutations)
    train_fitness = []
    for evaluation in export["evaluations"]:
        genome = genomes[evaluation["genome_id"]]
        if evaluation["data_split"] == "train" and genome["generation"] == 0 and evaluation["fitness"] is not None:
            train_fitness.append(evaluation["fitness"])
    assert len(set(train_fitness)) > 1
    test_evals = [item for item in export["evaluations"] if item["data_split"] == "test"]
    assert len(test_evals) == 1
    assert test_evals[0]["genome_id"] == result.champion_id
    types = [item["type"] for item in export["events"]]
    assert types.index("champion.frozen") < types.index("champion.tested")
    frozen_at = types.index("champion.frozen")
    assert "test" not in [
        (item.get("payload") or {}).get("split")
        for item in export["events"][:frozen_at]
        if item["type"] == "genome.evaluated"
    ]
    for row in result.summary["elite_preserved"]:
        stored = genomes[row["genome_id"]]
        assert stored["content_hash"] == row["content_hash"]
        assert stored["id"] in {
            genome_id
            for event in export["events"]
            if event["type"] == "generation.completed"
            for genome_id in event["payload"]["population_ids"]
        }


@pytest.mark.asyncio
async def test_malformed_and_duplicate_genomes_do_not_crash(tmp_path: Path) -> None:
    class ChaosModel(MockModelProvider):
        async def propose(self, instruction, schema, *, call_id):
            proposal = await super().propose(instruction, schema, call_id=call_id)
            if self.calls % 4 == 0:
                proposal.payload = {"nope": True}
            return proposal

    class ChaosAgent(MockAgentProvider):
        def __init__(self) -> None:
            super().__init__()
            self._fixed: dict | None = None

        async def run_workflow(self, workflow_name, inputs, *, call_id):
            proposal = await super().run_workflow(workflow_name, inputs, call_id=call_id)
            if self._fixed is None:
                self._fixed = {
                    "family": "weighted_policy",
                    "params": {"x": 0.2, "y": -0.6},
                    "description": "duplicate probe",
                }
                proposal.payload = dict(self._fixed)
                return proposal
            if self.calls % 3 == 0:
                proposal.payload = dict(self._fixed)
            return proposal

    config = EvolutionConfig(seed=42, population_size=6, generations=2, elite_count=1, max_model_calls=80)
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=42),
        config=config,
        providers={"model": ChaosModel(), "workflow": ChaosAgent()},
        mode="mock",
        repository=repository,
    )
    assert result.status == "completed"
    reasons = [
        event.payload.get("reason_code")
        for event in repository.events
        if event.type == "genome.rejected"
    ]
    assert "invalid_schema" in reasons
    assert "duplicate" in reasons


@pytest.mark.asyncio
async def test_cli_mock_run(tmp_path: Path) -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_toy.py",
            "--mode",
            "mock",
            "--seed",
            "42",
            "--population",
            "6",
            "--generations",
            "2",
            "--out",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["generations_completed"] == 2
    assert summary["mode"] == "mock"
    assert Path(summary["export_path"]).exists()
