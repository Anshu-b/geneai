from __future__ import annotations

import json
from pathlib import Path

import pytest

from geneai_core.experiments import execute_toy_run
from geneai_core.schemas import EVENT_TYPES
from geneai_core.store import verify_export


def _edges(genomes: list[dict]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for genome in genomes:
        for parent_id in genome["parent_ids"]:
            edges.add((parent_id, genome["id"]))
    return edges


def _assert_acyclic(genomes: dict[str, dict]) -> None:
    visiting: set[str] = set()

    def walk(genome_id: str) -> None:
        if genome_id in visiting:
            raise AssertionError(f"cycle at {genome_id}")
        visiting.add(genome_id)
        genome = genomes[genome_id]
        for parent_id in genome["parent_ids"]:
            assert parent_id in genomes
            assert genomes[parent_id]["generation"] < genome["generation"]
            walk(parent_id)
        visiting.remove(genome_id)

    for genome_id in genomes:
        walk(genome_id)


@pytest.mark.asyncio
async def test_jsonl_and_export_reconstruct_the_same_dag(tmp_path: Path) -> None:
    result = await execute_toy_run(mode="mock", seed=7, population_size=8, generations=3, root=tmp_path)
    export = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert verify_export(export)
    events = [json.loads(line) for line in Path(result.events_path).read_text(encoding="utf-8").splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert [event["type"] for event in events] == [event["type"] for event in export["events"]]
    assert {event["type"] for event in events} <= EVENT_TYPES
    genomes = {item["id"]: item for item in export["genomes"]}
    _assert_acyclic(genomes)
    created = []
    for event in events:
        if event["type"] in {"genome.seeded", "genome.mutated", "genome.crossed"}:
            created.append((event["payload"]["genome_id"], tuple(event["payload"]["parent_ids"])))
    export_edges = _edges(export["genomes"])
    event_edges = {(parent, child) for child, parents in created for parent in parents}
    assert event_edges == export_edges
    assert any(event["type"] == "selection.pruned" for event in events)
    assert any(event["type"] == "selection.elite" for event in events)
    assert any(event["type"] == "selection.tournament" for event in events)
    for event in events:
        if event["type"] in {"genome.seeded", "genome.mutated", "genome.crossed", "genome.evaluated"}:
            payload = event["payload"]
            assert "parent_ids" in payload
            assert "operator" in payload
            assert "family" in payload
            assert "species" in payload
            assert "provenance" in event
            assert event["provenance"]["mode"] == "mock"


@pytest.mark.asyncio
async def test_immigrants_have_no_parents(tmp_path: Path) -> None:
    from geneai_core.controller import EvolutionController
    from geneai_core.domains.toy import ToyOptimizationAdapter
    from geneai_core.experiments import toy_objective
    from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
    from geneai_core.schemas import EvolutionConfig
    from geneai_core.store import RunRepository

    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=3),
        config=EvolutionConfig(
            seed=3,
            population_size=6,
            generations=1,
            elite_count=1,
            mutation_probability=0.0,
            crossover_probability=0.0,
            random_immigrant_probability=1.0,
        ),
        providers={"model": MockModelProvider(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    assert result.status == "completed"
    export = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    immigrants = [item for item in export["genomes"] if item["metadata"].get("role") == "immigrant"]
    assert immigrants
    assert all(item["parent_ids"] == [] for item in immigrants)
    assert all(item["operator"] == "seed" for item in immigrants)
