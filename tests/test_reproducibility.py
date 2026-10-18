from __future__ import annotations

import json
from pathlib import Path

import pytest

from geneai_core.experiments import execute_toy_run


def _projection(export: dict) -> dict:
    return {
        "genomes": [
            {
                "id": item["id"],
                "generation": item["generation"],
                "operator": item["operator"],
                "parent_ids": item["parent_ids"],
                "payload": item["payload"],
                "content_hash": item["content_hash"],
                "provider": item["provider"],
                "role": item["metadata"].get("role"),
            }
            for item in export["genomes"]
        ],
        "train": [
            {
                "genome_id": item["genome_id"],
                "fitness": item["fitness"],
                "feasible": item["feasible"],
                "metrics": item["metrics"],
            }
            for item in export["evaluations"]
            if item["data_split"] == "train"
        ],
        "event_types": [item["type"] for item in export["events"]],
    }


@pytest.mark.asyncio
async def test_same_seed_is_reproducible(tmp_path: Path) -> None:
    first = await execute_toy_run(mode="mock", seed=42, population_size=8, generations=3, root=tmp_path / "a")
    second = await execute_toy_run(mode="mock", seed=42, population_size=8, generations=3, root=tmp_path / "b")
    left = json.loads(Path(first.export_path).read_text(encoding="utf-8"))
    right = json.loads(Path(second.export_path).read_text(encoding="utf-8"))
    assert first.run_id != second.run_id
    assert _projection(left) == _projection(right)


@pytest.mark.asyncio
async def test_different_seed_changes_the_population(tmp_path: Path) -> None:
    first = await execute_toy_run(mode="mock", seed=1, population_size=6, generations=2, root=tmp_path / "a")
    second = await execute_toy_run(mode="mock", seed=2, population_size=6, generations=2, root=tmp_path / "b")
    left = json.loads(Path(first.export_path).read_text(encoding="utf-8"))
    right = json.loads(Path(second.export_path).read_text(encoding="utf-8"))
    assert _projection(left)["genomes"] != _projection(right)["genomes"]
