"""15-city TSP operators and the mock homepage benchmark."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from itertools import permutations
from pathlib import Path

import pytest

from geneai_core.controller import EvolutionController
from geneai_core.diversity import content_hash
from geneai_core.domains.tsp import (
    HELD_OUT_SEEDS,
    N_CITIES,
    SHOWCASE_SEED,
    TspAdapter,
    canonical_tour,
    cities_for_seed,
    distance_matrix,
    edge_recombination,
    dfs_bnb,
    held_karp,
    is_permutation,
    order_crossover,
    random_swap,
    random_two_opt,
    repair_tour,
    run_classic_ga,
    tour_length,
    tsp_objective,
    two_opt_improve,
)
from geneai_core.providers.base import Proposal
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.schemas import EvolutionConfig, ObjectiveSpec
from geneai_core.store import RunRepository
import random


def _brute_length(cities: list[dict]) -> float:
    dist = distance_matrix(cities)
    n = len(cities)
    best = float("inf")
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        best = min(best, tour_length(list(tour), dist))
    return best


def test_held_karp_matches_brute_force_and_is_a_cycle() -> None:
    cities = cities_for_seed(7, n=8)
    tour, length = held_karp(cities)
    assert is_permutation(tour, 8)
    assert length == pytest.approx(_brute_length(cities), rel=1e-9, abs=1e-6)
    assert tour_length(tour, distance_matrix(cities)) == pytest.approx(length)


def test_dfs_bnb_matches_held_karp_on_showcase_and_held_out() -> None:
    """Exact DFS equals Held-Karp on the 15-city showcase and five held-out maps."""

    for seed in (SHOWCASE_SEED, *HELD_OUT_SEEDS):
        cities = cities_for_seed(seed)
        hk_tour, hk_length = held_karp(cities)
        result = dfs_bnb(cities)
        assert is_permutation(result["tour"], N_CITIES)
        assert result["length"] == pytest.approx(hk_length, rel=1e-9, abs=1e-6)
        assert tour_length(result["tour"], distance_matrix(cities)) == pytest.approx(result["length"])
        assert result["nodes_explored"] > 0
        assert result["time_ms"] >= 0
        del hk_tour


def test_canonical_tour_dedups_rotation_and_reversal() -> None:
    tour = list(range(N_CITIES))
    rotated = tour[4:] + tour[:4]
    reflected = [0] + list(reversed(tour[1:]))
    assert canonical_tour(tour) == canonical_tour(rotated) == canonical_tour(reflected)
    assert content_hash({"tour": tour, "description": "alpha"}) == content_hash(
        {"tour": rotated, "description": "beta"}
    )
    assert content_hash({"tour": tour}) == content_hash({"tour": reflected, "description": "other"})
    swapped = random_swap(tour, random.Random(1))
    assert content_hash({"tour": tour}) != content_hash({"tour": swapped})


def test_repair_fixes_a_bad_permutation_and_validate_does_not() -> None:
    adapter = TspAdapter(SHOWCASE_SEED)
    objective = tsp_objective(adapter)
    raw = {"tour": [0, 0, 2, "4", 99], "description": ""}
    snapshot = json.loads(json.dumps(raw))
    with pytest.raises(ValueError, match="permutation"):
        adapter.validate(raw, objective)
    assert raw == snapshot
    repaired, notes = adapter.prepare_payload(raw)
    assert is_permutation(repaired["tour"], N_CITIES)
    assert repaired["description"].strip()
    assert notes
    adapter.validate(repaired, objective)
    fixed, changed = repair_tour([0, 1, 1], N_CITIES)
    assert changed
    assert is_permutation(fixed, N_CITIES)


def test_ox_erx_swap_and_two_opt_stay_permutations() -> None:
    rng = random.Random(3)
    left = rng.sample(range(N_CITIES), N_CITIES)
    right = rng.sample(range(N_CITIES), N_CITIES)

    class _Cut(random.Random):
        def sample(self, population, k):  # type: ignore[override]
            del population, k
            return [2, 6]

    child = order_crossover(left, right, _Cut(0))
    assert child[2:7] == left[2:7]
    assert is_permutation(child, N_CITIES)
    other = edge_recombination(left, right, rng)
    assert is_permutation(other, N_CITIES)
    assert is_permutation(random_swap(left, rng), N_CITIES)
    assert is_permutation(random_two_opt(left, rng), N_CITIES)


def test_classic_ga_is_seeded_and_not_longer_than_a_random_start() -> None:
    cities = cities_for_seed(SHOWCASE_SEED)
    first = run_classic_ga(cities, SHOWCASE_SEED, population=12, generations=5)
    second = run_classic_ga(cities, SHOWCASE_SEED, population=12, generations=5)
    assert first["tour"] == second["tour"]
    assert first["length"] == pytest.approx(second["length"])
    assert is_permutation(first["tour"], N_CITIES)
    assert first["convergence"][0]["gen"] == 0
    assert first["convergence"][-1]["best_length"] <= first["convergence"][0]["best_length"] + 1e-9


def test_mock_benchmark_schema_and_hybrid_refuses_missing_keys(tmp_path: Path) -> None:
    out = tmp_path / "tsp_benchmark.json"
    completed = subprocess.run(
        [sys.executable, "scripts/tsp_benchmark.py", "--out", str(out), "--mode", "mock"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "geneai-tsp-v1" in completed.stdout
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema"] == "geneai-tsp-v1"
    assert len(document["cities"]) == N_CITIES
    assert {city["id"] for city in document["cities"]} == set(range(N_CITIES))
    optimum = document["optimum"]
    dist = distance_matrix(document["cities"])
    assert tour_length(optimum["tour"], dist) == pytest.approx(optimum["length"], abs=1e-6)
    keys = [row["key"] for row in document["methods"]]
    assert keys == ["dfs_bnb", "classic_ga", "llm_one_shot", "geneai"]
    assert "heuristic" not in keys
    by_key = {row["key"]: row for row in document["methods"]}
    held = by_key["dfs_bnb"]
    assert held["label"] == "Traditional: depth-first search (branch and bound)"
    assert held["gap_pct"] == 0
    assert held["valid"] is True
    assert held["llm_calls"] == 0
    assert held["convergence"] == [
        {"iter": 1, "best_length": optimum["length"], "elapsed_ms": held["time_ms"]}
    ]
    assert held["tour"] == optimum["tour"]
    assert held["time_ms"] == optimum["time_ms"]
    classic = by_key["classic_ga"]
    assert classic["valid"] is True
    assert classic["llm_calls"] == 0
    assert classic["length"] >= optimum["length"] - 1e-6
    assert classic["convergence"][0]["gen"] == 0
    one_shot = by_key["llm_one_shot"]
    assert one_shot["valid"] is False
    assert one_shot["tour"] is None
    assert one_shot["length"] is None
    assert one_shot["llm_calls"] == 0
    assert "mock" in one_shot["invalid_reason"]
    geneai = by_key["geneai"]
    assert geneai["valid"] is True
    assert geneai["provider"] == "mock"
    assert is_permutation(geneai["tour"], N_CITIES)
    assert tour_length(geneai["tour"], dist) == pytest.approx(geneai["length"], abs=1e-6)
    assert geneai["length"] >= optimum["length"] - 1e-6
    assert geneai["convergence"]
    assert geneai["convergence"][-1]["best_length"] == pytest.approx(geneai["length"], abs=1e-6)
    heldout = document["heldout"]
    assert heldout["n"] == 5
    assert [row["seed"] for row in heldout["instances"]] == list(HELD_OUT_SEEDS)
    for instance in heldout["instances"]:
        assert set(instance["by_method"]) == set(keys)
        assert instance["by_method"]["dfs_bnb"]["gap_pct"] == 0
        assert instance["by_method"]["dfs_bnb"]["valid"] is True
        assert instance["by_method"]["llm_one_shot"]["valid"] is False
        assert instance["by_method"]["geneai"]["valid"] is True
        assert instance["by_method"]["classic_ga"]["valid"] is True
    summary = heldout["summary"]
    assert summary["dfs_bnb"]["optimal_hits"] == 5
    assert summary["dfs_bnb"]["mean_gap_pct"] == 0
    assert summary["llm_one_shot"]["valid_count"] == 0
    assert summary["llm_one_shot"]["mean_gap_pct"] is None
    assert summary["geneai"]["valid_count"] == 5
    assert "branch-and-bound" in document["notes"]
    assert "Held-Karp" in document["notes"]
    assert "nearest" in document["notes"]
    assert "LLM-guided operators plus 2-opt local search" in document["notes"]
    assert max(point["gen"] for point in geneai["convergence"]) >= 20
    assert geneai["provenance_counts"]["algorithmic"] > 0
    assert isinstance(geneai.get("behavior_origin"), dict)
    assert geneai["behavior_origin"]["operator"]
    for instance in heldout["instances"]:
        row = instance["by_method"]["geneai"]
        assert row["provenance_counts"]["algorithmic"] > 0
        assert isinstance(row.get("behavior_origin"), dict)
    summaries = list((tmp_path / "tsp_runs").glob("**/summary.json"))
    assert summaries
    by_seed = {}
    for path in summaries:
        body = json.loads(path.read_text(encoding="utf-8"))
        by_seed[body["seed"]] = body
    assert by_seed[SHOWCASE_SEED]["generations_completed"] >= 20
    assert by_seed[SHOWCASE_SEED]["provenance_counts"]["algorithmic"] > 0
    for seed in HELD_OUT_SEEDS:
        assert by_seed[seed]["generations_completed"] >= 8
        assert by_seed[seed]["provenance_counts"]["algorithmic"] > 0
    blocked = tmp_path / "should_not_exist.json"
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("VERI_", "CREAO_", "FRONTIER_")):
            env.pop(name)
    refused = subprocess.run(
        [sys.executable, "scripts/tsp_benchmark.py", "--out", str(blocked), "--mode", "hybrid"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert refused.returncode == 3
    text = refused.stdout + refused.stderr
    assert "VERI_API_KEY" in text
    assert "CREAO_API_KEY" in text
    assert "FRONTIER_API_KEY" in text
    assert not blocked.exists()


def test_objective_lists_coordinates() -> None:
    adapter = TspAdapter(SHOWCASE_SEED)
    objective = tsp_objective(adapter)
    assert isinstance(objective, ObjectiveSpec)
    assert "0:(" in objective.prompt
    assert adapter.oneshot_hint().startswith("payload.tour")
    assert "not repaired" in adapter.oneshot_hint()


def test_prepare_runs_two_opt_and_validate_leaves_an_illegal_tour_alone() -> None:
    adapter = TspAdapter(SHOWCASE_SEED)
    scrambled = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13]
    before = tour_length(scrambled, adapter.dist)
    improved, changed = two_opt_improve(scrambled, adapter.dist)
    assert changed
    assert tour_length(improved, adapter.dist) < before - 1e-6
    raw = {"tour": scrambled, "description": "scrambled"}
    snapshot = json.loads(json.dumps(raw))
    adapter.validate(raw, tsp_objective(adapter))
    assert raw == snapshot
    prepared, notes = adapter.prepare_payload(raw)
    assert any("2-opt" in note for note in notes)
    assert tour_length(prepared["tour"], adapter.dist) < before - 1e-6


@pytest.mark.asyncio
async def test_memetic_tsp_keeps_evolving_when_the_provider_is_slow(tmp_path: Path) -> None:
    class Slow(MockModelProvider):
        async def propose(self, instruction, schema, *, call_id):
            await asyncio.sleep(30)
            return await super().propose(instruction, schema, call_id=call_id)

    adapter = TspAdapter(SHOWCASE_SEED)
    repository = RunRepository(tmp_path)
    started = time.perf_counter()
    result = await EvolutionController().evolve(
        objective=tsp_objective(adapter),
        adapter=adapter,
        config=EvolutionConfig(
            seed=SHOWCASE_SEED,
            population_size=4,
            generations=4,
            elite_count=1,
            strategist_calls_per_generation=0,
            max_concurrent_model_calls=2,
            max_model_calls=40,
            max_evaluations=200,
            max_wall_seconds=8,
        ),
        providers={"model": Slow(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    elapsed = time.perf_counter() - started
    assert result.generations_completed >= 4
    assert elapsed < 20
    counts = result.summary["provenance_counts"]
    assert counts["algorithmic"] > 0
    assert counts["llm:veri"] == 0
    assert counts["llm:creao"] == 0
    completed = [event for event in repository.events if event.type == "generation.completed"]
    assert completed
    assert all(len(event.payload["population_ids"]) == 4 for event in completed)


def test_realize_applies_a_short_edit_and_prepare_does_not_invent_a_tour() -> None:
    adapter = TspAdapter(SHOWCASE_SEED)
    parent = list(range(N_CITIES))
    payload = {"edit": {"op": "reverse", "i": 2, "j": 6}}
    left_alone, notes = adapter.prepare_payload(payload)
    assert left_alone == payload
    assert notes == []
    realized = adapter.realize_proposal(payload, [{"tour": parent, "description": "parent"}], random.Random(0))
    assert realized["tour"] == parent[:2] + list(reversed(parent[2:7])) + parent[7:]
    swapped = adapter.realize_proposal(
        {"edit": {"op": "swap", "i": 1, "j": 4}},
        [{"tour": parent}],
        random.Random(0),
    )
    assert swapped["tour"][1] == 4 and swapped["tour"][4] == 1
    moved = adapter.realize_proposal(
        {"edit": {"op": "move", "city": 0, "after": 3}},
        [{"tour": parent}],
        random.Random(0),
    )
    assert moved["tour"][3] == 0
    prepared, polish_notes = adapter.prepare_payload(realized)
    assert is_permutation(prepared["tour"], N_CITIES)
    assert any("2-opt" in note for note in polish_notes) or prepared["tour"] == realized["tour"]


@pytest.mark.asyncio
async def test_llm_children_are_accepted_and_population_stays_full(tmp_path: Path) -> None:
    class VeriTours(MockModelProvider):
        name = "veri"

        async def propose(self, instruction, schema, *, call_id):
            del instruction, schema
            self.calls += 1
            drawer = random.Random(10_000 + self.calls)
            tour = list(range(N_CITIES))
            drawer.shuffle(tour)
            return Proposal(
                payload={"tour": tour, "description": f"veri tour {self.calls}"},
                hypothesis=f"veri permutation {self.calls}",
                provider="veri",
                provider_run_id=f"veri-{call_id}",
                mode="live",
            )

    population_size = 8
    adapter = TspAdapter(SHOWCASE_SEED)
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=tsp_objective(adapter),
        adapter=adapter,
        config=EvolutionConfig(
            seed=SHOWCASE_SEED,
            population_size=population_size,
            generations=4,
            elite_count=1,
            strategist_calls_per_generation=0,
            max_concurrent_model_calls=4,
            max_model_calls=80,
            max_evaluations=400,
            max_wall_seconds=30,
            finalist_count=population_size,
            finalist_family_cap=population_size,
        ),
        providers={"model": VeriTours(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    assert result.generations_completed >= 4
    counts = result.summary["provenance_counts"]
    llm = int(counts["llm:veri"]) + int(counts["llm:creao"])
    total = llm + int(counts["algorithmic"])
    assert llm > 0
    assert total > 0
    assert llm / total >= 0.2
    credited = [
        genome
        for genome in repository.genomes.values()
        if genome.metadata.get("provenance") == "llm:veri"
    ]
    assert credited
    assert all(is_permutation(genome.payload.get("tour"), N_CITIES) for genome in credited)
    completed = [event for event in repository.events if event.type == "generation.completed"]
    assert len(completed) >= 5
    assert all(len(event.payload["population_ids"]) == population_size for event in completed)
