#!/usr/bin/env python3
"""15-city TSP homepage benchmark.

  uv run python scripts/tsp_benchmark.py --out tsp_benchmark.json --mode mock
  uv run python scripts/tsp_benchmark.py --out tsp_benchmark.json --mode hybrid

Mock makes no network calls. Hybrid reads VERI_*, CREAO_*, and FRONTIER_* the
same way the other domains do, and exits 3 when a credential is missing.
Held-Karp is the traditional method and the optimum. There is no nearest-neighbor arm.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.benchmark.oneshot import run_frontier_oneshot  # noqa: E402
from geneai_core.controller import EvolutionController  # noqa: E402
from geneai_core.domains.tsp import (  # noqa: E402
    HELD_OUT_SEEDS,
    N_CITIES,
    SHOWCASE_SEED,
    TspAdapter,
    cities_for_seed,
    distance_matrix,
    held_karp,
    run_classic_ga,
    tour_length,
    tsp_objective,
)
from geneai_core.experiments import build_providers  # noqa: E402
from geneai_core.providers.base import CredentialsError  # noqa: E402
from geneai_core.providers.creao import CreaoProvider  # noqa: E402
from geneai_core.providers.frontier import FrontierProvider  # noqa: E402
from geneai_core.providers.veri import VeriProvider  # noqa: E402
from geneai_core.schemas import EvolutionConfig  # noqa: E402
from geneai_core.store import RunRepository  # noqa: E402

METHOD_ORDER = ("held_karp", "classic_ga", "llm_one_shot", "geneai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="15-city TSP benchmark for the homepage")
    parser.add_argument("--out", required=True, help="Path of the geneai-tsp-v1 JSON file")
    parser.add_argument("--mode", choices=["hybrid", "mock"], required=True)
    return parser.parse_args()


def _assert_hybrid_credentials() -> None:
    errors: list[str] = []
    for factory in (VeriProvider.from_env, CreaoProvider.from_env, FrontierProvider.from_env):
        try:
            factory()
        except CredentialsError as exc:
            errors.append(str(exc))
    if errors:
        raise CredentialsError("\n\n".join(errors))


def _round_length(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _gap_pct(length: float | None, optimum: float) -> float | None:
    if length is None:
        return None
    if optimum == 0:
        return 0.0
    return round((length - optimum) / optimum * 100.0, 4)


def _geneai_config(mode: str, *, showcase: bool) -> EvolutionConfig:
    """Memetic TSP: 20 showcase generations inside about 150s, 8 held-out inside about 60s.

    The controller slices each generation, fills it with OX/ERX/swap/2-opt, and
    does not wait out a slow Veri or CREAO call. Mock uses the same generation
    counts. The slice is a timeout, not a sleep, so mock stays fast.
    """

    if showcase:
        return EvolutionConfig(
            seed=SHOWCASE_SEED,
            population_size=8 if mode == "hybrid" else 6,
            generations=20,
            elite_count=2,
            max_wall_seconds=150,
            max_model_calls=400,
            max_evaluations=2000,
            strategist_calls_per_generation=2 if mode == "hybrid" else 1,
            finalist_count=8 if mode == "hybrid" else 6,
            finalist_family_cap=8 if mode == "hybrid" else 6,
        )
    return EvolutionConfig(
        seed=0,
        population_size=6 if mode == "hybrid" else 4,
        generations=8,
        elite_count=1,
        max_wall_seconds=60,
        max_model_calls=200,
        max_evaluations=800,
        strategist_calls_per_generation=1,
        finalist_count=6 if mode == "hybrid" else 4,
        finalist_family_cap=6 if mode == "hybrid" else 4,
    )


def _method(
    *,
    key: str,
    label: str,
    provider: str,
    tour: list[int] | None,
    length: float | None,
    optimum: float,
    time_ms: int,
    llm_calls: int,
    valid: bool,
    invalid_reason: str | None,
    convergence: list[dict[str, Any]],
) -> dict[str, Any]:
    rounded = _round_length(length)
    return {
        "key": key,
        "label": label,
        "provider": provider,
        "tour": tour,
        "length": rounded,
        "gap_pct": _gap_pct(rounded, optimum) if valid else None,
        "time_ms": int(time_ms),
        "llm_calls": int(llm_calls),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "convergence": convergence,
    }


def _held_karp_method(cities: list[dict[str, Any]]) -> tuple[dict[str, Any], list[int], float]:
    started = time.perf_counter()
    tour, length = held_karp(cities)
    time_ms = int((time.perf_counter() - started) * 1000)
    rounded = round(length, 6)
    method = _method(
        key="held_karp",
        label="Traditional: dynamic programming (Held-Karp)",
        provider="local",
        tour=tour,
        length=rounded,
        optimum=rounded,
        time_ms=time_ms,
        llm_calls=0,
        valid=True,
        invalid_reason=None,
        convergence=[{"iter": 1, "best_length": rounded, "elapsed_ms": time_ms}],
    )
    method["gap_pct"] = 0
    return method, tour, rounded


def _classic_method(cities: list[dict[str, Any]], seed: int, optimum: float) -> dict[str, Any]:
    result = run_classic_ga(cities, seed)
    return _method(
        key="classic_ga",
        label="Classic genetic algorithm (OX crossover, swap mutation)",
        provider="local",
        tour=result["tour"],
        length=result["length"],
        optimum=optimum,
        time_ms=result["time_ms"],
        llm_calls=0,
        valid=True,
        invalid_reason=None,
        convergence=result["convergence"],
    )


def _blank_one_shot(reason: str, time_ms: int = 0, llm_calls: int = 0) -> dict[str, Any]:
    return {
        "key": "llm_one_shot",
        "label": "GPT-5.5 one-shot",
        "provider": "frontier",
        "tour": None,
        "length": None,
        "gap_pct": None,
        "time_ms": time_ms,
        "llm_calls": llm_calls,
        "valid": False,
        "invalid_reason": reason,
        "convergence": [],
    }


async def _one_shot_method(adapter: TspAdapter, optimum: float, *, mode: str) -> dict[str, Any]:
    if mode != "hybrid":
        return _blank_one_shot("mock mode does not call the frontier model")
    started = time.perf_counter()
    record = await run_frontier_oneshot(adapter=adapter, objective=tsp_objective(adapter))
    time_ms = int((time.perf_counter() - started) * 1000)
    calls = int(record.get("calls") or 0)
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else None
    tour = payload.get("tour") if isinstance(payload, dict) else None
    if record.get("status") == "skipped":
        return _blank_one_shot(str(record.get("reason") or "frontier one-shot skipped"), time_ms, calls)
    if not record.get("feasible"):
        shown = tour if isinstance(tour, list) else None
        return {
            "key": "llm_one_shot",
            "label": "GPT-5.5 one-shot",
            "provider": "frontier",
            "tour": shown,
            "length": None,
            "gap_pct": None,
            "time_ms": time_ms,
            "llm_calls": calls,
            "valid": False,
            "invalid_reason": str(record.get("reason") or "invalid tour"),
            "convergence": [],
        }
    if not isinstance(tour, list):
        return _blank_one_shot("frontier response had no tour", time_ms, calls)
    cities = [int(city) for city in tour]
    length = tour_length(cities, adapter.dist)
    return _method(
        key="llm_one_shot",
        label="GPT-5.5 one-shot",
        provider="frontier",
        tour=cities,
        length=length,
        optimum=optimum,
        time_ms=time_ms,
        llm_calls=calls,
        valid=True,
        invalid_reason=None,
        convergence=[{"iter": 1, "best_length": round(length, 6), "elapsed_ms": time_ms}],
    )


async def _geneai_method(
    *,
    mode: str,
    seed: int,
    optimum: float,
    work_root: Path,
    showcase: bool,
) -> dict[str, Any]:
    adapter = TspAdapter(seed=seed)
    config = _geneai_config(mode, showcase=showcase)
    config = config.model_copy(update={"seed": seed})
    objective = tsp_objective(adapter)
    repository = RunRepository(work_root / f"seed_{seed}")
    adapter.mark_start()
    started = time.perf_counter()
    result = await EvolutionController().evolve(
        objective=objective,
        adapter=adapter,
        config=config,
        providers=build_providers(mode, seed=seed),
        mode=mode,  # type: ignore[arg-type]
        repository=repository,
    )
    time_ms = int((time.perf_counter() - started) * 1000)
    budget = result.summary.get("budget") if isinstance(result.summary.get("budget"), dict) else {}
    llm_calls = int(budget.get("model_calls") or 0)
    payload = result.summary.get("champion_payload")
    tour = payload.get("tour") if isinstance(payload, dict) else None
    label = "GeneAI (Veri + CREAO)" if mode == "hybrid" else "GeneAI (mock providers)"
    provider = "veri+creao" if mode == "hybrid" else "mock"
    if not isinstance(tour, list):
        return {
            "key": "geneai",
            "label": label,
            "provider": provider,
            "tour": None,
            "length": None,
            "gap_pct": None,
            "time_ms": time_ms,
            "llm_calls": llm_calls,
            "valid": False,
            "invalid_reason": result.stop_reason or "no champion tour",
            "convergence": adapter.convergence(),
        }
    cities = [int(city) for city in tour]
    length = tour_length(cities, distance_matrix(adapter.cities))
    method = _method(
        key="geneai",
        label=label,
        provider=provider,
        tour=cities,
        length=length,
        optimum=optimum,
        time_ms=time_ms,
        llm_calls=llm_calls,
        valid=True,
        invalid_reason=None,
        convergence=adapter.convergence(),
    )
    counts = result.summary.get("provenance_counts")
    if isinstance(counts, dict):
        method["provenance_counts"] = counts
    return method


def _heldout_row(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "length": method["length"],
        "gap_pct": method["gap_pct"],
        "time_ms": method["time_ms"],
        "valid": method["valid"],
    }


def _summary(instances: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in METHOD_ORDER:
        gaps: list[float] = []
        times: list[int] = []
        hits = 0
        valid_count = 0
        for instance in instances:
            row = instance["by_method"][key]
            times.append(int(row["time_ms"]))
            if not row["valid"] or row["length"] is None:
                continue
            valid_count += 1
            if row["gap_pct"] is not None:
                gaps.append(float(row["gap_pct"]))
            if row["gap_pct"] is not None and abs(float(row["gap_pct"])) <= 1e-4:
                hits += 1
        summary[key] = {
            "mean_gap_pct": None if not gaps else round(sum(gaps) / len(gaps), 4),
            "optimal_hits": hits,
            "mean_time_ms": int(round(sum(times) / len(times))) if times else 0,
            "valid_count": valid_count,
        }
    return summary


async def build_benchmark(mode: str, work_root: Path) -> dict[str, Any]:
    showcase_cities = cities_for_seed(SHOWCASE_SEED, N_CITIES)
    held_karp_method, optimum_tour, optimum_length = _held_karp_method(showcase_cities)
    classic = _classic_method(showcase_cities, SHOWCASE_SEED, optimum_length)
    one_shot_adapter = TspAdapter(SHOWCASE_SEED)
    one_shot = await _one_shot_method(one_shot_adapter, optimum_length, mode=mode)
    geneai = await _geneai_method(
        mode=mode,
        seed=SHOWCASE_SEED,
        optimum=optimum_length,
        work_root=work_root,
        showcase=True,
    )
    instances: list[dict[str, Any]] = []
    for seed in HELD_OUT_SEEDS:
        cities = cities_for_seed(seed, N_CITIES)
        exact, _tour, length = _held_karp_method(cities)
        methods = {
            "held_karp": exact,
            "classic_ga": _classic_method(cities, seed, length),
            "llm_one_shot": await _one_shot_method(TspAdapter(seed), length, mode=mode),
            "geneai": await _geneai_method(
                mode=mode,
                seed=seed,
                optimum=length,
                work_root=work_root,
                showcase=False,
            ),
        }
        instances.append(
            {
                "seed": seed,
                "optimum_length": length,
                "by_method": {key: _heldout_row(methods[key]) for key in METHOD_ORDER},
            }
        )
    notes = (
        f"Fifteen Euclidean cities in [0, 100], drawn from random.Random(seed) and rounded to 4 decimals. "
        f"The showcase instance is seed {SHOWCASE_SEED}. Held-out seeds are {list(HELD_OUT_SEEDS)}. "
        "The traditional method is Held-Karp dynamic programming. optimum is that same search, not a second run, "
        "and there is no nearest-neighbor arm. "
        "classic_ga is a seeded permutation GA with population "
        f"{24}, {40} generations, order crossover, and swap mutation. It does not call a model. "
        "llm_one_shot is one frontier call with the coordinates. An illegal tour stays illegal. "
        "geneai is the shared evolution controller. Mock uses the mock providers. "
        "Hybrid uses Veri for most proposals and CREAO for the strategist slots. "
        "GeneAI uses LLM-guided operators plus 2-opt local search. "
        "The showcase run is 20 generations inside about 150s. "
        "Each held-out map is 8 generations inside about 60s. "
        "A slow provider is cancelled at the generation slice and the population is filled with algorithmic children. "
        "A TSP genome is a permutation. Swap and 2-opt are the mutations. OX and ERX are the crossovers. "
        "GeneAI repairs an invalid city list and runs 2-opt before scoring. The one-shot arm does not. "
        "Dedup hashes the canonical cycle (city 0 first, then the lexicographically smaller direction). "
        "Train, validation, and test all score the same instance. Fitness is negative tour length. "
        "time_ms is wall clock."
    )
    return {
        "schema": "geneai-tsp-v1",
        "cities": showcase_cities,
        "optimum": {"tour": optimum_tour, "length": optimum_length, "time_ms": held_karp_method["time_ms"]},
        "methods": [held_karp_method, classic, one_shot, geneai],
        "heldout": {"n": len(instances), "instances": instances, "summary": _summary(instances)},
        "notes": notes,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.mode == "hybrid":
        try:
            _assert_hybrid_credentials()
        except CredentialsError as exc:
            print(str(exc))
            raise SystemExit(3) from None
    out = Path(args.out)
    work = out.resolve().parent / "tsp_runs"
    work.mkdir(parents=True, exist_ok=True)
    document = asyncio.run(build_benchmark(args.mode, work))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    lengths = {row["key"]: row["length"] for row in document["methods"]}
    print(
        json.dumps(
            {
                "schema": document["schema"],
                "mode": args.mode,
                "out": str(out),
                "optimum_length": document["optimum"]["length"],
                "lengths": lengths,
            }
        )
    )


if __name__ == "__main__":
    main()
