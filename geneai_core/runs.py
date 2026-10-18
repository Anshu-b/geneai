"""Run a registered domain through the unchanged EvolutionController."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geneai_core.benchmark.oneshot import attach_baseline, frontier_blocker_message, run_frontier_oneshot
from geneai_core.controller import EvolutionController
from geneai_core.domains.base import DomainAdapter
from geneai_core.domains.cloud import CLOUD_SEARCH_CEILING, CloudSchedulingAdapter, cloud_objective
from geneai_core.domains.shopping import RECORDED_CATALOG_PATH, ShoppingAdapter, load_catalog, shopping_objective, write_recorded_catalog
from geneai_core.domains.toy import ToyOptimizationAdapter
from geneai_core.domains.tsp import TspAdapter, tsp_objective
from geneai_core.experiments import build_providers, toy_objective
from geneai_core.providers.base import CredentialsError
from geneai_core.providers.browser_use import BrowserUseClient, browser_use_blocker_message
from geneai_core.registry import DomainRegistry
from geneai_core.schemas import EvolutionConfig, Genome, ObjectiveSpec, RunResult
from geneai_core.store import RunRepository, export_integrity, verify_export


def make_adapter(domain: str, *, seed: int, shopping_constraints: dict[str, Any] | None = None, catalog: dict[str, Any] | None = None) -> tuple[DomainAdapter, ObjectiveSpec]:
    if domain == "toy":
        return ToyOptimizationAdapter(seed=seed), toy_objective()
    if domain == "cloud":
        return CloudSchedulingAdapter(seed=seed), cloud_objective()
    if domain == "shopping":
        adapter = ShoppingAdapter(catalog=catalog, constraints=shopping_constraints)
        constraints = adapter._objective_constraints
        objective = shopping_objective(
            palette=str(constraints.get("palette", "neutral")),
            budget=float(constraints.get("budget", 300)),
            occasions=list(constraints.get("occasions") or ["campus", "dinner"]),
        )
        return adapter, objective
    if domain == "tsp":
        adapter = TspAdapter(seed=seed)
        return adapter, tsp_objective(adapter)
    raise SystemExit(f"unknown domain {domain!r}. Use toy, cloud, shopping, or tsp.")


def _providers(mode: str) -> dict[str, object]:
    if mode == "mock":
        return build_providers("mock", seed=0)
    if mode in {"hybrid", "live"}:
        return build_providers("hybrid", seed=0)
    raise SystemExit(f"mode {mode!r} is not mock, hybrid, or live.")


async def execute_domain_run(
    *,
    domain: str,
    mode: str,
    seed: int = 42,
    population_size: int = 24,
    generations: int = 12,
    root: Path | str = "runs",
    elite_count: int = 2,
    strategist_calls_per_generation: int = 3,
    catalog_url: str | None = None,
    attach_oneshot: bool = True,
    max_wall_seconds: int = 900,
    catalog: str | None = None,
) -> RunResult:
    loaded_catalog = _load_shopping_catalog(domain, catalog)
    if domain == "shopping" and mode == "live" and loaded_catalog is None:
        if not catalog_url:
            raise SystemExit(
                "BLOCKED: live shopping was not executed. No browser call was made.\n"
                "Pass --catalog-url for a public page you are allowed to read. "
                "The fixture catalog is not labeled live, and this runner does not default to a retailer."
            )
        try:
            client = BrowserUseClient.from_env()
        except CredentialsError:
            raise SystemExit(browser_use_blocker_message()) from None
        catalog = await client.collect_catalog(
            catalog_url=catalog_url,
            query="neutral and athletic apparel with visible prices, sizes, image URLs, retailer, and product page URLs",
        )
        write_recorded_catalog(catalog)
        loaded_catalog = catalog
    adapter, objective = make_adapter(domain, seed=seed, catalog=loaded_catalog)
    if hasattr(adapter, "bind_objective"):
        adapter.bind_objective(objective)  # type: ignore[attr-defined]
    DomainRegistry.register(adapter)
    config = EvolutionConfig(
        seed=seed,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        strategist_calls_per_generation=strategist_calls_per_generation,
        max_model_calls=800,
        max_evaluations=900,
        max_wall_seconds=max_wall_seconds,
        finalist_count=6,
    )
    repository = RunRepository(root)
    result = await EvolutionController().evolve(
        objective=objective,
        adapter=DomainRegistry.get(adapter.name),
        config=config,
        providers=_providers(mode),
        mode=mode,  # type: ignore[arg-type]
        repository=repository,
    )
    if attach_oneshot and result.export_path:
        record = await run_frontier_oneshot(adapter=adapter, objective=objective)
        if record["status"] == "skipped":
            record["reason"] = frontier_blocker_message()
        attach_baseline(result.export_path, record)
        result.summary["one_shot_baseline"] = {
            "status": record["status"],
            "feasible": record.get("feasible"),
            "fitness": record.get("fitness"),
            "calls": record.get("calls"),
            "compute_matched": False,
            "comparison_label": record.get("comparison_label"),
            "reason": record.get("reason"),
        }
    result.summary["baselines"] = _baseline_report(adapter, objective)
    catalog_source = None
    if domain == "shopping":
        catalog_source = str(getattr(adapter, "catalog", {}).get("source") or "fixture")
    result.summary["split_note"] = _split_note(domain, catalog_source=catalog_source)
    result.summary["quant_domain"] = "skipped"
    if domain == "cloud" and result.export_path:
        _attach_export_field(result.export_path, "cloud_ceiling", CLOUD_SEARCH_CEILING)
        result.summary["cloud_ceiling"] = CLOUD_SEARCH_CEILING
    _write_summary(result)
    return result


def _load_shopping_catalog(domain: str, catalog: str | None) -> dict[str, Any] | None:
    if domain != "shopping" or not catalog or catalog == "fixture":
        return None
    if catalog == "recorded":
        if not RECORDED_CATALOG_PATH.is_file():
            raise SystemExit(
                "BLOCKED: --catalog recorded was requested but "
                f"{RECORDED_CATALOG_PATH} does not exist. "
                "Run shopping --mode live --catalog-url once to cache a collection, "
                "or pass a path. The fixture catalog was not substituted."
            )
        return load_catalog(RECORDED_CATALOG_PATH)
    path = Path(catalog)
    if not path.is_file():
        raise SystemExit(f"BLOCKED: catalog file {catalog} does not exist. No fixture was substituted.")
    return load_catalog(path)


def _attach_export_field(export_path: Path | str, key: str, value: dict[str, Any]) -> None:
    path = Path(export_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key] = value
    document["integrity_sha256"] = export_integrity(document)
    if not verify_export(document):
        raise RuntimeError(f"export integrity failed after attaching {key}")
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _split_note(domain: str, *, catalog_source: str | None = None) -> str:
    if domain == "shopping":
        source = catalog_source or "fixture"
        purchasable = "Fixture rows have no product URL and are not purchasable." if source == "fixture" else (
            "Recorded rows keep the URLs from the Browser Use collection. They are not a fixture."
            if source == "recorded"
            else f"Catalog source is {source}."
        )
        return (
            f"Train, validation, and test are disjoint shopper profiles over one catalog "
            f"(source={source}). "
            "Fitness is the mean subjective score on that split. "
            "The test shoppers are unread before champion.frozen. "
            "Scores are not comparable to cloud fitness. "
            f"{purchasable}"
        )
    if domain == "tsp":
        return (
            "One 15-city Euclidean instance. Train, validation, and test all score that "
            "same closed tour, so the champion length is the length of the tour shown. "
            "Fitness is the negative of the tour length."
        )
    if domain == "cloud":
        return (
            "Train, validation, and test are 18 disjoint workloads each, with the same "
            "burst and job-mix design. Generation 0 is seeded with the four hand-written "
            "DSL baselines. Champion selection uses validation only. "
            "The pool is 6 machines (CPU, high-memory, and one GPU) and 36 jobs. "
            "Fitness penalizes completion, SLA misses, hard-deadline tardiness, and energy."
        )
    return "Train, validation, and test are disjoint same-distribution shifts."


def _write_summary(result: RunResult) -> None:
    if not result.export_path:
        return
    path = Path(result.export_path).with_name("summary.json")
    path.write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")


def _baseline_report(adapter: DomainAdapter, objective: ObjectiveSpec) -> dict[str, Any]:
    if not hasattr(adapter, "baselines"):
        if adapter.name == "shopping":
            from geneai_core.domains.shopping import gym_outfit, neutral_outfit

            payloads = {"neutral_outfit": neutral_outfit(), "gym_outfit": gym_outfit()}
        else:
            return {}
    else:
        payloads = adapter.baselines()  # type: ignore[attr-defined]
    report: dict[str, Any] = {}
    for name, payload in payloads.items():
        genome = Genome(
            id=name,
            run_id="baseline",
            generation=0,
            operator="seed",
            payload=payload,
            hypothesis=name,
            provider="handwritten",
            created_at="",
            content_hash=name,
        )
        row: dict[str, Any] = {"provider": "handwritten"}
        for split in ("train", "validation", "test"):
            evaluation = adapter.evaluate(genome, split)
            row[split] = {
                "feasible": evaluation.feasible,
                "fitness": evaluation.fitness,
                "metrics": evaluation.metrics,
            }
        report[name] = row
    del objective
    return report
