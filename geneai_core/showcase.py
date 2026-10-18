"""One held-out test case, re-scored by the same evaluator that produced the export."""

from __future__ import annotations

import copy
from typing import Any

from geneai_core.domains.cloud import (
    MACHINES,
    TEST_INDEXES,
    CloudSchedulingAdapter,
    baseline_payload,
    round_robin_metrics,
    rules_plain,
    simulate,
    workload,
    workload_fitness,
)
from geneai_core.diversity import content_hash
from geneai_core.domains.shopping import (
    SHOPPERS,
    ShoppingAdapter,
    load_catalog,
    outfit_why,
)
from geneai_core.domains.shopping import RECORDED_CATALOG_PATH
from geneai_core.schemas import Genome

CASE_RULE = (
    "Not cherry-picked. For each held-out test case, margin is the champion fitness "
    "minus the best hand-written baseline on that same case (an infeasible score counts as missing). "
    "The case shown is the median margin. An even count uses the lower of the two central margins, "
    "then the smaller case id."
)

# simulate() adds machine kW draw times busy minutes. The showcase reports kWh.
_KWH_PER_KW_MINUTE = 1.0 / 60.0


def select_median_case(rows: list[tuple[Any, float]]) -> tuple[Any, float]:
    ordered = sorted(rows, key=lambda row: (row[1], str(row[0])))
    if not ordered:
        raise ValueError("no test cases")
    return ordered[(len(ordered) - 1) // 2]


def build_showcase(export: dict[str, Any]) -> dict[str, Any]:
    run = export.get("run") if isinstance(export.get("run"), dict) else {}
    domain = str(run.get("domain") or "")
    if domain == "cloud_scheduling":
        body = _cloud_showcase(export)
    elif domain == "shopping":
        body = _shopping_showcase(export)
    else:
        raise ValueError(f"showcase does not cover domain {domain!r}")
    body["schema"] = "geneai-showcase-v1"
    body["domain"] = domain
    body["run_id"] = run.get("id")
    body["export_integrity_sha256"] = export.get("integrity_sha256")
    body["split"] = "test"
    return body


def _round(value: float | None, places: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def _genome(genome_id: str, payload: dict[str, Any], provider: str) -> Genome:
    return Genome(
        id=genome_id,
        run_id="showcase",
        generation=0,
        operator="seed",
        payload=payload,
        hypothesis="showcase",
        provider=provider,
        created_at="",
        content_hash="showcase",
    )


def _champion(export: dict[str, Any]) -> dict[str, Any]:
    champion_id = export.get("champion_id")
    for genome in export.get("genomes") or []:
        if isinstance(genome, dict) and genome.get("id") == champion_id:
            return genome
    raise ValueError(f"champion {champion_id} is not in the export")


def _payload(genome: dict[str, Any]) -> dict[str, Any]:
    payload = genome.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"genome {genome.get('id')} has no payload")
    return copy.deepcopy(payload)


def _provider(value: Any, model: Any = None) -> str:
    text = str(value or "")
    if text in {"frontier", "openai"} or (isinstance(model, str) and model.startswith("gpt")):
        return "openai"
    return text or "unknown"


def _cloud_fitness(payload: dict[str, Any], seed: int, index: int) -> float | None:
    jobs = workload(seed, index)
    metrics = simulate(payload, jobs)
    if metrics["violations"] > 0 or metrics["completed"] < len(jobs):
        return None
    return workload_fitness(metrics, round_robin_metrics(seed, index))


def _best_named(scores: dict[str, float | None]) -> tuple[str, float | None]:
    feasible = {name: score for name, score in scores.items() if score is not None}
    if not feasible:
        name = sorted(scores)[0]
        return name, None
    name = max(feasible, key=lambda item: (feasible[item], item))
    return name, feasible[name]


def _cloud_showcase(export: dict[str, Any]) -> dict[str, Any]:
    seed = int((export.get("run") or {})["seed"])
    champion = _champion(export)
    champion_payload = _payload(champion)
    baselines = {
        "round_robin": baseline_payload("round_robin"),
        "shortest_job": baseline_payload("shortest_job"),
        "least_loaded": baseline_payload("least_loaded"),
        "deadline_weighted": baseline_payload("deadline_weighted"),
    }
    rows: list[tuple[int, float]] = []
    best_on_case: dict[int, str] = {}
    for index in TEST_INDEXES:
        champion_score = _cloud_fitness(champion_payload, seed, index)
        baseline_scores = {name: _cloud_fitness(payload, seed, index) for name, payload in baselines.items()}
        best_name, best_score = _best_named(baseline_scores)
        best_on_case[index] = best_name
        margin = -1e18 if champion_score is None or best_score is None else champion_score - best_score
        rows.append((index, margin))
    index, margin = select_median_case(rows)
    case_id = f"test-{index}"
    best_name = best_on_case[index]
    jobs = workload(seed, index)
    policies = [
        _cloud_policy(
            export,
            key="champion",
            label="GeneAI champion",
            provider=_provider(champion.get("provider")),
            genome_id=champion.get("id"),
            payload=champion_payload,
            seed=seed,
            index=index,
            jobs=jobs,
        ),
        _cloud_oneshot(export, seed, index, jobs),
        _cloud_policy(
            export,
            key=best_name,
            label=f"Hand-written {best_name.replace('_', ' ')}",
            provider="baseline",
            genome_id=_baseline_id(export, best_name),
            payload=baselines[best_name],
            seed=seed,
            index=index,
            jobs=jobs,
        ),
    ]
    if best_name != "round_robin":
        policies.append(
            _cloud_policy(
                export,
                key="round_robin",
                label="Round robin",
                provider="baseline",
                genome_id=_baseline_id(export, "round_robin"),
                payload=baselines["round_robin"],
                seed=seed,
                index=index,
                jobs=jobs,
            )
        )
    return {
        "case_id": case_id,
        "case_note": f"{CASE_RULE} Selected {case_id} (margin {margin:.6f}).",
        "cloud": {
            "units": {"time": "minutes", "energy": "kWh"},
            "machines": [_machine(machine) for machine in MACHINES],
            "jobs": [_job(job) for job in jobs],
            "policies": policies,
        },
    }


def _baseline_id(export: dict[str, Any], name: str) -> str | None:
    for genome in export.get("genomes") or []:
        if not isinstance(genome, dict):
            continue
        meta = genome.get("metadata") if isinstance(genome.get("metadata"), dict) else {}
        if meta.get("baseline_name") == name:
            return genome.get("id")
    return None


def _cloud_oneshot(export: dict[str, Any], seed: int, index: int, jobs: list[Any]) -> dict[str, Any]:
    record = export.get("one_shot_baseline") if isinstance(export.get("one_shot_baseline"), dict) else {}
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else None
    model = record.get("model")
    if payload is None:
        reason = str(record.get("reason") or "one-shot payload missing")
        row = _infeasible_policy(
            key="gpt55_one_shot",
            label="GPT-5.5 one-shot",
            provider=_provider(record.get("provider"), model),
            reason=reason,
            attempted=None,
        )
        row["aggregate"] = _empty_cloud_aggregate()
        return row
    return _cloud_policy(
        export,
        key="gpt55_one_shot",
        label="GPT-5.5 one-shot",
        provider=_provider(record.get("provider"), model),
        genome_id=None,
        payload=payload,
        seed=seed,
        index=index,
        jobs=jobs,
        attempted=payload,
    )


def _cloud_policy(
    export: dict[str, Any],
    *,
    key: str,
    label: str,
    provider: str,
    genome_id: str | None,
    payload: dict[str, Any],
    seed: int,
    index: int,
    jobs: list[Any],
    attempted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = CloudSchedulingAdapter(seed=seed)
    try:
        adapter.validate(copy.deepcopy(payload), adapter_objective())
    except ValueError as exc:
        row = _infeasible_policy(
            key=key,
            label=label,
            provider=provider,
            reason=str(exc),
            attempted=attempted if attempted is not None else payload,
        )
        row["genome_id"] = genome_id
        row["description"] = payload.get("description") if isinstance(payload.get("description"), str) else ""
        row["rules_plain"] = rules_plain(payload) if isinstance(payload.get("rules"), dict) else []
        row["aggregate"] = _empty_cloud_aggregate()
        _attach_behavior_origin(row, export, payload)
        return row
    trace: list[dict[str, Any]] = []
    metrics = simulate(payload, jobs, trace=trace)
    feasible = metrics["violations"] <= 0 and metrics["completed"] >= len(jobs)
    fitness = None if not feasible else workload_fitness(metrics, round_robin_metrics(seed, index))
    waits = [float(row["wait_min"]) for row in trace]
    ends = [float(row["end_min"]) for row in trace]
    sla_misses = sum(1 for row in trace if row["missed_deadline"])
    hard_misses = sum(1 for row in trace if row["missed_deadline"] and row["hard"])
    energy = None if metrics["violations"] > 0 else metrics["energy"] * _KWH_PER_KW_MINUTE
    row = {
        "key": key,
        "label": label,
        "provider": provider,
        "genome_id": genome_id,
        "description": payload.get("description") if isinstance(payload.get("description"), str) else "",
        "rules_plain": rules_plain(payload),
        "feasible": feasible,
        "infeasible_reason": None if feasible else "capacity violation or unfinished jobs",
        "schedule": [_schedule_row(item) for item in trace],
        "summary": {
            "avg_completion_min": _round(metrics["mean_completion"]) if metrics["completed"] else None,
            "p95_wait_min": _round(_percentile(waits, 0.95)) if waits else None,
            "sla_misses": sla_misses,
            "hard_deadline_misses": hard_misses,
            "energy_kwh": _round(energy),
            "makespan_min": _round(max(ends)) if ends else None,
            "capacity_violations": int(metrics["violations"]),
            "fitness": _round(fitness, 6),
        },
        "aggregate": _aggregate_cloud(payload, seed),
    }
    if not feasible:
        row["attempted_output"] = attempted if attempted is not None else payload
    _attach_behavior_origin(row, export, payload)
    return row


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _empty_cloud_aggregate() -> dict[str, Any]:
    return {
        "avg_completion_min": None,
        "p95_wait_min": None,
        "sla_misses": 0,
        "hard_deadline_misses": 0,
        "energy_kwh": None,
        "makespan_min": None,
        "capacity_violations": 0,
        "fitness": None,
        "n_cases": len(TEST_INDEXES),
    }


def _aggregate_cloud(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    """Means and totals over every held-out test workload, in the same units as summary."""

    completions: list[float] = []
    p95s: list[float] = []
    makespans: list[float] = []
    fitnesses: list[float] = []
    sla = 0
    hard = 0
    energy = 0.0
    energy_seen = False
    violations = 0
    for case_index in TEST_INDEXES:
        case_jobs = workload(seed, case_index)
        trace: list[dict[str, Any]] = []
        metrics = simulate(payload, case_jobs, trace=trace)
        violations += int(metrics["violations"])
        sla += sum(1 for item in trace if item["missed_deadline"])
        hard += sum(1 for item in trace if item["missed_deadline"] and item["hard"])
        if metrics["violations"] <= 0:
            energy += float(metrics["energy"]) * _KWH_PER_KW_MINUTE
            energy_seen = True
        feasible = metrics["violations"] <= 0 and metrics["completed"] >= len(case_jobs)
        if not feasible:
            continue
        waits = [float(item["wait_min"]) for item in trace]
        ends = [float(item["end_min"]) for item in trace]
        if metrics["completed"]:
            completions.append(float(metrics["mean_completion"]))
        if waits:
            p95s.append(_percentile(waits, 0.95))
        if ends:
            makespans.append(max(ends))
        fitnesses.append(workload_fitness(metrics, round_robin_metrics(seed, case_index)))
    return {
        "avg_completion_min": _round(_mean(completions)),
        "p95_wait_min": _round(_mean(p95s)),
        "sla_misses": sla,
        "hard_deadline_misses": hard,
        "energy_kwh": _round(energy) if energy_seen else None,
        "makespan_min": _round(_mean(makespans)),
        "capacity_violations": violations,
        "fitness": _round(_mean(fitnesses), 6),
        "n_cases": len(TEST_INDEXES),
    }


def _behavior_origin(export: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    """Earliest genome whose behavior payload matches. Generation, then id."""

    target = content_hash(payload)
    matches: list[dict[str, Any]] = []
    for genome in export.get("genomes") or []:
        if not isinstance(genome, dict):
            continue
        body = genome.get("payload")
        if not isinstance(body, dict):
            continue
        if content_hash(body) != target:
            continue
        matches.append(genome)
    if not matches:
        return None

    def origin_key(genome: dict[str, Any]) -> tuple[int, str]:
        generation = genome.get("generation")
        generation_n = generation if isinstance(generation, int) else 0
        return (generation_n, str(genome.get("id") or ""))

    earliest = min(matches, key=origin_key)
    return {
        "id": earliest.get("id"),
        "provider": earliest.get("provider"),
        "operator": earliest.get("operator"),
        "generation": earliest.get("generation"),
    }


def _attach_behavior_origin(row: dict[str, Any], export: dict[str, Any], payload: dict[str, Any]) -> None:
    if not row.get("genome_id"):
        return
    origin = _behavior_origin(export, payload)
    if origin is not None:
        row["behavior_origin"] = origin


def _schedule_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": item["job_id"],
        "machine_id": item["machine_id"],
        "start_min": _round(item["start_min"]),
        "end_min": _round(item["end_min"]),
        "wait_min": _round(item["wait_min"]),
        "deadline_min": _round(item["deadline_min"]),
        "missed_deadline": bool(item["missed_deadline"]),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _infeasible_policy(
    *,
    key: str,
    label: str,
    provider: str,
    reason: str,
    attempted: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "provider": provider,
        "genome_id": None,
        "description": "",
        "rules_plain": [],
        "feasible": False,
        "infeasible_reason": reason,
        "schedule": [],
        "summary": {
            "avg_completion_min": None,
            "p95_wait_min": None,
            "sla_misses": 0,
            "hard_deadline_misses": 0,
            "energy_kwh": None,
            "makespan_min": None,
            "capacity_violations": None,
            "fitness": None,
        },
    }
    if attempted is not None:
        row["attempted_output"] = attempted
    return row


def adapter_objective():
    from geneai_core.schemas import ObjectiveSpec

    return ObjectiveSpec(domain="cloud_scheduling", prompt="", evaluation_version="cloud-scheduler-v3")


def _machine(machine: dict[str, Any]) -> dict[str, Any]:
    kind = str(machine["kind"])
    if kind == "high_mem":
        kind = "highmem"
    return {
        "id": machine["id"],
        "kind": kind,
        "cpu": _whole(machine["cpu"]),
        "mem_gb": _whole(machine["mem"]),
        "gpu": _whole(machine["gpu"]),
    }


def _job(job: Any) -> dict[str, Any]:
    kind = str(job.kind)
    if kind == "high_mem":
        kind = "highmem"
    return {
        "id": f"j{job.jid:02d}",
        "class": kind,
        "cpu": _whole(job.cpu),
        "mem_gb": _whole(job.mem),
        "gpu": _whole(job.gpu),
        "arrival_min": _round(job.arrival),
        "duration_min": _round(job.duration),
        "deadline_min": _round(job.deadline),
        "hard_deadline": bool(job.hard),
    }


def _whole(value: Any) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _catalog_from_export(export: dict[str, Any]) -> dict[str, Any] | None:
    catalog = export.get("catalog")
    if not isinstance(catalog, dict):
        return None
    raw_items = catalog.get("items") or catalog.get("products") or []
    if not isinstance(raw_items, list) or not raw_items:
        return None
    source = catalog.get("source") or "export"
    products: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        product_id = item.get("id") or item.get("product_id")
        if not product_id:
            continue
        products.append(
            {
                "product_id": product_id,
                "title": item.get("name") or item.get("title"),
                "role": item.get("role"),
                "price": item.get("price") if item.get("price") is not None else item.get("price_usd"),
                "product_url": item.get("url") or item.get("product_url"),
                "image_url": item.get("image_url"),
                "colors": item.get("colors") or ["catalog_default"],
                "sizes": item.get("sizes") or ["M", "9", "One Size"],
                "style": item.get("style") or [],
                "occasions": item.get("occasions") or [],
                "availability": item.get("availability") or "unknown",
            }
        )
    if not products:
        return None
    return {
        "catalog_id": catalog.get("catalog_id") or "export",
        "source": source,
        "products": products,
    }


def _shopping_adapter(export: dict[str, Any]) -> ShoppingAdapter:
    wanted: set[str] = set()
    for genome in export.get("genomes") or []:
        payload = genome.get("payload") if isinstance(genome, dict) else None
        if not isinstance(payload, dict):
            continue
        for item in payload.get("items") or []:
            if isinstance(item, dict) and item.get("product_id"):
                wanted.add(str(item["product_id"]))
    record = export.get("one_shot_baseline") if isinstance(export.get("one_shot_baseline"), dict) else {}
    oneshot = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    for item in oneshot.get("items") or []:
        if isinstance(item, dict) and item.get("product_id"):
            wanted.add(str(item["product_id"]))
    embedded = _catalog_from_export(export)
    if embedded is not None and _covers(embedded, wanted):
        return ShoppingAdapter(embedded)
    fixture = load_catalog()
    if _covers(fixture, wanted):
        return ShoppingAdapter(fixture)
    if RECORDED_CATALOG_PATH.is_file():
        recorded = load_catalog(RECORDED_CATALOG_PATH)
        if _covers(recorded, wanted):
            return ShoppingAdapter(recorded)
    return ShoppingAdapter(fixture)


def _covers(catalog: dict[str, Any], wanted: set[str]) -> bool:
    if not wanted:
        return True
    have = {str(product.get("product_id")) for product in catalog.get("products") or []}
    return wanted <= have


def _shopping_showcase(export: dict[str, Any]) -> dict[str, Any]:
    adapter = _shopping_adapter(export)
    champion = _champion(export)
    champion_payload = _payload(champion)
    baselines = adapter.baselines()
    shoppers = list(SHOPPERS["test"])
    rows: list[tuple[str, float]] = []
    best_on_case: dict[str, str] = {}
    for shopper in shoppers:
        champion_score = _shop_score(adapter, champion_payload, shopper)
        scores = {name: _shop_score(adapter, payload, shopper) for name, payload in baselines.items()}
        best_name, best_score = _best_named(scores)
        best_on_case[str(shopper["id"])] = best_name
        margin = -1e18 if champion_score is None or best_score is None else champion_score - best_score
        rows.append((str(shopper["id"]), margin))
    case_id, margin = select_median_case(rows)
    shopper = next(item for item in shoppers if item["id"] == case_id)
    best_name = best_on_case[case_id]
    outfits = [
        _shop_outfit(
            export,
            adapter,
            key="champion",
            label="GeneAI champion",
            provider=_provider(champion.get("provider")),
            genome_id=champion.get("id"),
            payload=champion_payload,
            shopper=shopper,
        ),
        _shop_oneshot(export, adapter, shopper),
        _shop_outfit(
            export,
            adapter,
            key="best_baseline",
            label=f"Hand-written {best_name.replace('_', ' ')}",
            provider="baseline",
            genome_id=_baseline_id(export, best_name),
            payload=baselines[best_name],
            shopper=shopper,
        ),
    ]
    return {
        "case_id": case_id,
        "case_note": f"{CASE_RULE} Selected shopper {case_id} (margin {margin:.6f}).",
        "shopping": {"shopper": _shopper_card(shopper), "outfits": outfits},
    }


def _shop_score(adapter: ShoppingAdapter, payload: dict[str, Any], shopper: dict[str, Any]) -> float | None:
    try:
        adapter.validate(copy.deepcopy(payload), _shopping_objective())
    except ValueError:
        return None
    genome = _genome("case", copy.deepcopy(payload), "showcase")
    _fitness, _metrics, violations = adapter.score_shopper(genome, shopper)
    if violations:
        return None
    return float(_fitness)


def _shopping_objective():
    from geneai_core.domains.shopping import shopping_objective

    return shopping_objective()


def _shop_oneshot(export: dict[str, Any], adapter: ShoppingAdapter, shopper: dict[str, Any]) -> dict[str, Any]:
    record = export.get("one_shot_baseline") if isinstance(export.get("one_shot_baseline"), dict) else {}
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else None
    model = record.get("model")
    if payload is None:
        row = _infeasible_outfit(
            key="gpt55_one_shot",
            label="GPT-5.5 one-shot",
            provider=_provider(record.get("provider"), model),
            reason=str(record.get("reason") or "one-shot payload missing"),
            attempted=None,
            items=[],
        )
        row["aggregate"] = _aggregate_shopping(adapter, {})
        return row
    return _shop_outfit(
        export,
        adapter,
        key="gpt55_one_shot",
        label="GPT-5.5 one-shot",
        provider=_provider(record.get("provider"), model),
        genome_id=None,
        payload=payload,
        shopper=shopper,
        attempted=payload,
    )


def _shop_outfit(
    export: dict[str, Any],
    adapter: ShoppingAdapter,
    *,
    key: str,
    label: str,
    provider: str,
    genome_id: str | None,
    payload: dict[str, Any],
    shopper: dict[str, Any],
    attempted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = _outfit_items(adapter, payload)
    total = sum(float(item["price_usd"]) for item in items if isinstance(item.get("price_usd"), int | float))
    aggregate = _aggregate_shopping(adapter, payload)
    try:
        adapter.validate(copy.deepcopy(payload), _shopping_objective())
    except ValueError as exc:
        row = _infeasible_outfit(
            key=key,
            label=label,
            provider=provider,
            reason=str(exc),
            attempted=attempted if attempted is not None else payload,
            items=items,
            genome_id=genome_id,
            total=total,
        )
        row["aggregate"] = aggregate
        _attach_behavior_origin(row, export, payload)
        return row
    genome = _genome(genome_id or key, copy.deepcopy(payload), provider)
    fitness, metrics, violations = adapter.score_shopper(genome, shopper)
    resolved = adapter._resolved(genome.payload, str(shopper["palette"]))
    why = outfit_why(
        resolved,
        budget=float(shopper["budget"]),
        palette=str(shopper["palette"]),
        occasions=list(shopper["occasions"]),
        metrics=metrics,
        violations=violations,
    )
    if violations:
        row = _infeasible_outfit(
            key=key,
            label=label,
            provider=provider,
            reason="; ".join(violations),
            attempted=attempted if attempted is not None else payload,
            items=items,
            genome_id=genome_id,
            total=total,
            why=why,
        )
        row["aggregate"] = aggregate
        _attach_behavior_origin(row, export, payload)
        return row
    row = {
        "key": key,
        "label": label,
        "provider": provider,
        "genome_id": genome_id,
        "feasible": True,
        "infeasible_reason": None,
        "items": items,
        "total_usd": _round(total, 2),
        "score": _round(fitness, 6),
        "why": why,
        "aggregate": aggregate,
    }
    _attach_behavior_origin(row, export, payload)
    return row


def _aggregate_shopping(adapter: ShoppingAdapter, payload: dict[str, Any]) -> dict[str, Any]:
    shoppers = list(SHOPPERS["test"])
    scores: list[float] = []
    feasible = 0
    for shopper in shoppers:
        score = _shop_score(adapter, payload, shopper)
        if score is None:
            continue
        feasible += 1
        scores.append(score)
    items = _outfit_items(adapter, payload)
    prices = [float(item["price_usd"]) for item in items if isinstance(item.get("price_usd"), int | float)]
    total = sum(prices) if prices else None
    budgets = [float(shopper["budget"]) for shopper in shoppers]
    return {
        "score": _round(_mean(scores), 6),
        "n_feasible": feasible,
        "n_shoppers": len(shoppers),
        "mean_total_usd": _round(total, 2),
        "mean_budget_usd": _round(_mean(budgets), 2),
    }


def _infeasible_outfit(
    *,
    key: str,
    label: str,
    provider: str,
    reason: str,
    attempted: dict[str, Any] | None,
    items: list[dict[str, Any]],
    genome_id: str | None = None,
    total: float | None = None,
    why: list[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "provider": provider,
        "genome_id": genome_id,
        "feasible": False,
        "infeasible_reason": reason,
        "items": items,
        "total_usd": None if total is None else _round(total, 2),
        "score": None,
        "why": why or [reason],
    }
    if attempted is not None:
        row["attempted_output"] = attempted
    return row


def _outfit_items(adapter: ShoppingAdapter, payload: dict[str, Any]) -> list[dict[str, Any]]:
    source = str(adapter.catalog.get("source") or "fixture")
    rows: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        product = adapter._by_id.get(str(item.get("product_id")))
        if product is None:
            rows.append(
                {
                    "id": item.get("product_id"),
                    "name": None,
                    "role": item.get("role"),
                    "price_usd": None,
                    "url": None,
                    "image_url": None,
                    "source": source,
                }
            )
            continue
        rows.append(
            {
                "id": product["product_id"],
                "name": product.get("title"),
                "role": product.get("role"),
                "price_usd": product.get("price"),
                "url": product.get("product_url"),
                "image_url": product.get("image_url"),
                "source": source,
            }
        )
    return rows


def _shopper_card(shopper: dict[str, Any]) -> dict[str, Any]:
    palette = str(shopper["palette"])
    if palette == "neutral":
        likes = ["neutral colors", "relaxed and tailored pieces"]
        dislikes = ["neon", "athletic brights"]
    else:
        likes = ["bright colors", "athletic pieces"]
        dislikes = ["muted neutrals"]
    occasions = [str(item) for item in shopper["occasions"]]
    return {
        "id": shopper["id"],
        "style": palette,
        "budget_usd": shopper["budget"],
        "apparel_size": shopper["size"],
        "shoe_size": shopper["shoe_size"],
        "occasions": occasions,
        "likes": likes,
        "dislikes": dislikes,
        "notes": (
            f"Shopper {shopper['id']} wears apparel size {shopper['size']} and shoe size {shopper['shoe_size']}. "
            f"Budget ${shopper['budget']}. Palette {palette}. Occasions: {', '.join(occasions)}."
        ),
    }
