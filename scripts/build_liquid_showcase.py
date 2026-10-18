#!/usr/bin/env python3
"""Build web/public/fixtures/liquid.json (+ liquid_run.json) from one real Liquid run.

  uv run python scripts/build_liquid_showcase.py --run /path/to/run_xxx --web web/public/fixtures

Every number is recomputed by the Liquid evaluator from saved policies, or read
from the run export (lineage, providers, fitness by generation).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.domains.liquid_v2 import (  # noqa: E402
    BASELINE_LABELS,
    EMERGENCY_CENTS,
    EMERGENCY_SHORTFALL_RATE,
    EVALUATOR_VERSION,
    LAMBDA,
    MARGIN_CALL_PENALTY_CENTS,
    MAX_LTV,
    SPLITS,
    LiquidAdapter,
    base_scenario,
    baseline_policies,
    describe_policy,
    evaluate_scenario,
    normalize_policy,
    validate_policy,
)
from geneai_core.schemas import Genome  # noqa: E402


def _genome(payload: dict[str, Any], gid: str) -> Genome:
    return Genome(id=gid, run_id="showcase", generation=0, operator="seed", payload=payload, hypothesis=gid,
                  provider="showcase", created_at="", content_hash=gid)


def _splits(adapter: LiquidAdapter, payload: dict[str, Any], gid: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        ev = adapter.evaluate(_genome(payload, gid), split)
        out[split] = {"feasible": ev.feasible, "fitness": ev.fitness, "metrics": ev.metrics,
                      "violations": ev.constraint_violations}
    return out


def _candidate(adapter: LiquidAdapter, key: str, label: str, provider: str, payload: dict[str, Any] | None,
               *, valid_reason: str | None = None) -> dict[str, Any]:
    if payload is None:
        return {"key": key, "label": label, "provider": provider, "valid": False, "invalid_reason": valid_reason}
    try:
        validate_policy(payload)
    except ValueError as exc:
        return {"key": key, "label": label, "provider": provider, "valid": False, "invalid_reason": str(exc), "policy": payload}
    return {
        "key": key,
        "label": label,
        "provider": provider,
        "valid": True,
        "policy": payload,
        "summary": describe_policy(payload),
        "base": evaluate_scenario(payload, base_scenario()),
        "splits": _splits(adapter, payload, key),
        "variants": {
            split: [
                {"scenario_id": s["scenario_id"], "note": s["variant_note"], **{k: v for k, v in evaluate_scenario(payload, s).items() if k in {"feasible", "total_cost_cents", "risk_penalty_cents", "fitness", "stress_passes", "allocation_cents"}}}
                for s in rows
            ]
            for split, rows in SPLITS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--web", required=True, type=Path)
    args = parser.parse_args()
    export = json.loads((args.run / "export.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    adapter = LiquidAdapter()
    genomes = {g["id"]: g for g in export["genomes"]}
    champ_id = export["champion_id"]
    champ = genomes[champ_id]

    contributed = sorted({g["provider"] for g in export["genomes"]} & {"veri", "creao"})
    hybrid_label = "hybrid run · genomes from " + (" + ".join(contributed) or "no model") + (" · CREAO wrote none" if "creao" not in contributed else "")
    candidates = [_candidate(adapter, "geneai", "GeneAI champion", hybrid_label, champ["payload"])]
    one = export.get("one_shot_baseline") or {}
    one_payload = one.get("payload") if isinstance(one.get("payload"), dict) else None
    one_label = "GPT-5.5 one-shot"
    one_cand = _candidate(adapter, "one_shot", one_label, f"OpenAI · {one.get('model') or 'model not recorded'}", one_payload,
                          valid_reason=str(one.get("reason") or one.get("status") or "not run"))
    one_cand["oneshot_meta"] = {k: one.get(k) for k in ("status", "model", "latency_ms", "cost_usd", "prompt_tokens", "completion_tokens", "comparison_label", "compute_matched")}
    if one_payload is not None:
        one_cand["normalized_note"] = "Evaluated exactly as returned. Not repaired." if one_cand.get("valid") else None
    candidates.append(one_cand)
    for name, payload in baseline_policies().items():
        candidates.append(_candidate(adapter, name, BASELINE_LABELS[name], "handwritten rule", payload))

    # Fitness and providers per generation from the real run.
    evals_train: dict[str, dict[str, Any]] = {}
    for ev in export["evaluations"]:
        if ev.get("data_split") == "train":
            evals_train[ev["genome_id"]] = ev
    gens: dict[int, dict[str, Any]] = {}
    for g in export["genomes"]:
        row = gens.setdefault(g["generation"], {"generation": g["generation"], "born": 0, "by_provider": Counter(), "best_fitness": None, "best_id": None})
        row["born"] += 1
        row["by_provider"][g["provider"]] += 1
        ev = evals_train.get(g["id"])
        if ev and ev.get("feasible") and ev.get("fitness") is not None:
            if row["best_fitness"] is None or ev["fitness"] > row["best_fitness"]:
                row["best_fitness"] = ev["fitness"]
                row["best_id"] = g["id"]
    best_so_far = None
    generations = []
    for gen in sorted(gens):
        row = gens[gen]
        if row["best_fitness"] is not None:
            best_so_far = row["best_fitness"] if best_so_far is None else max(best_so_far, row["best_fitness"])
        generations.append({**row, "by_provider": dict(row["by_provider"]), "best_so_far": best_so_far})

    # Champion ancestry.
    path = []
    seen = set()
    frontier = [champ_id]
    while frontier:
        gid = frontier.pop(0)
        if gid in seen or gid not in genomes:
            continue
        seen.add(gid)
        g = genomes[gid]
        ev = evals_train.get(gid) or {}
        path.append({"id": gid, "generation": g["generation"], "operator": g["metadata"].get("origin_operator") or g["operator"],
                     "provider": g["provider"], "parent_ids": g["parent_ids"], "hypothesis": g["hypothesis"][:240],
                     "train_fitness": ev.get("fitness"), "baseline_name": g["metadata"].get("baseline_name")})
        frontier.extend(g["parent_ids"])
    path.sort(key=lambda r: (r["generation"], r["id"]))

    provider_counts = Counter(g["provider"] for g in export["genomes"])
    events = Counter(e["type"] for e in export["events"])
    providers_meta = export.get("providers") or []
    doc = {
        "schema": "geneai-liquid-v1",
        "mode": "saved_run_replay",
        "run": {
            "id": export["run"]["id"],
            "mode": export["run"]["mode"],
            "recorded_at": export.get("recorded_at"),
            "stop_reason": summary.get("stop_reason"),
            "generations_completed": summary.get("generations_completed"),
            "population_size": export["config"]["population_size"],
            "wall_seconds": (summary.get("budget") or {}).get("elapsed_seconds"),
            "model_calls": (summary.get("budget") or {}).get("model_calls"),
            "evaluations": (summary.get("budget") or {}).get("evaluations"),
            "git_sha": export["run"].get("git_sha"),
            "evaluator_version": EVALUATOR_VERSION,
            "champion_id": champ_id,
            "champion_selected_on": "validation",
        },
        "scenario": base_scenario(),
        "variants": {split: [{"scenario_id": s["scenario_id"], "note": s["variant_note"]} for s in rows] for split, rows in SPLITS.items()},
        "model": {
            "lambda": LAMBDA,
            "emergency_cents": EMERGENCY_CENTS,
            "emergency_shortfall_rate": EMERGENCY_SHORTFALL_RATE,
            "margin_call_penalty_cents": MARGIN_CALL_PENALTY_CENTS,
            "max_ltv": MAX_LTV,
            "cost_formula": "tax on realized gains (30% short-term, 20% long-term) + savings interest foregone (4.5% APY x 90/365) + dividends forgone on sold lots + securities-backed line interest (7% APR x 90/365) + card interest (0% for 60 days, then 24% for 30 days) + fees",
            "risk_formula": "5% of any shortfall below $2,000 of cash above the savings floor + extra interest if every credit APR rises 5 points + $150 margin call if a 30% market drop pushes the securities-backed line over 50% loan-to-value of the pledged lots",
            "fitness_formula": "fitness = -(modeled cost + lambda x risk penalty), lambda = 1. Mean over a split's scenario variants.",
        },
        "candidates": candidates,
        "generations": generations,
        "champion_lineage": path,
        "provenance": {
            "genomes_by_provider": dict(provider_counts),
            "run_provenance_counts": summary.get("provenance_counts"),
            "behavior_origin": summary.get("behavior_origin") or {"id": champ_id, "provider": champ["provider"], "operator": champ["metadata"].get("origin_operator") or champ["operator"], "generation": champ["generation"]},
            "champion_hypothesis": champ["hypothesis"],
            "providers": providers_meta,
            "event_counts": dict(events),
        },
    }
    args.web.mkdir(parents=True, exist_ok=True)
    (args.web / "liquid.json").write_text(json.dumps(doc, indent=2))
    # Bundled export for the lineage graph. Drop bulky event payload.task like the other bundles.
    slim = dict(export)
    slim_events = []
    for e in export["events"]:
        e2 = dict(e)
        if isinstance(e2.get("payload"), dict) and "task" in e2["payload"]:
            e2["payload"] = {k: v for k, v in e2["payload"].items() if k != "task"}
        slim_events.append(e2)
    slim["events"] = slim_events
    (args.web / "liquid_run.json").write_text(json.dumps(slim))
    print(json.dumps({"run": doc["run"], "provenance": doc["provenance"]["genomes_by_provider"], "behavior_origin": doc["provenance"]["behavior_origin"],
                      "table": [{"key": c["key"], "valid": c["valid"], "base_cost": c.get("base", {}).get("total_cost_cents"), "base_risk": c.get("base", {}).get("risk_penalty_cents"),
                                 "stress": c.get("base", {}).get("stress_passes"), "val": c.get("splits", {}).get("validation", {}).get("fitness"), "test": c.get("splits", {}).get("test", {}).get("fitness"),
                                 "train": c.get("splits", {}).get("train", {}).get("fitness")} for c in candidates]}, indent=1))


if __name__ == "__main__":
    main()
