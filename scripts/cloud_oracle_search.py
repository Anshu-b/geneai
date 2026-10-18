#!/usr/bin/env python3
"""Non-GeneAI ceiling for the cloud DSL.

Random samples plus local hill-climbing are scored on train. The best of those
candidates is chosen on validation. Test is read once, after that choice.
This is a reference search, not an evolutionary run and not a sponsor call.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.domains.cloud import (  # noqa: E402
    JOB_CLASSES,
    SCORE_TERMS,
    TIE_BREAKS,
    CloudSchedulingAdapter,
    baseline_payload,
)
from geneai_core.schemas import Genome  # noqa: E402


def _genome(payload: dict) -> Genome:
    return Genome(
        id="oracle",
        run_id="oracle",
        generation=0,
        operator="seed",
        payload=payload,
        hypothesis="oracle search",
        provider="oracle",
        created_at="",
        content_hash="oracle",
    )


def _weights(rng: random.Random, dense: bool) -> dict[str, float]:
    weights = {}
    for term in SCORE_TERMS:
        if not dense and rng.random() < 0.45:
            weights[term] = 0.0
        else:
            weights[term] = round(rng.uniform(-1.0, 1.0), 3)
    return weights


def _tie(rng: random.Random) -> list[str]:
    keys = list(TIE_BREAKS)
    rng.shuffle(keys)
    return keys[: rng.choice((3, 4))]


def random_payload(rng: random.Random) -> dict:
    classes = list(JOB_CLASSES)
    rng.shuffle(classes)
    overrides = []
    for job_class in classes[: rng.choice((0, 0, 1, 1, 2))]:
        overrides.append({"job_class": job_class, "score_weights": _weights(rng, dense=False)})
    return {
        "policy_family": "composed",
        "rules": {
            "max_queue_depth": rng.choice((8, 12, 18, 24, 36)),
            "score_weights": _weights(rng, dense=False),
            "tie_break": _tie(rng),
            "class_overrides": overrides,
        },
        "description": "oracle sample",
    }


def _nudge(value: float, rng: random.Random) -> float:
    updated = value + rng.choice((-1.0, -0.4, -0.15, 0.15, 0.4, 1.0))
    return round(max(-1.0, min(1.0, updated)), 3)


def neighbor(payload: dict, rng: random.Random) -> dict:
    child = copy.deepcopy(payload)
    child["policy_family"] = "composed"
    child["description"] = "oracle neighbor"
    rules = child["rules"]
    kind = rng.randrange(6)
    if kind == 0:
        term = rng.choice(SCORE_TERMS)
        rules["score_weights"][term] = _nudge(float(rules["score_weights"][term]), rng)
    elif kind == 1:
        term = rng.choice(SCORE_TERMS)
        rules["score_weights"][term] = 0.0 if rng.random() < 0.5 else round(rng.uniform(-1.0, 1.0), 3)
    elif kind == 2:
        rules["tie_break"] = _tie(rng)
    elif kind == 3:
        rules["max_queue_depth"] = rng.choice((8, 12, 18, 24, 36))
    elif kind == 4 and rules["class_overrides"]:
        if rng.random() < 0.35:
            rules["class_overrides"].pop(rng.randrange(len(rules["class_overrides"])))
        else:
            override = rules["class_overrides"][rng.randrange(len(rules["class_overrides"]))]
            term = rng.choice(SCORE_TERMS)
            override["score_weights"][term] = _nudge(float(override["score_weights"][term]), rng)
    else:
        used = {item["job_class"] for item in rules["class_overrides"]}
        free = [name for name in JOB_CLASSES if name not in used]
        if free and len(rules["class_overrides"]) < 2:
            rules["class_overrides"].append({"job_class": rng.choice(free), "score_weights": _weights(rng, dense=True)})
        elif rules["class_overrides"]:
            rules["class_overrides"].pop(rng.randrange(len(rules["class_overrides"])))
    return child


def _key(payload: dict) -> str:
    return json.dumps(payload["rules"], sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-GeneAI cloud DSL ceiling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--shortlist", type=int, default=25)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    adapter = CloudSchedulingAdapter(seed=args.seed)
    started = time.perf_counter()

    def train_fitness(payload: dict) -> float | None:
        evaluation = adapter.evaluate(_genome(payload), "train")
        if not evaluation.feasible or evaluation.fitness is None:
            return None
        return float(evaluation.fitness)

    sjf = baseline_payload("shortest_job")
    sjf_train = train_fitness(sjf)
    assert sjf_train is not None
    pool: list[tuple[float, dict]] = [(sjf_train, sjf)]
    seen = {_key(sjf)}
    spent = 1
    beat_train = 0
    for name in ("round_robin", "least_loaded", "deadline_weighted"):
        payload = baseline_payload(name)
        score = train_fitness(payload)
        spent += 1
        if score is None:
            continue
        seen.add(_key(payload))
        pool.append((score, payload))
        if score > sjf_train + 1e-6:
            beat_train += 1

    while spent < args.budget:
        if spent < int(args.budget * 0.55) or not pool:
            payload = random_payload(rng)
        else:
            pool.sort(key=lambda item: item[0], reverse=True)
            parent = pool[rng.randrange(min(12, len(pool)))][1]
            payload = neighbor(parent, rng)
        key = _key(payload)
        if key in seen:
            continue
        seen.add(key)
        score = train_fitness(payload)
        spent += 1
        if score is None:
            continue
        if score > sjf_train + 1e-6:
            beat_train += 1
        pool.append((score, payload))
        if len(pool) > 80:
            pool.sort(key=lambda item: item[0], reverse=True)
            pool = pool[:40]
        if spent % 500 == 0:
            best = max(item[0] for item in pool)
            print(f"train_evals={spent} best_train={best:.6f} sjf_train={sjf_train:.6f} beat_sjf={beat_train}", flush=True)

    pool.sort(key=lambda item: item[0], reverse=True)
    shortlist = pool[: args.shortlist]
    validated: list[tuple[float, float, dict, dict]] = []
    for train_score, payload in shortlist:
        evaluation = adapter.evaluate(_genome(payload), "validation")
        if evaluation.feasible and evaluation.fitness is not None:
            validated.append((float(evaluation.fitness), train_score, payload, dict(evaluation.metrics)))
    if not validated:
        raise SystemExit("no feasible validation candidate")
    validated.sort(key=lambda item: (item[0], item[1]), reverse=True)
    val_score, train_score, winner, winner_val_metrics = validated[0]
    sjf_val = adapter.evaluate(_genome(sjf), "validation")
    winner_test = adapter.evaluate(_genome(winner), "test")
    sjf_test = adapter.evaluate(_genome(sjf), "test")
    baselines = {}
    for name in ("round_robin", "shortest_job", "least_loaded", "deadline_weighted"):
        payload = baseline_payload(name)
        row = {}
        for split in ("train", "validation", "test"):
            evaluation = adapter.evaluate(_genome(payload), split)
            row[split] = evaluation.fitness
        baselines[name] = row
    report = {
        "label": "non-GeneAI reference",
        "method": "random samples plus hill-climb on train, select on validation, test read once",
        "train_evaluations": spent,
        "unique_policies": len(seen),
        "train_policies_beating_shortest_job": beat_train,
        "baselines": baselines,
        "shortest_job": {
            "train": sjf_train,
            "validation": sjf_val.fitness,
            "test": sjf_test.fitness,
        },
        "oracle": {
            "train": train_score,
            "validation": val_score,
            "test": winner_test.fitness,
            "payload": winner,
            "validation_metrics": winner_val_metrics,
            "test_metrics": winner_test.metrics,
        },
        "beats_shortest_job_on_validation": val_score > float(sjf_val.fitness or -1e18) + 1e-6,
        "beats_shortest_job_on_test": float(winner_test.fitness or -1e18) > float(sjf_test.fitness or -1e18) + 1e-6,
        "delta_validation": val_score - float(sjf_val.fitness or 0.0),
        "delta_test": float(winner_test.fitness or 0.0) - float(sjf_test.fitness or 0.0),
        "delta_validation_vs_deadline_weighted": val_score - float(baselines["deadline_weighted"]["validation"] or 0.0),
        "delta_test_vs_deadline_weighted": float(winner_test.fitness or 0.0) - float(baselines["deadline_weighted"]["test"] or 0.0),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
