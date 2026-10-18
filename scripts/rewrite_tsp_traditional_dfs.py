#!/usr/bin/env python3
"""Rewrite only the traditional TSP arm to dfs_bnb. Keep every other method byte-identical."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.domains.tsp import (  # noqa: E402
    HELD_OUT_SEEDS,
    SHOWCASE_SEED,
    cities_for_seed,
    dfs_bnb,
    held_karp,
    tour_length,
    distance_matrix,
)


def main() -> None:
    path = ROOT / "web/public/fixtures/tsp_benchmark.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    # Snapshot non-traditional methods for byte-identical check later
    other_before = json.dumps(
        [m for m in document["methods"] if m["key"] != "held_karp"],
        separators=(",", ":"),
    )

    report = []
    # Showcase
    cities = document["cities"]
    assert cities == cities_for_seed(SHOWCASE_SEED)
    result = dfs_bnb(cities)
    hk_tour, hk_len = held_karp(cities)
    assert abs(result["length"] - hk_len) < 1e-6
    rounded = round(float(result["length"]), 6)
    time_ms = int(result["time_ms"])
    tour = list(result["tour"])
    nodes = int(result["nodes_explored"])
    report.append({"seed": SHOWCASE_SEED, "length": rounded, "time_ms": time_ms, "nodes": nodes})

    dfs_method = {
        "key": "dfs_bnb",
        "label": "Traditional: depth-first search (branch and bound)",
        "provider": "local",
        "tour": tour,
        "length": rounded,
        "gap_pct": 0,
        "time_ms": time_ms,
        "llm_calls": 0,
        "valid": True,
        "invalid_reason": None,
        "convergence": [{"iter": 1, "best_length": rounded, "elapsed_ms": time_ms}],
        "nodes_explored": nodes,
    }

    new_methods = []
    replaced = False
    for method in document["methods"]:
        if method["key"] == "held_karp":
            new_methods.append(dfs_method)
            replaced = True
        else:
            new_methods.append(method)
    if not replaced:
        raise SystemExit("held_karp method not found in fixture")
    document["methods"] = new_methods
    document["optimum"] = {
        "tour": tour,
        "length": rounded,
        "time_ms": time_ms,
        "nodes_explored": nodes,
        "verified_with": "held_karp",
    }

    # Held-out instances
    for instance in document["heldout"]["instances"]:
        seed = int(instance["seed"])
        cities_i = cities_for_seed(seed)
        result_i = dfs_bnb(cities_i)
        _, hk_i = held_karp(cities_i)
        assert abs(result_i["length"] - hk_i) < 1e-6
        rounded_i = round(float(result_i["length"]), 6)
        # Keep optimum_length identical (same length)
        assert abs(rounded_i - float(instance["optimum_length"])) < 1e-6
        old = instance["by_method"].pop("held_karp")
        instance["by_method"]["dfs_bnb"] = {
            "length": rounded_i,
            "gap_pct": 0,
            "time_ms": int(result_i["time_ms"]),
            "valid": True,
            "nodes_explored": int(result_i["nodes_explored"]),
        }
        # Preserve key order roughly: dfs first
        ordered = {"dfs_bnb": instance["by_method"]["dfs_bnb"]}
        for key, row in instance["by_method"].items():
            if key != "dfs_bnb":
                ordered[key] = row
        instance["by_method"] = ordered
        report.append(
            {
                "seed": seed,
                "length": rounded_i,
                "time_ms": int(result_i["time_ms"]),
                "nodes": int(result_i["nodes_explored"]),
                "old_time_ms": old.get("time_ms"),
            }
        )

    # Summary: replace held_karp key with dfs_bnb, recompute time mean from new rows
    summary = document["heldout"]["summary"]
    old_sum = summary.pop("held_karp")
    times = [inst["by_method"]["dfs_bnb"]["time_ms"] for inst in document["heldout"]["instances"]]
    summary_dfs = {
        "mean_gap_pct": 0.0,
        "optimal_hits": 5,
        "mean_time_ms": int(round(sum(times) / len(times))) if times else 0,
        "valid_count": 5,
    }
    new_summary = {"dfs_bnb": summary_dfs}
    for key, row in summary.items():
        new_summary[key] = row
    document["heldout"]["summary"] = new_summary

    notes = document["notes"]
    notes = notes.replace(
        "The traditional method is Held-Karp dynamic programming. optimum is that same search, not a second run, and there is no nearest-neighbor arm.",
        "The traditional method is depth-first search with branch-and-bound. optimum is that same search, not a second run, and there is no nearest-neighbor arm. The optimum length was verified against Held-Karp on every instance.",
    )
    document["notes"] = notes

    other_after = json.dumps(
        [m for m in document["methods"] if m["key"] != "dfs_bnb"],
        separators=(",", ":"),
    )
    if other_before != other_after:
        raise SystemExit("non-traditional methods changed — aborting")

    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(path), "report": report, "old_summary_time": old_sum.get("mean_time_ms")}, indent=2))


if __name__ == "__main__":
    main()
