"""Showcase JSON and behavior-only content hashes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from geneai_core.diversity import content_hash
from geneai_core.domains.cloud import (
    TEST_INDEXES,
    CloudSchedulingAdapter,
    baseline_payload,
    round_robin_metrics,
    simulate,
    workload,
    workload_fitness,
)
from geneai_core.domains.shopping import ShoppingAdapter, gym_outfit, neutral_outfit
from geneai_core.schemas import Genome
from geneai_core.showcase import CASE_RULE, build_showcase, select_median_case


def test_content_hash_ignores_labels_and_keeps_behavior() -> None:
    base = baseline_payload("shortest_job")
    renamed = copy.deepcopy(base)
    renamed["description"] = "random perturbation of an elite"
    renamed["policy_family"] = "composed"
    assert content_hash(base) == content_hash(renamed)
    changed = copy.deepcopy(base)
    changed["rules"]["score_weights"]["short_job"] = 0.99
    assert content_hash(changed) != content_hash(base)

    outfit = neutral_outfit()
    labeled = copy.deepcopy(outfit)
    labeled["family"] = "bright_training"
    labeled["styling_notes"] = "not the same sentence"
    assert content_hash(outfit) == content_hash(labeled)
    swapped = copy.deepcopy(outfit)
    swapped["items"] = gym_outfit()["items"]
    assert content_hash(swapped) != content_hash(outfit)
    reordered = copy.deepcopy(outfit)
    reordered["items"] = list(reversed(outfit["items"]))
    assert content_hash(outfit) == content_hash(reordered)
    recolored = copy.deepcopy(outfit)
    for item in recolored["items"]:
        item["color_id"] = "catalog_default"
    assert content_hash(outfit) == content_hash(recolored)

    toy_a = {"family": "weighted_policy", "params": {"x": 0.1, "y": 0.2}, "description": "a"}
    toy_b = {"family": "weighted_policy", "params": {"x": 0.1, "y": 0.2}, "description": "b"}
    toy_c = {"family": "offset_policy", "params": {"x": 0.1, "y": 0.2}, "description": "a"}
    assert content_hash(toy_a) == content_hash(toy_b)
    assert content_hash(toy_a) != content_hash(toy_c)


def test_trace_does_not_change_fitness() -> None:
    payload = baseline_payload("deadline_weighted")
    jobs = workload(1, TEST_INDEXES[0])
    plain = simulate(payload, jobs)
    traced: list[dict] = []
    again = simulate(payload, jobs, trace=traced)
    assert plain == again
    assert len(traced) == int(plain["completed"])
    misses = sum(1 for row in traced if row["missed_deadline"])
    assert misses == pytest.approx(plain["sla_miss_rate"] * len(jobs))

    genome = Genome(
        id="rr",
        run_id="t",
        generation=0,
        operator="seed",
        payload=baseline_payload("round_robin"),
        hypothesis="rr",
        provider="baseline",
        created_at="",
        content_hash="rr",
    )
    evaluation = CloudSchedulingAdapter(seed=1).evaluate(genome, "test")
    fits = []
    for index in TEST_INDEXES:
        metrics = simulate(baseline_payload("round_robin"), workload(1, index))
        fits.append(workload_fitness(metrics, round_robin_metrics(1, index)))
    assert evaluation.fitness == pytest.approx(sum(fits) / len(fits))


def test_median_case_is_the_lower_central_margin() -> None:
    chosen, margin = select_median_case([("b", 3.0), ("a", 1.0), ("c", 2.0), ("d", 4.0)])
    assert (chosen, margin) == ("c", 2.0)
    assert "median" in CASE_RULE
    assert "Not cherry-picked" in CASE_RULE


def _cloud_export() -> dict:
    payload = baseline_payload("shortest_job")
    return {
        "integrity_sha256": "abc123",
        "run": {"id": "run_show", "domain": "cloud_scheduling", "seed": 1},
        "champion_id": "g0060",
        "genomes": [
            {
                "id": "g0051",
                "provider": "creao",
                "operator": "mutate",
                "generation": 1,
                "payload": payload,
                "metadata": {},
            },
            {
                "id": "g0060",
                "provider": "random",
                "operator": "seed",
                "generation": 2,
                "payload": copy.deepcopy(payload) | {"description": "random perturbation", "policy_family": "composed"},
                "metadata": {},
            },
        ],
        "one_shot_baseline": {
            "provider": "frontier",
            "model": "gpt-5.5",
            "payload": {"policy_family": "python", "rules": {}, "description": "bad"},
            "reason": "stored reason is not the re-check",
        },
    }


def test_cloud_showcase_uses_the_median_test_workload(tmp_path: Path) -> None:
    showcase = build_showcase(_cloud_export())
    assert showcase["schema"] == "geneai-showcase-v1"
    assert showcase["split"] == "test"
    assert showcase["export_integrity_sha256"] == "abc123"
    assert "Not cherry-picked" in showcase["case_note"]
    assert showcase["case_id"].startswith("test-")
    cloud = showcase["cloud"]
    assert cloud["units"] == {"time": "minutes", "energy": "kWh"}
    assert {row["id"] for row in cloud["machines"]} == {"a0", "a1", "b0", "b1", "m0", "g0"}
    assert len(cloud["jobs"]) == 36
    keys = [row["key"] for row in cloud["policies"]]
    assert keys[0] == "champion"
    assert "gpt55_one_shot" in keys
    assert "round_robin" in keys
    champion = cloud["policies"][0]
    assert champion["provider"] == "random"
    assert champion["feasible"] is True
    assert champion["rules_plain"]
    assert len(champion["schedule"]) == 36
    assert champion["summary"]["fitness"] is not None
    assert "avg_completion_min" in champion["summary"]
    assert champion["aggregate"]["n_cases"] == len(TEST_INDEXES)
    assert champion["aggregate"]["sla_misses"] >= champion["summary"]["sla_misses"]
    assert champion["aggregate"]["fitness"] is not None
    assert champion["behavior_origin"]["id"] == "g0051"
    assert champion["behavior_origin"]["provider"] == "creao"
    assert champion["behavior_origin"]["operator"] == "mutate"
    assert champion["behavior_origin"]["generation"] == 1
    oneshot_later = next(row for row in cloud["policies"] if row["key"] == "gpt55_one_shot")
    assert "behavior_origin" not in oneshot_later
    assert oneshot_later["aggregate"]["n_cases"] == len(TEST_INDEXES)
    oneshot = next(row for row in cloud["policies"] if row["key"] == "gpt55_one_shot")
    assert oneshot["feasible"] is False
    assert oneshot["attempted_output"]["policy_family"] == "python"
    assert oneshot["infeasible_reason"]
    assert "python" in oneshot["infeasible_reason"] or "policy_family" in oneshot["infeasible_reason"]
    index = int(showcase["case_id"].split("-")[1])
    direct = simulate(baseline_payload("shortest_job"), workload(1, index))
    reference = round_robin_metrics(1, index)
    assert champion["summary"]["fitness"] == pytest.approx(
        workload_fitness(direct, reference), abs=1e-6
    )
    script = Path("scripts/build_showcase.py")
    export_path = tmp_path / "export.json"
    out_path = tmp_path / "showcase.json"
    export_path.write_text(json.dumps(_cloud_export()), encoding="utf-8")
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(script), "--export", str(export_path), "--out", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_path.is_file()
    assert "test-" in completed.stdout


def test_shopping_showcase_scores_one_test_shopper() -> None:
    outfit = neutral_outfit()
    export = {
        "integrity_sha256": "shop",
        "run": {"id": "run_shop", "domain": "shopping", "seed": 1},
        "champion_id": "g0007",
        "genomes": [{"id": "g0007", "provider": "veri", "payload": outfit, "metadata": {}}],
        "one_shot_baseline": {
            "provider": "frontier",
            "model": "gpt-5.5",
            "payload": {
                "family": "neutral_layers",
                "items": [
                    {"product_id": "fx_top_01", "color_id": "black", "size": "M", "role": "top"},
                    {"product_id": "fx_top_02", "color_id": "bone", "size": "M", "role": "top"},
                ],
                "styling_notes": "two tops",
            },
        },
    }
    showcase = build_showcase(export)
    assert showcase["case_id"].startswith("s")
    assert "median" in showcase["case_note"]
    shopper = showcase["shopping"]["shopper"]
    assert shopper["apparel_size"] == "M"
    assert isinstance(shopper["shoe_size"], int)
    outfits = {row["key"]: row for row in showcase["shopping"]["outfits"]}
    assert outfits["champion"]["feasible"] is True
    assert outfits["champion"]["score"] is not None
    assert outfits["champion"]["why"]
    assert any(item["role"] == "shoes" for item in outfits["champion"]["items"])
    assert outfits["champion"]["items"][0]["source"] == "fixture"
    assert outfits["champion"]["aggregate"]["n_shoppers"] == 3
    assert outfits["champion"]["aggregate"]["n_feasible"] == 3
    assert outfits["champion"]["aggregate"]["mean_total_usd"] == outfits["champion"]["total_usd"]
    assert outfits["champion"]["aggregate"]["mean_budget_usd"] is not None
    assert outfits["champion"]["behavior_origin"]["id"] == "g0007"
    assert outfits["gpt55_one_shot"]["aggregate"]["n_feasible"] == 0
    assert outfits["best_baseline"]["provider"] == "baseline"
    oneshot = outfits["gpt55_one_shot"]
    assert oneshot["feasible"] is False
    assert oneshot["attempted_output"]["items"][0]["product_id"] == "fx_top_01"
    assert oneshot["infeasible_reason"]
    assert oneshot["score"] is None
    renamed = copy.deepcopy(export)
    snapshot_items = []
    for item in ShoppingAdapter().catalog_snapshot()["items"]:
        row = dict(item)
        if row["id"] == outfit["items"][0]["product_id"]:
            row["name"] = "Renamed Live Top"
        snapshot_items.append(row)
    renamed["catalog"] = {"catalog_id": "live", "source": "browser_use", "items": snapshot_items}
    named = build_showcase(renamed)
    champion_items = named["shopping"]["outfits"][0]["items"]
    assert champion_items[0]["name"] == "Renamed Live Top"
    assert champion_items[0]["source"] == "browser_use"
