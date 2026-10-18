"""Liquid funding-policy domain: hard invariants and a seeded mock run."""

from __future__ import annotations

import asyncio
import copy
import random

import pytest

from geneai_core.controller import EvolutionController
from geneai_core.domains.liquid import (
    SPLITS,
    SOURCE_IDS,
    LiquidAdapter,
    baseline_policies,
    base_scenario,
    decode,
    evaluate_scenario,
    liquid_objective,
    mutate_policy,
    normalize_policy,
    price,
    random_policy,
    validate_policy,
)
from geneai_core.experiments import build_providers
from geneai_core.schemas import EvolutionConfig
from geneai_core.store import RunRepository


def _policies(n: int = 200) -> list[dict]:
    rng = random.Random(7)
    pool = list(baseline_policies().values()) + [random_policy(rng) for _ in range(n)]
    pool += [mutate_policy(pool[i % len(pool)], rng)[1] for i in range(n)]
    return pool


def _all_scenarios() -> list[dict]:
    return [scenario for rows in SPLITS.values() for scenario in rows] + [base_scenario()]


def test_sums_exact_floors_kept_nvda_never_sold() -> None:
    for policy in _policies():
        for scenario in _all_scenarios():
            row = evaluate_scenario(policy, scenario)
            alloc = row["allocation_cents"]
            assert row["feasible"], row["constraint_violations"]
            assert sum(alloc.values()) == scenario["purchase_cents"]
            assert all(isinstance(v, int) and v >= 0 for v in alloc.values())
            assert alloc.get("nvda-1", 0) == 0
            assert scenario["savings"]["balance_cents"] - alloc.get("savings", 0) >= scenario["savings"]["minimum_cents"]
            assert alloc.get("checking", 0) <= scenario["checking"]["balance_cents"]
            assert alloc.get("credit", 0) <= scenario["credit"]["maximum_draw_cents"]


def test_nvda_first_policy_still_never_sells_nvda() -> None:
    policy = normalize_policy({"source_priority": ["nvda-1", "credit"], "borrow_cap_fraction_of_purchase": 0})[0]
    row = evaluate_scenario(policy, base_scenario())
    assert row["allocation_cents"].get("nvda-1", 0) == 0
    assert row["feasible"]


def test_loss_lot_gives_no_rebate_and_gain_tax_is_simple() -> None:
    scenario = base_scenario()
    loss = price({"msft-1": 300_000}, scenario)
    assert loss["costs_cents"]["estimated_tax"] == 0
    assert loss["loss_tax_benefit_cents"] == 0
    assert loss["total_cost_cents"] == 0
    assert loss["realized_loss_cents"] == 15_000
    gain = price({"aapl-1": 400_000}, scenario)
    assert gain["costs_cents"]["estimated_tax"] == 10_000  # $500 gain x 20%


def test_no_borrowing_when_disabled_or_unapproved() -> None:
    credit = baseline_policies()["credit_first"]
    for key in ("allow_borrowing", "approved"):
        scenario = base_scenario()
        if key == "allow_borrowing":
            scenario["constraints"]["allow_borrowing"] = False
        else:
            scenario["credit"]["approved"] = False
        row = evaluate_scenario(credit, scenario)
        assert row["allocation_cents"].get("credit", 0) == 0
        assert row["feasible"]


def test_apr_and_savings_monotone() -> None:
    scenario = base_scenario()
    low = price({"credit": 200_000, "checking": 50_000, "savings": 50_000}, scenario)
    scenario["credit"]["apr"] = 0.12
    high = price({"credit": 200_000, "checking": 50_000, "savings": 50_000}, scenario)
    assert high["costs_cents"]["borrowing_interest"] >= low["costs_cents"]["borrowing_interest"]
    a = price({"savings": 100_000, "msft-1": 200_000}, base_scenario())
    b = price({"savings": 200_000, "msft-1": 100_000}, base_scenario())
    assert b["costs_cents"]["foregone_savings_interest"] >= a["costs_cents"]["foregone_savings_interest"]


def test_infeasible_when_purchase_exceeds_hard_capacity() -> None:
    scenario = base_scenario()
    scenario["purchase_cents"] = 10_000_000
    row = evaluate_scenario(baseline_policies()["cash_savings_first"], scenario)
    assert not row["feasible"]
    assert row["fitness"] is None


def test_malformed_policy_rejected_and_normalized() -> None:
    with pytest.raises(ValueError):
        validate_policy({"source_priority": ["checking"]})
    fixed, notes = normalize_policy({"source_priority": ["NVDA", "bogus", "credit"], "risk_weight": 9, "lot_selection": "??"})
    validate_policy(fixed)
    assert sorted(fixed["source_priority"]) == sorted(SOURCE_IDS)
    assert fixed["risk_weight"] == 1.0
    assert notes


def test_deterministic() -> None:
    policy = baseline_policies()["tax_aware_sell_first"]
    assert evaluate_scenario(policy, base_scenario()) == evaluate_scenario(copy.deepcopy(policy), base_scenario())
    assert decode(policy, base_scenario()) == decode(policy, base_scenario())


def test_seeded_mock_run_has_lineage_and_feasible_champion(tmp_path) -> None:
    adapter = LiquidAdapter()
    config = EvolutionConfig(seed=3, population_size=8, generations=3, elite_count=2, max_wall_seconds=60, max_model_calls=60, max_evaluations=400)
    result = asyncio.run(
        EvolutionController().evolve(
            objective=liquid_objective(),
            adapter=adapter,
            config=config,
            providers=build_providers("mock", seed=3),
            mode="mock",
            repository=RunRepository(tmp_path),
        )
    )
    summary = result.summary
    assert summary["champion_id"]
    assert summary["champion_feasible"]
    validate_policy(summary["champion_payload"])
    assert summary["generations_completed"] >= 1
