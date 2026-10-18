"""liquid-laptop-v2 invariants: LTV, pledges, intro APR, dividends, baselines not optimal."""

from __future__ import annotations

import random

from geneai_core.domains.liquid_v2 import (
    SPLITS,
    base_scenario,
    baseline_policies,
    card_interest,
    evaluate_scenario,
    mutate_policy,
    normalize_policy,
    pledged_value,
    price,
    random_policy,
)


def _pool(n: int = 300) -> list[dict]:
    rng = random.Random(11)
    pool = list(baseline_policies().values()) + [random_policy(rng) for _ in range(n)]
    return pool + [mutate_policy(pool[i % len(pool)], rng)[1] for i in range(n)]


def _scenarios() -> list[dict]:
    return [s for rows in SPLITS.values() for s in rows]


def test_invariants_hold_for_every_decoded_policy() -> None:
    for p in _pool():
        for s in _scenarios():
            r = evaluate_scenario(p, s)
            a = r["allocation_cents"]
            if not r["feasible"]:
                continue
            assert sum(a.values()) == s["purchase_cents"]
            assert a.get("nvda-1", 0) == 0
            for lid in p["pledge"]:
                assert a.get(lid, 0) == 0, "pledged lot was sold"
            assert a.get("sbl", 0) <= pledged_value(p, s) * s["sbl"]["max_ltv"]
            assert s["savings"]["balance_cents"] - a.get("savings", 0) >= s["savings"]["minimum_cents"]
            assert a.get("intro_card", 0) <= s["intro_card"]["limit_cents"]


def test_sbl_needs_a_pledge() -> None:
    p = normalize_policy({"source_priority": ["sbl"], "pledge": []})[0]
    assert evaluate_scenario(p, base_scenario())["allocation_cents"].get("sbl", 0) == 0


def test_intro_card_math() -> None:
    card = base_scenario()["intro_card"]
    # 0% for 60 days, then 24% for 30 days on $1,000.
    assert card_interest(100_000, card, 90) == round(100_000 * 0.24 * 30 / 365)
    assert card_interest(100_000, card, 60) == 0


def test_dividend_forgone_and_short_term_tax() -> None:
    s = base_scenario()
    etf = price({"etf-1": 100_000}, normalize_policy({})[0], s)
    assert etf["costs_cents"]["dividends_forgone"] == round(100_000 * 0.03 * 90 / 365)
    assert etf["costs_cents"]["estimated_tax"] == 10_000  # $500 gain x 20%
    amzn = price({"amzn-1": 300_000}, normalize_policy({})[0], s)
    assert amzn["costs_cents"]["estimated_tax"] == 4_500  # $150 gain x 30%


def test_no_borrowing_when_disabled() -> None:
    s = base_scenario()
    s["constraints"]["allow_borrowing"] = False
    r = evaluate_scenario(baseline_policies()["credit_first"], s)
    assert r["allocation_cents"].get("sbl", 0) == 0 and r["allocation_cents"].get("intro_card", 0) == 0


def test_no_baseline_is_optimal_on_train() -> None:
    def fit(p: dict) -> float | None:
        vals = [evaluate_scenario(p, s)["fitness"] for s in SPLITS["train"]]
        return None if None in vals else sum(vals) / len(vals)

    best_baseline = max(fit(p) for p in baseline_policies().values())
    best_search = max(f for f in (fit(p) for p in _pool()) if f is not None)
    assert best_search > best_baseline + 1
