"""Liquid v2 (liquid-laptop-v2): a multi-term funding tradeoff.

No source is free. Savings forgoes 4.5% APY, every lot is at a gain (one
short-term at 30%), the ETF also forgoes dividends when sold, a
securities-backed line (SBL) needs pledged collateral (pledged lots cannot be
sold, the draw is capped at 50% loan-to-value, and a -30% market move that
pushes LTV over 50% is a margin call), and a 0%-for-60-days card jumps to 24%
for the rest of the 90-day horizon. Integer cents throughout.

Fictional data. Simplified tax model. Not financial advice.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.domains.liquid import _cents_times, _clamp, _round_div
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

EVALUATOR_VERSION = "liquid-model-v2"
SCENARIO_ID = "liquid-laptop-v2"
LAMBDA = 1.0
EMERGENCY_CENTS = 200_000
EMERGENCY_SHORTFALL_RATE = 0.05
MARGIN_CALL_PENALTY_CENTS = 15_000
MAX_LTV = 0.50
RATE_SHOCK_PP = 0.05
MARKET_DROP = 0.30
LOT_SELECTIONS = ("as_listed", "min_tax_first")
LOT_IDS = ("amzn-1", "aapl-1", "etf-1", "nvda-1")
SOURCE_IDS = ("checking", "savings", "amzn-1", "aapl-1", "etf-1", "nvda-1", "sbl", "intro_card")
CREDIT_IDS = ("sbl", "intro_card")

BASE_SCENARIO: dict[str, Any] = {
    "scenario_id": SCENARIO_ID,
    "currency": "USD",
    "purchase_cents": 600_000,
    "horizon_days": 90,
    "checking": {"balance_cents": 50_000, "minimum_cents": 0},
    "savings": {"balance_cents": 900_000, "minimum_cents": 500_000, "apy": 0.045},
    "lots": [
        {"lot_id": "amzn-1", "symbol": "AMZN", "market_value_cents": 300_000, "basis_cents": 285_000, "holding_period": "short", "dividend_yield": 0.0, "sellable": True, "pledgeable": True},
        {"lot_id": "aapl-1", "symbol": "AAPL", "market_value_cents": 600_000, "basis_cents": 510_000, "holding_period": "long", "dividend_yield": 0.005, "sellable": True, "pledgeable": True},
        {"lot_id": "etf-1", "symbol": "ETF", "market_value_cents": 800_000, "basis_cents": 400_000, "holding_period": "long", "dividend_yield": 0.03, "sellable": True, "pledgeable": True},
        {"lot_id": "nvda-1", "symbol": "NVDA", "market_value_cents": 700_000, "basis_cents": 200_000, "holding_period": "long", "dividend_yield": 0.0, "sellable": False, "pledgeable": False},
    ],
    "sbl": {"approved": True, "apr": 0.07, "max_ltv": MAX_LTV, "maximum_draw_cents": 500_000, "fee_cents": 0},
    "intro_card": {"approved": True, "intro_apr": 0.0, "intro_days": 60, "apr_after": 0.24, "limit_cents": 150_000, "fee_cents": 0},
    "assumptions": {"long_term_gain_rate": 0.20, "short_term_gain_rate": 0.30, "loss_tax_benefit_mode": "none", "trade_fee_cents": 0},
    "constraints": {"allow_borrowing": True, "protect": ["NVDA"]},
}


def _variant(name: str, note: str, edit: dict[str, Any]) -> dict[str, Any]:
    s = copy.deepcopy(BASE_SCENARIO)
    s["scenario_id"] = f"{SCENARIO_ID}:{name}"
    s["variant_note"] = note
    for key, value in edit.items():
        if key == "lot_basis":
            for lot in s["lots"]:
                if lot["lot_id"] in value:
                    lot["basis_cents"] = value[lot["lot_id"]]
        elif key == "market":
            for lot in s["lots"]:
                lot["market_value_cents"] = int(round(lot["market_value_cents"] * (1 + value)))
                lot["basis_cents"] = lot["basis_cents"]
        elif key == "sbl_apr":
            s["sbl"]["apr"] = value
        elif key == "card_apr":
            s["intro_card"]["apr_after"] = value
        elif key == "no_card":
            s["intro_card"]["approved"] = False
        elif key == "savings_balance":
            s["savings"]["balance_cents"] = value
        else:
            s[key] = value
    return s


SPLITS: dict[str, list[dict[str, Any]]] = {
    "train": [
        _variant("base", "the default $6,000 purchase", {}),
        _variant("big", "an $8,000 purchase", {"purchase_cents": 800_000}),
        _variant("cheap-sbl", "SBL at 5% and no intro card", {"sbl_apr": 0.05, "no_card": True}),
    ],
    "validation": [
        _variant("down10", "a $7,000 purchase after a 10% market dip", {"purchase_cents": 700_000, "market": -0.10}),
        _variant("pricey-card", "card jumps to 29% and AMZN has a bigger gain (basis $2,550)", {"card_apr": 0.29, "lot_basis": {"amzn-1": 255_000}}),
    ],
    "test": [
        _variant("small", "a $5,000 purchase with $7,000 in savings", {"purchase_cents": 500_000, "savings_balance": 700_000}),
        _variant("dear-sbl", "SBL at 9%, AAPL bought at $4,500, a $6,500 purchase", {"sbl_apr": 0.09, "lot_basis": {"aapl-1": 450_000}, "purchase_cents": 650_000}),
    ],
}


def base_scenario() -> dict[str, Any]:
    return copy.deepcopy(BASE_SCENARIO)


def _lot(s: dict[str, Any], lot_id: str) -> dict[str, Any] | None:
    return next((lot for lot in s["lots"] if lot["lot_id"] == lot_id), None)


def _protected(s: dict[str, Any], lot: dict[str, Any]) -> bool:
    return (not lot.get("sellable", True)) or lot["symbol"] in set(s["constraints"].get("protect") or [])


def lot_tax_rate(s: dict[str, Any], lot: dict[str, Any]) -> float:
    gain = lot["market_value_cents"] - lot["basis_cents"]
    if gain <= 0:
        return 0.0
    key = "long_term_gain_rate" if lot["holding_period"] == "long" else "short_term_gain_rate"
    return gain / lot["market_value_cents"] * float(s["assumptions"][key])


def interest(cents: int, rate: float, days: int) -> int:
    return int(round(cents * rate * days / 365.0))


def card_interest(cents: int, card: dict[str, Any], days: int) -> int:
    intro_days = min(days, int(card["intro_days"]))
    return interest(cents, float(card["intro_apr"]), intro_days) + interest(cents, float(card["apr_after"]), days - intro_days)


# ------------------------------------------------------------- genome

def normalize_policy(payload: Any) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    raw = payload if isinstance(payload, dict) else {}
    aliases = {"amzn": "amzn-1", "aapl": "aapl-1", "etf": "etf-1", "nvda": "nvda-1", "credit": "sbl", "securities_backed_line": "sbl", "card": "intro_card", "intro": "intro_card"}
    priority: list[str] = []
    for item in raw.get("source_priority") or []:
        token = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if token in SOURCE_IDS and token not in priority:
            priority.append(token)
    missing = [x for x in SOURCE_IDS if x not in priority]
    if missing and raw.get("source_priority"):
        notes.append(f"appended missing sources {missing}")
    priority.extend(missing)
    fractions: dict[str, float] = {}
    rf = raw.get("max_source_fraction") if isinstance(raw.get("max_source_fraction"), dict) else {}
    for key, value in rf.items():
        token = aliases.get(str(key).strip().lower(), str(key).strip().lower())
        if token in SOURCE_IDS:
            f = round(_clamp(value, 0.0, 1.0, 1.0), 2)
            if f < 1.0:
                fractions[token] = f
    pledge: list[str] = []
    for item in raw.get("pledge") or []:
        token = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if token in LOT_IDS and token != "nvda-1" and token not in pledge:
            pledge.append(token)
    lot_selection = str(raw.get("lot_selection") or "min_tax_first").strip().lower()
    if lot_selection not in LOT_SELECTIONS:
        lot_selection = "min_tax_first"
    buffer_cents = int(round(_clamp(raw.get("target_cash_buffer_cents", 0), 0, 400_000, 0) / 5_000.0)) * 5_000
    policy = {
        "source_priority": priority,
        "max_source_fraction": dict(sorted(fractions.items())),
        "target_cash_buffer_cents": buffer_cents,
        "pledge": sorted(pledge),
        "max_ltv_target": round(_clamp(raw.get("max_ltv_target", 0.35), 0.0, MAX_LTV, 0.35), 2),
        "lot_selection": lot_selection,
        "description": str(raw.get("description") or "").strip()[:200] or "funding policy",
    }
    return policy, notes


def validate_policy(p: Any) -> None:
    if not isinstance(p, dict):
        raise ValueError("payload must be an object")
    req = {"source_priority", "max_source_fraction", "target_cash_buffer_cents", "pledge", "max_ltv_target", "lot_selection", "description"}
    if set(p) != req:
        raise ValueError(f"policy fields must be exactly {sorted(req)}")
    if not isinstance(p["source_priority"], list) or sorted(p["source_priority"]) != sorted(SOURCE_IDS) or len(p["source_priority"]) != len(SOURCE_IDS):
        raise ValueError(f"source_priority must list each of {list(SOURCE_IDS)} once")
    if not isinstance(p["max_source_fraction"], dict) or any(k not in SOURCE_IDS or isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for k, v in p["max_source_fraction"].items()):
        raise ValueError("max_source_fraction must map sources to [0, 1]")
    if not isinstance(p["pledge"], list) or any(x not in LOT_IDS or x == "nvda-1" for x in p["pledge"]):
        raise ValueError("pledge must list sellable lot ids")
    b = p["target_cash_buffer_cents"]
    if isinstance(b, bool) or not isinstance(b, int) or not 0 <= b <= 400_000:
        raise ValueError("target_cash_buffer_cents must be an integer in [0, 400000]")
    v = p["max_ltv_target"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= MAX_LTV:
        raise ValueError(f"max_ltv_target must be within [0, {MAX_LTV}]")
    if p["lot_selection"] not in LOT_SELECTIONS:
        raise ValueError("bad lot_selection")
    if not isinstance(p["description"], str) or not p["description"].strip():
        raise ValueError("description must be non-empty")


# ------------------------------------------------------------- decoder

def effective_order(p: dict[str, Any], s: dict[str, Any]) -> list[str]:
    order = list(p["source_priority"])
    if p["lot_selection"] == "min_tax_first":
        pos = [i for i, x in enumerate(order) if x in LOT_IDS]
        lots = sorted((order[i] for i in pos), key=lambda lid: (lot_tax_rate(s, _lot(s, lid)) + float(_lot(s, lid)["dividend_yield"]) * s["horizon_days"] / 365.0, lid))
        for i, lid in zip(pos, lots):
            order[i] = lid
    return order


def pledged_value(p: dict[str, Any], s: dict[str, Any]) -> int:
    return sum(_lot(s, lid)["market_value_cents"] for lid in p["pledge"] if _lot(s, lid))


def hard_capacity(src: str, p: dict[str, Any], s: dict[str, Any]) -> int:
    if src == "checking":
        return max(0, s["checking"]["balance_cents"] - s["checking"]["minimum_cents"])
    if src == "savings":
        return max(0, s["savings"]["balance_cents"] - s["savings"]["minimum_cents"])
    if src == "sbl":
        if not (s["sbl"]["approved"] and s["constraints"]["allow_borrowing"]):
            return 0
        return max(0, min(s["sbl"]["maximum_draw_cents"], int(pledged_value(p, s) * float(s["sbl"]["max_ltv"]))))
    if src == "intro_card":
        if not (s["intro_card"]["approved"] and s["constraints"]["allow_borrowing"]):
            return 0
        return s["intro_card"]["limit_cents"]
    lot = _lot(s, src)
    if lot is None or _protected(s, lot) or src in p["pledge"]:
        return 0
    return lot["market_value_cents"]


def decode(p: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    purchase = int(s["purchase_cents"])
    order = effective_order(p, s)
    alloc = {x: 0 for x in SOURCE_IDS}
    remaining = purchase
    repairs: list[str] = []
    for src in order:
        if remaining <= 0:
            break
        cap = hard_capacity(src, p, s)
        if src == "savings":
            cap = max(0, cap - int(p["target_cash_buffer_cents"]))
        if src == "sbl":
            cap = min(cap, int(pledged_value(p, s) * float(p["max_ltv_target"])))
        f = p["max_source_fraction"].get(src)
        if f is not None:
            cap = min(cap, _round_div(purchase * int(round(float(f) * 100)), 100))
        take = min(cap, remaining)
        alloc[src] += take
        remaining -= take
    if remaining > 0:
        repairs.append(f"soft limits left ${remaining / 100:,.2f} unfunded; repair relaxed buffer, fractions and LTV target")
        for src in order:
            if remaining <= 0:
                break
            take = max(0, min(hard_capacity(src, p, s) - alloc[src], remaining))
            alloc[src] += take
            remaining -= take
    return {"allocation_cents": {k: v for k, v in alloc.items() if v > 0}, "order": order, "unfunded_cents": remaining, "repairs": repairs}


# ------------------------------------------------------------- evaluator

def check_invariants(alloc: dict[str, int], p: dict[str, Any], s: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if sum(alloc.values()) != s["purchase_cents"]:
        out.append("sources do not sum to the purchase")
    for src, c in alloc.items():
        if not isinstance(c, int) or c < 0:
            out.append(f"{src} negative or non-integer")
        elif c > hard_capacity(src, p, s):
            out.append(f"{src} exceeds capacity")
        if src in p["pledge"] and c > 0:
            out.append(f"{src} is pledged and cannot be sold")
        lot = _lot(s, src)
        if lot is not None and c > 0 and _protected(s, lot):
            out.append(f"{lot['symbol']} is protected")
    pv = pledged_value(p, s)
    if alloc.get("sbl", 0) > 0 and (pv == 0 or alloc["sbl"] > pv * float(s["sbl"]["max_ltv"]) + 1e-9):
        out.append("SBL draw exceeds the loan-to-value cap")
    return out


def price(alloc: dict[str, int], p: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    days = int(s["horizon_days"])
    tax = div = gain_total = 0
    sales = []
    for lot in s["lots"]:
        sold = int(alloc.get(lot["lot_id"], 0))
        if sold <= 0:
            continue
        basis = _round_div(lot["basis_cents"] * sold, lot["market_value_cents"])
        gain = sold - basis
        key = "long_term_gain_rate" if lot["holding_period"] == "long" else "short_term_gain_rate"
        t = _cents_times(max(0, gain), float(s["assumptions"][key]))
        d = interest(sold, float(lot["dividend_yield"]), days)
        tax += t
        div += d
        gain_total += gain
        sales.append({"lot_id": lot["lot_id"], "symbol": lot["symbol"], "proceeds_cents": sold, "gain_cents": gain, "tax_cents": t, "dividends_forgone_cents": d})
    ledger = {
        "estimated_tax": tax,
        "foregone_savings_interest": interest(int(alloc.get("savings", 0)), float(s["savings"]["apy"]), days),
        "dividends_forgone": div,
        "sbl_interest": interest(int(alloc.get("sbl", 0)), float(s["sbl"]["apr"]), days),
        "card_interest": card_interest(int(alloc.get("intro_card", 0)), s["intro_card"], days),
        "fees": (s["sbl"]["fee_cents"] if alloc.get("sbl") else 0) + (s["intro_card"]["fee_cents"] if alloc.get("intro_card") else 0),
    }
    return {"costs_cents": ledger, "total_cost_cents": sum(ledger.values()), "realized_gain_cents": gain_total, "realized_loss_cents": 0, "loss_tax_benefit_cents": 0, "sales": sales}


def stress(alloc: dict[str, int], p: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    days = int(s["horizon_days"])
    sbl = int(alloc.get("sbl", 0))
    card = int(alloc.get("intro_card", 0))
    shocked_card = dict(s["intro_card"], apr_after=float(s["intro_card"]["apr_after"]) + RATE_SHOCK_PP)
    extra = (interest(sbl, float(s["sbl"]["apr"]) + RATE_SHOCK_PP, days) - interest(sbl, float(s["sbl"]["apr"]), days)) + (card_interest(card, shocked_card, days) - card_interest(card, s["intro_card"], days))
    pv_drop = int(round(pledged_value(p, s) * (1 - MARKET_DROP)))
    ltv_after = None if sbl == 0 else (sbl / pv_drop if pv_drop else float("inf"))
    margin_call = sbl > 0 and ltv_after > MAX_LTV
    remaining_lots = sum(lot["market_value_cents"] - int(alloc.get(lot["lot_id"], 0)) for lot in s["lots"])
    checking_left = s["checking"]["balance_cents"] - int(alloc.get("checking", 0))
    savings_left = s["savings"]["balance_cents"] - int(alloc.get("savings", 0))
    above = checking_left + savings_left - s["savings"]["minimum_cents"]
    shortfall = max(0, EMERGENCY_CENTS - above)
    return {
        "rate_shock": {"label": "Credit APRs +5 points", "passed": extra == 0, "extra_interest_cents": extra},
        "market_drop": {"label": "Market falls 30%", "passed": not margin_call, "portfolio_after_drop_cents": int(round(remaining_lots * (1 - MARKET_DROP))), "loan_to_value_after_drop": None if ltv_after is None else round(ltv_after, 4), "margin_call": margin_call},
        "emergency": {"label": "$2,000 emergency after purchase", "passed": shortfall == 0, "cash_above_reserve_cents": above, "shortfall_cents": shortfall},
        "post_purchase": {"checking_cents": checking_left, "savings_cents": savings_left, "pledged_cents": pledged_value(p, s), "pledged_lots": list(p["pledge"])},
    }


def risk_penalty(st: dict[str, Any]) -> dict[str, int]:
    return {
        "emergency_shortfall": _cents_times(st["emergency"]["shortfall_cents"], EMERGENCY_SHORTFALL_RATE),
        "rate_shock_interest": int(st["rate_shock"]["extra_interest_cents"]),
        "margin_call": MARGIN_CALL_PENALTY_CENTS if st["market_drop"]["margin_call"] else 0,
    }


def evaluate_scenario(p: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    dec = decode(p, s)
    alloc = dec["allocation_cents"]
    viol = check_invariants(alloc, p, s)
    if dec["unfunded_cents"] > 0:
        viol.append(f"cannot fund ${dec['unfunded_cents'] / 100:,.2f} within hard limits")
    pr = price(alloc, p, s)
    st = stress(alloc, p, s)
    rk = risk_penalty(st)
    rt = sum(rk.values())
    ok = not viol
    return {
        "scenario_id": s["scenario_id"], "feasible": ok, "constraint_violations": viol, "allocation_cents": alloc,
        "order": dec["order"], "repairs": dec["repairs"], **pr, "stress": st, "risk_breakdown_cents": rk,
        "risk_penalty_cents": rt,
        "stress_passes": sum(1 for k in ("rate_shock", "market_drop", "emergency") if st[k]["passed"]),
        "fitness": -round((pr["total_cost_cents"] + LAMBDA * rt) / 100.0, 4) if ok else None,
        "lambda": LAMBDA, "assumptions_version": EVALUATOR_VERSION,
    }


# ------------------------------------------------------------- baselines

def baseline_policies() -> dict[str, dict[str, Any]]:
    n = lambda d: normalize_policy(d)[0]  # noqa: E731
    return {
        "cash_savings_first": n({"source_priority": ["checking", "savings", "amzn-1", "aapl-1", "etf-1", "intro_card", "sbl"], "lot_selection": "as_listed", "description": "Spend checking, then savings to the floor, then sell lots in order."}),
        "tax_aware_sell_first": n({"source_priority": ["amzn-1", "aapl-1", "etf-1", "checking", "savings", "intro_card", "sbl"], "lot_selection": "min_tax_first", "description": "Sell lots first, lowest tax first, then cash."}),
        "credit_first": n({"source_priority": ["intro_card", "sbl", "checking", "savings", "amzn-1", "aapl-1", "etf-1"], "pledge": ["etf-1"], "max_ltv_target": 0.5, "lot_selection": "as_listed", "description": "Borrow first: intro card, then an SBL against the ETF at full LTV, then cash."}),
        "proportional_split": n({"source_priority": ["checking", "savings", "amzn-1", "aapl-1", "intro_card", "sbl", "etf-1"], "max_source_fraction": {"checking": 0.2, "savings": 0.2, "amzn-1": 0.2, "aapl-1": 0.2, "intro_card": 0.2}, "pledge": ["etf-1"], "lot_selection": "as_listed", "description": "Take an equal fifth from checking, savings, AMZN, AAPL and the card."}),
    }


BASELINE_LABELS = {"cash_savings_first": "Cash/savings first", "tax_aware_sell_first": "Tax-aware sell first", "credit_first": "Credit first", "proportional_split": "Proportional split"}
NAMES = {"checking": "checking", "savings": "savings", "amzn-1": "AMZN (short-term)", "aapl-1": "AAPL", "etf-1": "the ETF", "nvda-1": "NVDA", "sbl": "the securities-backed line", "intro_card": "the 0%-intro card"}


def describe_policy(p: dict[str, Any], s: dict[str, Any] | None = None) -> str:
    s = s or BASE_SCENARIO
    order = [x for x in effective_order(p, s) if hard_capacity(x, p, s) > 0]
    parts = [f"Draw from {', then '.join(NAMES[x] for x in order[:5])}."]
    b = int(p["target_cash_buffer_cents"])
    parts.append(f"Keep an extra ${b / 100:,.0f} in savings above the floor." if b else "Savings may go down to the floor, never below.")
    if p["pledge"]:
        parts.append(f"Pledge {', '.join(NAMES[x] for x in p['pledge'])} as collateral (never sold) and keep the line under {float(p['max_ltv_target']):.0%} loan-to-value.")
    else:
        parts.append("Pledge nothing, so the securities-backed line is unused.")
    for src, f in sorted(p["max_source_fraction"].items()):
        if src == "nvda-1" or f >= 1.0:
            continue
        parts.append(f"Never use {NAMES[src]}." if f <= 0 else f"Take at most {f:.0%} of the price from {NAMES[src]}.")
    parts.append("NVDA is protected.")
    return " ".join(parts)


def mutate_policy(p: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    c = copy.deepcopy(p)
    move = rng.choice(["swap", "move", "buffer", "lot_selection", "fraction", "fraction", "pledge", "ltv"])
    o = c["source_priority"]
    if move == "swap":
        i, j = rng.sample(range(len(o)), 2)
        o[i], o[j] = o[j], o[i]
    elif move == "move":
        x = o.pop(rng.randrange(len(o)))
        o.insert(rng.randrange(len(o) + 1), x)
    elif move == "buffer":
        c["target_cash_buffer_cents"] = max(0, min(400_000, c["target_cash_buffer_cents"] + rng.choice([-50_000, -25_000, 25_000, 50_000, 100_000])))
    elif move == "lot_selection":
        c["lot_selection"] = "as_listed" if c["lot_selection"] == "min_tax_first" else "min_tax_first"
    elif move == "fraction":
        src = rng.choice([x for x in SOURCE_IDS if x != "nvda-1"])
        v = rng.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0])
        if v >= 1.0:
            c["max_source_fraction"].pop(src, None)
        else:
            c["max_source_fraction"][src] = v
    elif move == "pledge":
        lid = rng.choice(["amzn-1", "aapl-1", "etf-1"])
        c["pledge"] = [x for x in c["pledge"] if x != lid] if lid in c["pledge"] else c["pledge"] + [lid]
    else:
        c["max_ltv_target"] = rng.choice([0.1, 0.2, 0.3, 0.35, 0.4, 0.5])
    c["description"] = f"algorithmic {move} mutation"
    return f"mutate:{move}", normalize_policy(c)[0]


def crossover_policy(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    n = len(a["source_priority"])
    i, j = sorted(rng.sample(range(n), 2))
    head = a["source_priority"][i : j + 1]
    rest = [x for x in b["source_priority"] if x not in head]
    pick = lambda k: rng.choice([a, b])[k]  # noqa: E731
    return normalize_policy({"source_priority": rest[:i] + head + rest[i:], "max_source_fraction": dict(pick("max_source_fraction")), "target_cash_buffer_cents": pick("target_cash_buffer_cents"), "pledge": list(pick("pledge")), "max_ltv_target": pick("max_ltv_target"), "lot_selection": pick("lot_selection"), "description": "algorithmic crossover"})[0]


def random_policy(rng: random.Random) -> dict[str, Any]:
    o = list(SOURCE_IDS)
    rng.shuffle(o)
    return normalize_policy({"source_priority": o, "target_cash_buffer_cents": rng.choice([0, 100_000, 200_000]), "pledge": rng.sample(["amzn-1", "aapl-1", "etf-1"], rng.randrange(3)), "max_ltv_target": rng.choice([0.2, 0.35, 0.5]), "lot_selection": rng.choice(LOT_SELECTIONS), "description": "random policy"})[0]


def liquid_objective() -> ObjectiveSpec:
    return ObjectiveSpec(
        domain="liquid",
        prompt="Fund a $6,000 purchase over 90 days at the lowest modeled cost plus risk from checking, savings, taxable lots, a securities-backed line and a 0%-intro card.",
        constraints={"scenario_id": SCENARIO_ID, "sources": list(SOURCE_IDS), "protected": ["NVDA"], "lambda": LAMBDA},
        evaluation_version=EVALUATOR_VERSION,
    )


_SHAPE = ('{"source_priority":["checking","savings","amzn-1","intro_card","sbl","aapl-1","etf-1","nvda-1"],'
          '"max_source_fraction":{"savings":0.4},"target_cash_buffer_cents":100000,"pledge":["etf-1"],'
          '"max_ltv_target":0.3,"lot_selection":"min_tax_first","description":"one short line"}')
_FACTS = (
    "Buy $6,000 over 90 days. checking $500 (0%). savings $9,000, 4.5% APY, floor $5,000. "
    "Lots: AMZN $3,000 basis $2,850 short-term (30% tax); AAPL $6,000 basis $5,100 (20% tax, 0.5% dividend); "
    "ETF $8,000 basis $4,000 (20% tax, 3% dividend lost if sold); NVDA protected. "
    "sbl: 7% APR, draw up to 50% of pledged lots (pledged lots cannot be sold); max_ltv_target caps the draw. "
    "intro_card: up to $1,500, 0% for 60 days then 24%. "
    "Risk: 5% of any shortfall below $2,000 cash above the floor; extra interest if APRs +5pp; $150 margin call if a 30% drop pushes LTV over 50%. "
    "Fitness = -(cost + risk), averaged over other purchase sizes, APRs and markets."
)


class LiquidAdapter(DomainAdapter):
    name = "liquid"
    evaluator_version = EVALUATOR_VERSION

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def genome_schema(self) -> dict[str, Any]:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["source_priority", "max_source_fraction", "target_cash_buffer_cents", "pledge", "max_ltv_target", "lot_selection", "description"],
            "properties": {
                "source_priority": {"type": "array", "items": {"type": "string", "enum": list(SOURCE_IDS)}},
                "max_source_fraction": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}},
                "target_cash_buffer_cents": {"type": "integer", "minimum": 0, "maximum": 400000},
                "pledge": {"type": "array", "items": {"type": "string", "enum": ["amzn-1", "aapl-1", "etf-1"]}},
                "max_ltv_target": {"type": "number", "minimum": 0, "maximum": MAX_LTV},
                "lot_selection": {"type": "string", "enum": list(LOT_SELECTIONS)},
                "description": {"type": "string"},
            },
        }

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        return {f"{k}_hash": hashlib.sha256(f"{EVALUATOR_VERSION}|{k}|{seed}|{json.dumps(SPLITS[k], sort_keys=True)}".encode()).hexdigest() for k in ("train", "validation", "test")}

    def family_of(self, payload: dict[str, Any]) -> str:
        pr = payload.get("source_priority") if isinstance(payload, dict) else None
        first = pr[0] if isinstance(pr, list) and pr else "mixed"
        return "cash-first" if first in {"checking", "savings"} else "credit-first" if first in CREDIT_IDS else "sell-first"

    def baselines(self) -> dict[str, dict[str, Any]]:
        return baseline_policies()

    def seeded_baselines(self) -> list[dict[str, Any]]:
        return [{"name": k, "payload": v, "hypothesis": f"baseline {BASELINE_LABELS[k]}"} for k, v in baseline_policies().items()]

    def realize_proposal(self, payload: dict[str, Any], parent_payloads: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        del rng
        if not isinstance(payload, dict) or not parent_payloads or not isinstance(parent_payloads[0], dict):
            return payload
        merged = dict(parent_payloads[0])
        for k, v in payload.items():
            merged[k] = {**(merged.get(k) or {}), **v} if k == "max_source_fraction" and isinstance(v, dict) else v
        return merged

    def prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        return normalize_policy(payload)

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        del objective
        validate_policy(payload)

    def oneshot_hint(self) -> str:
        return ("Return one funding policy as payload. " + _FACTS + " source_priority must list each of " + json.dumps(list(SOURCE_IDS)) + " exactly once. An invalid policy is not repaired. Example: " + _SHAPE)

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return "Return JSON only. payload is one funding policy shaped like " + _SHAPE + ". hypothesis is one short line.\n" + _FACTS + f"\nObjective: {objective.prompt}"

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return ("Return JSON only. payload changes one or two fields of the parent policy; omitted fields are copied. hypothesis says why in one line.\n" + _FACTS + f"\nParent: {json.dumps(parent.payload, separators=(',', ':'))}\nFeedback: {feedback.feedback[:400]}")

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return ("Return JSON only. payload is one full child policy combining A and B. hypothesis is one short line.\n" + _FACTS + f"\nA: {json.dumps(a.payload, separators=(',', ':'))}\nB: {json.dumps(b.payload, separators=(',', ':'))}")

    def random_seed_payloads(self, rng: random.Random, count: int) -> list[dict[str, Any]]:
        return [random_policy(rng) for _ in range(count)]

    def uses_memetic_fallback(self) -> bool:
        return False

    def algorithmic_offspring(self, payloads: list[dict[str, Any]], rng: random.Random, count: int) -> list[tuple[str, dict[str, Any]]]:
        pool = [normalize_policy(x)[0] for x in payloads if isinstance(x, dict)] or [random_policy(rng)]
        out = []
        for i in range(max(0, count)):
            if len(pool) > 1 and i % 3 == 0:
                a, b = rng.sample(pool, 2)
                out.append(("crossover", crossover_policy(a, b, rng)))
            else:
                out.append(mutate_policy(rng.choice(pool), rng))
        return out

    def local_perturbations(self, payload: dict[str, Any], rng: random.Random, count: int) -> list[dict[str, Any]]:
        p = normalize_policy(payload)[0]
        return [mutate_policy(p, rng)[1] for _ in range(count)]

    def alter_duplicate(self, payload: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        return mutate_policy(normalize_policy(payload)[0], rng)[1]

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        t0 = time.perf_counter()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split}")

        def fail(r: str) -> Evaluation:
            return Evaluation(genome_id=genome.id, feasible=False, fitness=None, metrics={}, constraint_violations=[r], feedback=f"infeasible: {r}", evaluator_version=EVALUATOR_VERSION, data_split=split, elapsed_ms=int((time.perf_counter() - t0) * 1000))  # type: ignore[arg-type]

        try:
            validate_policy(genome.payload)
        except ValueError as exc:
            return fail(str(exc))
        rows = [evaluate_scenario(genome.payload, s) for s in SPLITS[split]]
        bad = [r for r in rows if not r["feasible"]]
        if bad:
            return fail(f"{bad[0]['scenario_id']}: {'; '.join(bad[0]['constraint_violations'])}")
        n = len(rows)
        cost = sum(r["total_cost_cents"] for r in rows) / n / 100
        risk = sum(r["risk_penalty_cents"] for r in rows) / n / 100
        fb = " | ".join(f"{r['scenario_id'].split(':')[-1]}: cost ${r['total_cost_cents'] / 100:.2f} risk ${r['risk_penalty_cents'] / 100:.2f} alloc {','.join(f'{k}={v // 100}' for k, v in r['allocation_cents'].items())}" for r in rows)
        return Evaluation(genome_id=genome.id, feasible=True, fitness=round(sum(r["fitness"] for r in rows) / n, 4),
                          metrics={"mean_cost_usd": round(cost, 4), "mean_risk_usd": round(risk, 4), "stress_passes": float(sum(r["stress_passes"] for r in rows)), "stress_checks": float(3 * n), "scenarios": float(n)},
                          constraint_violations=[], feedback=f"{split}: mean cost ${cost:.2f}, mean risk ${risk:.2f}. " + fb,
                          evaluator_version=EVALUATOR_VERSION, data_split=split, elapsed_ms=int((time.perf_counter() - t0) * 1000))  # type: ignore[arg-type]

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]:
        return {"domain": self.name, "genome_id": genome.id, "policy": genome.payload, "feasible": evaluation.feasible, "fitness": evaluation.fitness, "metrics": evaluation.metrics, "split": evaluation.data_split, "hypothesis": genome.hypothesis}
