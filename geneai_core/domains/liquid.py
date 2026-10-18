"""Liquid: fund a purchase from the cheapest dollars you own.

The genome is a heritable funding POLICY, not a dollar allocation. A
deterministic decoder turns the policy into an allocation for one scenario,
with hard-constraint repair. The evaluator then prices that allocation with a
simplified, disclosed cost model in integer cents:

  modeled cost = tax on realized gains (losses earn no rebate)
               + savings interest foregone over the horizon (simple interest)
               + borrowing interest over the horizon (simple interest)
               + trade and origination fees

Risk is a separate, disclosed penalty (``risk_breakdown``). Fitness is
``-(cost_usd + LAMBDA * risk_usd)``. Fitness is not a bill.

All data is fictional. This is a simplified illustration, not financial advice.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

EVALUATOR_VERSION = "liquid-model-v1"
SCENARIO_ID = "liquid-laptop-v1"
LAMBDA = 1.0
EMERGENCY_CENTS = 200_000
EMERGENCY_SHORTFALL_RATE = 0.05  # risk dollars per dollar of emergency shortfall
COLLATERAL_FAIL_PENALTY_CENTS = 10_000  # $100 when a financed plan breaches max LTV after a -30% drop
MAX_LTV_AFTER_DROP = 0.50
RATE_SHOCK_PP = 0.05
MARKET_DROP = 0.30
LOT_SELECTIONS = ("as_listed", "min_tax_first")
CASH_SOURCES = ("checking", "savings")

BASE_SCENARIO: dict[str, Any] = {
    "scenario_id": SCENARIO_ID,
    "currency": "USD",
    "purchase_cents": 300_000,
    "horizon_days": 90,
    "checking": {"balance_cents": 50_000, "minimum_cents": 0},
    "savings": {"balance_cents": 800_000, "minimum_cents": 500_000, "apy": 0.04},
    "lots": [
        {"lot_id": "nvda-1", "symbol": "NVDA", "market_value_cents": 500_000, "basis_cents": 200_000, "holding_period": "long", "sellable": False},
        {"lot_id": "aapl-1", "symbol": "AAPL", "market_value_cents": 400_000, "basis_cents": 350_000, "holding_period": "long", "sellable": True},
        {"lot_id": "msft-1", "symbol": "MSFT", "market_value_cents": 600_000, "basis_cents": 630_000, "holding_period": "long", "sellable": True},
        {"lot_id": "etf-1", "symbol": "ETF", "market_value_cents": 500_000, "basis_cents": 400_000, "holding_period": "long", "sellable": True},
    ],
    "credit": {"approved": True, "maximum_draw_cents": 300_000, "apr": 0.09, "fee_cents": 0},
    "assumptions": {"long_term_gain_rate": 0.20, "short_term_gain_rate": 0.30, "loss_tax_benefit_mode": "none", "trade_fee_cents": 0},
    "constraints": {"allow_borrowing": True, "max_modeled_borrow_cents": 300_000, "protect": ["NVDA"]},
}

SOURCE_IDS = ("checking", "savings", "nvda-1", "aapl-1", "msft-1", "etf-1", "credit")


def _variant(name: str, note: str, edit: dict[str, Any]) -> dict[str, Any]:
    scenario = copy.deepcopy(BASE_SCENARIO)
    scenario["scenario_id"] = f"{SCENARIO_ID}:{name}"
    scenario["variant_note"] = note
    for key, value in edit.items():
        if key == "lot_basis":
            for lot in scenario["lots"]:
                if lot["lot_id"] in value:
                    lot["basis_cents"] = value[lot["lot_id"]]
        elif key == "market":
            for lot in scenario["lots"]:
                lot["market_value_cents"] = int(round(lot["market_value_cents"] * (1 + value)))
        elif key == "apr":
            scenario["credit"]["apr"] = value
        elif key == "savings_balance":
            scenario["savings"]["balance_cents"] = value
        elif key == "checking_balance":
            scenario["checking"]["balance_cents"] = value
        else:
            scenario[key] = value
    return scenario


# Scenario variants per split. The champion has to be a policy that generalizes,
# not an allocation memorized for one balance sheet.
SPLITS: dict[str, list[dict[str, Any]]] = {
    "train": [
        _variant("base", "the default $3,000 laptop", {}),
        _variant("msft-gain", "MSFT has recovered to a gain (basis $4,500)", {"lot_basis": {"msft-1": 450_000}}),
        _variant("big-purchase", "a $5,000 purchase", {"purchase_cents": 500_000}),
    ],
    "validation": [
        _variant("thin-savings", "savings is only $6,500, MSFT is at a gain (basis $5,200), $3,500 purchase", {"savings_balance": 650_000, "purchase_cents": 350_000, "lot_basis": {"msft-1": 520_000}}),
        _variant("rate-up", "credit APR is 12% and the market is down 10%", {"apr": 0.12, "market": -0.10}),
    ],
    "test": [
        _variant("all-gains", "every lot is at a gain (MSFT basis $5,000)", {"lot_basis": {"msft-1": 500_000}, "purchase_cents": 400_000}),
        _variant("low-cash", "checking has $100 and the market is down 15%", {"checking_balance": 10_000, "market": -0.15}),
    ],
}


def base_scenario() -> dict[str, Any]:
    return copy.deepcopy(BASE_SCENARIO)


def _round_div(numer: int, denom: int) -> int:
    """Round half up for non-negative integers."""

    if denom <= 0:
        raise ValueError("denominator must be positive")
    return (2 * numer + denom) // (2 * denom)


def _cents_times(cents: int, rate: float) -> int:
    return int(round(cents * rate))


def _interest(cents: int, rate: float, days: int) -> int:
    return int(round(cents * rate * days / 365.0))


def _protected(scenario: dict[str, Any], lot: dict[str, Any]) -> bool:
    protect = set(scenario.get("constraints", {}).get("protect") or [])
    return (not lot.get("sellable", True)) or lot["symbol"] in protect


def _lot(scenario: dict[str, Any], lot_id: str) -> dict[str, Any] | None:
    for lot in scenario["lots"]:
        if lot["lot_id"] == lot_id:
            return lot
    return None


def credit_allowed(scenario: dict[str, Any]) -> bool:
    return bool(scenario["credit"].get("approved")) and bool(scenario["constraints"].get("allow_borrowing"))


def lot_tax_rate(scenario: dict[str, Any], lot: dict[str, Any]) -> float:
    """Estimated tax per dollar of proceeds. Loss lots are 0 (no rebate)."""

    gain = lot["market_value_cents"] - lot["basis_cents"]
    if gain <= 0 or lot["market_value_cents"] <= 0:
        return 0.0
    rate_key = "long_term_gain_rate" if lot.get("holding_period") == "long" else "short_term_gain_rate"
    return gain / lot["market_value_cents"] * float(scenario["assumptions"][rate_key])


# ---------------------------------------------------------------- genome repair

def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


def normalize_policy(payload: Any) -> tuple[dict[str, Any], list[str]]:
    """Repair a policy into the canonical schema. Notes record each change."""

    notes: list[str] = []
    raw = payload if isinstance(payload, dict) else {}
    if not isinstance(payload, dict):
        notes.append("payload was not an object")
    priority: list[str] = []
    for item in raw.get("source_priority") or []:
        token = str(item).strip().lower()
        aliases = {"nvda": "nvda-1", "aapl": "aapl-1", "msft": "msft-1", "etf": "etf-1", "credit_line": "credit", "loc": "credit"}
        token = aliases.get(token, token)
        if token in SOURCE_IDS and token not in priority:
            priority.append(token)
    missing = [source for source in SOURCE_IDS if source not in priority]
    if missing and raw.get("source_priority"):
        notes.append(f"appended missing sources {missing}")
    priority.extend(missing)
    fractions: dict[str, float] = {}
    raw_fractions = raw.get("max_source_fraction") if isinstance(raw.get("max_source_fraction"), dict) else {}
    for key, value in raw_fractions.items():
        token = str(key).strip().lower()
        token = {"nvda": "nvda-1", "aapl": "aapl-1", "msft": "msft-1", "etf": "etf-1"}.get(token, token)
        if token in SOURCE_IDS:
            fraction = round(_clamp(value, 0.0, 1.0, 1.0), 2)
            if fraction < 1.0:
                fractions[token] = fraction
    lot_selection = str(raw.get("lot_selection") or "min_tax_first").strip().lower()
    if lot_selection in {"min_estimated_tax_cost", "min_tax", "lowest_tax_first", "tax_aware"}:
        lot_selection = "min_tax_first"
    if lot_selection not in LOT_SELECTIONS:
        notes.append(f"lot_selection {lot_selection!r} -> min_tax_first")
        lot_selection = "min_tax_first"
    buffer_cents = int(round(_clamp(raw.get("target_cash_buffer_cents", 0), 0, 300_000, 0) / 5_000.0)) * 5_000
    policy = {
        "source_priority": priority,
        "max_source_fraction": dict(sorted(fractions.items())),
        "target_cash_buffer_cents": buffer_cents,
        "borrow_cap_fraction_of_purchase": round(_clamp(raw.get("borrow_cap_fraction_of_purchase", 0.0), 0.0, 1.0, 0.0), 2),
        "lot_selection": lot_selection,
        "risk_weight": round(_clamp(raw.get("risk_weight", 0.5), 0.0, 1.0, 0.5), 2),
        "description": str(raw.get("description") or "").strip()[:200] or "funding policy",
    }
    return policy, notes


def validate_policy(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    required = {"source_priority", "max_source_fraction", "target_cash_buffer_cents", "borrow_cap_fraction_of_purchase", "lot_selection", "risk_weight", "description"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"missing policy fields: {sorted(missing)}")
    extra = set(payload) - required
    if extra:
        raise ValueError(f"unknown policy fields: {sorted(extra)}")
    priority = payload["source_priority"]
    if not isinstance(priority, list) or sorted(priority) != sorted(SOURCE_IDS) or len(priority) != len(SOURCE_IDS):
        raise ValueError(f"source_priority must list each of {list(SOURCE_IDS)} exactly once")
    if not isinstance(payload["max_source_fraction"], dict):
        raise ValueError("max_source_fraction must be an object")
    for key, value in payload["max_source_fraction"].items():
        if key not in SOURCE_IDS:
            raise ValueError(f"unknown source in max_source_fraction: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError(f"max_source_fraction[{key}] must be within [0, 1]")
    buffer = payload["target_cash_buffer_cents"]
    if isinstance(buffer, bool) or not isinstance(buffer, int) or not 0 <= buffer <= 300_000:
        raise ValueError("target_cash_buffer_cents must be an integer in [0, 300000]")
    for key in ("borrow_cap_fraction_of_purchase", "risk_weight"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be within [0, 1]")
    if payload["lot_selection"] not in LOT_SELECTIONS:
        raise ValueError(f"lot_selection must be one of {list(LOT_SELECTIONS)}")
    if not isinstance(payload["description"], str) or not payload["description"].strip():
        raise ValueError("description must be a non-empty string")


# ---------------------------------------------------------------- decoder

def effective_order(policy: dict[str, Any], scenario: dict[str, Any]) -> list[str]:
    order = list(policy["source_priority"])
    if policy["lot_selection"] == "min_tax_first":
        positions = [index for index, source in enumerate(order) if source.endswith("-1")]
        lots = [order[index] for index in positions]
        lots.sort(key=lambda lot_id: (lot_tax_rate(scenario, _lot(scenario, lot_id) or {"market_value_cents": 1, "basis_cents": 1}), lot_id))
        for index, lot_id in zip(positions, lots):
            order[index] = lot_id
    if float(policy["risk_weight"]) >= 0.5 and "credit" in order:
        order.remove("credit")
        order.append("credit")
    return order


def _hard_capacity(source: str, scenario: dict[str, Any]) -> int:
    if source == "checking":
        return max(0, scenario["checking"]["balance_cents"] - scenario["checking"]["minimum_cents"])
    if source == "savings":
        return max(0, scenario["savings"]["balance_cents"] - scenario["savings"]["minimum_cents"])
    if source == "credit":
        if not credit_allowed(scenario):
            return 0
        return max(0, min(scenario["credit"]["maximum_draw_cents"], scenario["constraints"]["max_modeled_borrow_cents"]))
    lot = _lot(scenario, source)
    if lot is None or _protected(scenario, lot):
        return 0
    return max(0, lot["market_value_cents"])


def decode(policy: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Deterministic policy -> allocation. Pass 1 honors soft policy limits;
    pass 2 (repair) relaxes only soft limits, never floors, protection, or credit caps."""

    purchase = int(scenario["purchase_cents"])
    order = effective_order(policy, scenario)
    allocation = {source: 0 for source in SOURCE_IDS}
    remaining = purchase
    repairs: list[str] = []
    for source in order:
        if remaining <= 0:
            break
        cap = _hard_capacity(source, scenario)
        if source == "savings":
            cap = max(0, cap - int(policy["target_cash_buffer_cents"]))
        if source == "credit":
            cap = min(cap, _round_div(purchase * int(round(float(policy["borrow_cap_fraction_of_purchase"]) * 100)), 100))
        fraction = policy["max_source_fraction"].get(source)
        if fraction is not None:
            cap = min(cap, _round_div(purchase * int(round(float(fraction) * 100)), 100))
        take = min(cap, remaining)
        allocation[source] += take
        remaining -= take
    if remaining > 0:
        repairs.append(f"soft limits left ${remaining / 100:,.2f} unfunded; repair relaxed buffer, fractions and borrow cap")
        for source in order:
            if remaining <= 0:
                break
            room = _hard_capacity(source, scenario) - allocation[source]
            take = max(0, min(room, remaining))
            allocation[source] += take
            remaining -= take
    return {
        "allocation_cents": {key: value for key, value in allocation.items() if value > 0},
        "order": order,
        "unfunded_cents": remaining,
        "repairs": repairs,
    }


# ---------------------------------------------------------------- evaluator

def check_invariants(allocation: dict[str, int], scenario: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    total = sum(allocation.values())
    if total != scenario["purchase_cents"]:
        violations.append(f"sources sum to {total} cents, purchase is {scenario['purchase_cents']}")
    for source, cents in allocation.items():
        if not isinstance(cents, int) or cents < 0:
            violations.append(f"{source} has a negative or non-integer amount")
            continue
        if cents > _hard_capacity(source, scenario):
            violations.append(f"{source} exceeds its eligible capacity")
        lot = _lot(scenario, source)
        if lot is not None and cents > 0 and _protected(scenario, lot):
            violations.append(f"{lot['symbol']} is protected and cannot be sold")
    if allocation.get("credit", 0) > 0 and not credit_allowed(scenario):
        violations.append("borrowing is not approved or not allowed")
    return violations


def price(allocation: dict[str, int], scenario: dict[str, Any]) -> dict[str, Any]:
    days = int(scenario["horizon_days"])
    tax = 0
    realized_gain = 0
    realized_loss = 0
    fees = 0
    sales: list[dict[str, Any]] = []
    for lot in scenario["lots"]:
        sold = int(allocation.get(lot["lot_id"], 0))
        if sold <= 0:
            continue
        basis = _round_div(lot["basis_cents"] * sold, lot["market_value_cents"])
        gain = sold - basis
        rate_key = "long_term_gain_rate" if lot.get("holding_period") == "long" else "short_term_gain_rate"
        lot_tax = _cents_times(max(0, gain), float(scenario["assumptions"][rate_key]))
        tax += lot_tax
        if gain >= 0:
            realized_gain += gain
        else:
            realized_loss += -gain
        fees += int(scenario["assumptions"].get("trade_fee_cents", 0))
        sales.append({"lot_id": lot["lot_id"], "symbol": lot["symbol"], "proceeds_cents": sold, "basis_cents": basis, "gain_cents": gain, "tax_cents": lot_tax})
    savings_used = int(allocation.get("savings", 0))
    borrowed = int(allocation.get("credit", 0))
    foregone = _interest(savings_used, float(scenario["savings"]["apy"]), days)
    borrow_interest = _interest(borrowed, float(scenario["credit"]["apr"]), days)
    if borrowed > 0:
        fees += int(scenario["credit"].get("fee_cents", 0))
    ledger = {
        "estimated_tax": tax,
        "foregone_savings_interest": foregone,
        "borrowing_interest": borrow_interest,
        "fees": fees,
    }
    return {
        "costs_cents": ledger,
        "total_cost_cents": sum(ledger.values()),
        "realized_gain_cents": realized_gain,
        "realized_loss_cents": realized_loss,
        "loss_tax_benefit_cents": 0,
        "sales": sales,
    }


def stress(allocation: dict[str, int], scenario: dict[str, Any]) -> dict[str, Any]:
    days = int(scenario["horizon_days"])
    borrowed = int(allocation.get("credit", 0))
    apr = float(scenario["credit"]["apr"])
    extra_interest = _interest(borrowed, apr + RATE_SHOCK_PP, days) - _interest(borrowed, apr, days)
    remaining_lots = sum(lot["market_value_cents"] - int(allocation.get(lot["lot_id"], 0)) for lot in scenario["lots"])
    after_drop = int(round(remaining_lots * (1 - MARKET_DROP)))
    ltv = (borrowed / after_drop) if after_drop > 0 else (0.0 if borrowed == 0 else float("inf"))
    checking_left = scenario["checking"]["balance_cents"] - int(allocation.get("checking", 0))
    savings_left = scenario["savings"]["balance_cents"] - int(allocation.get("savings", 0))
    cash_above_floor = checking_left + (savings_left - scenario["savings"]["minimum_cents"])
    shortfall = max(0, EMERGENCY_CENTS - cash_above_floor)
    return {
        "rate_shock": {"label": "Credit APR +5 points", "passed": extra_interest == 0, "extra_interest_cents": extra_interest},
        "market_drop": {
            "label": "Market falls 30%",
            "passed": borrowed == 0 or ltv <= MAX_LTV_AFTER_DROP,
            "portfolio_after_drop_cents": after_drop,
            "loan_to_value_after_drop": None if borrowed == 0 else round(ltv, 4),
        },
        "emergency": {
            "label": "$2,000 emergency after purchase",
            "passed": shortfall == 0,
            "cash_above_reserve_cents": cash_above_floor,
            "shortfall_cents": shortfall,
        },
        "post_purchase": {"checking_cents": checking_left, "savings_cents": savings_left},
    }


def risk_penalty(stress_result: dict[str, Any]) -> dict[str, int]:
    emergency = _cents_times(stress_result["emergency"]["shortfall_cents"], EMERGENCY_SHORTFALL_RATE)
    rate = int(stress_result["rate_shock"]["extra_interest_cents"])
    collateral = 0 if stress_result["market_drop"]["passed"] else COLLATERAL_FAIL_PENALTY_CENTS
    return {"emergency_shortfall": emergency, "rate_shock_interest": rate, "collateral_breach": collateral}


def evaluate_scenario(policy: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    decoded = decode(policy, scenario)
    allocation = decoded["allocation_cents"]
    violations = check_invariants(allocation, scenario)
    if decoded["unfunded_cents"] > 0:
        violations.append(f"cannot fund ${decoded['unfunded_cents'] / 100:,.2f} within hard limits")
    priced = price(allocation, scenario)
    stressed = stress(allocation, scenario)
    risk = risk_penalty(stressed)
    risk_total = sum(risk.values())
    feasible = not violations
    fitness = None
    if feasible:
        fitness = -round((priced["total_cost_cents"] + LAMBDA * risk_total) / 100.0, 4)
    return {
        "scenario_id": scenario["scenario_id"],
        "feasible": feasible,
        "constraint_violations": violations,
        "allocation_cents": allocation,
        "order": decoded["order"],
        "repairs": decoded["repairs"],
        **priced,
        "stress": stressed,
        "risk_breakdown_cents": risk,
        "risk_penalty_cents": risk_total,
        "stress_passes": sum(1 for key in ("rate_shock", "market_drop", "emergency") if stressed[key]["passed"]),
        "fitness": fitness,
        "lambda": LAMBDA,
        "assumptions_version": EVALUATOR_VERSION,
    }


# ---------------------------------------------------------------- baselines

def baseline_policies() -> dict[str, dict[str, Any]]:
    return {
        "cash_savings_first": normalize_policy({
            "source_priority": ["checking", "savings", "etf-1", "aapl-1", "msft-1", "credit"],
            "lot_selection": "as_listed", "borrow_cap_fraction_of_purchase": 1.0, "risk_weight": 0.5,
            "description": "Spend checking, then savings down to the $5,000 floor, then sell lots in a fixed order, borrow last.",
        })[0],
        "tax_aware_sell_first": normalize_policy({
            "source_priority": ["aapl-1", "msft-1", "etf-1", "checking", "savings", "credit"],
            "lot_selection": "min_tax_first", "borrow_cap_fraction_of_purchase": 1.0, "risk_weight": 0.5,
            "description": "Sell eligible lots first, lowest estimated tax first, then cash, borrow last.",
        })[0],
        "credit_first": normalize_policy({
            "source_priority": ["credit", "checking", "savings", "aapl-1", "msft-1", "etf-1"],
            "lot_selection": "as_listed", "borrow_cap_fraction_of_purchase": 1.0, "risk_weight": 0.0,
            "description": "Borrow on the credit line first, then cash, then lots.",
        })[0],
    }


BASELINE_LABELS = {
    "cash_savings_first": "Cash/savings first",
    "tax_aware_sell_first": "Tax-aware sell first",
    "credit_first": "Credit first",
}


def describe_policy(policy: dict[str, Any], scenario: dict[str, Any] | None = None) -> str:
    """Plain-English summary built only from genome fields."""

    scenario = scenario or BASE_SCENARIO
    names = {"checking": "checking", "savings": "savings", "credit": "the credit line", "nvda-1": "NVDA", "aapl-1": "AAPL", "msft-1": "MSFT", "etf-1": "the ETF"}
    order = [source for source in effective_order(policy, scenario) if _hard_capacity(source, scenario) > 0]
    used_order = ", then ".join(names[source] for source in order[:4])
    parts = [f"Draw from {used_order}."]
    buffer = int(policy["target_cash_buffer_cents"])
    if buffer:
        parts.append(f"Keep an extra ${buffer / 100:,.0f} in savings above the $5,000 emergency floor.")
    else:
        parts.append("Savings may go down to the $5,000 emergency floor, never below.")
    parts.append("Sell the lowest-tax lots first." if policy["lot_selection"] == "min_tax_first" else "Sell lots in the listed order.")
    cap = float(policy["borrow_cap_fraction_of_purchase"])
    if cap <= 0:
        parts.append("Never borrow.")
    else:
        last = " and only as a last resort" if float(policy["risk_weight"]) >= 0.5 else ""
        parts.append(f"Borrow at most {cap:.0%} of the price{last}.")
    for source, fraction in sorted(policy["max_source_fraction"].items()):
        if source in {"nvda-1", "credit"} or _hard_capacity(source, scenario) == 0:
            continue
        parts.append(f"Take at most {fraction:.0%} of the price from {names[source]}.")
    parts.append("NVDA is protected and never sold.")
    return " ".join(parts)


# ---------------------------------------------------------------- operators

def mutate_policy(policy: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    child = copy.deepcopy(policy)
    move = rng.choice(["swap", "move", "buffer", "lot_selection", "fraction", "borrow", "risk"])
    order = child["source_priority"]
    if move == "swap":
        i, j = rng.sample(range(len(order)), 2)
        order[i], order[j] = order[j], order[i]
    elif move == "move":
        source = order.pop(rng.randrange(len(order)))
        order.insert(rng.randrange(len(order) + 1), source)
    elif move == "buffer":
        child["target_cash_buffer_cents"] = max(0, min(300_000, child["target_cash_buffer_cents"] + rng.choice([-50_000, -25_000, 25_000, 50_000, 100_000])))
    elif move == "lot_selection":
        child["lot_selection"] = "as_listed" if child["lot_selection"] == "min_tax_first" else "min_tax_first"
    elif move == "fraction":
        source = rng.choice(["checking", "savings", "aapl-1", "msft-1", "etf-1", "credit"])
        value = rng.choice([0.25, 0.5, 0.75, 1.0])
        if value >= 1.0:
            child["max_source_fraction"].pop(source, None)
        else:
            child["max_source_fraction"][source] = value
    elif move == "borrow":
        child["borrow_cap_fraction_of_purchase"] = rng.choice([0.0, 0.1, 0.25, 0.5, 1.0])
    else:
        child["risk_weight"] = round(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]), 2)
    child["description"] = f"algorithmic {move} mutation"
    return f"mutate:{move}", normalize_policy(child)[0]


def crossover_policy(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    n = len(a["source_priority"])
    i, j = sorted(rng.sample(range(n), 2))
    head = a["source_priority"][i : j + 1]
    rest = [source for source in b["source_priority"] if source not in head]
    order = rest[:i] + head + rest[i:]
    child = {
        "source_priority": order,
        "max_source_fraction": dict(rng.choice([a, b])["max_source_fraction"]),
        "target_cash_buffer_cents": rng.choice([a, b])["target_cash_buffer_cents"],
        "borrow_cap_fraction_of_purchase": rng.choice([a, b])["borrow_cap_fraction_of_purchase"],
        "lot_selection": rng.choice([a, b])["lot_selection"],
        "risk_weight": rng.choice([a, b])["risk_weight"],
        "description": "algorithmic crossover",
    }
    return normalize_policy(child)[0]


def random_policy(rng: random.Random) -> dict[str, Any]:
    order = list(SOURCE_IDS)
    rng.shuffle(order)
    return normalize_policy({
        "source_priority": order,
        "target_cash_buffer_cents": rng.choice([0, 50_000, 100_000, 200_000]),
        "borrow_cap_fraction_of_purchase": rng.choice([0.0, 0.25, 0.5, 1.0]),
        "lot_selection": rng.choice(LOT_SELECTIONS),
        "risk_weight": rng.choice([0.0, 0.5, 1.0]),
        "description": "random policy",
    })[0]


def liquid_objective() -> ObjectiveSpec:
    return ObjectiveSpec(
        domain="liquid",
        prompt=(
            "Fund a $3,000 laptop over 90 days at the lowest modeled cost from checking, savings "
            "(4% APY, $5,000 floor), taxable stock lots (NVDA protected) and an optional 9% credit line."
        ),
        constraints={
            "scenario_id": SCENARIO_ID,
            "sources": list(SOURCE_IDS),
            "protected": ["NVDA"],
            "savings_floor_cents": 500_000,
            "lambda": LAMBDA,
        },
        evaluation_version=EVALUATOR_VERSION,
    )


_POLICY_SHAPE = (
    '{"source_priority":["checking","savings","msft-1","aapl-1","etf-1","credit","nvda-1"],'
    '"max_source_fraction":{"credit":0.25},"target_cash_buffer_cents":100000,'
    '"borrow_cap_fraction_of_purchase":0.0,"lot_selection":"min_tax_first","risk_weight":0.7,'
    '"description":"one short line"}'
)

_FACTS = (
    "Sources: checking $500 (0%), savings $8,000 (4% APY, floor $5,000), lots AAPL $4,000 (basis $3,500), "
    "MSFT $6,000 (basis $6,300, a loss), ETF $5,000 (basis $4,000), NVDA protected, credit up to $3,000 at 9% APR. "
    "Cost = 20% tax on realized gains (losses give no rebate) + 4% savings interest foregone + 9% APR interest, 90 days. "
    "Risk = 5% of any shortfall below $2,000 cash above the savings floor + extra interest at APR+5pp + $100 if a loan breaks 50% LTV after a 30% drop. "
    "lot_selection is as_listed or min_tax_first. risk_weight>=0.5 moves credit last. "
    "The policy must also work for other purchase sizes, rates and market moves."
)


class LiquidAdapter(DomainAdapter):
    name = "liquid"
    evaluator_version = EVALUATOR_VERSION

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def genome_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_priority", "max_source_fraction", "target_cash_buffer_cents", "borrow_cap_fraction_of_purchase", "lot_selection", "risk_weight", "description"],
            "properties": {
                "source_priority": {"type": "array", "items": {"type": "string", "enum": list(SOURCE_IDS)}},
                "max_source_fraction": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}},
                "target_cash_buffer_cents": {"type": "integer", "minimum": 0, "maximum": 300000},
                "borrow_cap_fraction_of_purchase": {"type": "number", "minimum": 0, "maximum": 1},
                "lot_selection": {"type": "string", "enum": list(LOT_SELECTIONS)},
                "risk_weight": {"type": "number", "minimum": 0, "maximum": 1},
                "description": {"type": "string"},
            },
        }

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        def digest(split: str) -> str:
            raw = f"{self.evaluator_version}|{split}|{seed}|" + json.dumps(SPLITS[split], sort_keys=True)
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {"train_hash": digest("train"), "validation_hash": digest("validation"), "test_hash": digest("test")}

    def family_of(self, payload: dict[str, Any]) -> str:
        priority = payload.get("source_priority") if isinstance(payload, dict) else None
        first = priority[0] if isinstance(priority, list) and priority else "mixed"
        if first in {"checking", "savings"}:
            return "cash-first"
        if first == "credit":
            return "credit-first"
        return "sell-first"

    def baselines(self) -> dict[str, dict[str, Any]]:
        return baseline_policies()

    def seeded_baselines(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "payload": payload, "hypothesis": f"baseline {BASELINE_LABELS[name]}"}
            for name, payload in baseline_policies().items()
        ]

    def realize_proposal(self, payload: dict[str, Any], parent_payloads: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        """Fill fields an LLM edit left out from the first parent."""

        del rng
        if not isinstance(payload, dict) or not parent_payloads:
            return payload
        parent = parent_payloads[0] if isinstance(parent_payloads[0], dict) else {}
        merged = dict(parent)
        for key, value in payload.items():
            if key == "max_source_fraction" and isinstance(value, dict):
                merged[key] = {**(parent.get(key) or {}), **value}
            else:
                merged[key] = value
        return merged

    def prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        return normalize_policy(payload)

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        del objective
        validate_policy(payload)

    def oneshot_hint(self) -> str:
        return (
            "Return one funding policy as payload. " + _FACTS +
            " source_priority must list each of " + json.dumps(list(SOURCE_IDS)) + " exactly once. "
            "An invalid policy is reported as invalid and is not repaired. Example payload: " + _POLICY_SHAPE
        )

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return (
            "Return JSON only. payload is one funding policy shaped like " + _POLICY_SHAPE +
            ". hypothesis is one short line.\n" + _FACTS + f"\nObjective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Return JSON only. payload changes one or two fields of the parent policy; omitted fields are copied. "
            "hypothesis says why in one line.\n" + _FACTS +
            f"\nParent: {json.dumps(parent.payload, separators=(',', ':'))}"
            f"\nFeedback: {feedback.feedback[:400]}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Return JSON only. payload is one full child policy that combines the strengths of A and B. "
            "hypothesis is one short line.\n" + _FACTS +
            f"\nA: {json.dumps(a.payload, separators=(',', ':'))}\nB: {json.dumps(b.payload, separators=(',', ':'))}"
        )

    def random_seed_payloads(self, rng: random.Random, count: int) -> list[dict[str, Any]]:
        return [random_policy(rng) for _ in range(count)]

    def uses_memetic_fallback(self) -> bool:
        # Generations wait for model children. A Liquid evaluation takes microseconds,
        # so memetic slices would end before a 15 token/s model returned anything.
        return False

    def algorithmic_offspring(self, payloads: list[dict[str, Any]], rng: random.Random, count: int) -> list[tuple[str, dict[str, Any]]]:
        pool = [normalize_policy(p)[0] for p in payloads if isinstance(p, dict)] or [random_policy(rng)]
        children: list[tuple[str, dict[str, Any]]] = []
        for index in range(max(0, count)):
            if len(pool) > 1 and index % 3 == 0:
                a, b = rng.sample(pool, 2)
                children.append(("crossover", crossover_policy(a, b, rng)))
            else:
                children.append(mutate_policy(rng.choice(pool), rng))
        return children

    def local_perturbations(self, payload: dict[str, Any], rng: random.Random, count: int) -> list[dict[str, Any]]:
        policy = normalize_policy(payload)[0]
        return [mutate_policy(policy, rng)[1] for _ in range(count)]

    def alter_duplicate(self, payload: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        return mutate_policy(normalize_policy(payload)[0], rng)[1]

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        started = time.perf_counter()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split}")

        def failed(reason: str) -> Evaluation:
            return Evaluation(
                genome_id=genome.id, feasible=False, fitness=None, metrics={},
                constraint_violations=[reason], feedback=f"infeasible: {reason}",
                evaluator_version=self.evaluator_version, data_split=split,  # type: ignore[arg-type]
                elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
            )

        try:
            validate_policy(genome.payload)
        except ValueError as exc:
            return failed(str(exc))
        rows = [evaluate_scenario(genome.payload, scenario) for scenario in SPLITS[split]]
        bad = [row for row in rows if not row["feasible"]]
        if bad:
            return failed(f"{bad[0]['scenario_id']}: {'; '.join(bad[0]['constraint_violations'])}")
        n = len(rows)
        cost = sum(row["total_cost_cents"] for row in rows) / n / 100.0
        risk = sum(row["risk_penalty_cents"] for row in rows) / n / 100.0
        fitness = round(sum(row["fitness"] for row in rows) / n, 4)
        passes = sum(row["stress_passes"] for row in rows)
        parts = [
            f"{row['scenario_id'].split(':')[-1]}: cost ${row['total_cost_cents'] / 100:.2f}, risk ${row['risk_penalty_cents'] / 100:.2f}, "
            f"alloc {','.join(f'{k}={v // 100}' for k, v in row['allocation_cents'].items())}"
            for row in rows
        ]
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=fitness,
            metrics={
                "mean_cost_usd": round(cost, 4),
                "mean_risk_usd": round(risk, 4),
                "stress_passes": float(passes),
                "stress_checks": float(3 * n),
                "scenarios": float(n),
            },
            constraint_violations=[],
            feedback=f"{split}: mean cost ${cost:.2f}, mean risk ${risk:.2f} (lambda {LAMBDA}). " + " | ".join(parts),
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]:
        return {
            "domain": self.name,
            "genome_id": genome.id,
            "policy": genome.payload,
            "feasible": evaluation.feasible,
            "fitness": evaluation.fitness,
            "metrics": evaluation.metrics,
            "split": evaluation.data_split,
            "hypothesis": genome.hypothesis,
        }
