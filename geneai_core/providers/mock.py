"""Deterministic providers. They never open a network connection.

Mock output is labeled mode=mock. Callers must not present it as sponsor execution.
"""

from __future__ import annotations

import random
from typing import Any

from geneai_core.diversity import content_hash, normalize_payload
from geneai_core.operators import parse_context
from geneai_core.providers.base import Proposal

_MISSING = object()


def _schema_at(schema: dict[str, Any], path: list[str]) -> dict[str, Any]:
    current = schema
    for key in path:
        props = current.get("properties") or {}
        current = props.get(key) or {}
    return current


def _numeric_paths(schema: dict[str, Any], prefix: list[str] | None = None) -> list[list[str]]:
    prefix = prefix or []
    found: list[list[str]] = []
    kind = schema.get("type")
    if kind in {"number", "integer"} or "minimum" in schema and kind != "string":
        if kind in {"number", "integer"}:
            found.append(prefix)
    if schema.get("type") == "object" or "properties" in schema:
        for key, child in (schema.get("properties") or {}).items():
            found.extend(_numeric_paths(child, prefix + [key]))
    return found


def _get_path(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _set_path(payload: dict[str, Any], path: list[str], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        nxt = current.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            current[key] = nxt
        current = nxt
    current[path[-1]] = value


def _clamp(value: float, schema: dict[str, Any]) -> float | int:
    lo = schema.get("minimum")
    hi = schema.get("maximum")
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    if schema.get("type") == "integer":
        return int(round(value))
    return round(value, 6)


def fill_schema(schema: dict[str, Any], rng: random.Random) -> Any:
    if "enum" in schema:
        return schema["enum"][rng.randrange(len(schema["enum"]))]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        props: dict[str, Any] = schema.get("properties") or {}
        required = list(schema.get("required") or props.keys())
        built: dict[str, Any] = {}
        for key in required:
            if key in props:
                built[key] = fill_schema(props[key], rng)
        return built
    if kind == "integer":
        lo = int(schema.get("minimum", 0))
        hi = int(schema.get("maximum", 5))
        if hi < lo:
            hi = lo
        return rng.randint(lo, hi)
    if kind == "number":
        lo = float(schema.get("minimum", -1.0))
        hi = float(schema.get("maximum", 1.0))
        if hi < lo:
            hi = lo
        return round(rng.uniform(lo, hi), 6)
    if kind == "boolean":
        return bool(rng.randrange(2))
    if kind == "array":
        return []
    return f"candidate-{rng.randrange(1000)}"


def _payload_schema(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties") or {}
    if "payload" in props and isinstance(props["payload"], dict):
        return props["payload"]
    return schema


def mutate_one(parent: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Change one numeric feature. If none can move, mark the description."""

    child = normalize_payload(parent)
    if not isinstance(child, dict):
        return fill_schema(schema, rng)
    paths = _numeric_paths(schema)
    rng.shuffle(paths)
    for path in paths:
        current = _get_path(child, path)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            continue
        spec = _schema_at(schema, path)
        lo = float(spec.get("minimum", current - 1))
        hi = float(spec.get("maximum", current + 1))
        span = max(hi - lo, 1e-6)
        delta = (rng.random() - 0.5) * 0.35 * span
        if abs(delta) < 0.01 * span:
            delta = 0.05 * span if delta >= 0 else -0.05 * span
        updated = _clamp(float(current) + delta, spec)
        if updated == current:
            nudge = 1 if spec.get("type") == "integer" else round(0.05 * span, 6)
            updated = _clamp(float(current) + nudge, spec)
        if updated == current:
            continue
        _set_path(child, path, updated)
        return child
    description = str(child.get("description") or child.get("label") or "candidate")
    if "description" in child:
        child["description"] = f"{description} variant {rng.randrange(10000)}"
    elif "label" in child:
        child["label"] = f"{description} variant {rng.randrange(10000)}"
    else:
        child["note"] = f"variant {rng.randrange(10000)}"
    return child


def cross_payloads(
    left: dict[str, Any],
    right: dict[str, Any],
    schema: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    return _cross_node(left, right, schema, rng)


def _cross_node(left: Any, right: Any, schema: dict[str, Any], rng: random.Random) -> Any:
    if "enum" in schema:
        choices = [value for value in (left, right) if value in schema["enum"]]
        if choices:
            return choices[rng.randrange(len(choices))]
        return schema["enum"][rng.randrange(len(schema["enum"]))]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        props: dict[str, Any] = schema.get("properties") or {}
        required = list(schema.get("required") or props.keys())
        built: dict[str, Any] = {}
        left_map = left if isinstance(left, dict) else {}
        right_map = right if isinstance(right, dict) else {}
        for key in required:
            if key not in props:
                continue
            built[key] = _cross_node(left_map.get(key), right_map.get(key), props[key], rng)
        return built
    if kind in {"number", "integer"} and isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _clamp((float(left) + float(right)) / 2.0, schema)
    if left is not None and (rng.random() < 0.5 or right is None):
        return left
    if right is not None:
        return right
    return fill_schema(schema, rng)


def repair_payload(invalid: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    filled = fill_schema(schema, rng)
    if not isinstance(filled, dict):
        return {"value": filled}
    if not isinstance(invalid, dict):
        return filled
    return _repair_node(invalid, schema, rng, filled)


def _repair_node(invalid: Any, schema: dict[str, Any], rng: random.Random, fallback: Any) -> Any:
    if "enum" in schema:
        if invalid in schema["enum"]:
            return invalid
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        props: dict[str, Any] = schema.get("properties") or {}
        required = list(schema.get("required") or props.keys())
        source = invalid if isinstance(invalid, dict) else {}
        fallback_map = fallback if isinstance(fallback, dict) else {}
        built: dict[str, Any] = {}
        for key in required:
            if key not in props:
                continue
            if key in source:
                built[key] = _repair_node(source[key], props[key], rng, fallback_map.get(key))
            else:
                built[key] = fallback_map.get(key, fill_schema(props[key], rng))
        return built
    if kind in {"number", "integer"}:
        if isinstance(invalid, bool) or not isinstance(invalid, (int, float)):
            return fallback if fallback is not None else fill_schema(schema, rng)
        return _clamp(float(invalid), schema)
    if kind == "string":
        if isinstance(invalid, str) and invalid:
            return invalid
        return fallback if isinstance(fallback, str) else "repaired"
    return invalid if invalid is not None else fallback


def materialize(instruction: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    context = parse_context(instruction)
    token = int(context.get("token") or 0)
    rng = random.Random(token)
    payload_schema = _payload_schema(schema)
    operation = str(context.get("operation") or "seed")
    parents = context.get("parents") or []
    if operation == "mutation" and parents and isinstance(parents[0], dict):
        payload = mutate_one(parents[0], payload_schema, rng)
    elif operation == "crossover" and len(parents) >= 2 and isinstance(parents[0], dict):
        second = parents[1] if isinstance(parents[1], dict) else {}
        payload = cross_payloads(parents[0], second, payload_schema, rng)
    elif operation == "repair":
        invalid = context.get("invalid_payload") if isinstance(context.get("invalid_payload"), dict) else {}
        payload = repair_payload(invalid, payload_schema, rng)
    else:
        payload = fill_schema(payload_schema, rng)
    payload = normalize_payload(payload)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    if operation == "mutation" and parents and isinstance(parents[0], dict):
        if content_hash(payload) == content_hash(normalize_payload(parents[0])):
            payload = dict(payload)
            payload["description"] = str(payload.get("description") or "candidate") + f" #{token}"
            payload = normalize_payload(payload)
    hypothesis = f"{operation} proposal token {token}"
    return payload, hypothesis


class MockModelProvider:
    name = "mock"
    mode = "mock"

    def __init__(self, *, cost_per_call: float = 0.0) -> None:
        self.cost_per_call = cost_per_call
        self.calls = 0

    async def propose(self, instruction: str, schema: dict[str, Any], *, call_id: str) -> Proposal:
        self.calls += 1
        payload, hypothesis = materialize(instruction, schema)
        return Proposal(
            payload=payload,
            hypothesis=hypothesis,
            provider=self.name,
            provider_run_id=f"mock-{call_id}",
            model="mock-deterministic",
            cost_usd=self.cost_per_call,
            cost_is_estimate=self.cost_per_call > 0,
            latency_ms=0,
            mode="mock",
        )

    async def propose_json(self, instruction: str, schema: dict[str, Any]) -> dict[str, Any]:
        proposal = await self.propose(instruction, schema, call_id="direct")
        return {"payload": proposal.payload, "hypothesis": proposal.hypothesis}


class MockAgentProvider:
    """Same deterministic generator as the mock model, labeled as a workflow."""

    name = "mock-agent"
    mode = "mock"

    def __init__(self, *, model: MockModelProvider | None = None, cost_per_call: float = 0.0) -> None:
        self._model = model or MockModelProvider(cost_per_call=cost_per_call)
        self.cost_per_call = cost_per_call
        self.calls = 0

    async def run_workflow(
        self,
        workflow_name: str,
        inputs: dict[str, Any],
        *,
        call_id: str,
    ) -> Proposal:
        self.calls += 1
        instruction = str(
            inputs.get("mutation_goal") or inputs.get("instruction") or inputs.get("task") or workflow_name
        )
        schema = inputs.get("response_schema") or {"type": "object"}
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        proposal = await self._model.propose(instruction, schema, call_id=call_id)
        proposal.provider = self.name
        proposal.provider_run_id = f"mock-agent-{call_id}"
        proposal.extra = {"workflow_name": workflow_name}
        if self.cost_per_call:
            proposal.cost_usd = self.cost_per_call
            proposal.cost_is_estimate = True
        return proposal
