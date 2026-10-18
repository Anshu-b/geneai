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


def _schema_options(schema: dict[str, Any]) -> list[dict[str, Any]]:
    raw = schema.get("anyOf") or schema.get("oneOf") or []
    return [item for item in raw if isinstance(item, dict)]


def _const_values(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties") or {}
    values: dict[str, Any] = {}
    for key, spec in props.items():
        if isinstance(spec, dict) and "const" in spec:
            values[key] = spec["const"]
    return values


def _diversity_keys(item_schema: dict[str, Any]) -> list[str]:
    """Property names that are const on every alternative. Used to avoid clones."""

    options = _schema_options(item_schema)
    if len(options) < 2:
        return []
    shared: set[str] | None = None
    for option in options:
        consts = set(_const_values(option))
        shared = consts if shared is None else shared & consts
    return sorted(shared or [])


def _fill_array(schema: dict[str, Any], rng: random.Random) -> list[Any]:
    item_schema = schema.get("items") or {}
    minimum = int(schema.get("minItems") or 0)
    maximum = int(schema.get("maxItems") or max(minimum, 0))
    if maximum < minimum:
        maximum = minimum
    count = minimum if maximum == minimum else rng.randint(minimum, maximum)
    if schema.get("uniqueItems") and "enum" in item_schema:
        options = list(item_schema["enum"])
        rng.shuffle(options)
        take = min(count, len(options))
        return options[:take]
    keys = _diversity_keys(item_schema)
    options = _schema_options(item_schema)
    if not keys or not options:
        return [fill_schema(item_schema, rng) for _ in range(count)]
    used: dict[str, set[Any]] = {key: set() for key in keys}
    built: list[Any] = []
    guard = 0
    while len(built) < count and guard < 80:
        guard += 1
        choices = []
        for option in options:
            consts = _const_values(option)
            if any(consts.get(key) in used[key] for key in keys if key in consts):
                continue
            choices.append(option)
        if not choices:
            break
        option = choices[rng.randrange(len(choices))]
        for key, value in _const_values(option).items():
            if key in used:
                used[key].add(value)
        built.append(fill_schema(option, rng))
    while len(built) < minimum:
        built.append(fill_schema(item_schema, rng))
    return built


def _diversity_collision(items: list[Any], keys: list[str]) -> bool:
    seen: dict[str, set[Any]] = {key: set() for key in keys}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in keys:
            if key not in item:
                continue
            if item[key] in seen[key]:
                return True
            seen[key].add(item[key])
    return False


def fill_schema(schema: dict[str, Any], rng: random.Random) -> Any:
    if "const" in schema:
        return schema["const"]
    options = _schema_options(schema)
    if options:
        return fill_schema(options[rng.randrange(len(options))], rng)
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
        return _fill_array(schema, rng)
    return f"candidate-{rng.randrange(1000)}"


def _payload_schema(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties") or {}
    if "payload" in props and isinstance(props["payload"], dict):
        return props["payload"]
    return schema


def _mutate_numeric_jitter(child: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> bool:
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
        return True
    return False


def _mutate_term_presence(child: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> bool:
    """Turn a zero weight on, or a nonzero weight off. Bounds that exclude 0 are skipped."""

    zeros: list[tuple[list[str], dict[str, Any]]] = []
    nonzero: list[tuple[list[str], dict[str, Any]]] = []
    for path in _numeric_paths(schema):
        current = _get_path(child, path)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            continue
        spec = _schema_at(schema, path)
        lo = float(spec.get("minimum", -1))
        hi = float(spec.get("maximum", 1))
        if lo > 0 or hi < 0:
            continue
        if abs(float(current)) <= 1e-9:
            zeros.append((path, spec))
        else:
            nonzero.append((path, spec))
    rng.shuffle(zeros)
    rng.shuffle(nonzero)
    if zeros and (not nonzero or rng.random() < 0.5):
        path, spec = zeros[0]
        lo = float(spec.get("minimum", -1))
        hi = float(spec.get("maximum", 1))
        span = max(hi - lo, 1e-6)
        for _ in range(8):
            updated = _clamp(rng.uniform(lo, hi), spec)
            if abs(float(updated)) >= 0.15 * span:
                _set_path(child, path, updated)
                return True
        return False
    if nonzero:
        path, spec = nonzero[0]
        _set_path(child, path, _clamp(0.0, spec))
        return True
    return False


def _mutate_structure(child: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> bool:
    """Swap a tie-break key, or add/remove one class-override style array item."""

    paths = _array_paths(schema)
    rng.shuffle(paths)
    for path in paths:
        current = _get_path(child, path)
        if not isinstance(current, list):
            continue
        array_schema = _schema_at(schema, path)
        item_schema = array_schema.get("items") or {}
        if array_schema.get("uniqueItems") and isinstance(item_schema, dict) and "enum" in item_schema:
            if not current:
                continue
            used = set(current)
            choices = [option for option in item_schema["enum"] if option not in used]
            if not choices:
                continue
            updated = list(current)
            updated[rng.randrange(len(updated))] = choices[rng.randrange(len(choices))]
            _set_path(child, path, updated)
            return True
        options = _schema_options(item_schema)
        keys = _diversity_keys(item_schema)
        if not options or not keys:
            continue
        minimum = int(array_schema.get("minItems") or 0)
        maximum = int(array_schema.get("maxItems") or max(minimum, len(current)))
        can_add = len(current) < maximum
        can_remove = len(current) > minimum
        if can_add and (not can_remove or rng.random() < 0.5):
            used: dict[str, set[Any]] = {key: set() for key in keys}
            for item in current:
                if not isinstance(item, dict):
                    continue
                for key in keys:
                    if key in item:
                        used[key].add(item[key])
            choices = []
            for option in options:
                consts = _const_values(option)
                if any(consts.get(key) in used[key] for key in keys if key in consts):
                    continue
                choices.append(option)
            if not choices:
                continue
            updated = list(current)
            updated.append(fill_schema(choices[rng.randrange(len(choices))], rng))
            _set_path(child, path, updated)
            return True
        if can_remove:
            updated = list(current)
            updated.pop(rng.randrange(len(updated)))
            _set_path(child, path, updated)
            return True
    return False


def mutate_one(parent: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Jitter a weight, add or remove a term, or edit a tie-break / override list."""

    child = normalize_payload(parent)
    if not isinstance(child, dict):
        return fill_schema(schema, rng)
    roll = rng.random()
    if roll < 0.40:
        order = (_mutate_numeric_jitter, _mutate_term_presence, _mutate_structure, _mutate_array, _flip_enum)
    elif roll < 0.70:
        order = (_mutate_term_presence, _mutate_structure, _mutate_numeric_jitter, _mutate_array, _flip_enum)
    else:
        order = (_mutate_structure, _mutate_term_presence, _mutate_numeric_jitter, _mutate_array, _flip_enum)
    for edit in order:
        if edit(child, schema, rng):
            return child
    description = str(child.get("description") or child.get("label") or "candidate")
    if "description" in child:
        child["description"] = f"{description} variant {rng.randrange(10000)}"
    elif "label" in child:
        child["label"] = f"{description} variant {rng.randrange(10000)}"
    else:
        child["note"] = f"variant {rng.randrange(10000)}"
    return child


def _enum_paths(schema: dict[str, Any], prefix: list[str] | None = None) -> list[list[str]]:
    prefix = prefix or []
    found: list[list[str]] = []
    if "enum" in schema and prefix:
        found.append(prefix)
    if schema.get("type") == "object" or "properties" in schema:
        for key, child in (schema.get("properties") or {}).items():
            if isinstance(child, dict):
                found.extend(_enum_paths(child, prefix + [key]))
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        found.extend(_enum_paths(schema["items"], prefix + ["*"]))
    return found


def _array_paths(schema: dict[str, Any], prefix: list[str] | None = None) -> list[list[str]]:
    prefix = prefix or []
    found: list[list[str]] = []
    if schema.get("type") == "array" and prefix:
        found.append(prefix)
    if schema.get("type") == "object" or "properties" in schema:
        for key, child in (schema.get("properties") or {}).items():
            if isinstance(child, dict):
                found.extend(_array_paths(child, prefix + [key]))
    return found


def _mutate_array(payload: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> bool:
    paths = _array_paths(schema)
    rng.shuffle(paths)
    for path in paths:
        current = _get_path(payload, path)
        if not isinstance(current, list) or not current:
            continue
        item_schema = _schema_at(schema, path).get("items") or {}
        index = rng.randrange(len(current))
        keys = _diversity_keys(item_schema)
        options = _schema_options(item_schema)
        replacement: Any
        if options and keys:
            used: dict[str, set[Any]] = {key: set() for key in keys}
            for position, item in enumerate(current):
                if position == index or not isinstance(item, dict):
                    continue
                for key in keys:
                    if key in item:
                        used[key].add(item[key])
            choices = []
            for option in options:
                consts = _const_values(option)
                if any(consts.get(key) in used[key] for key in keys if key in consts):
                    continue
                choices.append(option)
            if not choices:
                continue
            replacement = fill_schema(choices[rng.randrange(len(choices))], rng)
        else:
            replacement = fill_schema(item_schema, rng)
        if replacement == current[index]:
            continue
        updated = list(current)
        updated[index] = replacement
        _set_path(payload, path, updated)
        return True
    return False


def _flip_enum(payload: dict[str, Any], schema: dict[str, Any], rng: random.Random) -> bool:
    paths = _enum_paths(schema)
    rng.shuffle(paths)
    for path in paths:
        if "*" in path:
            continue
        current = _get_path(payload, path)
        options = _schema_at(schema, path).get("enum") or []
        choices = [option for option in options if option != current]
        if not choices:
            continue
        _set_path(payload, path, choices[rng.randrange(len(choices))])
        return True
    return False


def cross_payloads(
    left: dict[str, Any],
    right: dict[str, Any],
    schema: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    return _cross_node(left, right, schema, rng)


def _matches_option(value: Any, option: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    for key, expected in _const_values(option).items():
        if value.get(key) != expected:
            return False
    props = option.get("properties") or {}
    for key, spec in props.items():
        if not isinstance(spec, dict) or "enum" not in spec or key not in value:
            continue
        if value[key] not in spec["enum"]:
            return False
    return True


def _cross_node(left: Any, right: Any, schema: dict[str, Any], rng: random.Random) -> Any:
    if "const" in schema:
        return schema["const"]
    options = _schema_options(schema)
    if options:
        matched = [value for value in (left, right) if any(_matches_option(value, option) for option in options)]
        if matched:
            return matched[rng.randrange(len(matched))]
        return fill_schema(schema, rng)
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
    if kind == "array":
        item_schema = schema.get("items") or {}
        left_list = left if isinstance(left, list) else []
        right_list = right if isinstance(right, list) else []
        if not left_list and not right_list:
            return fill_schema(schema, rng)
        length = max(len(left_list), len(right_list))
        cap = schema.get("maxItems")
        if isinstance(cap, int):
            length = min(length, cap)
        built = []
        for index in range(length):
            lv = left_list[index] if index < len(left_list) else None
            rv = right_list[index] if index < len(right_list) else None
            built.append(_cross_node(lv, rv, item_schema, rng))
        floor = schema.get("minItems")
        while isinstance(floor, int) and len(built) < floor:
            built.append(fill_schema(item_schema, rng))
        return built
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
    if "const" in schema:
        return schema["const"]
    options = _schema_options(schema)
    if options:
        for option in options:
            if _matches_option(invalid, option):
                filled = fill_schema(option, rng)
                return _repair_node(invalid, option, rng, filled)
        return fill_schema(schema, rng)
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
    if kind == "array":
        item_schema = schema.get("items") or {}
        source = invalid if isinstance(invalid, list) else []
        built = [_repair_node(item, item_schema, rng, None) for item in source]
        cap = schema.get("maxItems")
        if isinstance(cap, int):
            built = built[:cap]
        floor = int(schema.get("minItems") or 0)
        while len(built) < floor:
            built.append(fill_schema(item_schema, rng))
        keys = _diversity_keys(item_schema)
        if keys and _diversity_collision(built, keys):
            return fill_schema(schema, rng)
        return built
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
        proposal.provider = "mock"
        proposal.provider_run_id = f"mock-agent-{call_id}"
        proposal.extra = {"workflow_name": workflow_name}
        if self.cost_per_call:
            proposal.cost_usd = self.cost_per_call
            proposal.cost_is_estimate = True
        return proposal
