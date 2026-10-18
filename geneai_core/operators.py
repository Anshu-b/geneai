"""Variation schedule and domain-agnostic proposal helpers.

Numeric mutation and crossover live in the mock provider so tests stay
deterministic. Real providers receive the same instruction and schema.
"""

from __future__ import annotations

import json
import random
from typing import Any

from geneai_core.schemas import EvolutionConfig, Genome, ObjectiveSpec

MACHINE_MARK = "GENEAI_CONTEXT="


def sample_operation(config: EvolutionConfig, rng: random.Random) -> str:
    """Return mutation, crossover, or immigrant. Probabilities are normalized."""

    weights = [
        ("immigrant", config.random_immigrant_probability),
        ("crossover", config.crossover_probability),
        ("mutation", config.mutation_probability),
    ]
    total = sum(weight for _, weight in weights)
    if total <= 0:
        raise ValueError("variation probabilities must sum to a positive number")
    draw = rng.random() * total
    cursor = 0.0
    for name, weight in weights:
        cursor += weight
        if draw < cursor:
            return name
    return "mutation"


def attach_context(instruction: str, context: dict[str, Any]) -> str:
    blob = json.dumps(context, sort_keys=True, default=str, separators=(",", ":"))
    return instruction.rstrip() + "\n" + MACHINE_MARK + blob


def parse_context(instruction: str) -> dict[str, Any]:
    index = instruction.rfind(MACHINE_MARK)
    if index < 0:
        return {}
    raw = instruction[index + len(MACHINE_MARK) :]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def wrapper_schema(payload_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["payload", "hypothesis"],
        "properties": {
            "payload": payload_schema,
            "hypothesis": {
                "type": "string",
                "description": "Why this candidate should improve measured fitness",
            },
        },
    }


def seed_batch_note(*, count: int, avoid_hashes: list[str], temperature: float, sample_seed: int) -> str:
    """Tell a seed call to return several different genomes and to skip ones already kept."""

    shown = ", ".join(avoid_hashes[:8]) if avoid_hashes else "none"
    return (
        f"\nReturn {count} compact JSON candidates. hypothesis is one line. "
        f"Do not repeat avoid_hashes ({shown}). "
        f"temperature {temperature} sample_seed {sample_seed}."
    )


def repair_instructions(
    *,
    objective: ObjectiveSpec,
    payload_schema: dict[str, Any],
    invalid_payload: dict[str, Any],
    error: str,
) -> str:
    return (
        "Repair the candidate so it satisfies the JSON schema and the stated constraints. "
        "Change as little as possible. Return a JSON object with keys payload and hypothesis. "
        "Do not include markdown.\n"
        f"Objective: {objective.prompt}\n"
        f"Constraints: {json.dumps(objective.constraints, sort_keys=True)}\n"
        f"Validation error: {error}\n"
        f"Invalid payload: {json.dumps(invalid_payload, sort_keys=True, default=str)}\n"
        f"Payload schema: {json.dumps(payload_schema, sort_keys=True)}"
    )


def creation_event_type(operator: str, origin_operator: str) -> str:
    mapping = {
        "seed": "genome.seeded",
        "mutation": "genome.mutated",
        "crossover": "genome.crossed",
    }
    if operator == "repair":
        return mapping.get(origin_operator, "genome.mutated")
    try:
        return mapping[operator]
    except KeyError as exc:
        raise ValueError(f"no creation event for operator {operator}") from exc


def pick_two_parents(pool: list[Any], rng: random.Random) -> tuple[Any, Any]:
    first = rng.choice(pool)
    if len(pool) == 1:
        return first, first
    second = rng.choice(pool)
    for _ in range(8):
        if second.genome.id != first.genome.id:
            break
        second = rng.choice(pool)
    return first, second


def parent_payloads(genomes: list[Genome]) -> list[dict[str, Any]]:
    return [genome.payload for genome in genomes]
