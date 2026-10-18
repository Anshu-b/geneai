"""Structural diversity: canonical hashes, family quotas, novelty bonus."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Individual


def normalize_payload(obj: Any) -> Any:
    """Stable JSON-ish form. Floats round to 6 decimals so hashes dedup noise."""

    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return obj
        return round(obj, 6)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(key): normalize_payload(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [normalize_payload(value) for value in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(normalize_payload(payload)))


def family_of(adapter: DomainAdapter, payload: dict[str, Any]) -> str:
    try:
        name = adapter.family_of(payload)
    except Exception:
        name = None
    return str(name) if name else "default"


def species_of(adapter: DomainAdapter, payload: dict[str, Any]) -> str:
    try:
        name = adapter.species_of(payload)
    except Exception:
        name = None
    return str(name) if name else family_of(adapter, payload)


def novelty_bonus(individual: Individual, population: list[Individual], adapter: DomainAdapter) -> float:
    """Small bonus for rarer families. Does not override a large fitness gap."""

    fam = family_of(adapter, individual.genome.payload)
    same = sum(1 for other in population if family_of(adapter, other.genome.payload) == fam)
    return 1.0 / max(same, 1)


def trim_with_diversity(
    individuals: list[Individual],
    size: int,
    adapter: DomainAdapter,
) -> list[Individual]:
    """Keep the best of each family first, then fill by fitness."""

    if size <= 0 or not individuals:
        return []
    ranked = sorted(individuals, key=lambda ind: _rank_key(ind))
    chosen: list[Individual] = []
    seen: set[str] = set()
    for ind in ranked:
        fam = family_of(adapter, ind.genome.payload)
        if fam in seen:
            continue
        chosen.append(ind)
        seen.add(fam)
        if len(chosen) == size:
            return chosen
    chosen_ids = {ind.genome.id for ind in chosen}
    for ind in ranked:
        if ind.genome.id in chosen_ids:
            continue
        chosen.append(ind)
        chosen_ids.add(ind.genome.id)
        if len(chosen) == size:
            break
    return chosen


def _rank_key(ind: Individual) -> tuple[Any, ...]:
    feasible = bool(ind.evaluation.feasible and ind.evaluation.fitness is not None)
    fitness = ind.evaluation.fitness if ind.evaluation.fitness is not None else float("-inf")
    return (0 if feasible else 1, -fitness, ind.genome.id)
