"""Immutable lineage events (schema 1.0, section 13.1–13.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from geneai_core.schemas import EVENT_TYPES, PROMPT_VERSION, Genome, LineageEvent, Provenance


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def event_id_for(sequence: int) -> str:
    return f"evt_{sequence:04d}"


def lineage_payload(
    genome: Genome,
    *,
    extra: dict[str, Any] | None = None,
    evaluation_fitness: float | None = None,
    feasible: bool | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Fields a later lineage UI needs, without reading domain objects."""

    payload: dict[str, Any] = {
        "genome_id": genome.id,
        "generation": genome.generation,
        "parent_ids": list(genome.parent_ids),
        "operator": genome.operator,
        "origin_operator": genome.metadata.get("origin_operator", genome.operator),
        "role": genome.metadata.get("role"),
        "family": genome.metadata.get("family"),
        "species": genome.metadata.get("species"),
        "content_hash": genome.content_hash,
        "provider": genome.provider,
        "provider_role": genome.metadata.get("provider_role"),
        "hypothesis": genome.hypothesis,
        "fitness": evaluation_fitness,
        "feasible": feasible,
        "split": split,
    }
    if extra:
        payload.update(extra)
    return payload


def make_event(
    *,
    run_id: str,
    sequence: int,
    event_type: str,
    payload: dict[str, Any],
    provenance: Provenance,
    timestamp: str | None = None,
) -> LineageEvent:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {event_type}")
    if provenance.prompt_version is None:
        provenance = provenance.model_copy(update={"prompt_version": PROMPT_VERSION})
    return LineageEvent(
        event_id=event_id_for(sequence),
        run_id=run_id,
        sequence=sequence,
        timestamp=timestamp or utc_now(),
        type=event_type,
        payload=payload,
        provenance=provenance,
    )
