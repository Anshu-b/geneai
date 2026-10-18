"""Arm A: one frontier request. Arm B is the evolutionary run already exported.

This is a single attempt versus search. It is not a compute-matched comparison.
"""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.operators import wrapper_schema
from geneai_core.providers.base import CredentialsError, ProviderError
from geneai_core.providers.frontier import FrontierProvider
from geneai_core.schemas import Genome, ObjectiveSpec
from geneai_core.store import export_integrity, verify_export

COMPARISON_LABEL = "single attempt vs evolutionary search"


def frontier_blocker_message() -> str:
    return "\n".join(
        [
            "SKIPPED: frontier one-shot was not executed. No request was sent.",
            "Missing environment variables: FRONTIER_API_KEY, FRONTIER_BASE_URL, and/or FRONTIER_MODEL.",
            "Arm A needs an OpenAI-compatible endpoint you control. This repo does not pick a vendor URL.",
            "The evolutionary export stays valid with one_shot_baseline.status=skipped.",
            "This comparison is not compute-matched. GeneAI evaluates more candidates.",
        ]
    )


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "arm": "frontier_one_shot",
        "status": "skipped",
        "reason": reason,
        "calls": 0,
        "compute_matched": False,
        "comparison_label": COMPARISON_LABEL,
        "feasible": None,
        "fitness": None,
        "cost_usd": None,
    }


def _prompt(objective: ObjectiveSpec, schema: dict[str, Any], *, hint: str = "") -> str:
    extra = f"\n{hint.strip()}\n" if hint.strip() else "\n"
    return (
        "Propose one candidate. Reply with a single JSON object containing payload and hypothesis. "
        "payload must match the schema. No markdown and no second guess. "
        "If the domain is an outfit, use one item per role and never repeat a role. "
        "An invalid payload is reported as invalid. It is not repaired."
        f"{extra}"
        f"Objective: {objective.prompt}\n"
        f"Constraints: {json.dumps(objective.constraints, sort_keys=True)}\n"
        f"Schema: {json.dumps(schema, sort_keys=True)}"
    )


async def run_frontier_oneshot(
    *,
    adapter: DomainAdapter,
    objective: ObjectiveSpec,
    provider: FrontierProvider | None = None,
) -> dict[str, Any]:
    """One request. Invalid output stays invalid. No repair loop."""

    if provider is None:
        try:
            provider = FrontierProvider.from_env()
        except CredentialsError as exc:
            return _skipped(str(exc))
    schema = wrapper_schema(adapter.genome_schema())
    hint = adapter.oneshot_hint() if hasattr(adapter, "oneshot_hint") else ""
    prompt = _prompt(objective, schema, hint=str(hint or ""))
    started = time.perf_counter()
    try:
        proposal = await provider.propose(prompt, schema, call_id="oneshot")
    except ProviderError as exc:
        return {
            "arm": "frontier_one_shot",
            "status": "failed",
            "reason": str(exc),
            "calls": 1,
            "compute_matched": False,
            "comparison_label": COMPARISON_LABEL,
            "provider": "frontier",
            "model": getattr(provider, "model", None),
            "base_url": getattr(provider, "base_url", None),
            "feasible": False,
            "fitness": None,
            "cost_usd": None,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    finally:
        closer = getattr(provider, "aclose", None)
        if closer is not None:
            await closer()
    genome = Genome(
        id="oneshot",
        run_id="oneshot",
        generation=0,
        operator="seed",
        payload=proposal.payload if isinstance(proposal.payload, dict) else {},
        hypothesis=proposal.hypothesis,
        provider="frontier",
        created_at="",
        content_hash="oneshot",
    )
    try:
        adapter.validate(genome.payload, objective)
        if hasattr(adapter, "bind_objective"):
            adapter.bind_objective(objective)  # type: ignore[attr-defined]
        evaluation = adapter.evaluate(genome, "test")
    except ValueError as exc:
        evaluation = None
        invalid = str(exc)
    else:
        invalid = None
    feasible = bool(evaluation and evaluation.feasible) if evaluation is not None else False
    record: dict[str, Any] = {
        "arm": "frontier_one_shot",
        "status": "invalid" if not feasible else "completed",
        "reason": invalid or (evaluation.feedback if evaluation and not evaluation.feasible else None),
        "calls": 1,
        "compute_matched": False,
        "comparison_label": COMPARISON_LABEL,
        "provider": "frontier",
        "model": proposal.model,
        "base_url": provider.base_url,
        "provider_run_id": proposal.provider_run_id,
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "payload": genome.payload,
        "feasible": feasible,
        "fitness": None if evaluation is None else evaluation.fitness,
        "metrics": {} if evaluation is None else evaluation.metrics,
        "test_split": "test",
        "latency_ms": proposal.latency_ms,
        "cost_usd": proposal.cost_usd,
        "prompt_tokens": proposal.prompt_tokens,
        "completion_tokens": proposal.completion_tokens,
    }
    return record


def attach_baseline(export_path: Path | str, record: dict[str, Any]) -> dict[str, Any]:
    path = Path(export_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["one_shot_baseline"] = record
    document["integrity_sha256"] = export_integrity(document)
    if not verify_export(document):
        raise RuntimeError("export integrity failed after attaching the one-shot baseline")
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return document
