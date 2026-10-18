#!/usr/bin/env python3
"""One real GeneAI Mutator run on the app-scoped CREAO path, or a key-free blocker.

The report never prints the API key. CREAO_APP_ID is not a secret.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.providers.base import CredentialsError, ProviderError  # noqa: E402
from geneai_core.providers.creao import (  # noqa: E402
    EXAMPLE_APP_ID,
    CreaoProvider,
    creao_blocker_message,
)
from geneai_core.providers.creao_cli_unverified import UnverifiedCreaoCliProvider  # noqa: E402


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok", "source"],
    "properties": {
        "ok": {"type": "boolean"},
        "source": {"type": "string"},
    },
}


def _key_family(value: str) -> str:
    if value.startswith("capi_"):
        return "capi_"
    if value.startswith("cr_sk_"):
        return "cr_sk_"
    if not value:
        return "missing"
    return "other"


async def _run() -> int:
    load_dotenv()
    raw_key = os.environ.get("CREAO_API_KEY", "").strip()
    app_id = os.environ.get("CREAO_APP_ID", "").strip() or EXAMPLE_APP_ID
    ignored = []
    if os.environ.get("CREAO_AGENT_ID"):
        ignored.append("CREAO_AGENT_ID is ignored; this account uses CREAO_APP_ID")
    if os.environ.get("CREAO_ALLOW_UNVERIFIED_CLI"):
        ignored.append("CREAO_ALLOW_UNVERIFIED_CLI does not enable developer.creao.ai/v1")
    try:
        UnverifiedCreaoCliProvider()
    except CredentialsError as exc:
        cli_status = str(exc)
    else:
        cli_status = "unexpectedly constructed"
    if not raw_key:
        print(creao_blocker_message())
        print(json.dumps({"ignored": ignored, "cli_path": "disabled", "cli_note": cli_status}, indent=2))
        return 3
    if _key_family(raw_key) != "capi_":
        print(creao_blocker_message())
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "CREAO_API_KEY is not a capi_ app-scoped key. No request was sent.",
                    "key_family": _key_family(raw_key),
                },
                indent=2,
            )
        )
        return 3
    try:
        provider = CreaoProvider.from_env()
    except CredentialsError as exc:
        print(creao_blocker_message())
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2))
        return 3

    list_summary = await provider.list_runs()
    list_report = {key: value for key, value in list_summary.items() if key != "body"}
    inputs = {
        "task": (
            "Return only JSON of the form "
            '{"children":[{"genome":{"ok":true,"source":"creao"},"rationale":"smoke"}]}.'
        ),
        "parents": [{"genome": {"ok": False, "source": "seed"}, "fitness": 0.0}],
        "operator": "mutate",
        "response_schema": SCHEMA,
    }
    try:
        proposal = await provider.run_workflow("propose_mutation", inputs, call_id="smoke")
    except ProviderError as exc:
        print(
            json.dumps(
                {
                    "integration": "creao",
                    "path": "app-scoped-api-trigger",
                    "status": "failed",
                    "app_id": provider.app_id,
                    "example_app_id": EXAMPLE_APP_ID,
                    "key_family": "capi_",
                    "run_id": exc.provider_run_id,
                    "error": str(exc),
                    "code": exc.code,
                    "http_status": exc.status_code,
                    "list": list_report,
                    "ignored": ignored,
                    "note": "No API key is included in this report.",
                },
                indent=2,
            )
        )
        return 1
    excerpt = (proposal.raw_text or "")[:400]
    report = {
        "integration": "creao",
        "path": "app-scoped-api-trigger",
        "status": "completed",
        "provider_mode": "live",
        "agent": "GeneAI Mutator",
        "app_id": provider.app_id,
        "run_id": proposal.provider_run_id,
        "wire_operator": proposal.extra.get("wire_operator"),
        "thread_id": proposal.extra.get("thread_id"),
        "started_at": proposal.extra.get("started_at"),
        "completed_at": proposal.extra.get("completed_at"),
        "server_latency_ms": proposal.extra.get("server_latency_ms"),
        "answer_source": proposal.extra.get("answer_source"),
        "app_name": proposal.extra.get("app_name"),
        "child_count": proposal.extra.get("child_count"),
        "schema_repairs": proposal.extra.get("schema_repairs"),
        "credits": proposal.credits,
        "cost_usd": None,
        "latency_ms": proposal.latency_ms,
        "result_json_valid": True,
        "genome": proposal.payload,
        "rationale": proposal.hypothesis,
        "chat_text_excerpt": excerpt,
        "list": list_report,
        "ignored": ignored,
        "cli_path": "disabled",
        "note": "Live app-scoped run. cost_usd is not invented from credits. The API key is omitted.",
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
