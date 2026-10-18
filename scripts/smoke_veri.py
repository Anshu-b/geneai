#!/usr/bin/env python3
"""One real Veri chat completion, or a key-free blocker if credentials are absent."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.providers.base import CredentialsError, ProviderError  # noqa: E402
from geneai_core.providers.veri import VeriProvider, veri_blocker_message, veri_deployment_base_url  # noqa: E402


async def _run() -> int:
    load_dotenv()
    try:
        provider = VeriProvider.from_env()
    except CredentialsError:
        print(veri_blocker_message())
        return 3
    schema = {
        "type": "object",
        "required": ["payload", "hypothesis"],
        "properties": {
            "payload": {
                "type": "object",
                "required": ["ok", "source"],
                "properties": {"ok": {"type": "boolean"}, "source": {"type": "string"}},
            },
            "hypothesis": {"type": "string"},
        },
    }
    try:
        proposal = await provider.propose(
            'Return JSON {"payload": {"ok": true, "source": "veri"}, "hypothesis": "smoke"}.',
            schema,
            call_id="smoke",
        )
    except ProviderError as exc:
        print(
            json.dumps(
                {
                    "integration": "veri",
                    "status": "failed",
                    "error": str(exc),
                    "code": exc.code,
                    "http_status": exc.status_code,
                    "base_url": provider.base_url,
                    "deployment_id": provider.deployment_id,
                    "model": provider.model,
                },
                indent=2,
            )
        )
        return 1
    finally:
        await provider.aclose()
    report = {
        "integration": "veri",
        "status": "completed",
        "provider_mode": "live",
        "base_url": veri_deployment_base_url(provider.deployment_id or ""),
        "deployment_id": provider.deployment_id,
        "model": proposal.model,
        "completion_id": proposal.provider_run_id,
        "prompt_tokens": proposal.prompt_tokens,
        "completion_tokens": proposal.completion_tokens,
        "total_tokens": proposal.total_tokens,
        "cost_usd": proposal.cost_usd,
        "latency_ms": proposal.latency_ms,
        "payload": proposal.payload,
        "note": "This was a live Veri chat completion. cost_usd is null unless the response includes a price.",
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
