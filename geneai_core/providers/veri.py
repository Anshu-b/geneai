"""veri.studio OpenAI-compatible deployments.

Docs checked 2026-10-18:
https://docs.veri.studio/deployments/openai-compat
https://docs.veri.studio/quickstart

``base_url`` is ``https://api.veri.studio/v1/deployments/{deployment_id}``.
The OpenAI SDK appends ``/chat/completions``. ``model`` is the deployment
name (the OpenAI model field), not the Hugging Face repo id.

This project's deployment is ``bfe84d51e141`` / ``geneai-qwen7b``
(Qwen/Qwen2.5-7B-Instruct, vLLM, 1x L4-24GB, concurrency_target 8).
In-flight completions are capped at that target. 429 and 503 retry with
``Retry-After`` while a scaled-to-zero replica wakes.
"""

from __future__ import annotations

import os
from typing import Any

from geneai_core.providers.base import CredentialsError
from geneai_core.providers.openai_compat import OpenAICompatibleProvider

VERI_ORIGIN = "https://api.veri.studio"
# Not secrets. Verified externally (create returned 201) for this project.
EXAMPLE_DEPLOYMENT_ID = "bfe84d51e141"
EXAMPLE_MODEL = "geneai-qwen7b"
# OpenAI model field is the deployment name, not the Hugging Face repo.
EXAMPLE_BASE_URL = f"{VERI_ORIGIN}/v1/deployments/{EXAMPLE_DEPLOYMENT_ID}"
# Qwen/Qwen2.5-7B-Instruct on vLLM, 1x L4-24GB. concurrency_target is the
# per-replica in-flight setpoint. About $1.05/hr is deployment billing, not a
# per-completion price, so Proposal.cost_usd stays null unless a response
# actually carries a price.
VERI_CONCURRENCY_TARGET = 8
VERI_COST_PER_HOUR_USD = 1.05
VERI_WAKE_TIMEOUT_S = 180.0


def veri_deployment_base_url(deployment_id: str) -> str:
    segment = (deployment_id or "").strip()
    if not segment or any(char in segment for char in "/?# "):
        raise ValueError("VERI_DEPLOYMENT_ID must be a single path segment")
    if segment.endswith("chat") or "completions" in segment:
        raise ValueError("VERI_DEPLOYMENT_ID must be the deployment id, not a chat path")
    return f"{VERI_ORIGIN}/v1/deployments/{segment}"


class VeriProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        *,
        api_key: str,
        deployment_id: str,
        model: str,
        client: Any | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        in_flight_limit: int = VERI_CONCURRENCY_TARGET,
        wake_timeout_s: float = VERI_WAKE_TIMEOUT_S,
        sleeper: Any | None = None,
    ) -> None:
        if in_flight_limit < 1:
            raise ValueError("Veri in_flight_limit must be >= 1")
        super().__init__(
            name="veri",
            api_key=api_key,
            base_url=veri_deployment_base_url(deployment_id),
            model=model,
            deployment_id=deployment_id.strip(),
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="live",
            in_flight_limit=in_flight_limit,
            wake_timeout_s=wake_timeout_s,
            sleeper=sleeper,
        )

    @classmethod
    def from_env(cls) -> VeriProvider:
        api_key = os.environ.get("VERI_API_KEY", "").strip()
        deployment_id = os.environ.get("VERI_DEPLOYMENT_ID", "").strip() or EXAMPLE_DEPLOYMENT_ID
        model = os.environ.get("VERI_MODEL", "").strip() or EXAMPLE_MODEL
        if not api_key:
            raise CredentialsError(
                "VERI_API_KEY is missing. No request was sent. "
                f"VERI_DEPLOYMENT_ID defaults to {EXAMPLE_DEPLOYMENT_ID} and "
                f"VERI_MODEL defaults to {EXAMPLE_MODEL} when unset."
            )
        wake_timeout_s = float(os.environ.get("VERI_WAKE_TIMEOUT_SECONDS", str(VERI_WAKE_TIMEOUT_S)))
        raw_limit = os.environ.get("VERI_CONCURRENCY", "").strip()
        in_flight_limit = int(raw_limit) if raw_limit else VERI_CONCURRENCY_TARGET
        return cls(
            api_key=api_key,
            deployment_id=deployment_id,
            model=model,
            in_flight_limit=in_flight_limit,
            wake_timeout_s=wake_timeout_s,
        )


def veri_blocker_message() -> str:
    return "\n".join(
        [
            "BLOCKED: Veri smoke was not executed. No sponsor call was made.",
            "Missing environment variable: VERI_API_KEY.",
            f"VERI_DEPLOYMENT_ID defaults to {EXAMPLE_DEPLOYMENT_ID} when unset.",
            f"VERI_MODEL defaults to {EXAMPLE_MODEL} when unset (OpenAI model field = deployment name).",
            "Setup:",
            "1. Create an API key in the Veri dashboard. Keys use the vk_ prefix.",
            "2. Optional CLI: uv tool install 'veri-sdk[mcp]' && veri login --token \"$VERI_API_KEY\" && veri whoami",
            "3. Export VERI_API_KEY.",
            "4. Re-run: uv run python scripts/smoke_veri.py",
            f"Base URL: {EXAMPLE_BASE_URL}",
            "The OpenAI SDK appends /chat/completions. Do not include that suffix in the base URL.",
            "Deployment: Qwen/Qwen2.5-7B-Instruct on vLLM, 1x L4-24GB, concurrency_target 8, about $1.05/hr.",
            "In-flight chat calls are capped at 8. 429 and 503 (deployment_waking) retry with Retry-After.",
            "A cold start answers 503 for a few minutes. Default wake budget is 180 seconds.",
        ]
    )
