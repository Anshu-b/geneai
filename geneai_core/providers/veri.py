"""veri.studio OpenAI-compatible deployments.

Docs checked 2026-10-18:
https://docs.veri.studio/deployments/openai-compat
https://docs.veri.studio/quickstart

``base_url`` is ``https://api.veri.studio/v1/deployments/{deployment_id}``.
The OpenAI SDK appends ``/chat/completions``. ``model`` is the deployment's
served name (the proxy rewrites it on a single-model deployment).
"""

from __future__ import annotations

import os
from typing import Any

from geneai_core.providers.base import CredentialsError
from geneai_core.providers.openai_compat import OpenAICompatibleProvider

VERI_ORIGIN = "https://api.veri.studio"


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
    ) -> None:
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
        )

    @classmethod
    def from_env(cls) -> VeriProvider:
        api_key = os.environ.get("VERI_API_KEY", "").strip()
        deployment_id = os.environ.get("VERI_DEPLOYMENT_ID", "").strip()
        model = os.environ.get("VERI_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("VERI_API_KEY", api_key),
                ("VERI_DEPLOYMENT_ID", deployment_id),
                ("VERI_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise CredentialsError(
                "Veri credentials are incomplete (" + ", ".join(missing) + "). No request was sent."
            )
        return cls(api_key=api_key, deployment_id=deployment_id, model=model)


def veri_blocker_message() -> str:
    return "\n".join(
        [
            "BLOCKED: Veri smoke was not executed. No sponsor call was made.",
            "Missing environment variables: VERI_API_KEY, VERI_DEPLOYMENT_ID, and/or VERI_MODEL.",
            "Setup:",
            "1. Create an API key in the Veri dashboard. Keys use the vk_ prefix.",
            "2. Optional CLI: uv tool install 'veri-sdk[mcp]' && veri login --token \"$VERI_API_KEY\" && veri whoami",
            "3. Deploy a model and copy the deployment id and served model name.",
            "   https://docs.veri.studio/quickstart",
            "   https://docs.veri.studio/deployments/openai-compat",
            "4. Export VERI_API_KEY, VERI_DEPLOYMENT_ID, and VERI_MODEL.",
            "5. Re-run: uv run python scripts/smoke_veri.py",
            "Base URL: https://api.veri.studio/v1/deployments/$VERI_DEPLOYMENT_ID",
            "The OpenAI SDK appends /chat/completions. Do not include that suffix in the base URL.",
            "A deployment that is scaled to zero returns 503 deployment_waking with Retry-After.",
        ]
    )
