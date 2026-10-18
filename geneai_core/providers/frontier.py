"""Optional OpenAI-compatible client for a later one-shot baseline.

Phase 1 does not call this during evolution. Arm A in section 10 will.
``FRONTIER_BASE_URL`` must be a real base URL you control; this module does
not invent a vendor endpoint.
"""

from __future__ import annotations

import os
from typing import Any

from geneai_core.providers.base import CredentialsError
from geneai_core.providers.openai_compat import OpenAICompatibleProvider, reject_chat_completions_base


class FrontierProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            name="frontier",
            api_key=api_key,
            base_url=reject_chat_completions_base(base_url),
            model=model,
            client=client,
            mode="live",
        )

    @classmethod
    def from_env(cls) -> FrontierProvider:
        api_key = os.environ.get("FRONTIER_API_KEY", "").strip()
        base_url = os.environ.get("FRONTIER_BASE_URL", "").strip()
        model = os.environ.get("FRONTIER_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("FRONTIER_API_KEY", api_key),
                ("FRONTIER_BASE_URL", base_url),
                ("FRONTIER_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise CredentialsError(
                "Frontier baseline credentials are incomplete ("
                + ", ".join(missing)
                + "). No request was sent."
            )
        return cls(api_key=api_key, base_url=base_url, model=model)
