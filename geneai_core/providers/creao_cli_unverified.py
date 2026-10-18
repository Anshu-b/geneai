"""Unused CREAO CLI client.

This hackathon account authenticates with a ``capi_`` app-scoped key.
``developer.creao.ai/v1`` rejects that key. GeneAI does not construct this
provider from the environment and the smoke script does not call it.

The CLI contract, verified from docs but unverified for this account, is:

    POST https://developer.creao.ai/v1/runs
    Authorization: Bearer cr_sk_...
    {"agent_id": "<uuid>", "input": {...}}

    GET https://developer.creao.ai/v1/runs/{run_id}

Do not point ``CreaoProvider`` at those URLs.
"""

from __future__ import annotations

from geneai_core.providers.base import CredentialsError


class UnverifiedCreaoCliProvider:
    """Raises on construction. Present so the two route families stay separate."""

    name = "creao-cli-unverified"
    mode = "live"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise CredentialsError(
            "CREAO CLI /v1 is unverified for this account and is disabled. "
            "capi_ keys are rejected by developer.creao.ai/v1. "
            "Use geneai_core.providers.creao.CreaoProvider with CREAO_API_KEY and CREAO_APP_ID."
        )
