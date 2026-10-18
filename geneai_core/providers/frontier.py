"""Optional frontier client for the one-shot arm.

``FRONTIER_BASE_URL`` must be a real base URL. This module does not invent a
vendor endpoint. gpt-5.5 and other reasoning models use the Responses API
(``max_output_tokens``, no temperature). Other models use Chat Completions
with ``max_completion_tokens``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from geneai_core.providers.base import (
    CredentialsError,
    Proposal,
    ProviderError,
    coerce_proposal_body,
    message_text,
    parse_json_object,
)
from geneai_core.providers.openai_compat import (
    OpenAICompatibleProvider,
    _candidate_bodies,
    _is_timeout,
    _usage_int,
    reject_chat_completions_base,
)


def uses_responses_api(model: str) -> bool:
    """gpt-5 and o-series reasoning models use Responses. Chat Completions remains for other frontier models."""

    name = model.strip().lower()
    return name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4")


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
            max_tokens=4000,
            request_timeout_s=90.0,
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

    async def propose(self, instruction: str, schema: dict[str, Any], *, call_id: str) -> Proposal:
        if uses_responses_api(self.model):
            return await self._propose_responses(instruction, schema, call_id=call_id)
        text, response, used_format = await self._complete(
            _chat_messages(instruction, schema),
            token_field="max_completion_tokens",
        )
        return _proposal_from_text(
            self,
            text,
            response,
            call_id=call_id,
            used_format=used_format,
            api="chat.completions",
        )

    async def _propose_responses(self, instruction: str, schema: dict[str, Any], *, call_id: str) -> Proposal:
        started = time.perf_counter()
        client = self._client_or_create()
        use_format = True
        timeouts = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": (
                    "You propose one candidate genome. Reply with a single JSON object "
                    "containing keys payload and hypothesis. No markdown."
                ),
                "input": f"{instruction}\n\nJSON schema:\n{json.dumps(schema, sort_keys=True)}",
                "max_output_tokens": self.max_tokens,
                "store": False,
            }
            if use_format:
                kwargs["text"] = {"format": {"type": "json_object"}}
            try:
                response = await client.responses.create(**kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if use_format and status in {400, 422}:
                    use_format = False
                    continue
                if _is_timeout(exc) or status == 408:
                    timeouts += 1
                    if timeouts >= 3:
                        raise ProviderError(
                            f"frontier responses timed out after {timeouts} attempts",
                            retryable=True,
                            status_code=status or 408,
                            code="timeout",
                        ) from exc
                    await self._sleep(min(2 ** (timeouts - 1), 4))
                    continue
                raise ProviderError(
                    f"frontier responses failed: {exc.__class__.__name__}",
                    retryable=status in {429, 503, 408},
                    status_code=status,
                ) from exc
            text = _responses_text(response)
            if not text:
                raise ProviderError("frontier responses returned empty output")
            proposal = _proposal_from_text(
                self,
                text,
                response,
                call_id=call_id,
                used_format=use_format,
                api="responses",
            )
            proposal.latency_ms = int((time.perf_counter() - started) * 1000)
            proposal.prompt_tokens = _usage_int(getattr(response, "usage", None), "input_tokens") or proposal.prompt_tokens
            proposal.completion_tokens = (
                _usage_int(getattr(response, "usage", None), "output_tokens") or proposal.completion_tokens
            )
            return proposal


def _chat_messages(instruction: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You propose one candidate genome. Reply with a single JSON object "
                "containing keys payload and hypothesis. payload must match the schema. No markdown."
            ),
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nJSON schema:\n{json.dumps(schema, sort_keys=True)}",
        },
    ]


def _responses_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(str(text))
    return message_text("".join(parts)).strip()


def _proposal_from_text(
    provider: FrontierProvider,
    text: str,
    response: Any,
    *,
    call_id: str,
    used_format: bool,
    api: str,
) -> Proposal:
    parsed = parse_json_object(text)
    payload, hypothesis = coerce_proposal_body(parsed)
    candidates = _candidate_bodies(parsed, payload, hypothesis)
    stored = candidates[0]
    usage = getattr(response, "usage", None)
    return Proposal(
        payload=stored["payload"] if isinstance(stored.get("payload"), dict) else {"value": stored.get("payload")},
        hypothesis=str(stored.get("hypothesis") or hypothesis),
        provider=provider.name,
        provider_run_id=str(getattr(response, "id", None) or call_id),
        model=str(getattr(response, "model", None) or provider.model),
        prompt_tokens=_usage_int(usage, "prompt_tokens") or _usage_int(usage, "input_tokens"),
        completion_tokens=_usage_int(usage, "completion_tokens") or _usage_int(usage, "output_tokens"),
        total_tokens=_usage_int(usage, "total_tokens"),
        cost_usd=None,
        latency_ms=0,
        mode=provider.mode,
        raw_text=text,
        extra={
            "call_id": call_id,
            "base_url": provider.base_url,
            "api": api,
            "response_format": used_format,
            "candidates": candidates if len(candidates) > 1 else [],
        },
    )
