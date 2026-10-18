"""Shared OpenAI-compatible chat client used by Veri and the optional frontier baseline.

The OpenAI SDK appends ``/chat/completions``. Callers must pass the deployment
base URL, not the absolute chat URL.
"""

from __future__ import annotations

import json
import time
from typing import Any

from geneai_core.providers.base import Proposal, ProviderError, coerce_proposal_body, message_text, parse_json_object


def reject_chat_completions_base(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        raise ValueError(
            "base_url must not end in /chat/completions; the OpenAI SDK appends that path"
        )
    return cleaned


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        deployment_id: str | None = None,
        client: Any | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        max_attempts: int = 4,
        mode: str = "live",
    ) -> None:
        if not api_key:
            raise ProviderError(f"{name} api key is empty")
        if not model:
            raise ProviderError(f"{name} model is empty")
        self.name = name
        self.mode = mode
        self.model = model
        self.deployment_id = deployment_id
        self.base_url = reject_chat_completions_base(base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._client = client
        self._api_key = api_key

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self.base_url, timeout=120.0)
        return self._client

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            await client.close()

    async def propose(self, instruction: str, schema: dict[str, Any], *, call_id: str) -> Proposal:
        started = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "You propose one candidate genome. Reply with a single JSON object "
                    "containing keys payload and hypothesis. payload must match the schema. "
                    "No markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\nJSON schema:\n{json.dumps(schema, sort_keys=True)}"
                ),
            },
        ]
        text, response, used_response_format = await self._complete(messages)
        try:
            parsed = parse_json_object(text)
        except ProviderError:
            # One repair-oriented reparse is not a second generation. If the
            # text is still not JSON, the controller may request one repair call.
            raise
        payload, hypothesis = coerce_proposal_body(parsed)
        usage = getattr(response, "usage", None)
        extra = getattr(response, "model_extra", None) or {}
        if not isinstance(extra, dict):
            extra = {}
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Proposal(
            payload=payload if isinstance(payload, dict) else {"value": payload},
            hypothesis=hypothesis,
            provider=self.name,
            provider_run_id=str(getattr(response, "id", None) or call_id),
            model=str(getattr(response, "model", None) or self.model),
            deployment_id=self.deployment_id,
            prompt_tokens=_usage_int(usage, "prompt_tokens"),
            completion_tokens=_usage_int(usage, "completion_tokens"),
            total_tokens=_usage_int(usage, "total_tokens"),
            cost_usd=None,
            latency_ms=latency_ms,
            mode=self.mode,
            raw_text=text,
            extra={
                "call_id": call_id,
                "base_url": self.base_url,
                "response_format": used_response_format,
                "finish_reason": _finish_reason(response),
                "model_extra_keys": sorted(extra.keys()),
            },
        )

    async def propose_json(self, instruction: str, schema: dict[str, Any]) -> dict[str, Any]:
        proposal = await self.propose(instruction, schema, call_id="direct")
        return {"payload": proposal.payload, "hypothesis": proposal.hypothesis}

    async def _complete(self, messages: list[dict[str, str]]) -> tuple[str, Any, bool]:
        client = self._client_or_create()
        use_format = True
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if use_format:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if use_format and status in {400, 422}:
                    use_format = False
                    continue
                if status in {429, 503, 408} and attempt + 1 < self.max_attempts:
                    await _sleep_retry(exc, attempt)
                    continue
                raise ProviderError(
                    f"{self.name} chat completion failed: {exc.__class__.__name__}",
                    retryable=status in {429, 503, 408},
                    status_code=status,
                    code=_error_code(exc),
                ) from exc
            error_body = _embedded_error(response)
            if error_body is not None:
                raise ProviderError(
                    f"{self.name} returned an error body: {error_body.get('code') or 'error'}",
                    code=str(error_body.get("code") or "") or None,
                )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ProviderError(f"{self.name} returned no choices")
            text = message_text(getattr(choices[0].message, "content", None)).strip()
            if not text:
                raise ProviderError(f"{self.name} returned empty content")
            return text, response, use_format
        raise ProviderError(
            f"{self.name} chat completion failed after retries",
            retryable=True,
        ) from last_error


def _usage_int(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    if value is None:
        return None
    return int(value)


def _finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)


def _embedded_error(response: Any) -> dict[str, Any] | None:
    extra = getattr(response, "model_extra", None) or {}
    if isinstance(extra, dict) and isinstance(extra.get("error"), dict):
        choices = getattr(response, "choices", None)
        if not choices:
            return extra["error"]
    return None


def _error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code"):
            return str(error["code"])
        if isinstance(error, str):
            return error
    return None


async def _sleep_retry(exc: Exception, attempt: int) -> None:
    import asyncio

    delay = min(2**attempt, 8)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    retry_after = None
    if hasattr(headers, "get"):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is not None:
        try:
            delay = min(float(retry_after), 20.0)
        except (TypeError, ValueError):
            pass
    await asyncio.sleep(delay)
