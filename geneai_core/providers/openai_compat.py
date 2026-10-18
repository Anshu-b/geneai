"""Shared OpenAI-compatible chat client used by Veri and the optional frontier baseline.

The OpenAI SDK appends ``/chat/completions``. Callers must pass the deployment
base URL, not the absolute chat URL.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from geneai_core.operators import parse_context
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
        in_flight_limit: int | None = None,
        wake_timeout_s: float = 180.0,
        sleeper: Any | None = None,
        request_timeout_s: float = 30.0,
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
        self.wake_timeout_s = wake_timeout_s
        self.request_timeout_s = request_timeout_s
        self.in_flight_limit = in_flight_limit
        self._in_flight = asyncio.Semaphore(in_flight_limit) if in_flight_limit else None
        self._sleeper = sleeper
        self._client = client
        self._api_key = api_key

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        # The SDK's own retries give up before a GPU cold start. This client
        # retries 429/503 itself and honors Retry-After.
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
            max_retries=0,
        )
        return self._client

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            await client.close()

    async def propose(self, instruction: str, schema: dict[str, Any], *, call_id: str) -> Proposal:
        started = time.perf_counter()
        context = parse_context(instruction)
        batch_size = _positive_int(context.get("batch_size"), default=1, cap=8)
        temperature = context.get("temperature")
        sample_seed = context.get("sample_seed")
        if batch_size > 1:
            system = (
                f"You propose {batch_size} distinct candidate genomes. "
                'Reply with one JSON object {"candidates":[{"payload":{...},"hypothesis":"..."}]}. '
                "payload must match the schema. The payloads must not be copies. No markdown."
            )
            token_cap = min(3200, max(self.max_tokens, 800 * batch_size))
        else:
            system = (
                "You propose one candidate genome. Reply with a single JSON object "
                "containing keys payload and hypothesis. payload must match the schema. "
                "No markdown."
            )
            token_cap = self.max_tokens
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{instruction}\n\nJSON schema:\n{json.dumps(schema, sort_keys=True)}",
            },
        ]
        text, response, used_response_format = await self._complete(
            messages,
            temperature=None if temperature is None else float(temperature),
            sample_seed=None if sample_seed is None else int(sample_seed),
            max_tokens=token_cap,
        )
        try:
            parsed = parse_json_object(text)
        except ProviderError:
            # One repair-oriented reparse is not a second generation. If the
            # text is still not JSON, the controller may request one repair call.
            raise
        payload, hypothesis = coerce_proposal_body(parsed)
        candidates = _candidate_bodies(parsed, payload, hypothesis)
        usage = getattr(response, "usage", None)
        extra = getattr(response, "model_extra", None) or {}
        if not isinstance(extra, dict):
            extra = {}
        latency_ms = int((time.perf_counter() - started) * 1000)
        stored = dict(candidates[0])
        return Proposal(
            payload=stored["payload"] if isinstance(stored.get("payload"), dict) else {"value": stored.get("payload")},
            hypothesis=str(stored.get("hypothesis") or hypothesis),
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
                "candidates": candidates if len(candidates) > 1 else [],
                "temperature": self.temperature if temperature is None else float(temperature),
                "sample_seed": None if sample_seed is None else int(sample_seed),
            },
        )

    async def propose_json(self, instruction: str, schema: dict[str, Any]) -> dict[str, Any]:
        proposal = await self.propose(instruction, schema, call_id="direct")
        return {"payload": proposal.payload, "hypothesis": proposal.hypothesis}

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        sample_seed: int | None = None,
        max_tokens: int | None = None,
        token_field: str = "max_tokens",
    ) -> tuple[str, Any, bool]:
        client = self._client_or_create()
        use_format = True
        send_seed = sample_seed is not None
        slept = 0.0
        timeouts = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                token_field: self.max_tokens if max_tokens is None else max_tokens,
            }
            if send_seed and sample_seed is not None:
                kwargs["seed"] = int(sample_seed)
            if use_format:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = await self._create(client, kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                code = _error_code(exc)
                if use_format and status in {400, 422}:
                    use_format = False
                    continue
                if send_seed and status in {400, 422} and "seed" in str(exc).lower():
                    send_seed = False
                    continue
                if _is_timeout(exc) or status == 408:
                    timeouts += 1
                    if timeouts >= 3:
                        raise ProviderError(
                            f"{self.name} chat completion timed out after {timeouts} attempts",
                            retryable=True,
                            status_code=status or 408,
                            code="timeout",
                        ) from exc
                    await self._sleep(min(2 ** (timeouts - 1), 4))
                    continue
                if status in {429, 503}:
                    waking = status == 503
                    delay = _retry_delay(exc, waking=waking)
                    if self.wake_timeout_s <= 0 or slept + delay > self.wake_timeout_s:
                        raise ProviderError(
                            f"{self.name} chat completion failed after retries: {code or status}",
                            retryable=True,
                            status_code=status,
                            code=code or ("deployment_waking" if waking else None),
                        ) from exc
                    await self._sleep(delay)
                    slept += delay
                    continue
                raise ProviderError(
                    f"{self.name} chat completion failed: {exc.__class__.__name__}",
                    retryable=status in {429, 503, 408},
                    status_code=status,
                    code=code,
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

    async def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        if self._in_flight is None:
            return await client.chat.completions.create(**kwargs)
        async with self._in_flight:
            return await client.chat.completions.create(**kwargs)

    async def _sleep(self, delay: float) -> None:
        sleeper = self._sleeper or asyncio.sleep
        await sleeper(delay)


def _positive_int(value: Any, *, default: int, cap: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(1, min(cap, int(value)))


def _candidate_bodies(parsed: dict[str, Any], payload: Any, hypothesis: str) -> list[dict[str, Any]]:
    raw = parsed.get("candidates")
    if raw is None:
        raw = parsed.get("children")
    bodies: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("payload"), dict):
                bodies.append(
                    {
                        "payload": item["payload"],
                        "hypothesis": str(item.get("hypothesis") or hypothesis),
                    }
                )
            elif "payload" not in item and "hypothesis" not in item:
                bodies.append({"payload": item, "hypothesis": hypothesis})
    if bodies:
        return bodies
    if isinstance(payload, dict):
        return [{"payload": payload, "hypothesis": hypothesis}]
    return [{"payload": {"value": payload}, "hypothesis": hypothesis}]


def _is_timeout(exc: Exception) -> bool:
    name = exc.__class__.__name__
    return name in {
        "APITimeoutError",
        "APIConnectionError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutException",
    } or isinstance(exc, TimeoutError)


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


def _retry_delay(exc: Exception, *, waking: bool) -> float:
    """Seconds to wait before another 429 or 503. Waking GPUs advertise ~30s."""

    header = _header_retry_after(exc)
    body_seconds = _body_retry_after(exc)
    if header is not None:
        delay = header
    elif body_seconds is not None:
        delay = body_seconds
    elif waking:
        delay = 30.0
    else:
        delay = 2.0
    if delay <= 0:
        delay = 30.0 if waking else 2.0
    cap = 60.0 if waking else 20.0
    return min(delay, cap)


def _header_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    if not hasattr(headers, "get"):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _body_retry_after(exc: Exception) -> float | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    raw = None
    if isinstance(error, dict):
        raw = error.get("retry_after_seconds")
    if raw is None:
        raw = body.get("retry_after_seconds")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
