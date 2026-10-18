"""Provider protocols. Concrete sponsor clients live beside this module."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        code: str | None = None,
        provider_run_id: str | None = None,
        repair_attempted: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.code = code
        self.provider_run_id = provider_run_id
        self.repair_attempted = repair_attempted


class CredentialsError(ProviderError):
    """Raised before any network call when configuration is incomplete or the wrong path."""


class Proposal(BaseModel):
    payload: dict[str, Any]
    hypothesis: str
    provider: str
    provider_run_id: str | None = None
    model: str | None = None
    deployment_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_is_estimate: bool = False
    credits: float | None = None
    latency_ms: int = 0
    mode: str = "mock"
    raw_text: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(Protocol):
    name: str
    mode: str

    async def propose(self, instruction: str, schema: dict[str, Any], *, call_id: str) -> Proposal: ...

    async def propose_json(self, instruction: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class AgentWorkflowProvider(Protocol):
    name: str
    mode: str

    async def run_workflow(
        self,
        workflow_name: str,
        inputs: dict[str, Any],
        *,
        call_id: str,
    ) -> Proposal: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_json_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = _FENCE.sub("", raw).strip()
    return raw


def first_balanced_json(text: str) -> Any:
    """First balanced JSON object or array, ignoring fences and surrounding prose."""

    raw = strip_json_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char in "}]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blob = raw[start : index + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    start = None
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    parsed = first_balanced_json(text)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                parsed = item
                break
    if not isinstance(parsed, dict):
        raise ProviderError("provider response was not a JSON object")
    return parsed


def coerce_proposal_body(parsed: dict[str, Any]) -> tuple[dict[str, Any], str]:
    hypothesis = parsed.get("hypothesis") or ""
    payload = parsed.get("payload")
    if isinstance(payload, dict):
        return payload, str(hypothesis)
    # Some agents return the domain object at the top level.
    return parsed, str(hypothesis)


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "".join(parts)
    return str(content)
