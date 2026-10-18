"""CREAO app-scoped API Trigger for the GeneAI Mutator agent.

This account's key starts with ``capi_`` and is rejected by
``developer.creao.ai/v1``. Traffic stays on:

    POST https://agent.creao.ai/api/v1/apps/{appId}/runs
    GET  https://agent.creao.ai/api/v1/apps/{appId}/runs/{runId}
    GET  https://agent.creao.ai/api/v1/apps/{appId}/runs

Public route shapes were checked on 2026-10-18 against
https://docs.creao.ai/developer-api/api-paths and
https://docs.creao.ai/pro/api-trigger.
Terminal poll statuses: completed, failed, cancelled.
Create responses use ``id``. The answer is ``result.chatText`` (fenced JSON
plus prose). ``result.text`` echoes the inputs and is not parsed. A
``children.json`` artifact is the fallback. Failures expose ``errorCode``.

A live list call, verified outside this environment, returned
``200 {"runs": [], "limit": 20, "offset": 0}`` for app id
``0ed64fee-5003-4d56-a83b-c5f369d7a39c``.

That app is the GeneAI Mutator. Its input and output are not the generic
CLI ``input`` object:

    inputs: {task, parents: [{genome, fitness}], operator: mutate|crossover}
    result text: {children: [{genome, rationale}]}

The provider validates ``genome`` against the caller schema, applies a local
structural repair (unwrap aliases, clamp ranges, case-fold enums), and if the
genome is still illegal raises ``ProviderError`` with ``repair_attempted``
left false so the controller can issue one follow-up mutate. A second
automatic HTTP call inside this client is intentionally not made: the
controller owns the single repair budget.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import httpx

from geneai_core.diversity import normalize_payload
from geneai_core.providers.base import CredentialsError, Proposal, ProviderError

APP_ORIGIN = "https://agent.creao.ai"
# Not a secret. This is the GeneAI Mutator app id for the hackathon account.
EXAMPLE_APP_ID = "0ed64fee-5003-4d56-a83b-c5f369d7a39c"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
MUTATOR_OPERATORS = frozenset({"mutate", "crossover"})


def assert_app_scoped_key(api_key: str) -> None:
    if not api_key or not api_key.startswith("capi_"):
        raise CredentialsError(
            "CREAO_API_KEY must be an app-scoped capi_ key from the agent page. "
            "A cr_sk_ Developer Console key is the CLI path and is rejected for this account. "
            "No request was sent."
        )


def assert_app_id(app_id: str) -> str:
    segment = (app_id or "").strip()
    if not segment or any(char in segment for char in "/?#& "):
        raise CredentialsError("CREAO_APP_ID must be a single path segment. No request was sent.")
    return segment


def app_runs_collection(app_id: str) -> str:
    return f"{APP_ORIGIN}/api/v1/apps/{assert_app_id(app_id)}/runs"


def app_run_status(app_id: str, run_id: str) -> str:
    if not run_id or any(char in run_id for char in "/?#& "):
        raise ProviderError("CREAO run id is missing or not a single path segment")
    return f"{app_runs_collection(app_id)}/{run_id}"


def normalize_operator(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"mutate", "mutation", "repair"}:
        return "mutate"
    if text == "crossover":
        return "crossover"
    raise ProviderError(
        "GeneAI Mutator operator must be mutate or crossover. "
        f"Refusing to send {value!r}."
    )


def normalize_parent(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ProviderError("each GeneAI Mutator parent must be an object")
    if isinstance(item.get("genome"), dict):
        genome = item["genome"]
        fitness = item.get("fitness")
    elif isinstance(item.get("payload"), dict):
        genome = item["payload"]
        fitness = item.get("fitness")
    else:
        raise ProviderError("each parent needs a genome object")
    if isinstance(fitness, bool) or not isinstance(fitness, (int, float)):
        fitness = None
    return {"genome": normalize_payload(genome), "fitness": fitness}


def mutator_wire_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Project a workflow call onto the Mutator's three fields. Nothing else is sent."""

    if "task" not in inputs:
        raise ProviderError("GeneAI Mutator requires task. No request was sent.")
    operator = normalize_operator(inputs.get("operator"))
    parents = [normalize_parent(item) for item in inputs.get("parents") or []]
    if operator == "crossover" and len(parents) < 2:
        raise ProviderError("crossover requires two parents. No request was sent.")
    if operator == "mutate" and len(parents) < 1:
        raise ProviderError("mutate requires a parent genome. No request was sent.")
    return {
        "task": str(inputs["task"]),
        "parents": parents,
        "operator": operator,
    }


def build_create_body(inputs: dict[str, Any], *, webhook_url: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"inputs": mutator_wire_inputs(inputs)}
    if webhook_url:
        body["webhookUrl"] = webhook_url
    if "input" in body or "agent_id" in body:
        raise ProviderError("refusing to send CLI fields on the app-scoped path")
    return body


def run_id_from_create(document: dict[str, Any]) -> str:
    run_id = document.get("id") or document.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise ProviderError("app-scoped create response did not include id")
    return run_id


def chat_answer_text(document: dict[str, Any]) -> str:
    """Mutator answer. Never ``result.text``, which echoes the request."""

    result = document.get("result")
    if isinstance(result, dict):
        chat = result.get("chatText") or result.get("chat_text")
        if isinstance(chat, str):
            return chat
    return ""


def debug_result_text(document: dict[str, Any]) -> str:
    """Echoed inputs. Kept for logs. Not a genome proposal."""

    result = document.get("result")
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"]
    return ""


def children_artifact(document: dict[str, Any]) -> dict[str, Any] | None:
    result = document.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list):
        return None
    for item in result["artifacts"]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        artifact_id = str(item.get("id") or "")
        if "children.json" in title or artifact_id.endswith("children.json"):
            return item
    return None


def server_interval_ms(document: dict[str, Any]) -> int | None:
    started = document.get("startedAt")
    completed = document.get("completedAt")
    if not isinstance(started, str) or not isinstance(completed, str):
        return None
    try:
        delta = _parse_zulu(completed) - _parse_zulu(started)
    except ValueError:
        return None
    return int(delta.total_seconds() * 1000)


def _parse_zulu(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_completed_run(
    document: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a finished GET body. Parses ``chatText`` only."""

    genome, rationale, notes = interpret_mutator_output(chat_answer_text(document), schema)
    return {
        "genome": genome,
        "rationale": rationale,
        "notes": notes,
        "answer_source": "chatText",
        "thread_id": document.get("threadId"),
        "run_id": document.get("id") or document.get("runId"),
        "started_at": document.get("startedAt"),
        "completed_at": document.get("completedAt"),
        "server_latency_ms": server_interval_ms(document),
        "debug_text": debug_result_text(document),
    }


def error_code_from(document: dict[str, Any]) -> str | None:
    code = document.get("errorCode")
    if isinstance(code, str) and code:
        return code
    error = document.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, dict) and error.get("code"):
        return str(error["code"])
    return None


def extract_json_values(text: str) -> list[Any]:
    """Pull JSON objects out of raw model text, fenced blocks, and surrounding prose."""

    found: list[Any] = []
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    for candidate in (stripped, text):
        try:
            found.append(json.loads(candidate))
        except json.JSONDecodeError:
            pass
    found.extend(_scan_balanced(text))
    return found


def _scan_balanced(text: str) -> list[Any]:
    values: list[Any] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    opener = ""
    for index, char in enumerate(text):
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
                opener = char
            depth += 1
            continue
        if char in "}]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start : index + 1]
                try:
                    values.append(json.loads(blob))
                except json.JSONDecodeError:
                    pass
                start = None
    return values


def _as_child(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    genome = item.get("genome")
    if not isinstance(genome, dict) and isinstance(item.get("payload"), dict):
        genome = item["payload"]
    if not isinstance(genome, dict):
        return None
    rationale = item.get("rationale")
    if rationale is None:
        rationale = item.get("hypothesis") or item.get("description") or ""
    return {"genome": genome, "rationale": str(rationale)}


def children_from_parsed(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        children = parsed.get("children")
        if isinstance(children, list):
            return [child for item in children if (child := _as_child(item))]
        direct = _as_child(parsed)
        if direct:
            return [direct]
        if isinstance(parsed.get("payload"), dict) or _looks_like_genome(parsed):
            rationale = str(parsed.get("hypothesis") or parsed.get("rationale") or "")
            genome = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else parsed
            if isinstance(genome, dict):
                return [{"genome": genome, "rationale": rationale}]
    if isinstance(parsed, list):
        return [child for item in parsed if (child := _as_child(item))]
    return []


def _looks_like_genome(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("family", "params", "kind", "policy_family"))


def payload_schema_of(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {}
    props = schema.get("properties") or {}
    if isinstance(props.get("payload"), dict):
        return props["payload"]
    return schema


def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    if not schema:
        return []
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} must be one of {schema['enum']}"]
    kind = schema.get("type")
    if kind == "object" or (not kind and "properties" in schema):
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        errors: list[str] = []
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        for key, child in props.items():
            if key in value and isinstance(child, dict):
                errors.extend(schema_errors(value[key], child, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}.{key} is not allowed")
        return errors
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{path} must be an integer"]
        return _range_errors(float(value), schema, path)
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path} must be a number"]
        return _range_errors(float(value), schema, path)
    if kind == "string":
        if not isinstance(value, str):
            return [f"{path} must be a string"]
        return []
    if kind == "boolean":
        if not isinstance(value, bool):
            return [f"{path} must be a boolean"]
        return []
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        return []
    return []


def _range_errors(value: float, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if "minimum" in schema and value < float(schema["minimum"]):
        errors.append(f"{path} is below minimum {schema['minimum']}")
    if "maximum" in schema and value > float(schema["maximum"]):
        errors.append(f"{path} is above maximum {schema['maximum']}")
    return errors


def repair_to_schema(value: Any, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """Clamp and unwrap. Does not invent missing required fields."""

    notes: list[str] = []
    repaired = _repair(value, schema, "$", notes)
    return repaired, notes


def _repair(value: Any, schema: dict[str, Any], path: str, notes: list[str]) -> Any:
    if not schema:
        return value
    if "enum" in schema:
        if value in schema["enum"]:
            return value
        if isinstance(value, str):
            for option in schema["enum"]:
                if str(option).lower() == value.lower():
                    notes.append(f"{path} enum case folded to {option}")
                    return option
        return value
    kind = schema.get("type")
    if kind == "object" or (not kind and "properties" in schema):
        if not isinstance(value, dict):
            return value
        props = schema.get("properties") or {}
        rebuilt = dict(value)
        for key, child in props.items():
            if key in rebuilt and isinstance(child, dict):
                rebuilt[key] = _repair(rebuilt[key], child, f"{path}.{key}", notes)
        if schema.get("additionalProperties") is False:
            for key in list(rebuilt):
                if key not in props:
                    notes.append(f"dropped {path}.{key}")
                    del rebuilt[key]
        return rebuilt
    if kind in {"number", "integer"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        lo = schema.get("minimum")
        hi = schema.get("maximum")
        clamped = number
        if lo is not None and clamped < float(lo):
            clamped = float(lo)
        if hi is not None and clamped > float(hi):
            clamped = float(hi)
        if clamped != number:
            notes.append(f"clamped {path}")
        if kind == "integer":
            return int(round(clamped))
        return round(clamped, 6)
    return value


def interpret_mutator_output(
    text: str,
    schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return genome, rationale, and notes. Raise when no child satisfies the schema."""

    payload_schema = payload_schema_of(schema)
    candidates = extract_json_values(text)
    children: list[dict[str, Any]] = []
    for parsed in candidates:
        children.extend(children_from_parsed(parsed))
    if not children:
        raise ProviderError(
            "GeneAI Mutator output did not contain children[].genome",
            code="INVALID_PROPOSAL_JSON",
        )
    failures: list[str] = []
    for index, child in enumerate(children):
        genome = child["genome"]
        repaired, notes = repair_to_schema(genome, payload_schema)
        if not isinstance(repaired, dict):
            failures.append(f"child {index} genome was not an object")
            continue
        repaired = normalize_payload(repaired)
        errors = schema_errors(repaired, payload_schema)
        if errors:
            failures.append(f"child {index}: " + "; ".join(errors))
            continue
        return repaired, child["rationale"], {
            "repairs": notes,
            "child_count": len(children),
            "selected_index": index,
            "schema_valid": True,
        }
    raise ProviderError(
        "GeneAI Mutator children failed schema validation: " + " | ".join(failures[:4]),
        code="SCHEMA_INVALID",
    )


class CreaoProvider:
    """App-scoped GeneAI Mutator client. This is not the lineage graph."""

    name = "creao"
    mode = "live"

    def __init__(
        self,
        *,
        api_key: str,
        app_id: str,
        webhook_url: str | None = None,
        timeout_s: float = 180.0,
        poll_s: float = 2.0,
        max_http_attempts: int = 4,
        client: httpx.AsyncClient | None = None,
        sleeper: Any | None = None,
    ) -> None:
        assert_app_scoped_key(api_key)
        self.api_key = api_key
        self.app_id = assert_app_id(app_id)
        self.webhook_url = webhook_url
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.max_http_attempts = max_http_attempts
        self._client = client
        self._sleeper = sleeper

    @classmethod
    def from_env(cls) -> CreaoProvider:
        api_key = os.environ.get("CREAO_API_KEY", "").strip()
        app_id = os.environ.get("CREAO_APP_ID", "").strip() or EXAMPLE_APP_ID
        if not api_key:
            raise CredentialsError(
                "CREAO_API_KEY is missing. No request was sent. "
                f"CREAO_APP_ID defaults to {EXAMPLE_APP_ID} when unset."
            )
        timeout_s = float(os.environ.get("CREAO_TIMEOUT_SECONDS", "180"))
        poll_s = float(os.environ.get("CREAO_POLL_SECONDS", "2"))
        webhook = os.environ.get("CREAO_WEBHOOK_URL", "").strip() or None
        return cls(
            api_key=api_key,
            app_id=app_id,
            webhook_url=webhook,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )

    async def aclose(self) -> None:
        return None

    async def run_workflow(
        self,
        workflow_name: str,
        inputs: dict[str, Any],
        *,
        call_id: str,
    ) -> Proposal:
        started = time.perf_counter()
        created = await self._request(
            "POST",
            app_runs_collection(self.app_id),
            json_body=build_create_body(inputs, webhook_url=self.webhook_url),
        )
        run_id = run_id_from_create(created)
        document = await self._poll(run_id)
        status = str(document.get("status") or "")
        if status != "completed":
            raise ProviderError(
                f"CREAO run {run_id} ended with status {status or 'unknown'}",
                code=error_code_from(document),
                provider_run_id=run_id,
            )
        schema = inputs.get("response_schema") if isinstance(inputs.get("response_schema"), dict) else None
        genome, rationale, notes, answer_source, text = await self._read_answer(document, schema, run_id)
        result = document.get("result") if isinstance(document.get("result"), dict) else {}
        artifacts = result.get("artifacts") if isinstance(result, dict) else None
        local_ms = int((time.perf_counter() - started) * 1000)
        server_ms = server_interval_ms(document)
        latency_ms = server_ms if server_ms is not None else local_ms
        return Proposal(
            payload=genome,
            hypothesis=rationale or f"{workflow_name} via creao",
            provider=self.name,
            provider_run_id=run_id,
            model=None,
            latency_ms=latency_ms,
            mode="live",
            raw_text=text,
            credits=_credits(document),
            cost_usd=None,
            extra={
                "call_id": call_id,
                "path": "app-scoped-api-trigger",
                "agent": "GeneAI Mutator",
                "app_id": self.app_id,
                "app_name": document.get("appName") or created.get("appName"),
                "thread_id": document.get("threadId"),
                "started_at": document.get("startedAt"),
                "completed_at": document.get("completedAt"),
                "server_latency_ms": server_ms,
                "local_latency_ms": local_ms,
                "answer_source": answer_source,
                "debug_text": debug_result_text(document)[:2000],
                "status": status,
                "error_code": error_code_from(document),
                "artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
                "create_status": created.get("status"),
                "workflow_name": workflow_name,
                "wire_operator": mutator_wire_inputs(inputs)["operator"],
                "child_count": notes.get("child_count"),
                "selected_index": notes.get("selected_index"),
                "schema_repairs": notes.get("repairs") or [],
            },
        )

    async def _read_answer(
        self,
        document: dict[str, Any],
        schema: dict[str, Any] | None,
        run_id: str,
    ) -> tuple[dict[str, Any], str, dict[str, Any], str, str]:
        chat = chat_answer_text(document)
        chat_error: ProviderError | None = None
        if chat.strip():
            try:
                genome, rationale, notes = interpret_mutator_output(chat, schema)
                return genome, rationale, notes, "chatText", chat
            except ProviderError as exc:
                chat_error = exc
        artifact_text = await self._download_children_artifact(document)
        if artifact_text and artifact_text.strip():
            try:
                genome, rationale, notes = interpret_mutator_output(artifact_text, schema)
                return genome, rationale, notes, "artifact", artifact_text
            except ProviderError as exc:
                chat_error = chat_error or exc
        if chat_error is not None:
            chat_error.provider_run_id = run_id
            raise chat_error
        raise ProviderError(
            f"CREAO run {run_id} completed without result.chatText or a children.json artifact. "
            "result.text was not parsed.",
            code="EMPTY_RESULT",
            provider_run_id=run_id,
        )

    async def _download_children_artifact(self, document: dict[str, Any]) -> str | None:
        artifact = children_artifact(document)
        if artifact is None:
            return None
        url = artifact.get("url")
        if not isinstance(url, str) or not url.startswith("https://agent.creao.ai/"):
            return None
        async with self._open_client() as client:
            response = await client.get(url, headers=self._headers())
        if response.status_code >= 400:
            return None
        return response.text

    async def list_runs(self) -> dict[str, Any]:
        """GET the collection. Live shape: ``{"runs": [], "limit": 20, "offset": 0}``."""

        url = app_runs_collection(self.app_id)
        async with self._open_client() as client:
            response = await client.get(url, headers=self._headers())
        try:
            parsed = response.json()
        except json.JSONDecodeError:
            parsed = None
        summary = summarize_run_list(parsed)
        summary["http_status"] = response.status_code
        summary["body"] = parsed
        return summary

    async def _poll(self, run_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        url = app_run_status(self.app_id, run_id)
        while True:
            document = await self._request("GET", url)
            status = str(document.get("status") or "")
            if status in TERMINAL_STATUSES:
                return document
            if time.monotonic() >= deadline:
                raise ProviderError(
                    f"CREAO run {run_id} did not reach a terminal status within {self.timeout_s:.0f}s",
                    code="TIMEOUT",
                    retryable=True,
                    provider_run_id=run_id,
                )
            await self._sleep(self.poll_s)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cli_path = "/v1/runs" in url and "/api/v1/apps/" not in url
        if "developer.creao.ai" in url or cli_path:
            raise ProviderError("refusing to call the CREAO CLI host")
        last_error: ProviderError | None = None
        for attempt in range(self.max_http_attempts):
            async with self._open_client() as client:
                response = await client.request(method, url, headers=self._headers(), json=json_body)
            if response.status_code in {429, 503} and attempt + 1 < self.max_http_attempts:
                await self._sleep(_retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                code = _http_error_code(response)
                raise ProviderError(
                    f"CREAO {method} {response.status_code} {code or ''}".strip(),
                    retryable=response.status_code in {429, 503},
                    status_code=response.status_code,
                    code=code,
                )
            try:
                document = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"CREAO {method} returned non-JSON",
                    status_code=response.status_code,
                ) from exc
            if not isinstance(document, dict):
                raise ProviderError(f"CREAO {method} returned a {type(document).__name__}, expected an object")
            return document
        raise last_error or ProviderError("CREAO request failed")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _open_client(self) -> _ClientContext:
        return _ClientContext(self._client)

    async def _sleep(self, seconds: float) -> None:
        if self._sleeper is not None:
            await self._sleeper(seconds)
            return
        import asyncio

        await asyncio.sleep(seconds)


class _ClientContext:
    def __init__(self, client: httpx.AsyncClient | None) -> None:
        self._client = client
        self._owned: httpx.AsyncClient | None = None

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        self._owned = httpx.AsyncClient(timeout=60.0)
        return self._owned

    async def __aexit__(self, *exc: object) -> None:
        if self._owned is not None:
            await self._owned.aclose()


def summarize_run_list(parsed: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "top_level_type": type(parsed).__name__,
        "top_level_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
        "count": None,
        "limit": None,
        "offset": None,
    }
    if isinstance(parsed, dict):
        runs = parsed.get("runs")
        if isinstance(runs, list):
            summary["count"] = len(runs)
        if isinstance(parsed.get("limit"), int):
            summary["limit"] = parsed["limit"]
        if isinstance(parsed.get("offset"), int):
            summary["offset"] = parsed["offset"]
    elif isinstance(parsed, list):
        summary["count"] = len(parsed)
    return summary


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), 20.0)
        except ValueError:
            pass
    return float(min(2**attempt, 8))


def _http_error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, dict) and error.get("code"):
            return str(error["code"])
        if isinstance(body.get("errorCode"), str):
            return str(body["errorCode"])
    return None


def _credits(document: dict[str, Any]) -> float | None:
    usage = document.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("credits"), (int, float)):
        return float(usage["credits"])
    if isinstance(document.get("credits"), (int, float)):
        return float(document["credits"])
    return None


def creao_blocker_message() -> str:
    return "\n".join(
        [
            "BLOCKED: CREAO smoke was not executed. No sponsor call was made.",
            "This account uses the app-scoped API Trigger only (capi_ key).",
            "Missing environment variable: CREAO_API_KEY.",
            f"CREAO_APP_ID defaults to {EXAMPLE_APP_ID} when unset.",
            "Setup:",
            "1. On the GeneAI Mutator agent page in agent.creao.ai, create an app-scoped key (prefix capi_).",
            "   Do not use a cr_sk_ Developer Console key. developer.creao.ai/v1 rejects capi_ keys.",
            "2. Export CREAO_API_KEY=capi_...",
            f"3. Optional: export CREAO_APP_ID={EXAMPLE_APP_ID}",
            "4. Re-run: uv run python scripts/smoke_creao.py",
            "Mutator contract sent as inputs (not CLI input/agent_id):",
            '  {"task": "...", "parents": [{"genome": {...}, "fitness": 0.0}], "operator": "mutate"|"crossover"}',
            "Answer is result.chatText (fenced JSON). result.text echoes inputs and is not parsed.",
            "Optional fallback: result.artifacts children.json.",
            "Expected chatText JSON:",
            '  {"children": [{"genome": {...}, "rationale": "..."}]}',
            "Verified HTTP surface:",
            f"  POST https://agent.creao.ai/api/v1/apps/{EXAMPLE_APP_ID}/runs",
            f"  GET  https://agent.creao.ai/api/v1/apps/{EXAMPLE_APP_ID}/runs/{{runId}}",
            f"  GET  https://agent.creao.ai/api/v1/apps/{EXAMPLE_APP_ID}/runs",
            'Live list shape observed outside this environment: {"runs": [], "limit": 20, "offset": 0}.',
            "Terminal poll statuses: completed, failed, cancelled.",
        ]
    )
