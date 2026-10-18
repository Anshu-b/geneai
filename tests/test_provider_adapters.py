from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from geneai_core.providers.base import CredentialsError, ProviderError
from geneai_core.providers.creao import (
    EXAMPLE_APP_ID,
    CreaoProvider,
    build_create_body,
    debug_result_text,
    interpret_mutator_output,
    parse_completed_run,
    summarize_run_list,
)
from geneai_core.providers.creao_cli_unverified import UnverifiedCreaoCliProvider
from geneai_core.providers.frontier import FrontierProvider
from geneai_core.providers.mock import MockModelProvider
from geneai_core.providers.openai_compat import reject_chat_completions_base
from geneai_core.providers.veri import (
    EXAMPLE_BASE_URL,
    EXAMPLE_DEPLOYMENT_ID,
    EXAMPLE_MODEL,
    VERI_CONCURRENCY_TARGET,
    veri_deployment_base_url,
)
from geneai_core.schemas import EvolutionConfig


TOY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["family", "params", "description"],
    "properties": {
        "family": {"type": "string", "enum": ["weighted_policy", "offset_policy", "scaled_policy"]},
        "params": {
            "type": "object",
            "additionalProperties": False,
            "required": ["x", "y"],
            "properties": {
                "x": {"type": "number", "minimum": -5.12, "maximum": 5.12},
                "y": {"type": "number", "minimum": -5.12, "maximum": 5.12},
            },
        },
        "description": {"type": "string"},
    },
}


def test_veri_base_url_matches_docs() -> None:
    assert veri_deployment_base_url(EXAMPLE_DEPLOYMENT_ID) == EXAMPLE_BASE_URL
    assert EXAMPLE_DEPLOYMENT_ID == "bfe84d51e141"
    assert EXAMPLE_MODEL == "geneai-qwen7b"
    assert VERI_CONCURRENCY_TARGET == 8
    assert EvolutionConfig().max_concurrent_model_calls == VERI_CONCURRENCY_TARGET
    assert veri_deployment_base_url("dep_123") == "https://api.veri.studio/v1/deployments/dep_123"
    with pytest.raises(ValueError):
        veri_deployment_base_url("dep/chat/completions")
    with pytest.raises(ValueError):
        reject_chat_completions_base("https://example.test/v1/chat/completions")


def test_frontier_rejects_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    with pytest.raises(CredentialsError):
        FrontierProvider.from_env()


def test_cli_provider_is_disabled() -> None:
    with pytest.raises(CredentialsError, match="unverified"):
        UnverifiedCreaoCliProvider(api_key="cr_sk_test", agent_id="00000000-0000-4000-8000-000000000000")


def test_creao_wire_body_is_the_mutator_contract() -> None:
    body = build_create_body(
        {
            "task": "nudge x toward the basin",
            "parents": [{"genome": {"family": "weighted_policy", "params": {"x": 0.2, "y": -0.4}}, "fitness": 0.3}],
            "operator": "mutation",
            "response_schema": TOY_SCHEMA,
            "mutation_goal": "not sent",
        }
    )
    assert set(body) == {"inputs"}
    assert set(body["inputs"]) == {"task", "parents", "operator"}
    assert body["inputs"]["operator"] == "mutate"
    assert body["inputs"]["parents"][0]["fitness"] == 0.3
    assert "agent_id" not in body
    assert "input" not in body
    crossover = build_create_body(
        {
            "task": "combine",
            "parents": [
                {"genome": {"a": 1}, "fitness": 0.1},
                {"payload": {"a": 2}, "fitness": 0.2},
            ],
            "operator": "crossover",
        }
    )
    assert crossover["inputs"]["operator"] == "crossover"
    assert crossover["inputs"]["parents"][1]["genome"] == {"a": 2}
    with pytest.raises(ProviderError):
        build_create_body({"task": "seed", "parents": [], "operator": "mutate"})


def test_mutator_output_extraction_and_schema_repair() -> None:
    text = """Sure, here is the child:
```json
{"children":[{"genome":{"family":"Weighted_Policy","params":{"x":9,"y":-0.2,"extra":1},"description":"closer"},"rationale":"clamped"}]}
```
"""
    genome, rationale, notes = interpret_mutator_output(text, TOY_SCHEMA)
    assert rationale == "clamped"
    assert genome["family"] == "weighted_policy"
    assert genome["params"]["x"] == 5.12
    assert "extra" not in genome["params"]
    assert notes["repairs"]
    prose = 'prefix {"children":[{"genome":{"ok":true,"source":"creao"},"rationale":"smoke"}]} tail'
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "source"],
        "properties": {"ok": {"type": "boolean"}, "source": {"type": "string"}},
    }
    genome, rationale, _notes = interpret_mutator_output(prose, schema)
    assert genome == {"ok": True, "source": "creao"}
    assert rationale == "smoke"
    with pytest.raises(ProviderError):
        interpret_mutator_output('{"children":[{"genome":{},"rationale":"missing"}]}', schema)


def test_list_shape_matches_live_observation() -> None:
    observed = {"runs": [], "limit": 20, "offset": 0}
    summary = summarize_run_list(observed)
    assert summary["count"] == 0
    assert summary["limit"] == 20
    assert summary["offset"] == 0
    assert EXAMPLE_APP_ID == "0ed64fee-5003-4d56-a83b-c5f369d7a39c"


@pytest.mark.asyncio
async def test_creao_http_contract_uses_app_scoped_routes() -> None:
    seen: list[httpx.Request] = []
    app_id = EXAMPLE_APP_ID

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer capi_test_key"
        assert "developer.creao.ai" not in str(request.url)
        if request.method == "POST":
            assert str(request.url) == f"https://agent.creao.ai/api/v1/apps/{app_id}/runs"
            body = json.loads(request.content.decode())
            assert "inputs" in body and "input" not in body and "agent_id" not in body
            assert body["inputs"]["operator"] == "mutate"
            return httpx.Response(
                200,
                json={
                    "id": "11111111-1111-4111-8111-111111111111",
                    "status": "pending",
                    "appId": app_id,
                    "appName": "GeneAI Mutator",
                },
            )
        assert str(request.url).endswith("/runs/11111111-1111-4111-8111-111111111111")
        return httpx.Response(
            200,
            json={
                "id": "11111111-1111-4111-8111-111111111111",
                "status": "completed",
                "appId": app_id,
                "appName": "GeneAI Mutator",
                "threadId": "33333333-3333-4333-8333-333333333333",
                "result": {
                    "text": json.dumps(
                        {
                            "input": "{\"task\":\"nudge\"}",
                            "task": "nudge",
                            "operator": "mutate",
                        }
                    ),
                    "chatText": (
                        "```json\n"
                        + json.dumps(
                            {
                                "children": [
                                    {
                                        "genome": {
                                            "family": "weighted_policy",
                                            "params": {"x": 0.1, "y": -0.2},
                                            "description": "from creao",
                                        },
                                        "rationale": "moved x",
                                    }
                                ]
                            }
                        )
                        + "\n```\n\nMoved x."
                    ),
                    "artifacts": [],
                },
                "startedAt": "2026-10-18T03:38:24.878Z",
                "completedAt": "2026-10-18T03:39:08.616Z",
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = CreaoProvider(api_key="capi_test_key", app_id=app_id, client=client, poll_s=0)
        proposal = await provider.run_workflow(
            "propose_mutation",
            {
                "task": "nudge",
                "parents": [{"genome": {"family": "weighted_policy", "params": {"x": 0, "y": 0}, "description": "p"}, "fitness": 0.1}],
                "operator": "mutate",
                "response_schema": TOY_SCHEMA,
            },
            call_id="t1",
        )
    assert proposal.provider_run_id == "11111111-1111-4111-8111-111111111111"
    assert proposal.payload["description"] == "from creao"
    assert proposal.hypothesis == "moved x"
    assert proposal.mode == "live"
    assert proposal.extra["answer_source"] == "chatText"
    assert proposal.extra["thread_id"] == "33333333-3333-4333-8333-333333333333"
    assert proposal.extra["server_latency_ms"] == 43738
    assert "input" in proposal.extra["debug_text"]
    assert seen[0].method == "POST"
    assert seen[1].method == "GET"
    assert len(seen) == 2


def test_live_creao_sample_reads_chat_text_not_the_input_echo() -> None:
    document = json.loads(
        (Path(__file__).parent / "fixtures" / "creao_run_0897c113.json").read_text(encoding="utf-8")
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "source"],
        "properties": {"ok": {"type": "boolean"}, "source": {"type": "string"}},
    }
    parsed = parse_completed_run(document, schema)
    assert parsed["genome"] == {"ok": True, "source": "creao"}
    assert parsed["rationale"] == "smoke"
    assert parsed["answer_source"] == "chatText"
    assert parsed["run_id"] == "0897c113-58b0-43bf-8a7c-e56f705efa24"
    assert parsed["thread_id"] == "42424caf-56fd-4069-8fec-e71e7f281ca0"
    assert parsed["started_at"] == "2026-10-18T03:38:24.878Z"
    assert parsed["completed_at"] == "2026-10-18T03:39:08.616Z"
    assert parsed["server_latency_ms"] == 43738
    assert "input" in parsed["debug_text"]
    with pytest.raises(ProviderError):
        interpret_mutator_output(debug_result_text(document), schema)


@pytest.mark.asyncio
async def test_creao_falls_back_to_children_json_artifact() -> None:
    artifact_url = "https://agent.creao.ai/api/v1/runs/run-art/artifacts/download?artifactId=children.json"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer capi_test_key"
        if request.method == "POST":
            return httpx.Response(200, json={"id": "run-art", "status": "pending"})
        if request.url.path.endswith("/runs/run-art"):
            return httpx.Response(
                200,
                json={
                    "id": "run-art",
                    "status": "completed",
                    "threadId": "thread-art",
                    "startedAt": "2026-10-18T03:38:24.878Z",
                    "completedAt": "2026-10-18T03:39:08.616Z",
                    "result": {
                        "text": "{\"input\":\"echo\"}",
                        "chatText": "The file is attached.",
                        "artifacts": [
                            {
                                "id": "files/geneai_children.json",
                                "url": artifact_url,
                                "title": "GeneAI Mutator — children.json",
                            }
                        ],
                    },
                },
            )
        assert str(request.url) == artifact_url
        return httpx.Response(
            200,
            json={"children": [{"genome": {"ok": True, "source": "artifact"}, "rationale": "file"}]},
        )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "source"],
        "properties": {"ok": {"type": "boolean"}, "source": {"type": "string"}},
    }
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = CreaoProvider(api_key="capi_test_key", app_id=EXAMPLE_APP_ID, client=client, poll_s=0)
        proposal = await provider.run_workflow(
            "propose_mutation",
            {
                "task": "nudge",
                "parents": [{"genome": {"ok": False, "source": "seed"}, "fitness": 0.0}],
                "operator": "mutate",
                "response_schema": schema,
            },
            call_id="art",
        )
    assert proposal.payload == {"ok": True, "source": "artifact"}
    assert proposal.hypothesis == "file"
    assert proposal.extra["answer_source"] == "artifact"
    assert proposal.extra["thread_id"] == "thread-art"
    assert proposal.latency_ms == 43738
    assert artifact_url in seen


def test_creao_from_env_defaults_the_public_app_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREAO_API_KEY", "capi_test_key")
    monkeypatch.delenv("CREAO_APP_ID", raising=False)
    provider = CreaoProvider.from_env()
    assert provider.app_id == EXAMPLE_APP_ID


def test_creao_rejects_non_capi_key_without_a_client_call() -> None:
    with pytest.raises(CredentialsError):
        CreaoProvider(api_key="cr_sk_not_for_this_account", app_id=EXAMPLE_APP_ID)


@pytest.mark.asyncio
async def test_mock_provider_is_deterministic_and_labeled() -> None:
    provider = MockModelProvider()
    schema = {"type": "object", "properties": {"payload": TOY_SCHEMA, "hypothesis": {"type": "string"}}}
    instruction = 'GENEAI_CONTEXT={"operation":"seed","parents":[],"slot":0,"token":99}'
    first = await provider.propose(instruction, schema, call_id="a")
    second = await provider.propose(instruction, schema, call_id="b")
    assert first.payload == second.payload
    assert first.mode == "mock"
    assert first.provider == "mock"


@pytest.mark.asyncio
async def test_veri_provider_parses_a_fake_completion() -> None:
    from geneai_core.providers.veri import VeriProvider

    class Completions:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content='{"payload":{"ok":true},"hypothesis":"checked"}')
            choice = SimpleNamespace(message=message, finish_reason="stop")
            usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
            return SimpleNamespace(id="chatcmpl-test", model="served-model", choices=[choice], usage=usage, model_extra={})

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), close=None)
    provider = VeriProvider(api_key="vk_test", deployment_id="dep_123", model="served-model", client=client)
    proposal = await provider.propose("say json", {"type": "object"}, call_id="s")
    assert provider.base_url == "https://api.veri.studio/v1/deployments/dep_123"
    assert completions.kwargs["model"] == "served-model"
    assert proposal.payload == {"ok": True}
    assert proposal.provider_run_id == "chatcmpl-test"
    assert proposal.prompt_tokens == 3
    assert proposal.mode == "live"


def test_veri_from_env_defaults_the_public_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    from geneai_core.providers.veri import VeriProvider

    monkeypatch.setenv("VERI_API_KEY", "vk_test")
    monkeypatch.delenv("VERI_DEPLOYMENT_ID", raising=False)
    monkeypatch.delenv("VERI_MODEL", raising=False)
    provider = VeriProvider.from_env()
    assert provider.deployment_id == EXAMPLE_DEPLOYMENT_ID
    assert provider.model == EXAMPLE_MODEL
    assert provider.base_url == EXAMPLE_BASE_URL
    assert provider.in_flight_limit == VERI_CONCURRENCY_TARGET


def _completion(content: str, completion_id: str = "chatcmpl-wake") -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
    return SimpleNamespace(id=completion_id, model=EXAMPLE_MODEL, choices=[choice], usage=usage, model_extra={})


class _StatusError(Exception):
    def __init__(self, status_code: int, code: str, retry_after: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.body = {"error": {"code": code, "retry_after_seconds": int(retry_after)}}
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


@pytest.mark.asyncio
async def test_veri_retries_503_deployment_waking_then_succeeds() -> None:
    from geneai_core.providers.veri import VeriProvider

    calls = {"n": 0}
    delays: list[float] = []

    class Completions:
        async def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _StatusError(503, "deployment_waking", "30")
            return _completion('{"payload":{"ok":true},"hypothesis":"awake"}')

    async def sleeper(delay: float) -> None:
        delays.append(delay)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()), close=None)
    provider = VeriProvider(
        api_key="vk_test",
        deployment_id=EXAMPLE_DEPLOYMENT_ID,
        model=EXAMPLE_MODEL,
        client=client,
        sleeper=sleeper,
        wake_timeout_s=180,
    )
    proposal = await provider.propose("say json", {"type": "object"}, call_id="wake")
    assert calls["n"] == 3
    assert delays == [30.0, 30.0]
    assert proposal.payload == {"ok": True}
    assert proposal.hypothesis == "awake"


@pytest.mark.asyncio
async def test_veri_stops_retrying_when_the_wake_budget_is_spent() -> None:
    from geneai_core.providers.veri import VeriProvider

    class Completions:
        async def create(self, **kwargs):
            raise _StatusError(503, "deployment_waking", "30")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()), close=None)
    provider = VeriProvider(
        api_key="vk_test",
        deployment_id=EXAMPLE_DEPLOYMENT_ID,
        model=EXAMPLE_MODEL,
        client=client,
        sleeper=lambda _delay: _async_none(),
        wake_timeout_s=30,
    )
    with pytest.raises(ProviderError) as caught:
        await provider.propose("say json", {"type": "object"}, call_id="cold")
    assert caught.value.retryable is True
    assert caught.value.status_code == 503
    assert caught.value.code == "deployment_waking"


async def _async_none() -> None:
    return None


@pytest.mark.asyncio
async def test_veri_caps_in_flight_calls_at_the_deployment_target() -> None:
    from geneai_core.providers.veri import VeriProvider

    state = {"current": 0, "max_seen": 0}
    release = asyncio.Event()
    entered = asyncio.Event()

    class Completions:
        async def create(self, **kwargs):
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
            if state["current"] >= VERI_CONCURRENCY_TARGET:
                entered.set()
            await release.wait()
            state["current"] -= 1
            return _completion('{"payload":{"ok":true},"hypothesis":"slot"}')

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()), close=None)
    provider = VeriProvider(
        api_key="vk_test",
        deployment_id=EXAMPLE_DEPLOYMENT_ID,
        model=EXAMPLE_MODEL,
        client=client,
    )
    tasks = [
        asyncio.create_task(provider.propose("say json", {"type": "object"}, call_id=f"c{i}"))
        for i in range(VERI_CONCURRENCY_TARGET + 4)
    ]
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert state["current"] == VERI_CONCURRENCY_TARGET
    release.set()
    proposals = await asyncio.gather(*tasks)
    assert state["max_seen"] == VERI_CONCURRENCY_TARGET
    assert len(proposals) == VERI_CONCURRENCY_TARGET + 4
