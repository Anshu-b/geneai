from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from geneai_core.providers.base import CredentialsError, ProviderError
from geneai_core.providers.creao import (
    EXAMPLE_APP_ID,
    CreaoProvider,
    build_create_body,
    interpret_mutator_output,
    summarize_run_list,
)
from geneai_core.providers.creao_cli_unverified import UnverifiedCreaoCliProvider
from geneai_core.providers.frontier import FrontierProvider
from geneai_core.providers.mock import MockModelProvider
from geneai_core.providers.openai_compat import reject_chat_completions_base
from geneai_core.providers.veri import veri_deployment_base_url


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
                    ),
                    "artifacts": [],
                },
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
    assert seen[0].method == "POST"
    assert seen[1].method == "GET"


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
