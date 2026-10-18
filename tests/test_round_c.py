"""Live re-verification fixes at 25f98cd."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from geneai_core.budgets import Budget
from geneai_core.controller import EvolutionController
from geneai_core.domains.cloud import (
    CLOUD_SEARCH_CEILING,
    CloudSchedulingAdapter,
    baseline_payload,
    cloud_objective,
    normalize_cloud_payload,
)
from geneai_core.domains.shopping import ShoppingAdapter, _hand_baselines
from geneai_core.domains.toy import ToyOptimizationAdapter
from geneai_core.experiments import toy_objective
from geneai_core.providers.base import Proposal, ProviderError, parse_json_object
from geneai_core.providers.browser_use import BrowserUseClient
from geneai_core.providers.creao import CreaoProvider, interpret_mutator_output
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.providers.openai_compat import repair_token_cap
from geneai_core.providers.veri import EXAMPLE_DEPLOYMENT_ID, EXAMPLE_MODEL, VeriProvider
from geneai_core.schemas import EvolutionConfig, Genome, ObjectiveSpec
from geneai_core.store import RunRepository

ARTIFACT = Path(__file__).parent / "fixtures" / "creao_artifact_a605.json"


def test_creao_artifact_payload_list_parses() -> None:
    text = ARTIFACT.read_text(encoding="utf-8")
    genome, rationale, notes = interpret_mutator_output(text)
    assert notes["child_count"] == 5
    assert notes["selected_index"] == 0
    assert genome["family"] == "neutral_layers"
    assert genome["items"][0]["product_id"] == "bu_003"
    assert "crossover" in rationale
    wrapped = interpret_mutator_output(
        json.dumps({"children": [{"genome": {"ok": True, "source": "creao"}, "rationale": "child"}]})
    )
    assert wrapped[0] == {"ok": True, "source": "creao"}
    bare = interpret_mutator_output(json.dumps({"family": "neutral_layers", "items": [], "styling_notes": "one"}))
    assert bare[0]["family"] == "neutral_layers"
    listed = interpret_mutator_output(
        json.dumps([{"policy_family": "composed", "rules": {}, "description": "a", "rationale": "row"}])
    )
    assert listed[0]["policy_family"] == "composed"
    assert "row" in listed[1]


def test_json_extractor_strips_fences_and_takes_the_first_object() -> None:
    text = 'Sure.\n```json\n{"payload": {"ok": true}, "hypothesis": "kept"}\n```\ntrailing {not json'
    parsed = parse_json_object(text)
    assert parsed["payload"] == {"ok": True}
    assert parsed["hypothesis"] == "kept"


@pytest.mark.asyncio
async def test_creao_artifact_download_follows_redirects_and_rejects_non_2xx() -> None:
    body = json.dumps({"children": [{"genome": {"ok": True, "source": "creao"}, "rationale": "via s3"}]})
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url).endswith("/runs"):
            return httpx.Response(200, json={"id": "11111111-1111-4111-8111-111111111111", "status": "pending"})
        if str(request.url).endswith("/artifact"):
            return httpx.Response(
                302,
                headers={"Location": "https://agent.creao.ai/api/v1/apps/x/final-artifact"},
            )
        if str(request.url).endswith("/final-artifact"):
            return httpx.Response(200, text=body)
        return httpx.Response(
            200,
            json={
                "id": "11111111-1111-4111-8111-111111111111",
                "status": "completed",
                "result": {
                    "artifacts": [
                        {
                            "id": "geneai_mutate_result.json",
                            "type": "application/json",
                            "url": "https://agent.creao.ai/api/v1/apps/x/artifact",
                        }
                    ]
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = CreaoProvider(api_key="capi_test", app_id="0ed64fee-5003-4d56-a83b-c5f369d7a39c", client=client, poll_s=0)
        proposal = await provider.run_workflow(
            "propose_mutation",
            {
                "task": "nudge",
                "parents": [{"genome": {"ok": True, "source": "creao"}, "fitness": 0.1}],
                "operator": "mutate",
                "response_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ok", "source"],
                    "properties": {"ok": {"type": "boolean"}, "source": {"type": "string"}},
                },
            },
            call_id="redirect",
        )
    assert proposal.payload == {"ok": True, "source": "creao"}
    assert proposal.extra["answer_source"] == "artifact"
    assert any(url.endswith("/final-artifact") for url in seen)

    def failing(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "22222222-2222-4222-8222-222222222222", "status": "pending"})
        if str(request.url).endswith("/artifact"):
            return httpx.Response(302, headers={"Location": "https://agent.creao.ai/api/v1/apps/x/missing"})
        if str(request.url).endswith("/missing"):
            return httpx.Response(403, text="denied")
        return httpx.Response(
            200,
            json={
                "id": "22222222-2222-4222-8222-222222222222",
                "status": "completed",
                "result": {
                    "artifacts": [
                        {
                            "id": "geneai_mutate_result.json",
                            "type": "application/json",
                            "url": "https://agent.creao.ai/api/v1/apps/x/artifact",
                        }
                    ]
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        provider = CreaoProvider(api_key="capi_test", app_id="0ed64fee-5003-4d56-a83b-c5f369d7a39c", client=client, poll_s=0)
        with pytest.raises(ProviderError, match="application/json"):
            await provider.run_workflow(
                "propose_mutation",
                {
                    "task": "nudge",
                    "parents": [{"genome": {"ok": True}, "fitness": 0.1}],
                    "operator": "mutate",
                },
                call_id="denied",
            )


def test_cloud_normalizer_repairs_the_live_reject_reasons() -> None:
    adapter = CloudSchedulingAdapter(seed=42)
    objective = ObjectiveSpec(domain=adapter.name, prompt="schedule", evaluation_version=adapter.evaluator_version)
    broken = {
        "policy_family": "composed",
        "rules": {
            "score_weights": {"short_job": 1.4, "gpu": 0.2, "not_a_term": 3},
            "tie_break": ["gpu", "short_job", "duration", "duration", "arrival", "job_id", "free_cpu", "deadline"],
        },
        "description": "model draft",
    }
    normalized, notes = normalize_cloud_payload(broken)
    joined = " ".join(notes)
    assert "class_overrides defaulted" in joined
    assert "max_queue_depth defaulted" in joined
    assert "dropped unknown tie_break key gpu" in joined
    assert "dropped unknown tie_break key short_job" in joined
    assert "truncated tie_break" in joined
    assert "clamped score_weights.short_job" in joined
    adapter.validate(normalized, objective)
    assert normalized["rules"]["class_overrides"] == []
    assert 1 <= normalized["rules"]["max_queue_depth"] <= 36
    assert 3 <= len(normalized["rules"]["tie_break"]) <= 4
    seed = adapter.seed_instructions(cloud_objective())
    assert "Allowed tie_break keys only" in seed
    assert "duration" in seed and "machine_id" in seed
    assert "class_overrides is required" in seed
    assert "up to two" not in seed
    assert "Example genome" in seed
    mutate = adapter.mutate_instructions(
        Genome(
            id="g",
            run_id="r",
            generation=0,
            operator="seed",
            payload=baseline_payload("deadline_weighted"),
            hypothesis="h",
            provider="baseline",
            created_at="",
            content_hash="c",
        ),
        adapter.evaluate(
            Genome(
                id="g",
                run_id="r",
                generation=0,
                operator="seed",
                payload=baseline_payload("deadline_weighted"),
                hypothesis="h",
                provider="baseline",
                created_at="",
                content_hash="c",
            ),
            "train",
        ),
    )
    assert '"children"' in mutate
    assert "genome" in mutate


@pytest.mark.asyncio
async def test_cloud_normalizations_are_on_the_event(tmp_path: Path) -> None:
    class Messy(MockModelProvider):
        async def propose(self, instruction, schema, *, call_id):
            proposal = await super().propose(instruction, schema, call_id=call_id)
            proposal.payload = {
                "policy_family": "weighted_queue",
                "rules": {"score_weights": {"short_job": 0.4}, "tie_break": ["gpu", "arrival"]},
                "description": "messy veri seed",
            }
            proposal.extra = {}
            return proposal

    repository = RunRepository(tmp_path)
    await EvolutionController().evolve(
        objective=cloud_objective(),
        adapter=CloudSchedulingAdapter(seed=42),
        config=EvolutionConfig(
            seed=42,
            population_size=6,
            generations=1,
            elite_count=1,
            strategist_calls_per_generation=0,
            max_model_calls=6,
            max_evaluations=80,
            max_wall_seconds=120,
        ),
        providers={"model": Messy(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    recorded = [
        event
        for event in repository.events
        if event.type in {"genome.seeded", "genome.mutated"} and event.payload.get("normalizations")
    ]
    assert recorded
    blob = " ".join(note for event in recorded for note in event.payload["normalizations"])
    assert "class_overrides" in blob
    assert "tie_break" in blob


@pytest.mark.asyncio
async def test_repair_budget_is_larger_and_json_object_is_requested() -> None:
    seen: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs):
            seen["kwargs"] = kwargs
            message = type("M", (), {"content": '{"payload":{"policy_family":"composed"},"hypothesis":"fixed"}'})()
            choice = type("C", (), {"message": message, "finish_reason": "stop"})()
            usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})()
            return type("R", (), {"id": "chat", "model": "served", "choices": [choice], "usage": usage, "model_extra": {}})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})(), "close": None})()
    provider = VeriProvider(
        api_key="vk_test",
        deployment_id=EXAMPLE_DEPLOYMENT_ID,
        model=EXAMPLE_MODEL,
        client=client,
    )
    instruction = 'Repair it\nGENEAI_CONTEXT={"operation":"repair","batch_size":1}'
    proposal = await provider.propose(instruction, {"type": "object"}, call_id="repair")
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["max_tokens"] == repair_token_cap() == 1400
    assert kwargs["response_format"] == {"type": "json_object"}
    assert proposal.payload["policy_family"] == "composed"


def test_shopping_role_is_corrected_and_missing_bottoms_still_seed() -> None:
    adapter = ShoppingAdapter()
    payload = {
        "family": "neutral_layers",
        "items": [
            {"product_id": "fx_top_01", "color_id": "black", "size": "M", "role": "shoes"},
            {"product_id": "fx_bottom_01", "color_id": "sage", "size": "M", "role": "top"},
            {"product_id": "fx_shoes_01", "color_id": "black", "size": "9", "role": "shoes"},
        ],
        "styling_notes": "roles were wrong",
    }
    objective = ObjectiveSpec(domain="shopping", prompt="outfit", constraints={"size": "M"}, evaluation_version="t")
    prepared, notes = adapter.prepare_payload(payload)
    assert any("corrected to top" in note for note in notes)
    adapter.validate(prepared, objective)
    assert prepared["items"][0]["role"] == "top"
    assert prepared["items"][1]["role"] == "bottom"
    catalog = {
        "catalog_id": "partial",
        "source": "recorded",
        "products": [
            {"product_id": "p_top", "role": "top", "colors": ["black"], "sizes": ["M"]},
            {"product_id": "p_shoes", "role": "shoes", "colors": ["black"], "sizes": ["M"]},
            {"product_id": "p_acc", "role": "accessory", "colors": ["black"], "sizes": ["M"]},
        ],
    }
    outfits = _hand_baselines(catalog)
    assert outfits
    roles = {item["role"] for _name, payload in outfits for item in payload["items"]}
    assert "bottom" not in roles
    assert len(outfits[0][1]["items"]) >= 2
    partial = ShoppingAdapter(catalog)
    for _name, body in outfits:
        partial.validate(body, objective)


@pytest.mark.asyncio
async def test_browser_task_covers_categories_and_stops_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
    monkeypatch.delenv("BROWSER_USE_TIMEOUT_SECONDS", raising=False)
    client = BrowserUseClient.from_env()
    assert client.timeout_s == 420
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.method == "POST" and str(request.url).endswith("/tasks"):
            body = json.loads(request.content)
            calls.append(("task", str(body["task"])))
            return httpx.Response(200, json={"id": "task_slow", "sessionId": "sess"})
        if request.method == "PATCH":
            body = json.loads(request.content)
            assert body == {"action": "stop"}
            return httpx.Response(500, json={"error": "stop failed"})
        if str(request.url).endswith("/status"):
            return httpx.Response(200, json={"id": "task_slow", "status": "running"})
        return httpx.Response(200, json={"id": "task_slow", "status": "running"})

    slow = BrowserUseClient(
        api_key="bu_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timeout_s=0,
        poll_s=0,
    )
    with pytest.raises(ProviderError, match="did not finish") as caught:
        await slow.collect_catalog(catalog_url="https://shop.example/c", query="tees")
    assert caught.value.code == "TIMEOUT"
    task = next(text for method, text in calls if method == "task")
    assert "tops" in task and "bottoms" in task and "shoes" in task and "accessories" in task
    assert "department" not in task.lower()
    assert "Stay on this site" in task
    assert "product page" in task
    assert any(method == "PATCH" and url.rstrip("/").endswith("/tasks/task_slow") for method, url in calls)


@pytest.mark.asyncio
async def test_slow_strategist_falls_back_and_finishes_four_generations(tmp_path: Path) -> None:
    class SlowCreao:
        name = "creao"
        mode = "live"
        timeout_s = 0.2
        calls = 0

        async def run_workflow(self, workflow_name, inputs, *, call_id):
            del workflow_name, inputs, call_id
            self.calls += 1
            await asyncio.sleep(30)
            raise AssertionError("strategist call should have timed out")

        async def aclose(self):
            return None

    workflow = SlowCreao()
    started = time.perf_counter()
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=1),
        config=EvolutionConfig(
            seed=1,
            population_size=4,
            generations=4,
            elite_count=1,
            strategist_calls_per_generation=1,
            max_concurrent_model_calls=4,
            max_model_calls=80,
            max_evaluations=80,
            max_wall_seconds=30,
        ),
        providers={"model": MockModelProvider(), "workflow": workflow},
        mode="mock",
        repository=RunRepository(tmp_path),
    )
    elapsed = time.perf_counter() - started
    assert workflow.calls >= 1
    assert result.generations_completed == 4
    assert result.status == "completed"
    assert elapsed < 8


@pytest.mark.asyncio
async def test_wall_clock_does_not_issue_calls_it_cannot_finish(tmp_path: Path) -> None:
    class Slow(MockModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.request_timeout_s = 2.0

        async def propose(self, instruction, schema, *, call_id):
            await asyncio.sleep(2)
            return await super().propose(instruction, schema, call_id=call_id)

    started = time.perf_counter()
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=1),
        config=EvolutionConfig(
            seed=1,
            population_size=4,
            generations=2,
            elite_count=1,
            strategist_calls_per_generation=0,
            max_model_calls=20,
            max_evaluations=40,
            max_wall_seconds=1,
        ),
        providers={"model": Slow(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=RunRepository(tmp_path),
    )
    elapsed = time.perf_counter() - started
    assert result.summary["budget"]["model_calls"] == 0
    assert elapsed < 1.5
    assert result.summary["budget"]["elapsed_seconds"] == pytest.approx(elapsed, abs=0.3)
    assert result.stop_reason == "max_wall_seconds"


@pytest.mark.asyncio
async def test_in_flight_wait_is_capped_at_the_remaining_budget(tmp_path: Path) -> None:
    class Sleepy(MockModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.request_timeout_s = 0.2

        async def propose(self, instruction, schema, *, call_id):
            self.calls += 1
            await asyncio.sleep(2)
            return await super().propose(instruction, schema, call_id=call_id)

    model = Sleepy()
    started = time.perf_counter()
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=1),
        config=EvolutionConfig(
            seed=1,
            population_size=4,
            generations=2,
            elite_count=1,
            strategist_calls_per_generation=0,
            max_concurrent_model_calls=2,
            max_model_calls=20,
            max_evaluations=40,
            max_wall_seconds=1,
        ),
        providers={"model": model, "workflow": MockAgentProvider()},
        mode="mock",
        repository=RunRepository(tmp_path),
    )
    elapsed = time.perf_counter() - started
    assert model.calls >= 1
    assert elapsed < 1.6
    assert result.summary["budget"]["elapsed_seconds"] < 1.6
    budget = Budget(EvolutionConfig(max_wall_seconds=5))
    budget.started -= 10
    assert budget.allows_model_call() is False
    assert budget.elapsed_seconds() < 1


class _Down(MockModelProvider):
    async def propose(self, instruction, schema, *, call_id):
        self.calls += 1
        raise ProviderError("veri down", code="timeout")


def test_hand_biased_local_seeds_are_gone() -> None:
    cloud_source = Path("geneai_core/domains/cloud.py").read_text(encoding="utf-8")
    controller_source = Path("geneai_core/controller.py").read_text(encoding="utf-8")
    assert "_LOCAL_SAMPLE_BIAS" not in cloud_source
    assert "objective_local_payload" not in cloud_source
    assert "objective-biased" not in controller_source
    assert "_seed_local_samples" not in controller_source
