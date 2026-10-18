"""Fixes from the live hybrid verification at 6bbf16d."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from geneai_core.benchmark.oneshot import run_frontier_oneshot
from geneai_core.controller import EvolutionController, _provider_mode_note
from geneai_core.domains.cloud import CLOUD_SEARCH_CEILING, CloudSchedulingAdapter
from geneai_core.domains.shopping import (
    ShoppingAdapter,
    _catalog_role,
    products_from_browser_output,
)
from geneai_core.domains.toy import ToyOptimizationAdapter
from geneai_core.experiments import toy_objective
from geneai_core.providers.base import Proposal, ProviderError
from geneai_core.providers.browser_use import BrowserUseClient
from geneai_core.providers.creao import EXAMPLE_APP_ID, CreaoProvider, children_artifact
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.providers.openai_compat import completion_token_cap, prompt_schema_text
from geneai_core.providers.veri import (
    EXAMPLE_DEPLOYMENT_ID,
    EXAMPLE_MODEL,
    VERI_REQUEST_TIMEOUT_S,
    VeriProvider,
)
from geneai_core.runs import _split_note, execute_domain_run
from geneai_core.schemas import EvolutionConfig, ObjectiveSpec
from geneai_core.store import RunRepository, verify_export

SAMPLE = Path(__file__).parent / "fixtures" / "creao_failed_run_703a.json"


def test_veri_timeout_defaults_to_120_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = VeriProvider(api_key="vk_test", deployment_id=EXAMPLE_DEPLOYMENT_ID, model=EXAMPLE_MODEL)
    assert provider.request_timeout_s == VERI_REQUEST_TIMEOUT_S == 120
    assert provider.max_tokens == 400
    monkeypatch.setenv("VERI_API_KEY", "vk_test")
    monkeypatch.setenv("VERI_TIMEOUT_SECONDS", "90")
    loaded = VeriProvider.from_env()
    assert loaded.request_timeout_s == 90


def test_completion_budget_is_400_tokens_per_genome_and_at_most_two() -> None:
    assert completion_token_cap(batch_size=1, max_tokens=800) == 400
    assert completion_token_cap(batch_size=2, max_tokens=800) == 800
    assert completion_token_cap(batch_size=4, max_tokens=3200) == 800
    schema = CloudSchedulingAdapter(seed=1).genome_schema()
    full = json.dumps(schema)
    compact = prompt_schema_text(schema)
    assert len(compact) < len(full)
    assert len(compact) <= 1500


def test_provider_note_follows_the_run_mode() -> None:
    mock_note = _provider_mode_note("mock", [{"name": "mock", "mode": "mock"}])
    live_note = _provider_mode_note(
        "live",
        [{"name": "veri", "mode": "live"}, {"name": "creao", "mode": "live"}],
    )
    assert mock_note.startswith("mode=mock")
    assert "not sponsor execution" in mock_note
    assert live_note.startswith("mode=live")
    assert "veri=live" in live_note
    assert "creao=live" in live_note
    assert "mode=mock" not in live_note


def test_creao_sample_selects_geneai_mutate_result_json() -> None:
    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert "chatText" not in document["result"]
    artifact = children_artifact(document)
    assert artifact is not None
    assert str(artifact["id"]).endswith("geneai_mutate_result.json")
    assert artifact["type"] == "application/json"
    preferred = children_artifact(
        {
            "result": {
                "artifacts": [
                    {"id": "notes.json", "type": "application/json", "url": "https://agent.creao.ai/notes.json"},
                    {
                        "id": "files/children.json",
                        "type": "application/json",
                        "url": "https://agent.creao.ai/children.json",
                    },
                ]
            }
        }
    )
    assert preferred is not None
    assert preferred["id"] == "files/children.json"


@pytest.mark.asyncio
async def test_creao_sample_without_chattext_reads_json_artifact() -> None:
    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
    artifact_url = document["result"]["artifacts"][0]["url"]

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == artifact_url
        return httpx.Response(
            200,
            json={"children": [{"genome": {"ok": True, "source": "creao"}, "rationale": "artifact"}]},
        )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "source"],
        "properties": {"ok": {"type": "boolean"}, "source": {"type": "string"}},
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CreaoProvider(api_key="capi_test", app_id=EXAMPLE_APP_ID, client=client, poll_s=0)
        genome, rationale, _notes, source, _text = await provider._read_answer(document, schema, document["id"])
    assert source == "artifact"
    assert genome == {"ok": True, "source": "creao"}
    assert rationale == "artifact"


@pytest.mark.asyncio
async def test_short_model_seed_fills_random_and_continues(tmp_path: Path) -> None:
    class DownModel(MockModelProvider):
        async def propose(self, instruction, schema, *, call_id):
            self.calls += 1
            raise ProviderError("timeout", code="timeout")

    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=toy_objective(),
        adapter=ToyOptimizationAdapter(seed=3),
        config=EvolutionConfig(
            seed=3,
            population_size=4,
            generations=1,
            elite_count=1,
            max_model_calls=40,
            max_evaluations=40,
        ),
        providers={"model": DownModel(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    assert result.status == "degraded"
    assert result.generations_completed == 1
    assert "random" in result.summary["degraded_reason"]
    assert any(genome.provider == "random" for genome in repository.genomes.values())
    assert result.summary["provider_mode_note"].startswith("mode=mock")


def test_unknown_catalog_role_becomes_accessory_and_tags_come_from_the_row() -> None:
    assert _catalog_role("hat") == "accessory"
    assert _catalog_role("top") == "top"
    catalog = products_from_browser_output(
        {
            "products": [
                {
                    "title": "Court shoe",
                    "price": 40,
                    "product_url": "https://shop.example/p/shoe",
                    "role": "hat",
                    "style": ["athletic"],
                    "occasions": ["gym"],
                    "colors": ["neon"],
                },
                {
                    "title": "Silk blouse",
                    "price": 80,
                    "product_url": "https://shop.example/p/silk",
                    "role": "top",
                    "colors": ["bone"],
                },
            ]
        },
        run_id="run",
        session_id=None,
        model=None,
    )
    hat, silk = catalog["products"]
    assert hat["role"] == "accessory"
    assert hat["style"] == ["athletic"]
    assert hat["occasions"] == ["gym"]
    assert silk["style"] == ["neutral"]
    assert "dinner" in silk["occasions"]
    assert silk["occasions"] != ["campus", "dinner"] or "neutral" in silk["style"]


def test_shopping_baselines_are_hand_built_and_valid() -> None:
    adapter = ShoppingAdapter()
    baselines = adapter.seeded_baselines()
    names = [item["name"] for item in baselines]
    assert names == ["neutral_outfit", "gym_outfit"]
    objective = ObjectiveSpec(domain="shopping", prompt="outfit", constraints={"size": "M"}, evaluation_version="t")
    for item in baselines:
        adapter.validate(item["payload"], objective)
        roles = [row["role"] for row in item["payload"]["items"]]
        assert len(roles) == len(set(roles))


def test_split_note_names_the_catalog_source() -> None:
    fixture = _split_note("shopping", catalog_source="fixture")
    recorded = _split_note("shopping", catalog_source="recorded")
    assert "source=fixture" in fixture
    assert "not purchasable" in fixture
    assert "source=recorded" in recorded
    assert "not a fixture" in recorded


@pytest.mark.asyncio
async def test_browser_task_asks_for_roles_style_and_a_target_count() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "task_1", "sessionId": "sess"})
        if str(request.url).endswith("/status"):
            return httpx.Response(200, json={"status": "finished"})
        return httpx.Response(
            200,
            json={
                "status": "finished",
                "output": json.dumps(
                    {
                        "products": [
                            {
                                "title": "Tee",
                                "price": 10,
                                "product_url": "https://shop.example/p/1",
                                "image_url": "https://shop.example/i/1.jpg",
                                "retailer": "Example",
                                "role": "top",
                                "colors": ["black"],
                                "style": ["neutral"],
                                "availability": "in_stock",
                            }
                        ]
                    }
                ),
            },
        )

    client = BrowserUseClient(api_key="bu_test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.collect_catalog(catalog_url="https://shop.example/c", query="tees")
    body = seen["body"]
    assert isinstance(body, dict)
    task = str(body["task"])
    assert "30 to 40" in task
    assert "accessory" in task
    assert "style" in task
    schema = json.loads(str(body["structuredOutput"]))
    required = schema["properties"]["products"]["items"]["required"]
    assert "role" in required and "style" in required and "colors" in required


@pytest.mark.asyncio
async def test_recorded_catalog_is_replayable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    recorded = {
        "catalog_id": "shopping-recorded-test",
        "source": "recorded",
        "label": "recorded",
        "mode": "recorded",
        "products": [
            {
                "product_id": "rec_top",
                "title": "Recorded knit",
                "price": 40,
                "currency": "USD",
                "product_url": "https://shop.example/p/top",
                "image_url": "https://shop.example/i/top.jpg",
                "retailer": "Example",
                "role": "top",
                "colors": ["black"],
                "sizes": ["M"],
                "style": ["neutral"],
                "occasions": ["campus"],
                "availability": "in_stock",
            },
            {
                "product_id": "rec_bottom",
                "title": "Recorded pant",
                "price": 50,
                "currency": "USD",
                "product_url": "https://shop.example/p/bottom",
                "image_url": None,
                "retailer": "Example",
                "role": "bottom",
                "colors": ["black"],
                "sizes": ["M"],
                "style": ["neutral"],
                "occasions": ["campus"],
                "availability": "in_stock",
            },
        ],
    }
    path = tmp_path / "shopping_recorded.json"
    path.write_text(json.dumps(recorded), encoding="utf-8")
    result = await execute_domain_run(
        domain="shopping",
        mode="mock",
        seed=1,
        population_size=4,
        generations=1,
        elite_count=1,
        strategist_calls_per_generation=0,
        root=tmp_path / "run",
        attach_oneshot=False,
        catalog=str(path),
    )
    assert "source=recorded" in result.summary["split_note"]
    document = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert any(genome["provider"] == "baseline" for genome in document["genomes"])
    assert verify_export(document)


@pytest.mark.asyncio
async def test_oneshot_reports_duplicate_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)

    class FakeFrontier:
        model = "gpt-5.5"
        base_url = "https://api.openai.com/v1"

        async def propose(self, instruction, schema, *, call_id):
            assert "one item per role" in instruction or "never repeat a role" in instruction
            return Proposal(
                payload={
                    "family": "neutral_layers",
                    "items": [
                        {"product_id": "fx_top_01", "color_id": "black", "size": "M", "role": "top"},
                        {"product_id": "fx_top_02", "color_id": "bone", "size": "M", "role": "top"},
                    ],
                    "styling_notes": "two tops",
                },
                hypothesis="invalid",
                provider="frontier",
                mode="live",
            )

        async def aclose(self) -> None:
            return None

    adapter = ShoppingAdapter()
    record = await run_frontier_oneshot(
        adapter=adapter,
        objective=ObjectiveSpec(
            domain="shopping",
            prompt="one item per role",
            constraints={"size": "M"},
            evaluation_version=adapter.evaluator_version,
        ),
        provider=FakeFrontier(),  # type: ignore[arg-type]
    )
    assert record["status"] == "invalid"
    assert record["feasible"] is False
    assert "duplicate role" in str(record["reason"])


@pytest.mark.asyncio
async def test_cloud_export_carries_non_geneai_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    result = await execute_domain_run(
        domain="cloud",
        mode="mock",
        seed=1,
        population_size=4,
        generations=1,
        elite_count=1,
        strategist_calls_per_generation=0,
        root=tmp_path,
        attach_oneshot=False,
    )
    document = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    ceiling = document["cloud_ceiling"]
    assert ceiling["label"] == "non-GeneAI reference"
    assert ceiling["oracle"]["train"] == CLOUD_SEARCH_CEILING["oracle"]["train"]
    assert ceiling["rounded"]["oracle"] == {"train": -1.431, "validation": -1.585, "test": -1.582}
    assert ceiling["rounded"]["deadline_weighted"] == {"train": -1.527, "validation": -1.634, "test": -1.601}
    assert verify_export(document)
