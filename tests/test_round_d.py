"""Round D: unbiased cloud search, complete outfits, verified catalog images."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from geneai_core.controller import EvolutionController
from geneai_core.diversity import content_hash
from geneai_core.domains.shopping import (
    ShoppingAdapter,
    _hand_baselines,
    gym_outfit,
    neutral_outfit,
)
from geneai_core.providers.browser_use import BrowserUseClient
from geneai_core.providers.creao import CreaoProvider
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.runs import _load_shopping_catalog, _split_note
from geneai_core.schemas import EvolutionConfig, ObjectiveSpec
from geneai_core.store import RunRepository, verify_export


def test_fixture_catalog_badge_and_complete_budget_baselines() -> None:
    assert _load_shopping_catalog("shopping", "fixture") is None
    note = _split_note("shopping", catalog_source="fixture")
    assert "source=fixture" in note
    assert "not purchasable" in note
    adapter = ShoppingAdapter()
    objective = ObjectiveSpec(
        domain="shopping",
        prompt="outfit",
        constraints={"size": "M", "shoe_size": 9},
        evaluation_version=adapter.evaluator_version,
    )
    report = adapter.baselines()
    assert set(report) == {"neutral_outfit", "gym_outfit"}
    for name, payload in report.items():
        adapter.validate(payload, objective)
        roles = [item["role"] for item in payload["items"]]
        assert roles.count("top") == 1
        assert roles.count("bottom") == 1
        assert roles.count("shoes") == 1
        assert roles.count("accessory") <= 1
        shoes = next(item for item in payload["items"] if item["role"] == "shoes")
        assert shoes["size"] == "9"
        spend = sum(adapter._by_id[item["product_id"]]["price"] for item in payload["items"])
        assert spend <= 260
        assert name in {neutral_outfit() and "neutral_outfit", "gym_outfit"}
    assert gym_outfit()["items"]
    sock = {
        "family": "bright_training",
        "items": [
            {"product_id": "fx_bottom_03", "color_id": "black", "size": "M", "role": "bottom"},
            {"product_id": "fx_acc_01", "color_id": "black", "size": "One Size", "role": "accessory"},
        ],
        "styling_notes": "not an outfit",
    }
    with pytest.raises(ValueError, match="exactly one top"):
        adapter.validate(sock, objective)


def test_schema_keeps_shoes_and_one_size_accessories() -> None:
    adapter = ShoppingAdapter()
    roles = set()
    for branch in adapter._item_branches():
        props = branch["properties"]
        roles.add(props["role"]["const"])
        if props["role"]["const"] == "shoes":
            assert "9" in props["size"]["enum"]
            assert "M" not in props["size"]["enum"]
        if props["role"]["const"] == "accessory":
            assert "One Size" in props["size"]["enum"]
    assert {"top", "bottom", "shoes", "accessory"} <= roles
    shopper = __import__("geneai_core.domains.shopping", fromlist=["SHOPPERS"]).SHOPPERS["train"][0]
    assert isinstance(shopper["shoe_size"], int)


def test_unknown_family_maps_to_the_enum() -> None:
    adapter = ShoppingAdapter()
    prepared, notes = adapter.prepare_payload({"family": "gym_set", "items": []})
    assert prepared["family"] == "bright_training"
    assert any("normalized to bright_training" in note for note in notes)
    campus, _notes = adapter.prepare_payload({"family": "campus wear", "items": []})
    assert campus["family"] == "campus_uniform"
    other, _notes = adapter.prepare_payload({"family": "mystery", "items": []})
    assert other["family"] == "neutral_layers"
    moves = {
        adapter.mutate_instructions(
            type("G", (), {"payload": payload})(),
            type("E", (), {"feedback": "ok"})(),
        )
        for payload in ({"family": "neutral_layers", "items": [1]}, {"family": "bright_training", "items": [2]})
    }
    assert len(moves) == 2
    prompt = adapter.seed_instructions(
        ObjectiveSpec(domain="shopping", prompt="outfit", constraints={"size": "M", "shoe_size": 9}, evaluation_version="t")
    )
    assert "neutral_layers" in prompt and "campus_uniform" in prompt and "bright_training" in prompt
    assert "Example child:" in prompt
    assert "diverse outfit" in prompt
    contract = adapter._prompt_contract()
    assert "<top_id>" in contract and "<bottom_id>" in contract and "<shoe_id>" in contract
    assert "fx_" not in contract
    blank = neutral_outfit()
    blank["styling_notes"] = ""
    prepared, notes = adapter.prepare_payload(blank)
    assert prepared["styling_notes"].strip()
    assert prepared["family"] in prepared["styling_notes"]
    assert any("styling_notes defaulted" in note for note in notes)
    objective = ObjectiveSpec(
        domain="shopping",
        prompt="outfit",
        constraints={"size": "M", "shoe_size": 9},
        evaluation_version=adapter.evaluator_version,
    )
    adapter.validate(prepared, objective)


def test_partial_catalog_does_not_invent_a_missing_role() -> None:
    catalog = {
        "catalog_id": "partial",
        "source": "recorded",
        "products": [
            {"product_id": "p_top", "role": "top", "colors": ["black"], "sizes": ["M"], "price": 20},
            {"product_id": "p_shoes", "role": "shoes", "colors": ["black"], "sizes": ["9"], "price": 40},
        ],
    }
    outfits = _hand_baselines(catalog, budget=100, shoe_size=9)
    roles = {item["role"] for _name, payload in outfits for item in payload["items"]}
    assert "bottom" not in roles
    assert {"top", "shoes"} <= roles


@pytest.mark.asyncio
async def test_image_verification_drops_non_images_and_stop_is_patch() -> None:
    def api(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "task_img", "sessionId": "sess"})
        if request.method == "PATCH":
            body = json.loads(request.content)
            assert body == {"action": "stop"}
            return httpx.Response(500, json={"error": "no"})
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
                                "image_url": "https://cdn.example/ok.jpg",
                                "role": "top",
                            },
                            {
                                "title": "Bad",
                                "price": 12,
                                "product_url": "https://shop.example/p/2",
                                "image_url": "https://cdn.example/no.jpg",
                                "role": "bottom",
                            },
                        ]
                    }
                ),
            },
        )

    def images(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("ok.jpg"):
            return httpx.Response(200, content=b"\xff\xd8", headers={"content-type": "image/jpeg"})
        return httpx.Response(200, content=b"nope", headers={"content-type": "text/html"})

    client = BrowserUseClient(
        api_key="bu_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
        image_client=httpx.AsyncClient(transport=httpx.MockTransport(images)),
        timeout_s=420,
    )
    catalog = await client.collect_catalog(catalog_url="https://shop.example/c", query="tees")
    assert catalog["image_verified_count"] == 1
    assert catalog["image_dropped_count"] == 1
    assert catalog["products"][0]["product_url"] == "https://shop.example/p/1"
    task = (
        "Stay on this site: https://shop.example/c. "
        + Path("geneai_core/providers/browser_use.py").read_text(encoding="utf-8")
    )
    assert "department store" not in task


@pytest.mark.asyncio
async def test_creao_wait_is_capped_at_min_180_and_remaining() -> None:
    seen: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "run_slow", "status": "running"})
        return httpx.Response(200, json={"id": "run_slow", "status": "running"})

    provider = CreaoProvider(
        api_key="capi_test",
        app_id="0ed64fee-5003-4d56-a83b-c5f369d7a39c",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timeout_s=180,
        poll_s=0,
    )
    assert provider._bounded_timeout(50) == 50
    assert provider._bounded_timeout(400) == 180
    assert provider._bounded_timeout(999) == 180


class _CopyNeutral(MockModelProvider):
    async def propose(self, instruction: str, schema: dict, *, call_id: str):
        proposal = await super().propose(instruction, schema, call_id=call_id)
        proposal.payload = neutral_outfit()
        proposal.extra = {}
        return proposal


@pytest.mark.asyncio
async def test_baseline_copies_are_duplicates_and_catalog_is_exported(tmp_path: Path) -> None:
    adapter = ShoppingAdapter()
    objective = ObjectiveSpec(
        domain=adapter.name,
        prompt="Neutral outfit, size M, budget 300.",
        constraints={"budget": 300, "size": "M", "palette": "neutral", "occasions": ["campus", "dinner"], "shoe_size": 9},
        evaluation_version=adapter.evaluator_version,
    )
    adapter.bind_objective(objective)
    repository = RunRepository(tmp_path)
    result = await EvolutionController().evolve(
        objective=objective,
        adapter=adapter,
        config=EvolutionConfig(seed=5, population_size=4, generations=1, elite_count=1, max_model_calls=16),
        providers={"model": _CopyNeutral(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=repository,
    )
    assert result.status in {"completed", "degraded"}, result.summary
    baseline_hash = content_hash(neutral_outfit())
    copies = [
        genome
        for genome in repository.genomes.values()
        if genome.provider != "baseline" and genome.content_hash == baseline_hash
    ]
    assert copies == []
    reasons = [event.payload.get("reason_code") for event in repository.events if event.type == "genome.rejected"]
    assert "duplicate" in reasons
    document = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert verify_export(document)
    catalog = document["catalog"]
    assert catalog["items"]
    required = {"id", "name", "role", "price", "url", "image_url", "source"}
    assert required <= set(catalog["items"][0])
