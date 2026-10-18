"""Phase 2: cloud and shopping share the Phase 1 controller. No sponsor calls."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from geneai_core.benchmark.oneshot import attach_baseline, run_frontier_oneshot
from geneai_core.controller import EvolutionController
from geneai_core.domains.cloud import (
    BASELINE_NAMES,
    SPLIT_INDEXES,
    CloudSchedulingAdapter,
    baseline_payload,
    simulate,
    workload,
)
from geneai_core.domains.shopping import (
    SHOPPERS,
    ShoppingAdapter,
    gym_outfit,
    load_catalog,
    neutral_outfit,
    products_from_browser_output,
)
from geneai_core.providers.base import CredentialsError, Proposal, ProviderError
from geneai_core.providers.browser_use import API_ORIGIN, BrowserUseClient
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.runs import execute_domain_run
from geneai_core.schemas import EvolutionConfig, Genome, ObjectiveSpec
from geneai_core.store import RunRepository, verify_export


def _genome(payload: dict) -> Genome:
    return Genome(
        id="g",
        run_id="r",
        generation=0,
        operator="seed",
        payload=payload,
        hypothesis="hand",
        provider="handwritten",
        created_at="",
        content_hash="g",
    )


def test_cloud_is_deterministic_and_capacity_safe() -> None:
    payload = baseline_payload("shortest_job")
    jobs = workload(7, 3)
    first = simulate(payload, jobs)
    second = simulate(payload, jobs)
    assert first == second
    assert first["violations"] == 0
    assert first["completed"] == len(jobs)
    adapter = CloudSchedulingAdapter(seed=7)
    objective = ObjectiveSpec(domain=adapter.name, prompt="schedule", evaluation_version=adapter.evaluator_version)
    for name in ("round_robin", "shortest_job", "least_loaded", "deadline_weighted"):
        adapter.validate(baseline_payload(name), objective)
        for split in ("train", "validation", "test"):
            evaluation = adapter.evaluate(_genome(baseline_payload(name)), split)
            assert evaluation.feasible, evaluation.feedback
            assert evaluation.metrics["violations"] == 0
    train = set(SPLIT_INDEXES["train"])
    validation = set(SPLIT_INDEXES["validation"])
    test = set(SPLIT_INDEXES["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_cloud_affinity_beats_shortest_job_on_train() -> None:
    """A policy that combines class, deadline, and energy signals beats pure SJF.

    Checked on train only. Validation and test stay unread here.
    """

    from geneai_core.domains.cloud import MACHINES, _rules, _weight_vector

    adapter = CloudSchedulingAdapter(seed=42)
    combined = {
        "policy_family": "composed",
        "rules": _rules(
            _weight_vector(
                short_job=0.55,
                deadline_urgency=0.4,
                energy_frugal=0.25,
                class_affinity=0.85,
                hard_deadline=0.45,
            ),
            ["deadline", "duration", "machine_cost", "job_id"],
        ),
        "description": "keep the GPU and high-memory node for jobs that need them",
    }
    sjf = adapter.evaluate(_genome(baseline_payload("shortest_job")), "train")
    better = adapter.evaluate(_genome(combined), "train")
    assert sjf.feasible and better.feasible
    assert sjf.fitness is not None and better.fitness is not None
    assert better.fitness > sjf.fitness + 0.02
    assert better.metrics["gpu_foreign_work"] < sjf.metrics["gpu_foreign_work"]
    kinds = {str(machine["kind"]) for machine in MACHINES}
    assert {"cpu", "high_mem", "gpu"} <= kinds
    from geneai_core.domains.cloud import Job, _fits

    gpu_job = Job(0, 0.0, 4.0, 8.0, 1.0, 4.0, 6.0, True, "gpu")
    cpu = next(machine for machine in MACHINES if machine["kind"] == "cpu")
    gpu = next(machine for machine in MACHINES if machine["kind"] == "gpu")
    assert not _fits(cpu, [], gpu_job)
    assert _fits(gpu, [], gpu_job)


def test_mock_mutation_edits_terms_ties_and_overrides() -> None:
    import random

    from geneai_core.providers.mock import mutate_one
    from geneai_core.schemas import ObjectiveSpec

    adapter = CloudSchedulingAdapter(seed=42)
    schema = adapter.genome_schema()
    objective = ObjectiveSpec(domain=adapter.name, prompt="schedule", evaluation_version=adapter.evaluator_version)
    parent = baseline_payload("shortest_job")
    added = removed = ties = overrides = 0
    for token in range(120):
        child = mutate_one(parent, schema, random.Random(token))
        adapter.validate(child, objective)
        parent_weights = parent["rules"]["score_weights"]
        for term, value in child["rules"]["score_weights"].items():
            if parent_weights[term] == 0 and value != 0:
                added += 1
            if parent_weights[term] != 0 and value == 0:
                removed += 1
        if child["rules"]["tie_break"] != parent["rules"]["tie_break"]:
            ties += 1
        if len(child["rules"]["class_overrides"]) != len(parent["rules"]["class_overrides"]):
            overrides += 1
    assert added > 0
    assert removed > 0
    assert ties > 0
    assert overrides > 0


def test_cloud_baselines_and_bad_rules() -> None:
    adapter = CloudSchedulingAdapter(seed=42)
    scores = {}
    for name in ("round_robin", "shortest_job", "deadline_weighted"):
        evaluation = adapter.evaluate(_genome(baseline_payload(name)), "train")
        assert evaluation.feasible, evaluation.feedback
        assert evaluation.fitness is not None
        scores[name] = evaluation.fitness
    assert scores["shortest_job"] >= scores["round_robin"]
    assert scores["deadline_weighted"] >= scores["round_robin"]
    objective = ObjectiveSpec(domain="cloud_scheduling", prompt="x", evaluation_version=adapter.evaluator_version)
    broken = baseline_payload("shortest_job")
    broken["rules"]["score_weights"]["short_job"] = 2.0
    with pytest.raises(ValueError):
        adapter.validate(broken, objective)
    foreign = baseline_payload("round_robin")
    foreign["policy_family"] = "python"
    with pytest.raises(ValueError):
        adapter.validate(foreign, objective)
    cells = {index % 9 for index in range(18)}
    assert cells == set(range(9))
    assert {index % 9 for index in range(18, 36)} == cells
    assert {index % 9 for index in range(36, 54)} == cells


def test_shopping_fixture_constraints_and_ranking() -> None:
    catalog = load_catalog()
    assert catalog["source"] == "fixture"
    assert catalog["browser_use_run_id"] is None
    for product in catalog["products"]:
        assert product["product_url"] is None
        assert product["image_url"] is None
        assert product["availability"] == "unknown"
    adapter = ShoppingAdapter()
    neutral_profile = {"id": "neutral", "palette": "neutral", "occasions": ["campus", "dinner"], "budget": 300, "size": "M"}
    bright_profile = {"id": "bright", "palette": "bright", "occasions": ["gym"], "budget": 300, "size": "M"}
    neutral_on_neutral, _, neutral_violations = adapter.score_shopper(_genome(neutral_outfit()), neutral_profile)
    gym_on_neutral, _, gym_violations = adapter.score_shopper(_genome(gym_outfit()), neutral_profile)
    gym_on_bright, _, _ = adapter.score_shopper(_genome(gym_outfit()), bright_profile)
    neutral_on_bright, _, _ = adapter.score_shopper(_genome(neutral_outfit()), bright_profile)
    assert not neutral_violations and not gym_violations
    assert neutral_on_neutral > gym_on_neutral
    assert gym_on_bright > neutral_on_bright
    split_ids = {split: {shopper["id"] for shopper in SHOPPERS[split]} for split in SHOPPERS}
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    outfit = _genome(neutral_outfit())
    split_scores = [adapter.evaluate(outfit, split).fitness for split in ("train", "validation", "test")]
    assert all(score is not None for score in split_scores)
    assert len({round(score, 6) for score in split_scores}) > 1
    assert adapter.evaluate(outfit, "train").metrics["purchasable"] == 0.0
    over = {
        "family": "neutral_layers",
        "items": [
            {"product_id": "fx_top_04", "color_id": "bone", "size": "M", "role": "top"},
            {"product_id": "fx_bottom_04", "color_id": "black", "size": "M", "role": "bottom"},
            {"product_id": "fx_layer_02", "color_id": "black", "size": "M", "role": "layer"},
        ],
        "styling_notes": "over budget",
    }
    assert not adapter.evaluate(_genome(over), "validation").feasible
    objective = ObjectiveSpec(
        domain="shopping",
        prompt="outfit",
        constraints={"size": "M"},
        evaluation_version="shopping-outfit-v2",
    )
    with pytest.raises(ValueError, match="unknown product"):
        adapter.validate(
            {
                "family": "neutral_layers",
                "items": [
                    {"product_id": "not_a_product", "color_id": "black", "size": "M", "role": "top"},
                    {"product_id": "fx_bottom_01", "color_id": "sage", "size": "M", "role": "bottom"},
                ],
                "styling_notes": "missing",
            },
            objective,
        )
    with pytest.raises(ValueError, match="size"):
        adapter.validate(
            {
                "family": "neutral_layers",
                "items": [
                    {"product_id": "fx_top_04", "color_id": "black", "size": "S", "role": "top"},
                    {"product_id": "fx_bottom_01", "color_id": "sage", "size": "M", "role": "bottom"},
                ],
                "styling_notes": "wrong size",
            },
            objective,
        )


def test_browser_output_drops_rows_without_urls() -> None:
    catalog = products_from_browser_output(
        {
            "products": [
                {
                    "title": "Known tee",
                    "price": 20,
                    "currency": "USD",
                    "product_url": "https://shop.example/p/1",
                    "role": "top",
                    "availability": "in_stock",
                },
                {"title": "No page", "price": 10, "currency": "USD", "role": "bottom"},
                {"title": "Bad price", "price": "free", "product_url": "https://shop.example/p/2"},
            ]
        },
        run_id="run_test",
        session_id="sess",
        model="browser-use",
    )
    assert catalog["source"] == "browser_use"
    assert catalog["dropped_without_url"] == 2
    assert len(catalog["products"]) == 1
    assert catalog["products"][0]["product_url"] == "https://shop.example/p/1"


@pytest.mark.asyncio
async def test_browser_use_client_parses_v4_and_refuses_other_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["x-browser-use-api-key"] == "bu_test"
        path = request.url.path
        if request.method == "POST" and path.endswith("/runs"):
            return httpx.Response(200, json={"id": "run_1", "status": "queued", "sessionId": "sess", "model": "bu"})
        if path.endswith("/status"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": {
                    "products": [
                        {
                            "title": "Known tee",
                            "price": 20,
                            "currency": "USD",
                            "product_url": "https://shop.example/p/1",
                            "image_url": "https://shop.example/i/1.jpg",
                            "role": "top",
                            "availability": "in_stock",
                        },
                        {"title": "Dropped", "price": 5, "currency": "USD", "role": "bottom"},
                    ]
                },
                "model": "bu-model",
                "sessionId": "sess",
                "updatedAt": "2026-10-18T00:00:00Z",
                "totalCostUsd": "0.02",
            },
        )

    client = BrowserUseClient(api_key="bu_test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    catalog = await client.collect_catalog(catalog_url="https://shop.example/c", query="tees")
    assert catalog["browser_use_run_id"] == "run_1"
    assert catalog["source"] == "browser_use"
    assert len(catalog["products"]) == 1
    assert calls["n"] == 3
    with pytest.raises(ProviderError, match="non Browser Use"):
        await client._request("GET", "https://evil.example/api/v2/tasks", None)
    with pytest.raises(CredentialsError):
        BrowserUseClient.from_env()


@pytest.mark.asyncio
async def test_browser_use_empty_catalog_does_not_substitute_the_fixture() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "run_empty", "status": "queued"})
        if str(request.url).endswith("/status"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(
            200,
            json={"status": "completed", "output": {"products": [{"title": "no url", "price": 1}]}},
        )

    client = BrowserUseClient(api_key="bu_test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError, match="fixture catalog was not substituted") as caught:
        await client.collect_catalog(catalog_url="https://shop.example/c", query="tees")
    assert caught.value.code == "EMPTY_CATALOG"


@pytest.mark.asyncio
async def test_oneshot_skips_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)

    def explode(*_args, **_kwargs):
        raise AssertionError("OpenAI client must not be constructed")

    monkeypatch.setattr("openai.AsyncOpenAI", explode)
    adapter = CloudSchedulingAdapter(seed=1)
    record = await run_frontier_oneshot(
        adapter=adapter,
        objective=ObjectiveSpec(domain=adapter.name, prompt="schedule", evaluation_version=adapter.evaluator_version),
    )
    assert record["status"] == "skipped"
    assert record["calls"] == 0
    assert record["compute_matched"] is False


@pytest.mark.asyncio
async def test_oneshot_one_call_invalid_stays_invalid_and_export_verifies(tmp_path: Path) -> None:
    class FakeFrontier:
        model = "fake-model"
        base_url = "https://frontier.example/v1"
        calls = 0

        async def propose(self, instruction: str, schema: dict, *, call_id: str) -> Proposal:
            del instruction, schema
            self.calls += 1
            assert call_id == "oneshot"
            return Proposal(
                payload={"policy_family": "not-a-policy", "rules": {}, "description": "bad"},
                hypothesis="one try",
                provider="frontier",
                provider_run_id="fake-1",
                model=self.model,
                mode="live",
            )

        async def aclose(self) -> None:
            return None

    provider = FakeFrontier()
    adapter = CloudSchedulingAdapter(seed=3)
    objective = ObjectiveSpec(domain=adapter.name, prompt="schedule", evaluation_version=adapter.evaluator_version)
    record = await run_frontier_oneshot(adapter=adapter, objective=objective, provider=provider)  # type: ignore[arg-type]
    assert provider.calls == 1
    assert record["status"] == "invalid"
    assert record["feasible"] is False
    assert record["compute_matched"] is False
    assert record["calls"] == 1
    document = {
        "schema_version": "1.0",
        "one_shot_baseline": None,
        "run": {"id": "local"},
    }
    from geneai_core.store import export_integrity

    document["integrity_sha256"] = export_integrity(document)
    path = tmp_path / "export.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    updated = attach_baseline(path, record)
    assert updated["one_shot_baseline"]["status"] == "invalid"
    assert updated["one_shot_baseline"]["compute_matched"] is False
    assert verify_export(updated)


@pytest.mark.asyncio
async def test_cloud_and_shopping_share_one_controller(tmp_path: Path) -> None:
    controller = EvolutionController()
    providers = {"model": MockModelProvider(), "workflow": MockAgentProvider()}
    cloud = CloudSchedulingAdapter(seed=5)
    shopping = ShoppingAdapter()
    shopping.bind_objective(
        ObjectiveSpec(
            domain=shopping.name,
            prompt="bound",
            constraints={"budget": 300, "size": "M", "palette": "neutral", "occasions": ["campus", "dinner"]},
            evaluation_version=shopping.evaluator_version,
        )
    )
    cloud_result = await controller.evolve(
        objective=ObjectiveSpec(
            domain=cloud.name,
            prompt="Schedule without exceeding CPU or memory.",
            evaluation_version=cloud.evaluator_version,
        ),
        adapter=cloud,
        config=EvolutionConfig(seed=5, population_size=4, generations=1, elite_count=1, max_model_calls=80),
        providers=providers,
        mode="mock",
        repository=RunRepository(tmp_path / "cloud"),
    )
    shopping_result = await controller.evolve(
        objective=ObjectiveSpec(
            domain=shopping.name,
            prompt="Neutral outfit, size M, budget 300.",
            constraints={"budget": 300, "size": "M", "palette": "neutral", "occasions": ["campus", "dinner"]},
            evaluation_version=shopping.evaluator_version,
        ),
        adapter=shopping,
        config=EvolutionConfig(seed=5, population_size=4, generations=1, elite_count=1, max_model_calls=80),
        providers={"model": MockModelProvider(), "workflow": MockAgentProvider()},
        mode="mock",
        repository=RunRepository(tmp_path / "shopping"),
    )
    assert cloud_result.status == "completed", cloud_result.summary
    assert shopping_result.status == "completed", shopping_result.summary
    assert cloud_result.summary["domain"] == "cloud_scheduling"
    assert shopping_result.summary["domain"] == "shopping"
    assert cloud_result.champion_id and shopping_result.champion_id
    for item in shopping_result.summary["champion_payload"]["items"]:
        assert item["product_id"].startswith("fx_")
        assert item["product_id"] in shopping._by_id


@pytest.mark.asyncio
async def test_domain_run_exports_replay_and_skips_oneshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    result = await execute_domain_run(
        domain="shopping",
        mode="mock",
        seed=4,
        population_size=4,
        generations=1,
        root=tmp_path,
        elite_count=1,
    )
    assert result.status == "completed"
    document = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert document["schema_version"] == "1.0"
    assert document["run"]["domain"] == "shopping"
    assert document["run"]["mode"] == "mock"
    assert document["one_shot_baseline"]["status"] == "skipped"
    assert document["one_shot_baseline"]["compute_matched"] is False
    assert document["one_shot_baseline"]["comparison_label"] == "single attempt vs evolutionary search"
    assert verify_export(document)
    assert "neutral_outfit" in result.summary["baselines"]
    assert result.summary["baselines"]["neutral_outfit"]["train"]["feasible"] is True
    train_fit = result.summary["baselines"]["neutral_outfit"]["train"]["fitness"]
    test_fit = result.summary["baselines"]["neutral_outfit"]["test"]["fitness"]
    assert train_fit != test_fit


@pytest.mark.asyncio
async def test_seeded_baselines_and_child_provider_are_on_the_export(tmp_path: Path) -> None:
    result = await execute_domain_run(
        domain="cloud",
        mode="mock",
        seed=3,
        population_size=6,
        generations=1,
        root=tmp_path,
        elite_count=1,
        attach_oneshot=False,
    )
    assert result.status == "completed", result.summary
    document = json.loads(Path(result.export_path).read_text(encoding="utf-8"))
    assert verify_export(document)
    seeded = [
        genome
        for genome in document["genomes"]
        if genome["generation"] == 0 and genome["metadata"].get("seeded_baseline")
    ]
    assert {genome["metadata"]["baseline_name"] for genome in seeded} == set(BASELINE_NAMES)
    assert all(genome["provider"] == "baseline" for genome in seeded)
    allowed = {"mock", "baseline", "creao", "veri"}
    for genome in document["genomes"]:
        assert genome["provider"] in allowed
    for event in document["events"]:
        if event["type"] in {"genome.seeded", "genome.mutated", "genome.crossed", "genome.evaluated", "selection.pruned"}:
            assert event["payload"].get("provider") in allowed
        if event["type"] == "selection.elite":
            assert event["payload"]["elites"]
            assert all(row["provider"] in allowed for row in event["payload"]["elites"])
        if event["type"] == "selection.tournament":
            assert event["payload"]["winner_provider"] in allowed
            assert all(row["provider"] in allowed for row in event["payload"]["participants"])
    seeded_events = [
        event
        for event in document["events"]
        if event["type"] == "genome.seeded" and event["payload"].get("seeded_baseline")
    ]
    assert len(seeded_events) == len(BASELINE_NAMES)
    assert all(event["payload"]["provider"] == "baseline" for event in seeded_events)
