"""Phase 1 toy entrypoint. Live sponsor clients are constructed only for hybrid mode."""

from __future__ import annotations

from pathlib import Path

from geneai_core.controller import EvolutionController
from geneai_core.domains.toy import ToyOptimizationAdapter
from geneai_core.providers.base import CredentialsError
from geneai_core.providers.creao import CreaoProvider, creao_blocker_message
from geneai_core.providers.mock import MockAgentProvider, MockModelProvider
from geneai_core.providers.veri import VeriProvider, veri_blocker_message
from geneai_core.registry import DomainRegistry
from geneai_core.schemas import EvolutionConfig, ObjectiveSpec, RunResult
from geneai_core.store import RunRepository


def toy_objective() -> ObjectiveSpec:
    return ObjectiveSpec(
        domain="toy_optimization",
        prompt=(
            "Search the square [-5.12, 5.12]^2 for a point with high shifted-Rastrigin quality. "
            "Hard constraint: coordinates stay inside the square. "
            "Families are weighted_policy, offset_policy, and scaled_policy."
        ),
        constraints={"x_min": -5.12, "x_max": 5.12, "y_min": -5.12, "y_max": 5.12},
        evaluation_version="toy-rastrigin-v1",
    )


def build_providers(mode: str, *, seed: int) -> dict[str, object]:
    if mode == "mock":
        model = MockModelProvider()
        return {"model": model, "workflow": MockAgentProvider(model=MockModelProvider())}
    if mode == "hybrid":
        try:
            model = VeriProvider.from_env()
            workflow = CreaoProvider.from_env()
        except CredentialsError:
            raise SystemExit(veri_blocker_message() + "\n\n" + creao_blocker_message()) from None
        return {"model": model, "workflow": workflow}
    raise SystemExit(
        f"mode {mode!r} is not a Phase 1 toy mode. Use mock or hybrid. "
        "live domain runs are Phase 2. replay is Phase 3."
    )


async def execute_toy_run(
    *,
    mode: str,
    seed: int = 42,
    population_size: int = 12,
    generations: int = 5,
    root: Path | str = "runs",
    elite_count: int = 2,
    max_model_calls: int = 100,
    max_evaluations: int = 150,
    max_wall_seconds: int = 300,
    max_usd_estimate: float = 5.0,
) -> RunResult:
    config = EvolutionConfig(
        seed=seed,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        max_model_calls=max_model_calls,
        max_evaluations=max_evaluations,
        max_wall_seconds=max_wall_seconds,
        max_usd_estimate=max_usd_estimate,
    )
    adapter = ToyOptimizationAdapter(seed=seed)
    DomainRegistry.register(adapter)
    providers = build_providers(mode, seed=seed)
    repository = RunRepository(root)
    controller = EvolutionController()
    return await controller.evolve(
        objective=toy_objective(),
        adapter=DomainRegistry.get("toy_optimization"),
        config=config,
        providers=providers,
        mode=mode,  # type: ignore[arg-type]
        repository=repository,
    )
