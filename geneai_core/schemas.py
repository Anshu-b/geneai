"""Typed contracts shared by the engine, adapters, and Phase 3 replay export.

Genome ``payload`` is domain-defined JSON. The core stores and hashes it
without interpreting domain nouns.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "geneai-phase1-v1"

RunMode = Literal["mock", "hybrid", "live", "replay", "fixture"]
SplitName = Literal["train", "validation", "test"]
OperatorName = Literal["seed", "mutation", "crossover", "repair", "elite"]

# Section 13.2. Phase 1 emits these names only; baseline.evaluated is reserved
# for the Phase 2 one-shot arm and is accepted so exports stay compatible.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "run.created",
        "run.started",
        "run.completed",
        "run.cancelled",
        "run.failed",
        "model.requested",
        "model.completed",
        "model.failed",
        "agent.requested",
        "agent.completed",
        "agent.failed",
        "genome.seeded",
        "genome.mutated",
        "genome.crossed",
        "genome.rejected",
        "genome.evaluated",
        "selection.tournament",
        "selection.elite",
        "selection.pruned",
        "selection.immigrant",
        "generation.started",
        "generation.completed",
        "budget.updated",
        "checkpoint.saved",
        "champion.frozen",
        "champion.tested",
        "baseline.evaluated",
    }
)


class ObjectiveSpec(BaseModel):
    domain: str
    prompt: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    evaluation_version: str


class Genome(BaseModel):
    id: str
    run_id: str
    generation: int
    parent_ids: list[str] = Field(default_factory=list)
    operator: OperatorName
    payload: dict[str, Any]
    hypothesis: str
    provider: str
    created_at: str
    content_hash: str
    # Extension beyond the illustrative spec: lineage UI and provenance need
    # family/species, the pre-repair operator, and provider mode without
    # forcing those keys into domain payloads.
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evaluation(BaseModel):
    genome_id: str
    feasible: bool
    fitness: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    constraint_violations: list[str] = Field(default_factory=list)
    feedback: str = ""
    evaluator_version: str
    data_split: SplitName
    elapsed_ms: int


class EvolutionConfig(BaseModel):
    seed: int = 42
    population_size: int = 12
    generations: int = 5
    elite_count: int = 2
    mutation_probability: float = 0.65
    crossover_probability: float = 0.25
    random_immigrant_probability: float = 0.10
    max_model_calls: int = 100
    max_evaluations: int = 150
    max_wall_seconds: int = 300
    max_usd_estimate: float = 5.0
    # In-flight cap for provider calls. 8 matches the Veri deployment
    # concurrency_target so a fan-out cannot exceed that GPU setpoint.
    max_concurrent_model_calls: int = 8
    # Workflow-provider calls per reproductive generation. CREAO is about 44s
    # per call, so the default keeps it to crossover plus one or two mutations.
    # The model provider handles the rest. Zero sends every variation to the model.
    strategist_calls_per_generation: int = 3
    finalist_count: int = 4
    finalist_family_cap: int = 2

    @model_validator(mode="after")
    def _check_bounds(self) -> EvolutionConfig:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must satisfy 0 <= elite_count < population_size")
        if self.max_model_calls < 1 or self.max_evaluations < 1:
            raise ValueError("model and evaluation budgets must be >= 1")
        if self.max_wall_seconds < 1:
            raise ValueError("max_wall_seconds must be >= 1")
        if self.max_usd_estimate < 0:
            raise ValueError("max_usd_estimate must be >= 0")
        if self.max_concurrent_model_calls < 1:
            raise ValueError("max_concurrent_model_calls must be >= 1")
        if self.strategist_calls_per_generation < 0:
            raise ValueError("strategist_calls_per_generation must be >= 0")
        for name in (
            "mutation_probability",
            "crossover_probability",
            "random_immigrant_probability",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        total = (
            self.mutation_probability
            + self.crossover_probability
            + self.random_immigrant_probability
        )
        if total <= 0:
            raise ValueError("at least one variation probability must be positive")
        if self.finalist_count < 1:
            raise ValueError("finalist_count must be >= 1")
        return self


class Provenance(BaseModel):
    """Honest mode labels. ``mode`` is the run mode; ``provider_mode`` is this call.

    A mock provider call inside any run stays ``provider_mode=mock`` and must
    not be presented as sponsor execution.
    """

    mode: str
    model_provider: str | None = None
    workflow_provider: str | None = None
    provider_mode: str | None = None
    provider_run_id: str | None = None
    evaluator_version: str | None = None
    deployment_id: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    thread_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    latency_ms: int | None = None


class LineageEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    run_id: str
    sequence: int
    timestamp: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance

    @model_validator(mode="after")
    def _known_type(self) -> LineageEvent:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {self.type}")
        return self


class Individual(BaseModel):
    genome: Genome
    evaluation: Evaluation


class BudgetSnapshot(BaseModel):
    model_calls: int
    evaluations: int
    spend_usd: float
    elapsed_seconds: float
    exhausted_reason: str | None = None
    cancelled: bool = False


class RunResult(BaseModel):
    run_id: str
    status: Literal["completed", "cancelled", "failed"]
    stop_reason: str
    mode: str
    champion_id: str | None = None
    generations_completed: int
    summary: dict[str, Any]
    export_path: str
    events_path: str
