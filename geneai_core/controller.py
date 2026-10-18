"""Domain-agnostic evolutionary controller.

Concrete adapters and sponsor clients are passed in. This module does not
import an adapter implementation or a concrete provider.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from geneai_core.budgets import Budget
from geneai_core.diversity import content_hash, family_of, normalize_payload, species_of
from geneai_core.domains.base import DomainAdapter
from geneai_core.events import lineage_payload, utc_now
from geneai_core.operators import (
    attach_context,
    creation_event_type,
    pick_two_parents,
    repair_instructions,
    sample_operation,
    seed_batch_note,
    wrapper_schema,
)
from geneai_core.providers.base import Proposal, ProviderError
from geneai_core.schemas import (
    PROMPT_VERSION,
    Evaluation,
    EvolutionConfig,
    Genome,
    Individual,
    ObjectiveSpec,
    Provenance,
    RunResult,
)
from geneai_core.selection import (
    diversity_aware_tournament_selection,
    next_population,
    pruned_individuals,
    select_elites,
)
from geneai_core.store import RunRepository

# One seed call asks for several genomes. Temperatures rotate so repeated
# calls are not samples from the same narrow distribution.
_SEED_BATCH = 4
_SEED_TEMPERATURES = (0.4, 0.8, 1.1, 0.6)


def _proposal_bodies(proposal: Proposal) -> list[tuple[dict[str, Any], str]]:
    raw = (proposal.extra or {}).get("candidates")
    bodies: list[tuple[dict[str, Any], str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("payload"), dict):
                bodies.append((item["payload"], str(item.get("hypothesis") or proposal.hypothesis or "")))
    if bodies:
        return bodies
    payload = proposal.payload if isinstance(proposal.payload, dict) else {}
    return [(payload, proposal.hypothesis or "")]


class EvolutionController:
    def __init__(self, *, clock: Any | None = None) -> None:
        self._clock = clock or time.monotonic

    async def evolve(
        self,
        *,
        objective: ObjectiveSpec,
        adapter: DomainAdapter,
        config: EvolutionConfig,
        providers: dict[str, Any],
        mode: str,
        repository: RunRepository,
        budget: Budget | None = None,
    ) -> RunResult:
        if mode not in {"mock", "hybrid", "live", "replay", "fixture"}:
            raise ValueError(f"unknown mode {mode}")
        model = providers["model"]
        workflow = providers["workflow"]
        rng = random.Random(config.seed)
        budget = budget or Budget(config)
        if self._clock is not time.monotonic:
            budget.started = self._clock()
            budget.elapsed_seconds = lambda: self._clock() - budget.started  # type: ignore[method-assign]
        datasets = adapter.dataset_hashes(config.seed)
        repository.bind(
            objective=objective,
            config=config,
            mode=mode,
            datasets=datasets,
            providers_meta=_provider_meta(model, workflow),
        )
        provenance = Provenance(
            mode=mode,
            evaluator_version=adapter.evaluator_version,
            prompt_version=PROMPT_VERSION,
        )
        repository.emit(
            "run.created",
            {
                "domain": objective.domain,
                "seed": config.seed,
                "mode": mode,
                "git_sha": repository.git_sha,
                "prompt_version": PROMPT_VERSION,
                "evaluator_version": adapter.evaluator_version,
                "datasets": datasets,
                "objective": objective.model_dump(mode="json"),
                "config": config.model_dump(mode="json"),
            },
            provenance,
        )
        repository.emit("run.started", {"domain": objective.domain, "mode": mode}, provenance)
        schema = wrapper_schema(adapter.genome_schema())
        seen_hashes: set[str] = set()
        counters = {"mutation": 0, "crossover": 0, "immigrant": 0, "seed": 0, "repair": 0}
        call_serial = 0
        strategist_used = 0
        # Default 8 matches the Veri deployment concurrency_target. Seed waves
        # and each generation's variation wave run this many calls at once.
        semaphore = asyncio.Semaphore(config.max_concurrent_model_calls)
        population: list[Individual] = []
        generations_completed = 0
        stop_reason = "generation_limit"
        status = "completed"
        champion_id: str | None = None
        elite_preserved: list[dict[str, Any]] = []

        async def provider_call(
            channel: str,
            operation: str,
            instruction: str,
            parents: list[Individual],
            *,
            task: str,
            invalid_payload: dict[str, Any] | None = None,
            provider_role: str | None = None,
        ) -> Proposal | None:
            nonlocal call_serial
            if not budget.allows_model_call():
                return None
            call_serial += 1
            call_id = f"c{call_serial:04d}"
            event_prefix = "agent" if channel == "workflow" else "model"
            provider = workflow if channel == "workflow" else model
            call_provenance = provenance.model_copy(
                update={
                    "model_provider": getattr(model, "name", None) if channel == "model" else None,
                    "workflow_provider": getattr(workflow, "name", None) if channel == "workflow" else None,
                    "provider_mode": getattr(provider, "mode", None),
                    "model": getattr(provider, "model", None),
                    "deployment_id": getattr(provider, "deployment_id", None),
                }
            )
            repository.emit(
                f"{event_prefix}.requested",
                {
                    "call_id": call_id,
                    "operation": operation,
                    "provider": getattr(provider, "name", channel),
                    "provider_mode": getattr(provider, "mode", None),
                    "instruction_sha256": _sha(instruction),
                    "task": task,
                    "provider_role": provider_role,
                },
                call_provenance,
            )
            budget.consume_model_call()
            try:
                async with semaphore:
                    if channel == "workflow":
                        wire_parents = _wire_parents(parents, invalid_payload)
                        proposal = await provider.run_workflow(
                            _workflow_name(operation),
                            {
                                "task": task,
                                "parents": wire_parents,
                                "operator": "crossover" if operation == "crossover" else "mutate",
                                "response_schema": adapter.genome_schema(),
                                "mutation_goal": instruction,
                            },
                            call_id=call_id,
                        )
                    else:
                        proposal = await provider.propose(instruction, schema, call_id=call_id)
            except ProviderError as exc:
                repository.emit(
                    f"{event_prefix}.failed",
                    {
                        "call_id": call_id,
                        "operation": operation,
                        "error": str(exc),
                        "code": exc.code,
                        "provider_run_id": exc.provider_run_id,
                        "repair_attempted": exc.repair_attempted,
                    },
                    call_provenance.model_copy(update={"provider_run_id": exc.provider_run_id}),
                )
                return None
            budget.note_cost(proposal.cost_usd)
            repository.emit(
                f"{event_prefix}.completed",
                {
                    "call_id": call_id,
                    "operation": operation,
                    "provider": proposal.provider,
                    "provider_mode": proposal.mode,
                    "provider_run_id": proposal.provider_run_id,
                    "model": proposal.model,
                    "deployment_id": proposal.deployment_id,
                    "prompt_tokens": proposal.prompt_tokens,
                    "completion_tokens": proposal.completion_tokens,
                    "total_tokens": proposal.total_tokens,
                    "cost_usd": proposal.cost_usd,
                    "cost_is_estimate": proposal.cost_is_estimate,
                    "credits": proposal.credits,
                    "latency_ms": proposal.latency_ms,
                    "provider_role": provider_role,
                    "thread_id": proposal.extra.get("thread_id"),
                    "started_at": proposal.extra.get("started_at"),
                    "completed_at": proposal.extra.get("completed_at"),
                    "answer_source": proposal.extra.get("answer_source"),
                    "schema_repairs": proposal.extra.get("schema_repairs", []),
                },
                call_provenance.model_copy(update=_proposal_provenance(proposal)),
            )
            return proposal

        def accept_payload(
            proposal: Proposal,
            *,
            operator: str,
            origin: str,
            parent_ids: list[str],
            generation: int,
            role: str | None,
            provider_role: str | None = None,
        ) -> tuple[Genome | None, str | None]:
            payload = normalize_payload(proposal.payload)
            if not isinstance(payload, dict):
                repository.emit(
                    "genome.rejected",
                    {"reason_code": "invalid_schema", "reason": "payload was not an object", "operator": operator},
                    provenance,
                )
                return None, "invalid_schema"
            payload = _project_payload(payload, adapter.genome_schema())
            try:
                adapter.validate(payload, objective)
            except ValueError as exc:
                repository.emit(
                    "genome.rejected",
                    {
                        "reason_code": "invalid_schema",
                        "reason": str(exc),
                        "operator": operator,
                        "generation": generation,
                    },
                    provenance,
                )
                return None, "invalid_schema"
            digest = content_hash(payload)
            if digest in seen_hashes:
                repository.emit(
                    "genome.rejected",
                    {
                        "reason_code": "duplicate",
                        "reason": "duplicate canonical hash",
                        "content_hash": digest,
                        "operator": operator,
                        "generation": generation,
                    },
                    provenance,
                )
                return None, "duplicate"
            genome = Genome(
                id=repository.next_genome_id(),
                run_id=repository.run_id,
                generation=generation,
                parent_ids=list(parent_ids),
                operator=operator,  # type: ignore[arg-type]
                payload=payload,
                hypothesis=proposal.hypothesis or f"{origin} candidate",
                provider=proposal.provider,
                created_at=utc_now(),
                content_hash=digest,
                metadata={
                    "origin_operator": origin,
                    "role": role,
                    "family": family_of(adapter, payload),
                    "species": species_of(adapter, payload),
                    "provider_mode": proposal.mode,
                    "provider_run_id": proposal.provider_run_id,
                    "provider_role": provider_role,
                    "seeded_baseline": bool(proposal.extra.get("seeded_baseline")),
                    "baseline_name": proposal.extra.get("baseline_name"),
                    "model": proposal.model,
                    "deployment_id": proposal.deployment_id,
                    "thread_id": proposal.extra.get("thread_id"),
                },
            )
            try:
                repository.add_genome(genome)
            except Exception as exc:
                repository.emit(
                    "genome.rejected",
                    {"reason_code": "lineage", "reason": str(exc), "operator": operator},
                    provenance,
                )
                return None, "lineage"
            seen_hashes.add(digest)
            repository.emit(
                creation_event_type(operator, origin),
                lineage_payload(genome),
                provenance.model_copy(
                    update={
                        **_proposal_provenance(proposal),
                        "model_provider": None if provider_role == "strategist" else proposal.provider,
                        "workflow_provider": proposal.provider if provider_role == "strategist" else None,
                    }
                ),
            )
            if role == "immigrant":
                repository.emit(
                    "selection.immigrant",
                    lineage_payload(genome),
                    provenance,
                )
            return genome, None

        def evaluate_genome(genome: Genome, split: str, *, required: bool = False) -> Evaluation | None:
            if budget.cancelled:
                return None
            key = (genome.content_hash, split)
            cached = _cached(repository, key)
            if cached is not None:
                return cached
            if budget.evaluations >= budget.config.max_evaluations:
                budget.exhausted_reason = budget.exhausted_reason or "max_evaluations"
                return None
            # A genome already proposed, and the champion's validation score,
            # still run when the wall clock is the only thing that tripped.
            if not required and not budget.allows_evaluation():
                return None
            started = time.perf_counter()
            evaluation = adapter.evaluate(genome, split)
            evaluation.elapsed_ms = max(evaluation.elapsed_ms, int((time.perf_counter() - started) * 1000))
            evaluation.genome_id = genome.id
            budget.consume_evaluation()
            repository.add_evaluation(evaluation)
            repository.emit(
                "genome.evaluated",
                lineage_payload(
                    genome,
                    evaluation_fitness=evaluation.fitness,
                    feasible=evaluation.feasible,
                    split=split,
                    extra={
                        "metrics": evaluation.metrics,
                        "constraint_violations": evaluation.constraint_violations,
                        "feedback": evaluation.feedback,
                        "evaluation_split": split,
                    },
                ),
                provenance.model_copy(update={"evaluator_version": evaluation.evaluator_version}),
            )
            return evaluation

        async def propose_many(
            *,
            operation: str,
            generation: int,
            parents: list[Individual],
            channel: str,
            task: str,
            human: str,
            context_parents: list[Genome],
            invalid_payload: dict[str, Any] | None = None,
            role: str | None = None,
            provider_role: str | None = None,
            batch_size: int = 1,
            temperature: float | None = None,
            sample_seed: int | None = None,
        ) -> list[Individual]:
            nonlocal strategist_used
            token = rng.randrange(2**30)
            effective_batch = 1 if invalid_payload is not None else max(1, min(int(batch_size), 8))
            if operation not in {"seed", "immigrant"}:
                effective_batch = 1
            context: dict[str, Any] = {
                "operation": "repair" if invalid_payload is not None else operation,
                "slot": generation,
                "token": token,
                "parents": [item.payload for item in context_parents],
                "invalid_payload": invalid_payload,
                "batch_size": effective_batch,
                "avoid_hashes": sorted(seen_hashes)[:32],
            }
            if temperature is not None:
                context["temperature"] = float(temperature)
            if sample_seed is not None:
                context["sample_seed"] = int(sample_seed)
            note = ""
            if effective_batch > 1 and operation in {"seed", "immigrant"} and sample_seed is not None:
                note = seed_batch_note(
                    count=effective_batch,
                    avoid_hashes=list(context["avoid_hashes"]),
                    temperature=float(temperature if temperature is not None else 0.2),
                    sample_seed=int(sample_seed),
                )
            instruction = attach_context(human + note, context)
            proposal = await provider_call(
                channel,
                operation,
                instruction,
                parents,
                task=task,
                invalid_payload=invalid_payload,
                provider_role=provider_role,
            )
            if proposal is None:
                return []
            origin = operation if operation != "immigrant" else "seed"
            operator = "repair" if invalid_payload is not None else origin
            parent_ids = [item.id for item in context_parents]
            accepted: list[Individual] = []
            invalid_body: dict[str, Any] | None = None
            for payload, hypothesis in _proposal_bodies(proposal):
                cloned = proposal.model_copy(update={"payload": payload, "hypothesis": hypothesis})
                genome, reason = accept_payload(
                    cloned,
                    operator=operator,  # type: ignore[arg-type]
                    origin=origin,
                    parent_ids=parent_ids,
                    generation=generation,
                    role=role,
                    provider_role=provider_role,
                )
                if genome is None:
                    if reason == "invalid_schema" and invalid_body is None:
                        invalid_body = payload
                    continue
                evaluation = evaluate_genome(genome, "train", required=True)
                if evaluation is None:
                    continue
                if operator == "repair":
                    counters["repair"] += 1
                elif operation == "mutation":
                    counters["mutation"] += 1
                elif operation == "crossover":
                    counters["crossover"] += 1
                elif operation == "immigrant":
                    counters["immigrant"] += 1
                elif operation == "seed":
                    counters["seed"] += 1
                accepted.append(Individual(genome=genome, evaluation=evaluation))
            if accepted or invalid_payload is not None or invalid_body is None:
                return accepted
            # Schema failures get one repair. Duplicate hashes do not: a repair
            # that "changes as little as possible" reproduces the same genome.
            repository.emit(
                "genome.rejected",
                {"reason_code": "repair_scheduled", "reason": "scheduling one repair", "operator": operation},
                provenance,
            )
            repair_task = repair_instructions(
                objective=objective,
                payload_schema=adapter.genome_schema(),
                invalid_payload=invalid_body,
                error="schema rejection",
            )
            repair_channel = "model"
            repair_role = provider_role or "bulk"
            if channel == "workflow" and strategist_used < config.strategist_calls_per_generation:
                strategist_used += 1
                repair_channel = "workflow"
                repair_role = "strategist"
            return await propose_many(
                operation="repair",
                generation=generation,
                parents=parents,
                channel=repair_channel,
                task=repair_task,
                human=repair_task,
                context_parents=context_parents,
                invalid_payload=invalid_body,
                role=role,
                provider_role=repair_role,
                batch_size=1,
            )

        try:
            repository.emit("generation.started", {"generation": 0, "phase": "seed"}, provenance)
            for spec in adapter.seeded_baselines():
                if len(population) >= config.population_size or not budget.allows_evaluation():
                    break
                name = str(spec.get("name") or "baseline")
                payload = spec.get("payload")
                if not isinstance(payload, dict):
                    continue
                proposal = Proposal(
                    payload=payload,
                    hypothesis=str(spec.get("hypothesis") or f"seeded baseline {name}"),
                    provider="baseline",
                    provider_run_id=f"baseline-{name}",
                    mode="baseline",
                    extra={"seeded_baseline": True, "baseline_name": name},
                )
                genome, _reason = accept_payload(
                    proposal,
                    operator="seed",
                    origin="seed",
                    parent_ids=[],
                    generation=0,
                    role="seeded_baseline",
                    provider_role="baseline",
                )
                if genome is None:
                    continue
                evaluation = evaluate_genome(genome, "train")
                if evaluation is None:
                    continue
                counters["seed"] += 1
                population.append(Individual(genome=genome, evaluation=evaluation))
            attempts = 0
            attempt_cap = config.population_size * 4
            temperature_index = 0
            while len(population) < config.population_size and attempts < attempt_cap:
                if not budget.allows_model_call():
                    stop_reason = budget.exhausted_reason or "budget"
                    break
                if budget.evaluations >= config.max_evaluations:
                    stop_reason = budget.exhausted_reason or "max_evaluations"
                    break
                remaining = config.population_size - len(population)
                batch = min(_SEED_BATCH, remaining)
                room_calls = max(0, config.max_model_calls - budget.model_calls)
                wave = min(
                    config.max_concurrent_model_calls,
                    max(1, (remaining + batch - 1) // batch),
                    room_calls,
                    attempt_cap - attempts,
                )
                if wave < 1:
                    stop_reason = budget.exhausted_reason or "budget"
                    break
                coroutines = []
                for _ in range(wave):
                    attempts += 1
                    temperature = _SEED_TEMPERATURES[temperature_index % len(_SEED_TEMPERATURES)]
                    temperature_index += 1
                    sample_seed = rng.randrange(2**31)
                    # The GeneAI Mutator only accepts mutate|crossover, so seeds stay
                    # on the model provider. Hand-written baselines were inserted above.
                    human = adapter.seed_instructions(objective)
                    coroutines.append(
                        propose_many(
                            operation="seed",
                            generation=0,
                            parents=[],
                            channel="model",
                            task=human,
                            human=human,
                            context_parents=[],
                            role="seed",
                            provider_role="seed",
                            batch_size=batch,
                            temperature=temperature,
                            sample_seed=sample_seed,
                        )
                    )
                groups = await asyncio.gather(*coroutines)
                for group in groups:
                    for child in group:
                        if len(population) >= config.population_size:
                            break
                        population.append(child)
            if not population:
                status = "failed"
                stop_reason = budget.exhausted_reason or "no_valid_seeds"
                repository.emit(
                    "run.failed",
                    {"stop_reason": stop_reason, "reason": "no valid seed genomes"},
                    provenance,
                )
                summary = _summary(
                    repository, population, counters, budget, stop_reason, None, elite_preserved, generations_completed
                )
                repository.finalize(status=status, stop_reason=stop_reason, champion_id=None, summary=summary)
                return _result(repository, status, stop_reason, None, generations_completed, summary)
            _finish_generation(
                repository,
                provenance,
                generation=0,
                population=population,
                elites=[],
                offspring=population,
                pruned=[],
                budget=budget,
            )
            generations_completed = 0

            for generation in range(1, config.generations + 1):
                if budget.cancelled:
                    stop_reason = "cancelled"
                    status = "cancelled"
                    break
                if not budget.allows_model_call() or not budget.allows_evaluation():
                    stop_reason = budget.exhausted_reason or "budget"
                    break
                repository.emit("generation.started", {"generation": generation}, provenance)
                strategist_used = 0
                elites = select_elites(population, config.elite_count)
                repository.emit(
                    "selection.elite",
                    {
                        "generation": generation,
                        "elite_ids": [item.genome.id for item in elites],
                        "elites": [
                            {
                                "genome_id": item.genome.id,
                                "content_hash": item.genome.content_hash,
                                "family": item.genome.metadata.get("family"),
                                "species": item.genome.metadata.get("species"),
                                "fitness": item.evaluation.fitness,
                                "feasible": item.evaluation.feasible,
                                "operator": item.genome.operator,
                                "parent_ids": list(item.genome.parent_ids),
                                "provider": item.genome.provider,
                                "provider_role": item.genome.metadata.get("provider_role"),
                                "provider_mode": item.genome.metadata.get("provider_mode"),
                                "seeded_baseline": bool(item.genome.metadata.get("seeded_baseline")),
                                "baseline_name": item.genome.metadata.get("baseline_name"),
                            }
                            for item in elites
                        ],
                    },
                    provenance,
                )
                mating_pool, tournaments = diversity_aware_tournament_selection(population, rng, adapter)
                for record in tournaments:
                    repository.emit(
                        "selection.tournament",
                        {"generation": generation, **record},
                        provenance,
                    )
                desired = config.population_size - len(elites)
                children: list[Individual] = []
                child_attempts = 0
                attempt_limit = max(desired * 8, 8)
                while (
                    len(children) < desired
                    and child_attempts < attempt_limit
                    and budget.allows_model_call()
                    and budget.allows_evaluation()
                ):
                    room = min(
                        desired - len(children),
                        attempt_limit - child_attempts,
                        config.max_concurrent_model_calls,
                    )
                    plans: list[dict[str, Any]] = []
                    while len(plans) < room and budget.allows_model_call() and budget.allows_evaluation():
                        child_attempts += 1
                        plans.append(
                            _plan_variation(
                                operation_rng=rng,
                                config=config,
                                mating_pool=mating_pool,
                                adapter=adapter,
                                objective=objective,
                            )
                        )
                    if not plans:
                        break
                    _mark_strategist(plans, config.strategist_calls_per_generation - strategist_used)
                    strategist_used += sum(1 for plan in plans if plan["channel"] == "workflow")
                    produced = await asyncio.gather(
                        *[
                            propose_many(
                                operation=plan["operation"],
                                generation=generation,
                                parents=plan["parents"],
                                channel=plan["channel"],
                                task=plan["human"],
                                human=plan["human"],
                                context_parents=plan["context_parents"],
                                role=plan["role"],
                                provider_role=plan["provider_role"],
                                batch_size=int(plan.get("batch_size") or 1),
                                temperature=plan.get("temperature"),
                                sample_seed=plan.get("sample_seed"),
                            )
                            for plan in plans
                        ]
                    )
                    for group in produced:
                        for item in group:
                            if len(children) >= desired:
                                break
                            children.append(item)
                nxt = next_population(elites, children, config.population_size, adapter)
                removed = pruned_individuals(population, nxt)
                for item in removed:
                    repository.emit(
                        "selection.pruned",
                        lineage_payload(
                            item.genome,
                            evaluation_fitness=item.evaluation.fitness,
                            feasible=item.evaluation.feasible,
                            split="train",
                            extra={"reason": "not_selected", "generation": generation},
                        ),
                        provenance,
                    )
                for elite in elites:
                    if any(member.genome.id == elite.genome.id for member in nxt):
                        elite_preserved.append(
                            {
                                "genome_id": elite.genome.id,
                                "content_hash": elite.genome.content_hash,
                                "preserved_into_generation": generation,
                                "unchanged": True,
                            }
                        )
                population = nxt
                _finish_generation(
                    repository,
                    provenance,
                    generation=generation,
                    population=population,
                    elites=elites,
                    offspring=children,
                    pruned=removed,
                    budget=budget,
                )
                generations_completed = generation
                if budget.cancelled:
                    stop_reason = "cancelled"
                    status = "cancelled"
                    break
                if budget.exhausted_reason in {"max_model_calls", "max_evaluations", "max_usd_estimate", "max_wall_seconds"} and (
                    not budget.allows_model_call() or not budget.allows_evaluation()
                ):
                    # The schedule may still have generations left. Stop before the next one.
                    if generation < config.generations:
                        stop_reason = budget.exhausted_reason
                        break

            champion = _freeze_champion(
                repository,
                provenance,
                adapter,
                config,
                budget,
                evaluate_genome,
            )
            champion_id = None if champion is None else champion.genome.id
            if status != "cancelled":
                repository.emit(
                    "run.completed",
                    {
                        "stop_reason": stop_reason,
                        "champion_id": champion_id,
                        "generations_completed": generations_completed,
                        "mode": mode,
                    },
                    provenance,
                )
            else:
                repository.emit(
                    "run.cancelled",
                    {"stop_reason": "cancelled", "champion_id": champion_id},
                    provenance,
                )
        except Exception as exc:
            status = "failed"
            stop_reason = "exception"
            repository.emit("run.failed", {"stop_reason": stop_reason, "reason": str(exc)}, provenance)
            summary = _summary(
                repository, population, counters, budget, stop_reason, None, elite_preserved, generations_completed
            )
            repository.finalize(status=status, stop_reason=stop_reason, champion_id=None, summary=summary)
            raise
        finally:
            for provider in (model, workflow):
                closer = getattr(provider, "aclose", None)
                if closer is not None:
                    await closer()

        summary = _summary(
            repository,
            population,
            counters,
            budget,
            stop_reason,
            champion_id,
            elite_preserved,
            generations_completed,
        )
        repository.finalize(
            status=status,
            stop_reason=stop_reason,
            champion_id=champion_id,
            summary=summary,
        )
        return _result(repository, status, stop_reason, champion_id, generations_completed, summary)


def _plan_variation(
    *,
    operation_rng: random.Random,
    config: EvolutionConfig,
    mating_pool: list[Individual],
    adapter: DomainAdapter,
    objective: ObjectiveSpec,
) -> dict[str, Any]:
    operation = sample_operation(config, operation_rng)
    if operation == "crossover":
        first, second = pick_two_parents(mating_pool, operation_rng)
        if first.genome.id != second.genome.id:
            return {
                "operation": "crossover",
                "parents": [first, second],
                "context_parents": [first.genome, second.genome],
                "human": adapter.crossover_instructions(first.genome, second.genome),
                "role": None,
                "channel": "model",
                "provider_role": "bulk",
            }
        operation = "mutation"
    if operation == "immigrant":
        sample_seed = operation_rng.randrange(2**31)
        return {
            "operation": "immigrant",
            "parents": [],
            "context_parents": [],
            "human": adapter.seed_instructions(objective),
            "role": "immigrant",
            "channel": "model",
            "provider_role": "immigrant",
            "batch_size": _SEED_BATCH,
            "temperature": _SEED_TEMPERATURES[sample_seed % len(_SEED_TEMPERATURES)],
            "sample_seed": sample_seed,
        }
    parent = operation_rng.choice(mating_pool)
    return {
        "operation": "mutation",
        "parents": [parent],
        "context_parents": [parent.genome],
        "human": adapter.mutate_instructions(parent.genome, parent.evaluation),
        "role": None,
        "channel": "model",
        "provider_role": "bulk",
    }


def _mark_strategist(plans: list[dict[str, Any]], slots: int) -> None:
    """Send crossover first, then mutations, until the workflow budget is gone."""

    remaining = max(0, slots)
    for operation in ("crossover", "mutation"):
        for plan in plans:
            if remaining <= 0:
                return
            if plan["operation"] != operation:
                continue
            plan["channel"] = "workflow"
            plan["provider_role"] = "strategist"
            remaining -= 1


def _proposal_provenance(proposal: Proposal) -> dict[str, Any]:
    extra = proposal.extra or {}
    update: dict[str, Any] = {
        "provider_run_id": proposal.provider_run_id,
        "provider_mode": proposal.mode,
        "model": proposal.model,
        "deployment_id": proposal.deployment_id,
        "thread_id": extra.get("thread_id"),
        "started_at": extra.get("started_at"),
        "completed_at": extra.get("completed_at"),
    }
    if extra.get("server_latency_ms") is not None:
        update["latency_ms"] = extra["server_latency_ms"]
    return update


def _wire_parents(
    parents: list[Individual],
    invalid_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if invalid_payload is not None:
        return [{"genome": invalid_payload, "fitness": None}]
    wired = []
    for parent in parents:
        wired.append({"genome": parent.genome.payload, "fitness": parent.evaluation.fitness})
    return wired


def _workflow_name(operation: str) -> str:
    if operation == "crossover":
        return "propose_crossover"
    if operation == "repair":
        return "repair_genome"
    return "propose_mutation"


def _project_payload(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("additionalProperties") is False and isinstance(schema.get("properties"), dict):
        allowed = set(schema["properties"])
        return {key: value for key, value in payload.items() if key in allowed}
    return payload


def _cached(repository: RunRepository, key: tuple[str, str]) -> Evaluation | None:
    digest, split = key
    for evaluation in reversed(repository.evaluations):
        genome = repository.genomes.get(evaluation.genome_id)
        if genome and genome.content_hash == digest and evaluation.data_split == split:
            return evaluation
    return None


def _sha(text: str) -> str:
    from geneai_core.diversity import sha256_text

    return sha256_text(text)


def _provider_meta(model: Any, workflow: Any) -> list[dict[str, Any]]:
    rows = []
    for provider in (model, workflow):
        rows.append(
            {
                "name": getattr(provider, "name", "unknown"),
                "model": getattr(provider, "model", None),
                "deployment_id": getattr(provider, "deployment_id", None),
                "mode": getattr(provider, "mode", None),
                "app_id": getattr(provider, "app_id", None),
            }
        )
    return rows


def _finish_generation(
    repository: RunRepository,
    provenance: Provenance,
    *,
    generation: int,
    population: list[Individual],
    elites: list[Individual],
    offspring: list[Individual],
    pruned: list[Individual],
    budget: Budget,
) -> None:
    feasible = [item for item in population if item.evaluation.feasible and item.evaluation.fitness is not None]
    best = max(feasible, key=lambda item: (item.evaluation.fitness or float("-inf"), item.genome.id), default=None)
    payload = {
        "generation": generation,
        "population_ids": [item.genome.id for item in population],
        "elite_ids": [item.genome.id for item in elites],
        "offspring_ids": [item.genome.id for item in offspring],
        "pruned_ids": [item.genome.id for item in pruned],
        "best_fitness": None if best is None else best.evaluation.fitness,
        "best_genome_id": None if best is None else best.genome.id,
        "best_provider": None if best is None else best.genome.provider,
        "families": sorted({str(item.genome.metadata.get("family")) for item in population}),
    }
    repository.checkpoint(
        {
            "generation": generation,
            "population_ids": payload["population_ids"],
            "elite_ids": payload["elite_ids"],
            "budget": budget.snapshot().model_dump(mode="json"),
            "best_fitness": payload["best_fitness"],
        }
    )
    repository.emit("checkpoint.saved", {"generation": generation, "path": str(repository.checkpoint_path)}, provenance)
    repository.emit("generation.completed", payload, provenance)
    snapshot = budget.snapshot()
    repository.emit("budget.updated", snapshot.model_dump(mode="json"), provenance)
    best_fitness = "none" if payload["best_fitness"] is None else f"{payload['best_fitness']:.6f}"
    best_id = payload["best_genome_id"] or "-"
    print(
        f"generation {generation} complete  population={len(population)}  "
        f"best_train={best_fitness}  best={best_id}  "
        f"calls={snapshot.model_calls}  evals={snapshot.evaluations}  "
        f"elapsed_s={snapshot.elapsed_seconds:.1f}",
        flush=True,
    )


def _freeze_champion(
    repository: RunRepository,
    provenance: Provenance,
    adapter: DomainAdapter,
    config: EvolutionConfig,
    budget: Budget,
    evaluate_genome: Any,
) -> Individual | None:
    archive: list[Individual] = []
    train_by_id: dict[str, Evaluation] = {}
    for evaluation in repository.evaluations:
        if evaluation.data_split == "train":
            train_by_id[evaluation.genome_id] = evaluation
    for genome_id, evaluation in train_by_id.items():
        genome = repository.genomes.get(genome_id)
        if genome is not None:
            archive.append(Individual(genome=genome, evaluation=evaluation))
    ranked = sorted(
        archive,
        key=lambda item: (
            0 if item.evaluation.feasible and item.evaluation.fitness is not None else 1,
            -(item.evaluation.fitness or float("-inf")),
            item.genome.id,
        ),
    )
    finalists: list[Individual] = []
    family_counts: dict[str, int] = {}
    for item in ranked:
        if not item.evaluation.feasible:
            continue
        family = str(item.genome.metadata.get("family") or "default")
        if family_counts.get(family, 0) >= config.finalist_family_cap:
            continue
        finalists.append(item)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(finalists) >= config.finalist_count:
            break
    if not finalists and ranked:
        finalists = [ranked[0]]
    validated: list[tuple[Individual, Evaluation]] = []
    for item in finalists:
        evaluation = evaluate_genome(item.genome, "validation", required=True)
        if evaluation is not None and evaluation.fitness is not None:
            validated.append((item, evaluation))
    if not validated:
        return None
    winner, held = max(
        validated,
        key=lambda pair: (
            1 if pair[1].feasible else 0,
            pair[1].fitness or float("-inf"),
            pair[0].evaluation.fitness or float("-inf"),
            pair[0].genome.id,
        ),
    )
    repository.emit(
        "champion.frozen",
        lineage_payload(
            winner.genome,
            evaluation_fitness=held.fitness,
            feasible=held.feasible,
            split="validation",
            extra={"basis": "validation"},
        ),
        provenance,
    )
    tested = evaluate_genome(winner.genome, "test", required=True)
    if tested is not None:
        repository.emit(
            "champion.tested",
            lineage_payload(
                winner.genome,
                evaluation_fitness=tested.fitness,
                feasible=tested.feasible,
                split="test",
                extra={"metrics": tested.metrics, "basis": "validation"},
            ),
            provenance.model_copy(update={"evaluator_version": adapter.evaluator_version}),
        )
    return winner


def _summary(
    repository: RunRepository,
    population: list[Individual],
    counters: dict[str, int],
    budget: Budget,
    stop_reason: str,
    champion_id: str | None,
    elite_preserved: list[dict[str, Any]],
    generations_completed: int,
) -> dict[str, Any]:
    champion = repository.genomes.get(champion_id) if champion_id else None
    train = _latest_split(repository, champion_id, "train") if champion_id else None
    validation = _latest_split(repository, champion_id, "validation") if champion_id else None
    test = _latest_split(repository, champion_id, "test") if champion_id else None
    preserved_ids = []
    seen: set[str] = set()
    for row in elite_preserved:
        if row["genome_id"] not in seen:
            seen.add(row["genome_id"])
            preserved_ids.append(row)
    return {
        "run_id": repository.run_id,
        "mode": repository.mode,
        "domain": None if repository.objective is None else repository.objective.domain,
        "seed": None if repository.config is None else repository.config.seed,
        "stop_reason": stop_reason,
        "generations_completed": generations_completed,
        "champion_id": champion_id,
        "champion_family": None if champion is None else champion.metadata.get("family"),
        "champion_payload": None if champion is None else champion.payload,
        "champion_train_fitness": None if train is None else train.fitness,
        "champion_validation_fitness": None if validation is None else validation.fitness,
        "champion_test_fitness": None if test is None else test.fitness,
        "champion_feasible": None if train is None else train.feasible,
        "mutation_children_evaluated": counters["mutation"],
        "crossover_children_evaluated": counters["crossover"],
        "immigrant_children_evaluated": counters["immigrant"],
        "seed_evaluated": counters["seed"],
        "repair_evaluated": counters["repair"],
        "elite_preserved": preserved_ids,
        "population_ids": [item.genome.id for item in population],
        "budget": budget.snapshot().model_dump(mode="json"),
        "provider_mode_note": "provider_mode=mock is not sponsor execution",
    }


def _latest_split(repository: RunRepository, genome_id: str | None, split: str) -> Evaluation | None:
    if genome_id is None:
        return None
    for evaluation in reversed(repository.evaluations):
        if evaluation.genome_id == genome_id and evaluation.data_split == split:
            return evaluation
    return None


def _result(
    repository: RunRepository,
    status: str,
    stop_reason: str,
    champion_id: str | None,
    generations_completed: int,
    summary: dict[str, Any],
) -> RunResult:
    return RunResult(
        run_id=repository.run_id,
        status=status,  # type: ignore[arg-type]
        stop_reason=stop_reason,
        mode=repository.mode,
        champion_id=champion_id,
        generations_completed=generations_completed,
        summary=summary,
        export_path=str(repository.export_path),
        events_path=str(repository.events_path),
    )
