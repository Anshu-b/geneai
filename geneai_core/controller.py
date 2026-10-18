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
        # Default 8 matches the Veri deployment concurrency_target. Proposals
        # are awaited one at a time; the cap applies if calls overlap.
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
                    "schema_repairs": proposal.extra.get("schema_repairs", []),
                },
                call_provenance.model_copy(
                    update={
                        "provider_run_id": proposal.provider_run_id,
                        "provider_mode": proposal.mode,
                        "model": proposal.model,
                        "deployment_id": proposal.deployment_id,
                    }
                ),
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
        ) -> Genome | None:
            payload = normalize_payload(proposal.payload)
            if not isinstance(payload, dict):
                repository.emit(
                    "genome.rejected",
                    {"reason_code": "invalid_schema", "reason": "payload was not an object", "operator": operator},
                    provenance,
                )
                return None
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
                return None
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
                return None
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
                    "model": proposal.model,
                    "deployment_id": proposal.deployment_id,
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
                return None
            seen_hashes.add(digest)
            repository.emit(
                creation_event_type(operator, origin),
                lineage_payload(genome),
                provenance.model_copy(
                    update={
                        "provider_run_id": proposal.provider_run_id,
                        "provider_mode": proposal.mode,
                        "model_provider": proposal.provider if proposal.provider != "creao" else None,
                        "workflow_provider": proposal.provider if proposal.provider in {"creao", "mock-agent"} else None,
                    }
                ),
            )
            if role == "immigrant":
                repository.emit(
                    "selection.immigrant",
                    lineage_payload(genome),
                    provenance,
                )
            return genome

        def evaluate_genome(genome: Genome, split: str) -> Evaluation | None:
            if not budget.allows_evaluation():
                return None
            key = (genome.content_hash, split)
            cached = _cached(repository, key)
            if cached is not None:
                return cached
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

        async def propose_one(
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
        ) -> Individual | None:
            token = rng.randrange(2**30)
            context = {
                "operation": "repair" if invalid_payload is not None else operation,
                "slot": generation,
                "token": token,
                "parents": [item.payload for item in context_parents],
                "invalid_payload": invalid_payload,
            }
            instruction = attach_context(human, context)
            proposal = await provider_call(
                channel,
                operation,
                instruction,
                parents,
                task=task,
                invalid_payload=invalid_payload,
            )
            if proposal is None:
                return None
            origin = operation if operation != "immigrant" else "seed"
            operator = origin
            parent_ids = [item.id for item in context_parents]
            genome = accept_payload(
                proposal,
                operator=operator,  # type: ignore[arg-type]
                origin=origin,
                parent_ids=parent_ids,
                generation=generation,
                role=role,
            )
            if genome is None and invalid_payload is None:
                # One repair pass. The follow-up is mutate on the mutator contract.
                repository.emit(
                    "genome.rejected",
                    {"reason_code": "repair_scheduled", "reason": "scheduling one repair", "operator": operation},
                    provenance,
                )
                repair_task = repair_instructions(
                    objective=objective,
                    payload_schema=adapter.genome_schema(),
                    invalid_payload=proposal.payload if isinstance(proposal.payload, dict) else {},
                    error="schema or duplicate rejection",
                )
                return await propose_one(
                    operation="repair",
                    generation=generation,
                    parents=parents,
                    channel=channel,
                    task=repair_task,
                    human=repair_task,
                    context_parents=context_parents,
                    invalid_payload=proposal.payload if isinstance(proposal.payload, dict) else {},
                    role=role,
                )
            if genome is None:
                return None
            if genome.operator == "repair" or operation == "repair":
                counters["repair"] += 1
            evaluation = evaluate_genome(genome, "train")
            if evaluation is None:
                return None
            if operation == "mutation":
                counters["mutation"] += 1
            elif operation == "crossover":
                counters["crossover"] += 1
            elif operation == "immigrant":
                counters["immigrant"] += 1
            elif operation == "seed":
                counters["seed"] += 1
            return Individual(genome=genome, evaluation=evaluation)

        try:
            repository.emit("generation.started", {"generation": 0, "phase": "seed"}, provenance)
            attempts = 0
            while len(population) < config.population_size and attempts < config.population_size * 8:
                if not budget.allows_model_call() or not budget.allows_evaluation():
                    stop_reason = budget.exhausted_reason or "budget"
                    break
                attempts += 1
                # The GeneAI Mutator only accepts mutate|crossover, so seeds stay
                # on the model provider.
                human = adapter.seed_instructions(objective)
                child = await propose_one(
                    operation="seed",
                    generation=0,
                    parents=[],
                    channel="model",
                    task=human,
                    human=human,
                    context_parents=[],
                    role="seed",
                )
                if child is not None:
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
                while (
                    len(children) < desired
                    and child_attempts < max(desired * 8, 8)
                    and budget.allows_model_call()
                    and budget.allows_evaluation()
                ):
                    child_attempts += 1
                    operation = sample_operation(config, rng)
                    if operation == "crossover":
                        first, second = pick_two_parents(mating_pool, rng)
                        if first.genome.id == second.genome.id:
                            operation = "mutation"
                        else:
                            human = adapter.crossover_instructions(first.genome, second.genome)
                            child = await propose_one(
                                operation="crossover",
                                generation=generation,
                                parents=[first, second],
                                channel="workflow",
                                task=human,
                                human=human,
                                context_parents=[first.genome, second.genome],
                            )
                            if child is not None:
                                children.append(child)
                            continue
                    if operation == "immigrant":
                        human = adapter.seed_instructions(objective)
                        child = await propose_one(
                            operation="immigrant",
                            generation=generation,
                            parents=[],
                            channel="model",
                            task=human,
                            human=human,
                            context_parents=[],
                            role="immigrant",
                        )
                        if child is not None:
                            children.append(child)
                        continue
                    parent = rng.choice(mating_pool)
                    human = adapter.mutate_instructions(parent.genome, parent.evaluation)
                    child = await propose_one(
                        operation="mutation",
                        generation=generation,
                        parents=[parent],
                        channel="workflow",
                        task=human,
                        human=human,
                        context_parents=[parent.genome],
                    )
                    if child is not None:
                        children.append(child)
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
    repository.emit("budget.updated", budget.snapshot().model_dump(mode="json"), provenance)


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
        if not budget.allows_evaluation():
            break
        evaluation = evaluate_genome(item.genome, "validation")
        if evaluation is not None:
            validated.append((item, evaluation))
    if validated:
        winner, held = max(
            validated,
            key=lambda pair: (
                1 if pair[1].feasible and pair[1].fitness is not None else 0,
                pair[1].fitness or float("-inf"),
                pair[0].evaluation.fitness or float("-inf"),
                pair[0].genome.id,
            ),
        )
        basis = "validation"
        basis_fitness = held.fitness
        basis_feasible = held.feasible
    elif ranked:
        winner = ranked[0]
        basis = "train_fallback"
        basis_fitness = winner.evaluation.fitness
        basis_feasible = winner.evaluation.feasible
    else:
        return None
    repository.emit(
        "champion.frozen",
        lineage_payload(
            winner.genome,
            evaluation_fitness=basis_fitness,
            feasible=basis_feasible,
            split="validation" if basis == "validation" else "train",
            extra={"basis": basis},
        ),
        provenance,
    )
    if budget.allows_evaluation():
        tested = evaluate_genome(winner.genome, "test")
        if tested is not None:
            repository.emit(
                "champion.tested",
                lineage_payload(
                    winner.genome,
                    evaluation_fitness=tested.fitness,
                    feasible=tested.feasible,
                    split="test",
                    extra={"metrics": tested.metrics, "basis": basis},
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
