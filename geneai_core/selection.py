"""Diversity-aware tournament selection, elitism, and pruning."""

from __future__ import annotations

import math
import random
from typing import Any

from geneai_core.diversity import family_of, novelty_bonus, species_of, trim_with_diversity
from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Individual


def _feasible(ind: Individual) -> bool:
    return bool(ind.evaluation.feasible and ind.evaluation.fitness is not None)


def _fitness(ind: Individual) -> float:
    if ind.evaluation.fitness is None:
        return float("-inf")
    return float(ind.evaluation.fitness)


def select_elites(population: list[Individual], elite_count: int) -> list[Individual]:
    if elite_count <= 0 or not population:
        return []
    ranked = sorted(
        population,
        key=lambda ind: (0 if _feasible(ind) else 1, -_fitness(ind), ind.genome.id),
    )
    return ranked[: min(elite_count, len(ranked))]


def tournament_once(
    population: list[Individual],
    rng: random.Random,
    adapter: DomainAdapter,
) -> tuple[Individual, list[dict[str, Any]]]:
    """Sample 3 candidates and keep the best feasible one, plus a small novelty bonus."""

    if len(population) >= 3:
        contestants = rng.sample(population, 3)
    else:
        contestants = [rng.choice(population) for _ in range(3)]
    scored: list[tuple[tuple[Any, ...], Individual, dict[str, Any]]] = []
    for ind in contestants:
        novelty = novelty_bonus(ind, population, adapter)
        feasible = _feasible(ind)
        fitness = _fitness(ind) if feasible else None
        bonus = (fitness + 0.02 * novelty) if feasible and fitness is not None else None
        record = {
            "genome_id": ind.genome.id,
            "feasible": feasible,
            "fitness": fitness,
            "novelty": round(novelty, 6),
            "family": family_of(adapter, ind.genome.payload),
            "species": species_of(adapter, ind.genome.payload),
            "provider": ind.genome.provider,
            "provider_role": ind.genome.metadata.get("provider_role"),
            "adjusted": None if bonus is None else round(bonus, 6),
        }
        rank = (
            0 if feasible else 1,
            -(bonus if bonus is not None else -1e18),
            ind.genome.id,
        )
        scored.append((rank, ind, record))
    scored.sort(key=lambda item: item[0])
    winner = scored[0][1]
    return winner, [item[2] for item in scored]


def diversity_aware_tournament_selection(
    population: list[Individual],
    rng: random.Random,
    adapter: DomainAdapter,
    *,
    winners: int | None = None,
    family_fraction: float = 0.5,
) -> tuple[list[Individual], list[dict[str, Any]]]:
    """Build a mating pool. No family may fill more than half the slots until a reroll fails."""

    if not population:
        return [], []
    target = winners or len(population)
    cap = max(1, math.ceil(family_fraction * target))
    pool: list[Individual] = []
    logs: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for _ in range(target):
        chosen: Individual | None = None
        last_log: list[dict[str, Any]] = []
        for attempt in range(6):
            candidate, participants = tournament_once(population, rng, adapter)
            last_log = participants
            fam = family_of(adapter, candidate.genome.payload)
            if family_counts.get(fam, 0) < cap or attempt == 5:
                chosen = candidate
                family_counts[fam] = family_counts.get(fam, 0) + 1
                logs.append(
                    {
                        "participants": participants,
                        "winner_id": candidate.genome.id,
                        "winner_provider": candidate.genome.provider,
                        "family": fam,
                        "species": species_of(adapter, candidate.genome.payload),
                        "rerolls": attempt,
                    }
                )
                break
        if chosen is None:
            raise RuntimeError("tournament produced no winner")
        pool.append(chosen)
        _ = last_log
    return pool, logs


def next_population(
    elites: list[Individual],
    children: list[Individual],
    population_size: int,
    adapter: DomainAdapter,
) -> list[Individual]:
    """Elites are copied through unchanged. Remaining slots prefer diverse children."""

    if len(elites) > population_size:
        raise ValueError("elite count exceeds population size")
    slots = population_size - len(elites)
    chosen_children = trim_with_diversity(children, slots, adapter)
    elite_ids = {ind.genome.id for ind in elites}
    merged: list[Individual] = list(elites)
    for child in chosen_children:
        if child.genome.id in elite_ids:
            continue
        merged.append(child)
    return merged[:population_size]


def pruned_individuals(previous: list[Individual], nxt: list[Individual]) -> list[Individual]:
    living = {ind.genome.id for ind in nxt}
    return [ind for ind in previous if ind.genome.id not in living]
