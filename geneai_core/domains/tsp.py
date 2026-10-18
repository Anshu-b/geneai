"""15-city Euclidean TSP. The genome is a permutation tour.

Cities are drawn once from a fixed seed in the square [0, 100]. Tour length is
the closed Euclidean cycle. Fitness is the negative of that length so the
existing controller, which maximizes fitness, searches for shorter tours.

Mutations are a city swap or a 2-opt reversal. Crossover is order crossover
(OX) or edge recombination (ERX). An invalid city list is repaired into a
permutation for GeneAI only. The one-shot arm calls ``validate`` and does not
repair. Dedup uses the canonical cycle: rotate so city 0 is first, then keep
the lexicographically smaller direction.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

N_CITIES = 15
SHOWCASE_SEED = 1500
HELD_OUT_SEEDS = (1501, 1502, 1503, 1504, 1505)
COORD_SCALE = 100.0
CLASSIC_GA_POPULATION = 24
CLASSIC_GA_GENERATIONS = 40
CLASSIC_GA_ELITE = 2
CLASSIC_GA_SWAP_RATE = 0.8
CLASSIC_GA_TOURNAMENT = 3
# A few improving reversals, not a local optimum. Full 2-opt on 15 cities
# leaves two canonical tours, and canonical dedup then shrinks the population.
TSP_POLISH_MOVES = 3
_LOG = logging.getLogger("geneai_core.tsp")


def cities_for_seed(seed: int, n: int = N_CITIES) -> list[dict[str, Any]]:
    """Deterministic Euclidean cities. ``random.Random(seed)`` is the only source."""

    rng = random.Random(seed)
    cities: list[dict[str, Any]] = []
    for city_id in range(n):
        cities.append(
            {
                "id": city_id,
                "x": round(rng.random() * COORD_SCALE, 4),
                "y": round(rng.random() * COORD_SCALE, 4),
            }
        )
    return cities


def distance_matrix(cities: list[dict[str, Any]]) -> list[list[float]]:
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = float(cities[i]["x"])
        yi = float(cities[i]["y"])
        for j in range(i + 1, n):
            edge = math.hypot(xi - float(cities[j]["x"]), yi - float(cities[j]["y"]))
            dist[i][j] = edge
            dist[j][i] = edge
    return dist


def tour_length(tour: list[int], dist: list[list[float]]) -> float:
    if len(tour) < 2:
        return 0.0
    total = 0.0
    for index in range(len(tour) - 1):
        total += dist[tour[index]][tour[index + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


def is_permutation(tour: Any, n: int | None = None) -> bool:
    if not isinstance(tour, list) or not tour:
        return False
    cities: list[int] = []
    for item in tour:
        if isinstance(item, bool) or not isinstance(item, int):
            return False
        cities.append(item)
    size = n if n is not None else len(cities)
    return len(cities) == size and sorted(cities) == list(range(size))


def canonical_tour(tour: list[Any]) -> list[Any]:
    """Rotate a cycle so city 0 is first and the direction is lexicographically smaller.

    Anything that is not a permutation of ``0 .. n-1`` is returned unchanged.
    """

    if not is_permutation(tour):
        return list(tour)
    cities = [int(item) for item in tour]
    start = cities.index(0)
    rotated = cities[start:] + cities[:start]
    reflected = [rotated[0]] + list(reversed(rotated[1:]))
    if reflected < rotated:
        return reflected
    return rotated


def _city_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def repair_tour(raw: Any, n: int) -> tuple[list[int], bool]:
    """Keep the first legal copy of each city, then append the missing ids in order."""

    kept: list[int] = []
    used: set[int] = set()
    if isinstance(raw, list):
        for item in raw:
            city = _city_id(item)
            if city is None or city < 0 or city >= n or city in used:
                continue
            used.add(city)
            kept.append(city)
    changed = not (isinstance(raw, list) and list(raw) == kept and len(kept) == n)
    for city in range(n):
        if city not in used:
            kept.append(city)
            changed = True
    return kept, changed


def swap_move(tour: list[int], i: int, j: int) -> list[int]:
    child = list(tour)
    child[i], child[j] = child[j], child[i]
    return child


def two_opt_move(tour: list[int], i: int, j: int) -> list[int]:
    """Reverse the inclusive segment ``tour[i:j+1]``."""

    if i > j:
        i, j = j, i
    return tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]


def random_swap(tour: list[int], rng: random.Random) -> list[int]:
    i, j = rng.sample(range(len(tour)), 2)
    return swap_move(tour, i, j)


def random_two_opt(tour: list[int], rng: random.Random) -> list[int]:
    n = len(tour)
    i, j = sorted(rng.sample(range(n), 2))
    if j < i + 2:
        j = min(n - 1, i + 2)
    return two_opt_move(tour, i, j)


def two_opt_improve(
    tour: list[int],
    dist: list[list[float]],
    *,
    max_moves: int | None = None,
) -> tuple[list[int], bool]:
    """First-improvement 2-opt.

    ``max_moves`` stops after that many improving reversals. None runs until a
    local optimum or ``n * n`` moves. A full polish of a 15-city map collapses
    hundreds of starts onto two canonical cycles, so search uses a small limit.
    Reversing the whole cycle (the same tour walked backwards) is skipped.
    """

    best = [int(city) for city in tour]
    n = len(best)
    if n < 4:
        return best, False
    changed = False
    moves = 0
    limit = n * n if max_moves is None else max(0, int(max_moves))
    while moves < limit:
        improved = False
        for i in range(n - 1):
            # i == 0 stops before the closing reversal, which does not change the cycle.
            end = n - 1 if i == 0 else n
            a = best[i]
            b = best[i + 1]
            for j in range(i + 2, end):
                c = best[j]
                d = best[(j + 1) % n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-9:
                    best[i + 1 : j + 1] = reversed(best[i + 1 : j + 1])
                    changed = True
                    improved = True
                    moves += 1
                    break
            if improved:
                break
        if not improved:
            break
    return best, changed


def move_city_after(tour: list[int], city: int, after: int) -> list[int]:
    """Place ``city`` immediately after ``after``. A no-op when either id is absent."""

    child = list(tour)
    if city == after or city not in child or after not in child:
        return child
    child.remove(city)
    child.insert(child.index(after) + 1, city)
    return child


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text.lstrip("-").isdigit():
            return None
        return int(text)
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _edit_dict(payload: dict[str, Any]) -> dict[str, Any] | None:
    """A segment edit or operator choice, when the payload is not already a tour."""

    edit = payload.get("edit")
    if isinstance(edit, dict) and (edit.get("op") or edit.get("operator")):
        return edit
    if payload.get("op") or payload.get("operator"):
        return payload
    return None


def _parent_tours(payloads: list[dict[str, Any]], n: int) -> list[list[int]]:
    tours: list[list[int]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        tour = payload.get("tour")
        if is_permutation(tour, n):
            tours.append([int(city) for city in tour])
    return tours


def order_crossover(left: list[int], right: list[int], rng: random.Random) -> list[int]:
    """OX: copy a slice from the first parent and fill the rest in the second parent's order."""

    n = len(left)
    i, j = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[i : j + 1] = list(left[i : j + 1])
    used = set(left[i : j + 1])
    pos = (j + 1) % n
    for city in list(right[j + 1 :]) + list(right[: j + 1]):
        if city in used:
            continue
        while child[pos] is not None:
            pos = (pos + 1) % n
        child[pos] = city
        used.add(city)
        pos = (pos + 1) % n
    return [int(city) for city in child]


def edge_recombination(left: list[int], right: list[int], rng: random.Random) -> list[int]:
    """ERX: walk the union of undirected cycle edges, preferring the rarest edge."""

    n = len(left)
    edges: dict[int, set[int]] = {city: set() for city in range(n)}

    def add(tour: list[int]) -> None:
        for index, city in enumerate(tour):
            edges[city].add(tour[(index - 1) % n])
            edges[city].add(tour[(index + 1) % n])

    add(left)
    add(right)
    current = left[rng.randrange(n)]
    child = [current]
    used = {current}
    while len(child) < n:
        for pool in edges.values():
            pool.discard(current)
        options = [city for city in edges[current] if city not in used]
        if options:
            rarest = min(len(edges[city]) for city in options)
            tied = [city for city in options if len(edges[city]) == rarest]
            current = tied[rng.randrange(len(tied))]
        else:
            remaining = [city for city in range(n) if city not in used]
            current = remaining[rng.randrange(len(remaining))]
        child.append(current)
        used.add(current)
    return child


def held_karp(cities: list[dict[str, Any]]) -> tuple[list[int], float]:
    """Exact cycle. City 0 is fixed as the start so the DP covers each direction once.

    Returns the reconstructed tour and its closed Euclidean length.
    """

    n = len(cities)
    if n < 2:
        raise ValueError("Held-Karp needs at least 2 cities")
    dist = distance_matrix(cities)
    full = 1 << n
    inf = float("inf")
    dp = [inf] * (full * n)
    parent = [-1] * (full * n)
    dp[1 * n + 0] = 0.0
    for mask in range(full):
        if (mask & 1) == 0:
            continue
        base_index = mask * n
        for last in range(n):
            if (mask & (1 << last)) == 0:
                continue
            base = dp[base_index + last]
            if base == inf:
                continue
            row = dist[last]
            for nxt in range(n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                cand = base + row[nxt]
                idx = (mask | bit) * n + nxt
                if cand < dp[idx]:
                    dp[idx] = cand
                    parent[idx] = last
    complete = full - 1
    best = inf
    end = 1
    for last in range(1, n):
        cand = dp[complete * n + last] + dist[last][0]
        if cand < best:
            best = cand
            end = last
    tour_rev: list[int] = []
    mask = complete
    last = end
    while last != -1:
        tour_rev.append(last)
        prev = parent[mask * n + last]
        mask ^= 1 << last
        last = prev
        if mask == 0:
            break
    tour = list(reversed(tour_rev))
    if not is_permutation(tour, n):
        raise RuntimeError("Held-Karp reconstruction was not a permutation")
    length = tour_length(tour, dist)
    if not math.isclose(length, best, rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError(f"Held-Karp length {length} does not match the DP value {best}")
    return tour, length


def run_classic_ga(
    cities: list[dict[str, Any]],
    seed: int,
    *,
    population: int = CLASSIC_GA_POPULATION,
    generations: int = CLASSIC_GA_GENERATIONS,
) -> dict[str, Any]:
    """Seeded permutation GA. OX crossover, swap mutation, no model calls."""

    n = len(cities)
    dist = distance_matrix(cities)
    rng = random.Random(seed)
    pop = [rng.sample(range(n), n) for _ in range(population)]
    started = time.perf_counter()

    def best_tour() -> list[int]:
        return min(pop, key=lambda tour: tour_length(tour, dist))

    def point(gen: int) -> dict[str, Any]:
        tour = best_tour()
        return {
            "gen": gen,
            "best_length": round(tour_length(tour, dist), 6),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    convergence = [point(0)]
    for gen in range(1, generations + 1):
        ranked = sorted(pop, key=lambda tour: tour_length(tour, dist))
        children = [list(tour) for tour in ranked[:CLASSIC_GA_ELITE]]
        while len(children) < population:
            left = _tournament(ranked, dist, rng)
            right = _tournament(ranked, dist, rng)
            child = order_crossover(left, right, rng)
            if rng.random() < CLASSIC_GA_SWAP_RATE:
                child = random_swap(child, rng)
            children.append(child)
        pop = children
        convergence.append(point(gen))
    winner = best_tour()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "tour": winner,
        "length": tour_length(winner, dist),
        "time_ms": elapsed_ms,
        "convergence": convergence,
    }


def _tournament(ranked: list[list[int]], dist: list[list[float]], rng: random.Random) -> list[int]:
    picks = [ranked[rng.randrange(len(ranked))] for _ in range(CLASSIC_GA_TOURNAMENT)]
    return min(picks, key=lambda tour: tour_length(tour, dist))


def tsp_objective(adapter: TspAdapter) -> ObjectiveSpec:
    return ObjectiveSpec(
        domain=adapter.name,
        prompt=(
            "Find a short Hamiltonian cycle on these Euclidean cities. "
            "payload.tour must list every city id exactly once. "
            f"Cities: {adapter.coordinate_text()}."
        ),
        constraints={"n": adapter.n, "metric": "euclidean", "cities": adapter.cities},
        evaluation_version=adapter.evaluator_version,
    )


class TspAdapter(DomainAdapter):
    name = "tsp"
    evaluator_version = "tsp-euclid-v1"

    def __init__(self, seed: int = SHOWCASE_SEED) -> None:
        self.seed = seed
        self.cities = cities_for_seed(seed)
        self.n = len(self.cities)
        self.dist = distance_matrix(self.cities)
        self.trace: list[tuple[int, float, float]] = []
        self.started = time.perf_counter()

    def coordinate_text(self) -> str:
        return ", ".join(f"{city['id']}:({city['x']},{city['y']})" for city in self.cities)

    def mark_start(self) -> None:
        self.started = time.perf_counter()
        self.trace = []

    def genome_schema(self) -> dict[str, Any]:
        ids = list(range(self.n))
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["tour", "description"],
            "properties": {
                "tour": {
                    "type": "array",
                    "minItems": self.n,
                    "maxItems": self.n,
                    "uniqueItems": True,
                    "items": {"type": "integer", "enum": ids},
                },
                "description": {"type": "string"},
            },
        }

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        import hashlib

        packed = ",".join(f"{city['id']}:{city['x']}:{city['y']}" for city in self.cities)

        def digest(split: str) -> str:
            raw = f"{self.evaluator_version}|{split}|{self.seed}|{seed}|{packed}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {"train_hash": digest("train"), "validation_hash": digest("validation"), "test_hash": digest("test")}

    def family_of(self, payload: dict[str, Any]) -> str:
        del payload
        return "permutation"

    def realize_proposal(
        self,
        payload: dict[str, Any],
        parent_payloads: list[dict[str, Any]],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Apply a short edit to a parent tour. A full permutation is unchanged."""

        if not isinstance(payload, dict):
            return payload
        if is_permutation(payload.get("tour"), self.n):
            return payload
        edit = _edit_dict(payload)
        if edit is None:
            return payload
        parents = _parent_tours(parent_payloads, self.n)
        if not parents:
            return payload
        op = str(edit.get("op") or edit.get("operator") or "").strip().lower().replace("-", "_")
        base = list(parents[0])
        other = parents[1] if len(parents) > 1 else list(base)
        child: list[int] | None = None
        if op in {"reverse", "2_opt", "two_opt"}:
            i = _as_int(edit.get("i"))
            j = _as_int(edit.get("j"))
            if i is None or j is None or not (0 <= i < self.n and 0 <= j < self.n):
                return payload
            child = two_opt_move(base, i, j)
        elif op == "swap":
            i = _as_int(edit.get("i"))
            j = _as_int(edit.get("j"))
            if i is None or j is None or not (0 <= i < self.n and 0 <= j < self.n) or i == j:
                return payload
            child = swap_move(base, i, j)
        elif op == "move":
            city = _as_int(edit.get("city"))
            after = _as_int(edit.get("after"))
            if city is None or after is None:
                return payload
            child = move_city_after(base, city, after)
        elif op in {"ox", "order_crossover"}:
            child = order_crossover(base, other, rng)
        elif op in {"erx", "edge_recombination"}:
            child = edge_recombination(base, other, rng)
        else:
            return payload
        if not is_permutation(child, self.n):
            return payload
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            description = edit.get("description")
        if not isinstance(description, str) or not description.strip():
            description = f"{op} edit"
        return {"tour": child, "description": str(description)}

    def prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Repair an invalid city list, then a few 2-opt moves. ``validate`` does not call this.

        An operator edit without a tour is left alone. The controller applies it
        with ``realize_proposal`` before this runs again. Inventing a tour here
        would throw the edit away, because a strategist client prepares first.
        """

        notes: list[str] = []
        if not isinstance(payload, dict):
            return payload, notes
        if _edit_dict(payload) is not None and not is_permutation(payload.get("tour"), self.n):
            return payload, notes
        body = dict(payload)
        tour, changed = repair_tour(body.get("tour"), self.n)
        if changed:
            notes.append(f"tour repaired to a permutation of 0..{self.n - 1}")
            _LOG.info("tour repaired to a permutation of 0..%s", self.n - 1)
        before = tour_length(tour, self.dist)
        improved, shortened = two_opt_improve(tour, self.dist, max_moves=TSP_POLISH_MOVES)
        body["tour"] = improved
        if shortened:
            after = tour_length(improved, self.dist)
            notes.append(f"2-opt shortened the tour from {before:.4f} to {after:.4f}")
            _LOG.info("2-opt shortened the tour from %.4f to %.4f", before, after)
        description = body.get("description")
        if not isinstance(description, str) or not description.strip():
            body["description"] = "repaired permutation tour"
            notes.append("description defaulted")
            _LOG.info("description defaulted for a TSP tour")
        return body, notes

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        """Reject an illegal tour. Does not rewrite the payload."""

        del objective
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        extra = set(payload) - {"tour", "description"}
        if extra:
            raise ValueError(f"unknown payload keys: {sorted(extra)}")
        tour = payload.get("tour")
        if not is_permutation(tour, self.n):
            raise ValueError(
                f"tour must be a permutation of city ids 0..{self.n - 1} "
                "with each city exactly once"
            )
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")

    def oneshot_hint(self) -> str:
        return (
            f"payload.tour must be a permutation of city ids 0..{self.n - 1}. "
            "Each city appears once. Do not add a city and do not drop a city. "
            "An invalid tour is reported as invalid and is not repaired. "
            f"Coordinates: {self.coordinate_text()}."
        )

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return (
            "Return one permutation as JSON. No prose. "
            'payload is {"tour":[city ids 0..'
            f"{self.n - 1}"
            ' each once],"description":"one short line"}. '
            "hypothesis is one short line. "
            f"Cities: {self.coordinate_text()}.\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Return one short JSON edit of the parent tour. No prose and not the full tour. "
            'payload is one of {"edit":{"op":"reverse","i":2,"j":8}}, '
            '{"edit":{"op":"swap","i":1,"j":4}}, '
            '{"edit":{"op":"move","city":4,"after":9}}. '
            "i and j are indexes into the parent tour. city and after are city ids. "
            "hypothesis is one short line.\n"
            f"Parent tour: {parent.payload.get('tour')}\n"
            f"Train feedback: {feedback.feedback}\n"
            f"Train fitness: {feedback.fitness}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Return one short JSON operator choice. No prose. "
            'payload is {"edit":{"op":"ox"}} or {"edit":{"op":"erx"}} '
            "or {\"edit\":{\"op\":\"reverse\",\"i\":2,\"j\":8}}. "
            "hypothesis is one short line.\n"
            f"Parent A tour: {a.payload.get('tour')}\n"
            f"Parent B tour: {b.payload.get('tour')}"
        )

    def seeded_baselines(self) -> list[dict[str, Any]]:
        rng = random.Random(self.seed + 17)
        rows = []
        for name in ("seed_alpha", "seed_beta"):
            tour = list(range(self.n))
            rng.shuffle(tour)
            rows.append(
                {
                    "name": name,
                    "payload": {"tour": tour, "description": f"random permutation {name}"},
                    "hypothesis": f"seeded random tour {name}",
                }
            )
        return rows

    def uses_memetic_fallback(self) -> bool:
        return True

    def algorithmic_offspring(
        self,
        payloads: list[dict[str, Any]],
        rng: random.Random,
        count: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        """OX, ERX, swap, and 2-opt. A fresh random tour is mixed in so 2-opt does not collapse the pool."""

        tours: list[list[int]] = []
        for payload in payloads:
            tour = payload.get("tour") if isinstance(payload, dict) else None
            if is_permutation(tour, self.n):
                tours.append([int(city) for city in tour])
        if not tours:
            tours.append(rng.sample(range(self.n), self.n))
        operators = ("OX", "ERX", "swap", "2-opt")
        children: list[tuple[str, dict[str, Any]]] = []
        for index in range(max(0, count)):
            op = operators[index % len(operators)]
            left = tours[rng.randrange(len(tours))]
            if len(tours) == 1 or index % 5 == 4:
                right = rng.sample(range(self.n), self.n)
            else:
                right = tours[rng.randrange(len(tours))]
            if op == "OX":
                child = order_crossover(left, right, rng)
            elif op == "ERX":
                child = edge_recombination(left, right, rng)
            elif op == "swap":
                child = random_swap(left, rng)
            else:
                child = random_two_opt(left, rng)
            children.append((op, {"tour": child, "description": f"algorithmic {op}"}))
        return children

    def local_perturbations(self, payload: dict[str, Any], rng: random.Random, count: int) -> list[dict[str, Any]]:
        tour = payload.get("tour")
        if not is_permutation(tour, self.n):
            return []
        cities = [int(city) for city in tour]
        edits: list[dict[str, Any]] = []
        for index in range(count):
            if index % 2 == 0:
                child = random_swap(cities, rng)
                label = "swap"
            else:
                child = random_two_opt(cities, rng)
                label = "2-opt"
            edits.append({"tour": child, "description": f"{label} of an elite tour"})
        return edits

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        started = time.perf_counter()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {split}")
        try:
            self.validate(
                genome.payload,
                ObjectiveSpec(domain=self.name, prompt="", evaluation_version=self.evaluator_version),
            )
        except ValueError as exc:
            return Evaluation(
                genome_id=genome.id,
                feasible=False,
                fitness=None,
                metrics={},
                constraint_violations=[str(exc)],
                feedback=f"infeasible: {exc}",
                evaluator_version=self.evaluator_version,
                data_split=split,  # type: ignore[arg-type]
                elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
            )
        tour = [int(city) for city in genome.payload["tour"]]
        length = tour_length(tour, self.dist)
        self.trace.append((genome.generation, length, time.perf_counter()))
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=-length,
            metrics={"length": length, "n_cities": float(self.n)},
            constraint_violations=[],
            feedback=f"{split} closed tour length {length:.4f} on {self.n} Euclidean cities. Shorter is better.",
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    def observe_generation(self, generation: int, best_fitness: float | None) -> None:
        """Record the population best even when a generation accepts no new tour."""

        if isinstance(best_fitness, bool) or not isinstance(best_fitness, (int, float)):
            return
        self.trace.append((int(generation), -float(best_fitness), time.perf_counter()))

    def convergence(self) -> list[dict[str, Any]]:
        """Best tour length so far at the end of each generation, in wall-clock milliseconds."""

        if not self.trace:
            return []
        by_gen: dict[int, tuple[float, float]] = {}
        for generation, length, stamp in self.trace:
            previous = by_gen.get(generation)
            if previous is None:
                by_gen[generation] = (length, stamp)
                continue
            best_length, latest = previous
            by_gen[generation] = (min(best_length, length), max(latest, stamp))
        best = float("inf")
        points: list[dict[str, Any]] = []
        for generation in sorted(by_gen):
            length, stamp = by_gen[generation]
            best = min(best, length)
            points.append(
                {
                    "gen": generation,
                    "best_length": round(best, 6),
                    "elapsed_ms": max(0, int((stamp - self.started) * 1000)),
                }
            )
        return points

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]:
        return {
            "domain": self.name,
            "genome_id": genome.id,
            "tour": genome.payload.get("tour"),
            "feasible": evaluation.feasible,
            "fitness": evaluation.fitness,
            "metrics": evaluation.metrics,
            "split": evaluation.data_split,
            "hypothesis": genome.hypothesis,
        }
