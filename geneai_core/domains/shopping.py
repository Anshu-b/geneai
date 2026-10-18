"""Shopping outfit evaluator.

The committed catalog is a fixture: product_url and image_url are null and
nothing is marked purchasable. A Browser Use collection can replace that
catalog only when it returns real http(s) URLs from a run. This module does
not invent retailer links.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

ROLES = ("top", "bottom", "layer", "shoes")
FAMILIES = ("neutral_layers", "campus_uniform", "bright_training")
COLORS = ("catalog_default", "black", "bone", "sage", "navy", "neon")
SIZES = ("S", "M", "L")
NEUTRAL = {"black", "bone", "sage", "navy"}
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "shopping_catalog.json"

# Disjoint shoppers. Train is larger. Validation and test use the same menu of
# palettes, occasions, and budgets, in a different mix, so a held-out score is
# not a copy of the training objective. Size stays M. Budgets stay at or above
# 260 so a mid-price outfit is legal everywhere; the soft score is what moves.
_OCCASION_MENU = (
    ("campus", "dinner"),
    ("dinner",),
    ("gym", "campus"),
    ("campus",),
    ("gym",),
    ("dinner", "gym"),
)
_BUDGET_MENU = (260, 300, 340, 300, 260, 340)
_PALETTE_MENU = ("neutral", "bright")


def _shopper(index: int) -> dict[str, Any]:
    return {
        "id": f"s{index:02d}",
        "palette": _PALETTE_MENU[index % 2],
        "occasions": list(_OCCASION_MENU[index % len(_OCCASION_MENU)]),
        "budget": _BUDGET_MENU[index % len(_BUDGET_MENU)],
        "size": "M",
    }


SHOPPERS = {
    "train": tuple(_shopper(index) for index in range(0, 6)),
    "validation": tuple(_shopper(index) for index in range(6, 9)),
    "test": tuple(_shopper(index) for index in range(9, 12)),
}


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    document = json.loads((path or FIXTURE_PATH).read_text(encoding="utf-8"))
    if document.get("source") not in {"fixture", "browser_use"}:
        raise ValueError("catalog source must be fixture or browser_use")
    return document


def _http_url(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith("https://") or value.startswith("http://"))


def products_from_browser_output(output: Any, *, run_id: str, session_id: str | None, model: str | None) -> dict[str, Any]:
    """Keep only items whose product_url was returned by the browser run."""

    raw_products = []
    if isinstance(output, dict):
        raw_products = output.get("products") or []
    elif isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            raw_products = parsed.get("products") or []
    kept: list[dict[str, Any]] = []
    dropped = 0
    for item in raw_products if isinstance(raw_products, list) else []:
        if not isinstance(item, dict) or not _http_url(item.get("product_url")):
            dropped += 1
            continue
        role = item.get("role") if item.get("role") in ROLES else "top"
        price = item.get("price")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or float(price) < 0:
            dropped += 1
            continue
        image = item.get("image_url") if _http_url(item.get("image_url")) else None
        availability = item.get("availability") if item.get("availability") in {"in_stock", "out_of_stock", "unknown"} else "unknown"
        kept.append(
            {
                "product_id": f"bu_{len(kept):03d}",
                "title": str(item.get("title") or "Untitled"),
                "price": float(price),
                "currency": str(item.get("currency") or "USD"),
                "product_url": item["product_url"],
                "image_url": image,
                "role": role,
                "colors": [str(color) for color in item.get("colors") or ["black"]][:6],
                "sizes": [str(size) for size in item.get("sizes") or ["M"]][:6],
                "style": ["neutral"],
                "occasions": ["campus", "dinner"],
                "availability": availability,
                "source_product_url": item["product_url"],
            }
        )
    return {
        "catalog_id": f"browser-use-{run_id}",
        "source": "browser_use",
        "captured_at": None,
        "browser_use_run_id": run_id,
        "browser_use_session_id": session_id,
        "browser_use_model": model,
        "dropped_without_url": dropped,
        "label": "Collected by Browser Use. Only rows with an http(s) product_url from that run are kept.",
        "products": kept,
    }


def _index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["product_id"]): item for item in catalog.get("products") or []}


def _chosen_color(product: dict[str, Any], requested: str, palette: str) -> str:
    colors = list(product.get("colors") or [])
    if requested != "catalog_default" and requested in colors:
        return requested
    if palette == "bright":
        for color in colors:
            if color not in NEUTRAL:
                return color
    for color in colors:
        if color in NEUTRAL:
            return color
    return colors[0] if colors else "unknown"


def _score_outfit(
    records: list[tuple[dict[str, Any], str, str]],
    *,
    budget: float,
    palette: str,
    occasions: list[str],
) -> tuple[float, dict[str, float], list[str]]:
    spend = sum(float(item["price"]) for item, _color, _size in records)
    if spend > budget + 1e-9:
        return 0.0, {}, [f"over budget {spend:.2f} > {budget:.2f}"]
    roles = [item["role"] for item, _color, _size in records]
    if len(roles) != len(set(roles)):
        return 0.0, {}, ["duplicate role"]
    if any(item.get("availability") == "out_of_stock" for item, _c, _s in records):
        return 0.0, {}, ["verified out of stock"]
    colors = [color for _item, color, _size in records]
    if palette == "neutral":
        color_hit = sum(1 for color in colors if color in NEUTRAL) / len(colors)
    elif palette == "bright":
        color_hit = sum(1 for color in colors if color not in NEUTRAL) / len(colors)
    else:
        color_hit = 0.5
    wanted_style = {"neutral", "relaxed"} if palette == "neutral" else {"bright", "athletic"}
    style_hits = []
    for item, _color, _size in records:
        tags = set(item.get("style") or [])
        style_hits.append(len(tags & wanted_style) / max(1, len(wanted_style)))
    style_match = sum(style_hits) / len(style_hits)
    occasion_hits = []
    wanted = set(occasions)
    for item, _color, _size in records:
        tags = set(item.get("occasions") or [])
        occasion_hits.append(len(tags & wanted) / max(1, len(wanted)))
    occasion_fit = sum(occasion_hits) / len(occasion_hits)
    known = 0
    for item, _color, _size in records:
        if item.get("availability") == "in_stock" and _http_url(item.get("product_url")):
            known += 1
        elif item.get("availability") == "unknown":
            known += 0
    availability = known / len(records)
    efficiency = 1.0 - (spend / budget) if budget else 0.0
    fitness = (
        0.30 * style_match
        + 0.25 * occasion_fit
        + 0.20 * color_hit
        + 0.15 * availability
        + 0.10 * efficiency
    )
    metrics = {
        "style_match": style_match,
        "occasion_fit": occasion_fit,
        "color_compatibility": color_hit,
        "availability_confidence": availability,
        "budget_efficiency": efficiency,
        "spend": spend,
        "subjective_rubric": 1.0,
    }
    return fitness, metrics, []


class ShoppingAdapter(DomainAdapter):
    name = "shopping"
    evaluator_version = "shopping-outfit-v2"

    def __init__(self, catalog: dict[str, Any] | None = None, constraints: dict[str, Any] | None = None) -> None:
        self.catalog = catalog or load_catalog()
        self._by_id = _index(self.catalog)
        self._objective_constraints = dict(constraints or {
            "budget": 300,
            "size": "M",
            "palette": "neutral",
            "occasions": ["campus", "dinner"],
        })

    def _item_branches(self) -> list[dict[str, Any]]:
        """One schema branch per catalog row so product_id and role stay paired."""

        required_size = self._objective_constraints.get("size")
        branches: list[dict[str, Any]] = []
        for product in self.catalog.get("products") or []:
            sizes = [size for size in product.get("sizes") or [] if size in SIZES]
            if required_size:
                if required_size not in sizes:
                    continue
                sizes = [str(required_size)]
            if not sizes:
                continue
            colors = ["catalog_default"]
            for color in product.get("colors") or []:
                if color in COLORS and color not in colors:
                    colors.append(color)
            branches.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["product_id", "color_id", "size", "role"],
                    "properties": {
                        "product_id": {"const": product["product_id"]},
                        "role": {"const": product["role"]},
                        "color_id": {"type": "string", "enum": colors},
                        "size": {"type": "string", "enum": sizes},
                    },
                }
            )
        return branches

    def genome_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["family", "items", "styling_notes"],
            "properties": {
                "family": {"type": "string", "enum": list(FAMILIES)},
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 3,
                    "items": {"anyOf": self._item_branches()},
                },
                "styling_notes": {"type": "string"},
            },
        }

    def dataset_hashes(self, seed: int) -> dict[str, str]:
        del seed
        catalog_id = str(self.catalog.get("catalog_id"))
        source = str(self.catalog.get("source"))

        def digest(split: str) -> str:
            shopper_ids = [shopper["id"] for shopper in SHOPPERS[split]]
            raw = f"{self.evaluator_version}|{catalog_id}|{source}|{split}|{shopper_ids}"
            import hashlib

            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return {"train_hash": digest("train"), "validation_hash": digest("validation"), "test_hash": digest("test")}

    def family_of(self, payload: dict[str, Any]) -> str:
        return str(payload.get("family") or "outfit")

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if payload.get("family") not in FAMILIES:
            raise ValueError("family is not allowed")
        items = payload.get("items")
        if not isinstance(items, list) or not 2 <= len(items) <= 3:
            raise ValueError("items must contain 2 or 3 products")
        notes = payload.get("styling_notes")
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError("styling_notes must be a non-empty string")
        seen: set[str] = set()
        roles: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("each item must be an object")
            product_id = item.get("product_id")
            product = self._by_id.get(str(product_id))
            if product is None:
                raise ValueError(f"unknown product_id {product_id}")
            if product_id in seen:
                raise ValueError("duplicate product")
            seen.add(str(product_id))
            if item.get("role") != product["role"]:
                raise ValueError(f"{product_id} role must be {product['role']}")
            if item.get("role") in roles:
                raise ValueError("duplicate role")
            roles.add(item["role"])
            if item.get("size") not in (product.get("sizes") or []):
                raise ValueError(f"{product_id} does not offer size {item.get('size')}")
            color = item.get("color_id")
            if color != "catalog_default" and color not in (product.get("colors") or []):
                raise ValueError(f"{product_id} does not offer color {color}")
            required_size = (objective.constraints or {}).get("size")
            if required_size and item.get("size") != required_size:
                raise ValueError(f"size must be {required_size}")

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return (
            "Propose an outfit JSON object with keys payload and hypothesis. "
            "payload.items has 2 or 3 catalog products with distinct roles. "
            "Use product_id values from the fixture catalog only. "
            "color_id may be catalog_default. Do not invent product URLs.\n"
            f"Catalog ids: {sorted(self._by_id)}\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        return (
            "Swap one product, color, or size. Keep ids inside the catalog. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent: {parent.payload}\nFeedback: {feedback.feedback}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Inherit items from both parents without repeating a role. "
            "Return JSON with keys payload and hypothesis.\n"
            f"Parent A: {a.payload}\nParent B: {b.payload}"
        )

    def _resolved(self, payload: dict[str, Any], palette: str) -> list[tuple[dict[str, Any], str, str]]:
        resolved: list[tuple[dict[str, Any], str, str]] = []
        for item in payload["items"]:
            product = self._by_id[item["product_id"]]
            color = _chosen_color(product, str(item["color_id"]), palette)
            resolved.append((product, color, str(item["size"])))
        return resolved

    def score_shopper(self, genome: Genome, shopper: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
        """Score one shopper. Used by split means and by tests of a single profile."""

        resolved = self._resolved(genome.payload, str(shopper["palette"]))
        fitness, metrics, violations = _score_outfit(
            resolved,
            budget=float(shopper["budget"]),
            palette=str(shopper["palette"]),
            occasions=list(shopper["occasions"]),
        )
        return fitness, metrics, violations

    def evaluate(self, genome: Genome, split: str) -> Evaluation:
        started = time.perf_counter()
        if split not in SHOPPERS:
            raise ValueError(f"unknown split {split}")
        objective = ObjectiveSpec(
            domain=self.name,
            prompt="",
            constraints=self._objective_constraints,
            evaluation_version=self.evaluator_version,
        )
        try:
            self.validate(genome.payload, objective)
        except ValueError as exc:
            return self._bad(genome, split, str(exc), started)
        totals: dict[str, float] = {}
        fitnesses: list[float] = []
        shoppers = SHOPPERS[split]
        for shopper in shoppers:
            fitness, metrics, violations = self.score_shopper(genome, shopper)
            if violations:
                return self._bad(genome, split, f"{shopper['id']}: {'; '.join(violations)}", started)
            fitnesses.append(fitness)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        count = float(len(shoppers))
        metrics = {key: value / count for key, value in totals.items()}
        resolved = self._resolved(genome.payload, "neutral")
        purchasable = all(
            _http_url(item.get("product_url")) and item.get("availability") == "in_stock"
            for item, _c, _s in resolved
        )
        metrics["purchasable"] = 1.0 if purchasable else 0.0
        metrics["shoppers"] = count
        fitness = sum(fitnesses) / count
        return Evaluation(
            genome_id=genome.id,
            feasible=True,
            fitness=fitness,
            metrics=metrics,
            feedback=(
                f"{split} mean subjective fitness={fitness:.4f} over {int(count)} shoppers "
                f"spend={metrics['spend']:.2f}. Catalog source={self.catalog.get('source')}. "
                "Scores are not comparable to cloud milliseconds. "
                + ("Purchasable links were verified." if purchasable else "Not purchasable: missing verified URL or stock.")
            ),
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    def bind_objective(self, objective: ObjectiveSpec) -> None:
        self._objective_constraints = dict(objective.constraints or {})

    def summarize(self, genome: Genome, evaluation: Evaluation) -> dict[str, Any]:
        rows = []
        for item in genome.payload.get("items") or []:
            product = self._by_id.get(item.get("product_id"))
            if not product:
                continue
            rows.append(
                {
                    "product_id": product["product_id"],
                    "title": product["title"],
                    "price": product["price"],
                    "currency": product["currency"],
                    "product_url": product.get("product_url"),
                    "image_url": product.get("image_url"),
                    "availability": product.get("availability"),
                    "source": self.catalog.get("source"),
                    "purchasable": bool(_http_url(product.get("product_url")) and product.get("availability") == "in_stock"),
                }
            )
        return {
            "domain": self.name,
            "catalog_id": self.catalog.get("catalog_id"),
            "catalog_source": self.catalog.get("source"),
            "catalog_label": self.catalog.get("label"),
            "feasible": evaluation.feasible,
            "fitness": evaluation.fitness,
            "metrics": evaluation.metrics,
            "items": rows,
            "subjective": True,
        }

    def _bad(self, genome: Genome, split: str, reason: str, started: float) -> Evaluation:
        return Evaluation(
            genome_id=genome.id,
            feasible=False,
            fitness=None,
            metrics={},
            constraint_violations=[reason],
            feedback=f"infeasible: {reason}",
            evaluator_version=self.evaluator_version,
            data_split=split,  # type: ignore[arg-type]
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )


def shopping_objective(*, palette: str = "neutral", budget: float = 300.0, occasions: list[str] | None = None) -> ObjectiveSpec:
    del palette, budget, occasions
    return ObjectiveSpec(
        domain="shopping",
        prompt=(
            "Build a 2-3 piece outfit from the committed catalog, size M. "
            "Train, validation, and test are disjoint shopper profiles. "
            "Fitness on a split is the mean subjective score across that split's shoppers. "
            "Fixture rows have no product URL and are not purchasable."
        ),
        constraints={"size": "M", "currency": "USD"},
        evaluation_version="shopping-outfit-v2",
    )


def neutral_outfit() -> dict[str, Any]:
    return {
        "family": "neutral_layers",
        "items": [
            {"product_id": "fx_top_01", "color_id": "black", "size": "M", "role": "top"},
            {"product_id": "fx_bottom_01", "color_id": "sage", "size": "M", "role": "bottom"},
            {"product_id": "fx_layer_01", "color_id": "bone", "size": "M", "role": "layer"},
        ],
        "styling_notes": "neutral campus to dinner",
    }


def gym_outfit() -> dict[str, Any]:
    return {
        "family": "bright_training",
        "items": [
            {"product_id": "fx_top_03", "color_id": "neon", "size": "M", "role": "top"},
            {"product_id": "fx_bottom_03", "color_id": "black", "size": "M", "role": "bottom"},
        ],
        "styling_notes": "bright gym set",
    }
