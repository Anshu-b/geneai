"""Shopping outfit evaluator.

The committed catalog is a fixture: product_url and image_url are null and
nothing is marked purchasable. A Browser Use collection can replace that
catalog only when it returns real http(s) URLs from a run. This module does
not invent retailer links.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from geneai_core.domains.base import DomainAdapter
from geneai_core.schemas import Evaluation, Genome, ObjectiveSpec

ROLES = ("top", "bottom", "layer", "shoes", "accessory")
FAMILIES = ("neutral_layers", "campus_uniform", "bright_training")
COLORS = ("catalog_default", "black", "bone", "sage", "navy", "neon")
SIZES = ("S", "M", "L")
NEUTRAL = {"black", "bone", "sage", "navy"}
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "shopping_catalog.json"
RECORDED_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalogs" / "shopping_recorded.json"

# Disjoint shoppers. Train is larger. Validation and test use the same menu of
# palettes, occasions, and budgets, in a different mix, so a held-out score is
# not a copy of the training objective. Apparel size stays M. Shoe size is the
# same number on every profile so one genome can fit the split. Budgets stay
# at or above 260 so a mid-price outfit is legal everywhere; the soft score is
# what moves.
APPAREL_ROLES = ("top", "bottom", "layer")
REQUIRED_ROLES = ("top", "bottom", "shoes")
DEFAULT_SHOE_SIZE = 9
_LOG = logging.getLogger("geneai_core.shopping")
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
        "shoe_size": DEFAULT_SHOE_SIZE,
    }


SHOPPERS = {
    "train": tuple(_shopper(index) for index in range(0, 6)),
    "validation": tuple(_shopper(index) for index in range(6, 9)),
    "test": tuple(_shopper(index) for index in range(9, 12)),
}


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    document = json.loads((path or FIXTURE_PATH).read_text(encoding="utf-8"))
    if document.get("source") not in {"fixture", "browser_use", "recorded"}:
        raise ValueError("catalog source must be fixture, browser_use, or recorded")
    return document


def write_recorded_catalog(catalog: dict[str, Any], path: Path | None = None) -> Path:
    """Cache a real Browser Use collection. Does not invent rows."""

    if catalog.get("source") != "browser_use":
        raise ValueError("only a Browser Use collection can be cached as recorded")
    products = catalog.get("products") or []
    if not products:
        raise ValueError("refusing to cache an empty catalog")
    target = path or RECORDED_CATALOG_PATH
    document = dict(catalog)
    document["source"] = "recorded"
    document["label"] = "recorded"
    document["mode"] = "recorded"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return target


def _catalog_role(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in ROLES:
        return text
    return "accessory"


def _tag_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("/", ",").split(",")
        return [part.strip().lower() for part in parts if part.strip()]
    if isinstance(value, list):
        return [str(part).strip().lower() for part in value if str(part).strip()]
    return []


def _style_tags(item: dict[str, Any]) -> list[str]:
    tags = _tag_list(item.get("style") or item.get("styles") or item.get("style_tags"))
    if tags:
        return tags[:6]
    title = str(item.get("title") or "").lower()
    colors = [str(color).lower() for color in item.get("colors") or []]
    found: list[str] = []
    if any(word in title for word in ("train", "gym", "run", "sport", "track")):
        found.append("athletic")
    if any(color in {"neon", "red", "orange", "pink"} for color in colors):
        found.append("bright")
    if any(color in NEUTRAL for color in colors) or any(word in title for word in ("silk", "wool", "knit", "oxford")):
        found.append("neutral")
    return found[:6] or ["unspecified"]


def _occasion_tags(item: dict[str, Any]) -> list[str]:
    tags = _tag_list(item.get("occasions") or item.get("occasion"))
    if tags:
        return tags[:6]
    title = str(item.get("title") or "").lower()
    style = set(_style_tags(item))
    found: list[str] = []
    if "gym" in title or "athletic" in style:
        found.append("gym")
    if any(word in title for word in ("silk", "wool", "dinner", "blouse")):
        found.append("dinner")
    if any(word in title for word in ("campus", "oxford", "denim")):
        found.append("campus")
    return found[:6] or ["unspecified"]


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
        role = _catalog_role(item.get("role"))
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
                "retailer": str(item["retailer"]).strip() if isinstance(item.get("retailer"), str) and str(item.get("retailer")).strip() else None,
                "role": role,
                "colors": [str(color) for color in item.get("colors") or ["black"]][:6],
                "sizes": [str(size) for size in item.get("sizes") or ["M"]][:6],
                "style": _style_tags(item),
                "occasions": _occasion_tags(item),
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


def outfit_why(
    records: list[tuple[dict[str, Any], str, str]],
    *,
    budget: float,
    palette: str,
    occasions: list[str],
    metrics: dict[str, float],
    violations: list[str],
) -> list[str]:
    """Bullets from the same components ``_score_outfit`` uses. Not a second rubric."""

    if violations:
        return [str(item) for item in violations]
    count = len(records)
    if palette == "neutral":
        color_hits = sum(1 for _item, color, _size in records if color in NEUTRAL)
        color_line = f"Neutral colors: {color_hits}/{count} items"
    elif palette == "bright":
        color_hits = sum(1 for _item, color, _size in records if color not in NEUTRAL)
        color_line = f"Bright colors: {color_hits}/{count} items"
    else:
        color_line = f"Color compatibility {metrics.get('color_compatibility', 0):.2f}"
    wanted_style = {"neutral", "relaxed"} if palette == "neutral" else {"bright", "athletic"}
    style_hits = 0
    for item, _color, _size in records:
        tags = set(item.get("style") or [])
        if tags & wanted_style:
            style_hits += 1
    occasion_hits = 0
    wanted = set(occasions)
    for item, _color, _size in records:
        if set(item.get("occasions") or []) & wanted:
            occasion_hits += 1
    spend = float(metrics.get("spend") or 0.0)
    under = budget - spend
    budget_line = (
        f"Under budget by ${under:.2f}" if under >= -1e-9 else f"Over budget by ${-under:.2f}"
    )
    return [
        f"Style match {metrics.get('style_match', 0):.2f}: {style_hits}/{count} items share {palette} tags {sorted(wanted_style)}",
        f"Occasion fit {metrics.get('occasion_fit', 0):.2f}: {occasion_hits}/{count} items match {list(occasions)}",
        f"{color_line} (component {metrics.get('color_compatibility', 0):.2f})",
        f"Availability confidence {metrics.get('availability_confidence', 0):.2f}",
        f"{budget_line} (spend ${spend:.2f} of ${budget:.2f}, efficiency {metrics.get('budget_efficiency', 0):.2f})",
    ]


class ShoppingAdapter(DomainAdapter):
    name = "shopping"
    evaluator_version = "shopping-outfit-v3"

    def __init__(self, catalog: dict[str, Any] | None = None, constraints: dict[str, Any] | None = None) -> None:
        self.catalog = catalog or load_catalog()
        self._by_id = _index(self.catalog)
        self._objective_constraints = dict(constraints or {
            "budget": 300,
            "size": "M",
            "shoe_size": DEFAULT_SHOE_SIZE,
            "palette": "neutral",
            "occasions": ["campus", "dinner"],
        })

    def _catalog_roles(self) -> set[str]:
        found: set[str] = set()
        for product in self.catalog.get("products") or []:
            role = str(product.get("role") or "")
            if role in ROLES:
                found.add(role)
        return found

    def _required_roles(self) -> list[str]:
        present = self._catalog_roles()
        return [role for role in REQUIRED_ROLES if role in present]

    def _item_branches(self) -> list[dict[str, Any]]:
        """One schema branch per catalog row. Shoes and accessories stay in the schema."""

        apparel_size = self._objective_constraints.get("size")
        shoe_size = self._objective_constraints.get("shoe_size")
        branches: list[dict[str, Any]] = []
        for product in self.catalog.get("products") or []:
            role = str(product.get("role") or "")
            if role not in ROLES:
                continue
            sizes = _schema_sizes(product, apparel_size=apparel_size, shoe_size=shoe_size)
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
                    "minItems": self._min_items(),
                    "maxItems": self._max_items(),
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

    def _min_items(self) -> int:
        required = len(self._required_roles())
        return required if required else 1

    def _max_items(self) -> int:
        present = self._catalog_roles()
        count = len(self._required_roles())
        if "layer" in present and "top" in present:
            count += 1
        if "accessory" in present:
            count += 1
        return max(count, self._min_items())

    def family_of(self, payload: dict[str, Any]) -> str:
        return str(payload.get("family") or "outfit")

    def prepare_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Correct roles from the catalog and map an unknown family onto the enum."""

        notes: list[str] = []
        if not isinstance(payload, dict):
            return payload, notes
        body = dict(payload)
        family = body.get("family")
        if family not in FAMILIES:
            mapped = _closest_family(family)
            notes.append(f"family {family!r} normalized to {mapped}")
            _LOG.info("family %r normalized to %s", family, mapped)
            body["family"] = mapped
        items = body.get("items")
        if not isinstance(items, list):
            return body, notes
        corrected: list[Any] = []
        for item in items:
            if not isinstance(item, dict):
                corrected.append(item)
                continue
            row = dict(item)
            product = self._by_id.get(str(row.get("product_id")))
            if product is not None and row.get("role") != product["role"]:
                notes.append(
                    f"{row.get('product_id')} role {row.get('role')!r} corrected to {product['role']}"
                )
                row["role"] = product["role"]
            if product is not None and product.get("role") == "accessory":
                if str(row.get("size") or "").strip().lower() == "one size" and row.get("size") != "One Size":
                    notes.append(f"{row.get('product_id')} size normalized to One Size")
                    row["size"] = "One Size"
            if product is not None and product.get("role") == "shoes" and _is_number(row.get("size")):
                text = _size_text(row.get("size"))
                if text != row.get("size"):
                    notes.append(f"{row.get('product_id')} shoe size normalized to {text}")
                    row["size"] = text
            corrected.append(row)
        body["items"] = corrected
        return body, notes

    def validate(self, payload: dict[str, Any], objective: ObjectiveSpec) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        prepared, _notes = self.prepare_payload(payload)
        payload.clear()
        payload.update(prepared)
        if payload.get("family") not in FAMILIES:
            raise ValueError("family is not allowed")
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("items must contain a product for each required role")
        notes = payload.get("styling_notes")
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError("styling_notes must be a non-empty string")
        seen: set[str] = set()
        roles: list[str] = []
        constraints = objective.constraints or {}
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
            roles.append(str(item["role"]))
            _check_item_size(item, product, constraints)
            color = item.get("color_id")
            if color != "catalog_default" and color not in (product.get("colors") or []):
                raise ValueError(f"{product_id} does not offer color {color}")
        problem = _completeness_problem(roles, self._catalog_roles())
        if problem:
            raise ValueError(problem)

    def seeded_baselines(self) -> list[dict[str, Any]]:
        outfits = _hand_baselines(
            self.catalog,
            budget=_baseline_budget(),
            apparel_size=str(self._objective_constraints.get("size") or "M"),
            shoe_size=self._objective_constraints.get("shoe_size", DEFAULT_SHOE_SIZE),
        )
        return [
            {
                "name": name,
                "payload": payload,
                "hypothesis": f"seeded baseline {name}",
            }
            for name, payload in outfits
        ]

    def baselines(self) -> dict[str, dict[str, Any]]:
        return {spec["name"]: spec["payload"] for spec in self.seeded_baselines()}

    def oneshot_hint(self) -> str:
        return (
            "Build one complete outfit: exactly one top, one bottom, and one pair of shoes "
            "when the catalog has those roles. A layer is optional and only with a top. "
            "At most one accessory. One item per role. Never repeat a role. "
            "A repeated role is invalid and will be reported as invalid."
        )

    def _prompt_contract(self) -> str:
        example = _example_child(self.catalog)
        return (
            f"Allowed family values: {', '.join(FAMILIES)}. "
            "Use one of those strings exactly. "
            "A valid outfit has exactly one top, one bottom, and one pair of shoes when the catalog "
            "includes that role. A layer is optional and only together with a top. "
            "At most one accessory. One item per role. Never repeat a role. "
            "Tops, bottoms, and layers use the shopper apparel size. "
            f"Shoes use numeric shoe size {self._objective_constraints.get('shoe_size', DEFAULT_SHOE_SIZE)}. "
            "Accessories accept One Size. "
            "Example child: "
            + json.dumps({"genome": example, "rationale": "complete outfit"}, separators=(",", ":"))
        )

    def seed_instructions(self, objective: ObjectiveSpec) -> str:
        return (
            "Propose an outfit JSON object with keys payload and hypothesis. "
            "One item per role. Never repeat a role. "
            "Use product_id values from the catalog only. "
            "color_id may be catalog_default. Do not invent product URLs. "
            "hypothesis is one line. "
            f"{self._prompt_contract()}\n"
            f"Catalog ids: {sorted(self._by_id)}\n"
            f"Objective: {objective.prompt}"
        )

    def mutate_instructions(self, parent: Genome, feedback: Evaluation) -> str:
        move = _mutation_move(parent.payload)
        return (
            f"{move} Keep ids inside the catalog. "
            "One item per role. role must match that product's catalog role. "
            f"{self._prompt_contract()}\n"
            'Return JSON exactly as {"children":[{"genome": <one payload object>, "rationale": "one line"}]}.\n'
            f"Parent: {parent.payload}\nFeedback: {feedback.feedback}"
        )

    def crossover_instructions(self, a: Genome, b: Genome) -> str:
        return (
            "Inherit items from both parents without repeating a role. "
            "Keep a complete outfit: one top, one bottom, and one pair of shoes when those roles exist. "
            "One item per role. role must match that product's catalog role. "
            f"{self._prompt_contract()}\n"
            'Return JSON exactly as {"children":[{"genome": <one payload object>, "rationale": "one line"}]}.\n'
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
            "Build one complete outfit from the catalog. Apparel size is M. Shoe size is 9. "
            "Include one top, one bottom, and one pair of shoes when the catalog has that role. "
            "A layer is optional. At most one accessory. One item per role. "
            "Train, validation, and test are disjoint shopper profiles. "
            "Fitness on a split is the mean subjective score across that split's shoppers. "
            "Fixture rows have no product URL and are not purchasable."
        ),
        constraints={"size": "M", "shoe_size": DEFAULT_SHOE_SIZE, "currency": "USD"},
        evaluation_version="shopping-outfit-v3",
    )


def _baseline_budget() -> float:
    return min(float(shopper["budget"]) for split in SHOPPERS.values() for shopper in split)


def _hand_baselines(
    catalog: dict[str, Any],
    *,
    budget: float | None = None,
    apparel_size: str = "M",
    shoe_size: Any = DEFAULT_SHOE_SIZE,
) -> list[tuple[str, dict[str, Any]]]:
    """Cheapest complete outfit under the budget, one per style. Roles the catalog lacks are skipped."""

    limit = _baseline_budget() if budget is None else float(budget)
    by_role: dict[str, list[dict[str, Any]]] = {}
    present: set[str] = set()
    for product in catalog.get("products") or []:
        if isinstance(product, dict) and product.get("role") in ROLES:
            role = str(product["role"])
            by_role.setdefault(role, []).append(product)
            present.add(role)
    required = [role for role in REQUIRED_ROLES if role in present]
    optional = [role for role in ("layer", "accessory") if role in present and not (role == "layer" and "top" not in present)]
    outfits: list[tuple[str, dict[str, Any]]] = []
    plans = (("neutral_outfit", "neutral"), ("gym_outfit", "gym"))
    for name, kind in plans:
        items = _budget_items(
            by_role,
            required=required,
            optional=optional,
            kind=kind,
            budget=limit,
            apparel_size=apparel_size,
            shoe_size=shoe_size,
        )
        if len(items) < len(required) or not items:
            continue
        family = "neutral_layers" if name == "neutral_outfit" else "bright_training"
        outfits.append(
            (
                name,
                {
                    "family": family,
                    "items": items,
                    "styling_notes": f"hand baseline {name} from catalog {catalog.get('catalog_id')}",
                },
            )
        )
    return outfits


def _example_child(catalog: dict[str, Any]) -> dict[str, Any]:
    outfits = _hand_baselines(catalog)
    if outfits:
        return outfits[0][1]
    return {
        "family": "neutral_layers",
        "items": [],
        "styling_notes": "complete outfit",
    }


def _closest_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    compact = text.replace(" ", "_")
    if compact in FAMILIES:
        return compact
    if any(word in text for word in ("gym", "bright", "train", "sport", "athletic")):
        return "bright_training"
    if any(word in text for word in ("campus", "uniform", "school")):
        return "campus_uniform"
    return "neutral_layers"


_MUTATION_MOVES = (
    "Swap only the shoes for a different catalog shoe at the numeric shoe size.",
    "Swap only the bottom for a different catalog bottom at the apparel size.",
    "Swap only the top. Keep exactly one top. A layer is optional and only with that top.",
    "Change one color_id to another color that product offers, or catalog_default.",
    "Add one accessory if the outfit has none, or remove the accessory if it has one. Do not add a second accessory.",
    "Add one layer if the outfit has a top and no layer, or remove the layer. Do not drop the top.",
)


def _mutation_move(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    import hashlib

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return _MUTATION_MOVES[int(digest[:8], 16) % len(_MUTATION_MOVES)]


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _size_text(value: Any) -> str:
    if _is_number(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _numbers_match(offered: Any, required: Any) -> bool:
    if _is_number(offered) or _is_number(required):
        try:
            return float(offered) == float(required)
        except (TypeError, ValueError):
            return False
    return str(offered) == str(required)


def _offered_sizes(product: dict[str, Any]) -> list[str]:
    return [str(size) for size in product.get("sizes") or []]


def _schema_sizes(product: dict[str, Any], *, apparel_size: Any, shoe_size: Any) -> list[str]:
    role = str(product.get("role") or "")
    raw = _offered_sizes(product)
    if role in APPAREL_ROLES:
        sizes = [size for size in raw if size in SIZES]
        if apparel_size:
            sizes = [size for size in sizes if size == str(apparel_size)]
        return sizes
    if role == "shoes":
        if shoe_size is not None:
            matched = [size for size in raw if _numbers_match(size, shoe_size)]
            if matched:
                return matched
        return raw
    sizes = list(raw)
    if "One Size" not in sizes:
        sizes.append("One Size")
    return sizes


def _size_for(product: dict[str, Any], *, apparel_size: str, shoe_size: Any) -> str | None:
    role = str(product.get("role") or "")
    sizes = _schema_sizes(product, apparel_size=apparel_size, shoe_size=shoe_size)
    if role == "shoes" and not sizes:
        sizes = _offered_sizes(product)
    if not sizes:
        return None
    if role == "accessory" and "One Size" in sizes:
        return "One Size"
    return sizes[0]


def _product_price(product: dict[str, Any]) -> float:
    price = product.get("price")
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return 0.0
    return float(price)


def _style_prefers(product: dict[str, Any], kind: str) -> bool:
    style = set(_style_tags(product))
    occasions = set(_occasion_tags(product))
    if kind == "neutral":
        return bool(style & {"neutral", "relaxed", "tailored"}) or bool(occasions & {"campus", "dinner"})
    return bool(style & {"bright", "athletic"}) or "gym" in occasions


def _pick_product(
    choices: list[dict[str, Any]],
    *,
    kind: str,
    apparel_size: str,
    shoe_size: Any,
    spend: float,
    budget: float,
    required: bool,
) -> dict[str, Any] | None:
    sized = [product for product in choices if _size_for(product, apparel_size=apparel_size, shoe_size=shoe_size)]
    if not sized:
        return None
    ranked = sorted(sized, key=lambda product: (_product_price(product), str(product.get("product_id"))))
    preferred = [product for product in ranked if _style_prefers(product, kind)]
    pool = preferred or ranked
    for product in pool:
        if spend + _product_price(product) <= budget + 1e-9:
            return product
    for product in ranked:
        if spend + _product_price(product) <= budget + 1e-9:
            return product
    if required:
        return ranked[0]
    return None


def _budget_items(
    by_role: dict[str, list[dict[str, Any]]],
    *,
    required: list[str],
    optional: list[str],
    kind: str,
    budget: float,
    apparel_size: str,
    shoe_size: Any,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    spend = 0.0
    for role in list(required) + list(optional):
        picked = _pick_product(
            by_role.get(role) or [],
            kind=kind,
            apparel_size=apparel_size,
            shoe_size=shoe_size,
            spend=spend,
            budget=budget,
            required=role in required,
        )
        if picked is None:
            continue
        price = _product_price(picked)
        if role not in required and spend + price > budget + 1e-9:
            continue
        size = _size_for(picked, apparel_size=apparel_size, shoe_size=shoe_size)
        if size is None:
            continue
        items.append(_item_from_product(picked, size=size))
        spend += price
    return items


def _item_from_product(product: dict[str, Any], *, size: str | None = None) -> dict[str, Any]:
    colors = list(product.get("colors") or [])
    color = "catalog_default" if "catalog_default" in colors or not colors else str(colors[0])
    chosen = size if size is not None else (_offered_sizes(product) or ["M"])[0]
    return {"product_id": product["product_id"], "color_id": color, "size": chosen, "role": product["role"]}


def _check_item_size(item: dict[str, Any], product: dict[str, Any], constraints: dict[str, Any]) -> None:
    role = str(item.get("role") or product.get("role") or "")
    size = item.get("size")
    offered = _offered_sizes(product)
    product_id = product.get("product_id")
    if role == "accessory" and str(size) == "One Size":
        return
    if not any(_numbers_match(size, candidate) or str(size) == candidate for candidate in offered):
        raise ValueError(f"{product_id} does not offer size {size}")
    if role in APPAREL_ROLES:
        required = constraints.get("size")
        if required and str(size) != str(required):
            raise ValueError(f"size must be {required}")
        return
    if role == "shoes" and constraints.get("shoe_size") is not None:
        if not _numbers_match(size, constraints.get("shoe_size")):
            raise ValueError(f"shoe size must be {constraints.get('shoe_size')}")


def _completeness_problem(roles: list[str], catalog_roles: set[str]) -> str | None:
    counts: dict[str, int] = {}
    for role in roles:
        counts[role] = counts.get(role, 0) + 1
    if any(count > 1 for count in counts.values()):
        return "duplicate role"
    required = [role for role in REQUIRED_ROLES if role in catalog_roles]
    for role in required:
        if counts.get(role, 0) != 1:
            return f"outfit must include exactly one {role}"
    if counts.get("layer", 0) > 1:
        return "at most one layer"
    if counts.get("layer", 0) and "top" in catalog_roles and counts.get("top", 0) != 1:
        return "a layer requires a top"
    if counts.get("accessory", 0) > 1:
        return "at most one accessory"
    allowed = set(required)
    if "layer" in catalog_roles:
        allowed.add("layer")
    if "accessory" in catalog_roles:
        allowed.add("accessory")
    extra = [role for role in counts if role not in allowed]
    if extra:
        return f"role not in catalog outfit: {extra[0]}"
    return None


def neutral_outfit() -> dict[str, Any]:
    outfits = {name: payload for name, payload in _hand_baselines(load_catalog())}
    return outfits["neutral_outfit"]


def gym_outfit() -> dict[str, Any]:
    outfits = {name: payload for name, payload in _hand_baselines(load_catalog())}
    return outfits["gym_outfit"]
