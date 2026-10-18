#!/usr/bin/env python3
"""Run toy, cloud, or shopping on the shared evolution controller.

Examples:
  uv run python scripts/run_domain.py --domain cloud --mode mock --seed 42
  uv run python scripts/run_domain.py --domain shopping --mode mock --seed 42
  uv run python scripts/run_domain.py --domain toy --mode hybrid --seed 42
  uv run python scripts/run_domain.py --domain shopping --mode live --catalog-url https://example.com/catalog
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.providers.base import ProviderError  # noqa: E402
from geneai_core.runs import execute_domain_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeneAI domain run on the unchanged controller")
    parser.add_argument("--domain", choices=["cloud", "shopping", "toy", "tsp", "liquid"], required=True)
    parser.add_argument("--mode", choices=["mock", "hybrid", "live"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--strategist-calls", type=int, default=3)
    parser.add_argument(
        "--max-wall-seconds",
        type=int,
        default=900,
        help="Stop after this many seconds. The current generation still finishes its evaluations, and the champion is scored on validation.",
    )
    parser.add_argument("--catalog-url", default=None, help="Required for shopping --mode live. No retailer default.")
    parser.add_argument(
        "--catalog",
        default=None,
        help="fixture, recorded, or a path. recorded replays data/catalogs/shopping_recorded.json and does not call Browser Use.",
    )
    parser.add_argument("--out", default="runs")
    parser.add_argument("--skip-oneshot", action="store_true")
    return parser.parse_args()


def _credential_block(message: str) -> int:
    print(message)
    return 3


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        result = asyncio.run(
            execute_domain_run(
                domain=args.domain,
                mode=args.mode,
                seed=args.seed,
                population_size=args.population,
                generations=args.generations,
                elite_count=args.elite_count,
                strategist_calls_per_generation=args.strategist_calls,
                max_wall_seconds=args.max_wall_seconds,
                root=args.out,
                catalog_url=args.catalog_url,
                catalog=args.catalog,
                attach_oneshot=not args.skip_oneshot,
            )
        )
    except SystemExit as exc:
        if isinstance(exc.code, int):
            raise
        message = str(exc.code or exc)
        lowered = message.lower()
        if "blocked" in lowered or "missing environment" in lowered or "credentials" in lowered:
            raise SystemExit(_credential_block(message)) from None
        print(message)
        raise SystemExit(1) from None
    except ProviderError as exc:
        print(str(exc))
        raise SystemExit(1) from None
    summary = dict(result.summary)
    summary["export_path"] = result.export_path
    summary["events_path"] = result.events_path
    summary["status"] = result.status
    print(json.dumps(summary, indent=2, default=str))
    if result.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
