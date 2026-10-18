#!/usr/bin/env python3
"""Run the toy_optimization domain on the shared evolution controller.

Examples:
  uv run python scripts/run_toy.py --mode mock --seed 42 --population 12 --generations 5
  uv run python scripts/run_toy.py --mode hybrid --seed 42 --population 8 --generations 3
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

from geneai_core.experiments import execute_toy_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeneAI Phase 1 toy evolutionary run")
    parser.add_argument("--mode", choices=["mock", "hybrid"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=100)
    parser.add_argument("--max-evaluations", type=int, default=150)
    parser.add_argument("--out", default="runs")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    result = asyncio.run(
        execute_toy_run(
            mode=args.mode,
            seed=args.seed,
            population_size=args.population,
            generations=args.generations,
            elite_count=args.elite_count,
            max_model_calls=args.max_model_calls,
            max_evaluations=args.max_evaluations,
            root=args.out,
        )
    )
    summary = dict(result.summary)
    summary["export_path"] = result.export_path
    summary["events_path"] = result.events_path
    summary["status"] = result.status
    print(json.dumps(summary, indent=2, default=str))
    if result.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
