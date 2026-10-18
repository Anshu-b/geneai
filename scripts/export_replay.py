#!/usr/bin/env python3
"""Validate a Phase 1 run export and print lineage counts. Does not relabel mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.store import verify_export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a GeneAI replay export")
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    document = json.loads(args.export.read_text(encoding="utf-8"))
    ok = verify_export(document)
    genomes = document.get("genomes") or []
    edges = sum(len(item.get("parent_ids") or []) for item in genomes)
    print(
        json.dumps(
            {
                "valid_integrity": ok,
                "mode": (document.get("run") or {}).get("mode"),
                "run_id": (document.get("run") or {}).get("id"),
                "events": len(document.get("events") or []),
                "genomes": len(genomes),
                "parent_edges": edges,
                "champion_id": document.get("champion_id"),
                "one_shot_baseline": document.get("one_shot_baseline"),
            },
            indent=2,
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
