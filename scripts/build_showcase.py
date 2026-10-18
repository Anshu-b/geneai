#!/usr/bin/env python3
"""Write a geneai-showcase-v1 file for one held-out test case.

The case is the median champion-minus-best-baseline margin on the test split.
Numbers are re-scored by the domain evaluator. Nothing is sampled from a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geneai_core.showcase import build_showcase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GeneAI showcase JSON from an export")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    document = json.loads(args.export.read_text(encoding="utf-8"))
    showcase = build_showcase(document)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(showcase, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} case={showcase['case_id']} domain={showcase['domain']}")


if __name__ == "__main__":
    main()
