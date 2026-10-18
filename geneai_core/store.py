"""Append-only run store: JSONL events, checkpoints, and section 13.3 exports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from geneai_core.diversity import canonical_json, sha256_text
from geneai_core.events import make_event, utc_now
from geneai_core.schemas import (
    SCHEMA_VERSION,
    Evaluation,
    EvolutionConfig,
    Genome,
    LineageEvent,
    ObjectiveSpec,
    Provenance,
)


class LineageError(ValueError):
    pass


def git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_integrity(document: dict[str, Any]) -> str:
    body = {key: value for key, value in document.items() if key != "integrity_sha256"}
    return sha256_text(canonical_json(body))


def verify_export(document: dict[str, Any]) -> bool:
    claimed = document.get("integrity_sha256")
    if not isinstance(claimed, str) or not claimed:
        return False
    return export_integrity(document) == claimed


class RunRepository:
    """One evolutionary run. Events are immutable and sequence numbers are dense."""

    def __init__(self, root: Path | str, run_id: str | None = None) -> None:
        self.run_id = run_id or f"run_{uuid4().hex[:12]}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.checkpoint_path = self.dir / "checkpoint.json"
        self.export_path = self.dir / "export.json"
        self.summary_path = self.dir / "summary.json"
        self.events: list[LineageEvent] = []
        self.genomes: dict[str, Genome] = {}
        self.evaluations: list[Evaluation] = []
        self._sequence = 0
        self._genome_counter = 0
        self.objective: ObjectiveSpec | None = None
        self.config: EvolutionConfig | None = None
        self.mode: str = "mock"
        self.datasets: dict[str, str] = {}
        self.providers_meta: list[dict[str, Any]] = []
        self.git_sha: str | None = None
        self.status: str = "created"
        self.stop_reason: str | None = None
        self.champion_id: str | None = None
        self.created_at: str = utc_now()
        self.summary: dict[str, Any] = {}
        self.catalog: dict[str, Any] | None = None
        self._handle = self.events_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def next_genome_id(self) -> str:
        self._genome_counter += 1
        return f"g{self._genome_counter:04d}"

    def bind(
        self,
        *,
        objective: ObjectiveSpec,
        config: EvolutionConfig,
        mode: str,
        datasets: dict[str, str],
        providers_meta: list[dict[str, Any]],
    ) -> None:
        self.objective = objective
        self.config = config
        self.mode = mode
        self.datasets = datasets
        self.providers_meta = providers_meta
        self.git_sha = git_sha()

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        provenance: Provenance,
    ) -> LineageEvent:
        self._sequence += 1
        if self.events and self._sequence != self.events[-1].sequence + 1:
            raise LineageError("event sequence is not monotonic")
        event = make_event(
            run_id=self.run_id,
            sequence=self._sequence,
            event_type=event_type,
            payload=payload,
            provenance=provenance,
        )
        self.events.append(event)
        self._handle.write(event.model_dump_json() + "\n")
        self._handle.flush()
        return event

    def add_genome(self, genome: Genome) -> None:
        if genome.id in self.genomes:
            raise LineageError(f"duplicate genome id {genome.id}")
        if genome.run_id != self.run_id:
            raise LineageError("genome run_id does not match this run")
        seen: set[str] = set()
        for parent_id in genome.parent_ids:
            if parent_id not in self.genomes:
                raise LineageError(f"parent {parent_id} does not exist")
            parent = self.genomes[parent_id]
            if parent.generation >= genome.generation:
                raise LineageError(
                    f"parent {parent_id} generation {parent.generation} "
                    f"is not older than child {genome.id} generation {genome.generation}"
                )
            self._assert_no_cycle(parent_id, seen, stack=set())
        self.genomes[genome.id] = genome

    def _assert_no_cycle(self, genome_id: str, seen: set[str], stack: set[str]) -> None:
        if genome_id in stack:
            raise LineageError(f"lineage cycle at {genome_id}")
        if genome_id in seen:
            return
        stack.add(genome_id)
        genome = self.genomes[genome_id]
        for parent_id in genome.parent_ids:
            self._assert_no_cycle(parent_id, seen, stack)
        stack.remove(genome_id)
        seen.add(genome_id)

    def add_evaluation(self, evaluation: Evaluation) -> None:
        if evaluation.genome_id not in self.genomes:
            raise LineageError(f"evaluation for unknown genome {evaluation.genome_id}")
        self.evaluations.append(evaluation)

    def checkpoint(self, payload: dict[str, Any]) -> None:
        document = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **payload}
        self.checkpoint_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    def read_jsonl(self) -> list[dict[str, Any]]:
        self._handle.flush()
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def build_export(self, *, recorded_at: str | None = None) -> dict[str, Any]:
        if self.objective is None or self.config is None:
            raise LineageError("run is not bound")
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "id": self.run_id,
                "domain": self.objective.domain,
                "mode": self.mode,
                "seed": self.config.seed,
                "status": self.status,
                "stop_reason": self.stop_reason,
                "git_sha": self.git_sha,
            },
            "objective": self.objective.model_dump(mode="json"),
            "config": self.config.model_dump(mode="json"),
            "datasets": {
                "train_hash": self.datasets.get("train_hash", ""),
                "validation_hash": self.datasets.get("validation_hash", ""),
                "test_hash": self.datasets.get("test_hash", ""),
            },
            "providers": self.providers_meta,
            "genomes": [self.genomes[key].model_dump(mode="json") for key in sorted(self.genomes)],
            "evaluations": [item.model_dump(mode="json") for item in self.evaluations],
            "events": [item.model_dump(mode="json") for item in self.events],
            "champion_id": self.champion_id,
            "one_shot_baseline": None,
            "recorded_at": recorded_at or utc_now(),
        }
        if self.catalog is not None:
            document["catalog"] = self.catalog
        document["integrity_sha256"] = export_integrity(document)
        return document

    def finalize(self, *, status: str, stop_reason: str, champion_id: str | None, summary: dict[str, Any]) -> dict[str, Any]:
        self.status = status
        self.stop_reason = stop_reason
        self.champion_id = champion_id
        self.summary = summary
        document = self.build_export()
        if not verify_export(document):
            raise LineageError("export integrity check failed immediately after signing")
        self.export_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        self.close()
        return document
