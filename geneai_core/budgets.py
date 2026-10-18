"""Resource budgets. Hitting any cap stops new work and keeps partial results."""

from __future__ import annotations

import time

from geneai_core.schemas import BudgetSnapshot, EvolutionConfig


class Budget:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config
        self.model_calls = 0
        self.evaluations = 0
        self.spend_usd = 0.0
        self.wall_started = time.monotonic()
        self.started = self.wall_started
        self._clock = time.monotonic
        self.exhausted_reason: str | None = None
        self.cancelled = False

    def elapsed_seconds(self) -> float:
        """True wall time. A test clock does not replace this."""

        return time.monotonic() - self.wall_started

    def decision_elapsed(self) -> float:
        """Clock used for the wall cap. Tests may replace ``_clock``."""

        return self._clock() - self.started

    def remaining_seconds(self) -> float:
        return max(0.0, float(self.config.max_wall_seconds) - self.decision_elapsed())

    def _mark_wall(self) -> bool:
        if self.decision_elapsed() >= self.config.max_wall_seconds:
            self.exhausted_reason = self.exhausted_reason or "max_wall_seconds"
            return True
        return False

    def blocked(self) -> bool:
        if self.cancelled:
            self.exhausted_reason = "cancelled"
            return True
        return self._mark_wall() or self.exhausted_reason is not None

    def allows_model_call(self) -> bool:
        if self.cancelled:
            self.exhausted_reason = "cancelled"
            return False
        if self._mark_wall():
            return False
        if self.exhausted_reason in {"max_model_calls", "max_usd_estimate", "max_wall_seconds", "cancelled"}:
            return False
        if self.model_calls >= self.config.max_model_calls:
            self.exhausted_reason = "max_model_calls"
            return False
        if self.spend_usd >= self.config.max_usd_estimate:
            self.exhausted_reason = "max_usd_estimate"
            return False
        return True

    def allows_evaluation(self) -> bool:
        if self.cancelled:
            self.exhausted_reason = "cancelled"
            return False
        if self._mark_wall():
            return False
        if self.exhausted_reason in {"max_evaluations", "max_wall_seconds", "cancelled"}:
            return False
        if self.evaluations >= self.config.max_evaluations:
            self.exhausted_reason = "max_evaluations"
            return False
        return True

    def consume_model_call(self) -> None:
        self.model_calls += 1
        if self.model_calls >= self.config.max_model_calls:
            self.exhausted_reason = self.exhausted_reason or "max_model_calls"

    def note_cost(self, amount: float | None) -> None:
        if amount is None:
            return
        self.spend_usd += float(amount)
        if self.spend_usd >= self.config.max_usd_estimate:
            self.exhausted_reason = self.exhausted_reason or "max_usd_estimate"

    def consume_evaluation(self) -> None:
        self.evaluations += 1
        if self.evaluations >= self.config.max_evaluations:
            self.exhausted_reason = self.exhausted_reason or "max_evaluations"

    def cancel(self) -> None:
        self.cancelled = True
        self.exhausted_reason = "cancelled"

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            model_calls=self.model_calls,
            evaluations=self.evaluations,
            spend_usd=round(self.spend_usd, 8),
            elapsed_seconds=round(self.elapsed_seconds(), 6),
            exhausted_reason=self.exhausted_reason,
            cancelled=self.cancelled,
        )
