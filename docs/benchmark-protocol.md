# Benchmark protocol

Two arms, same adapter, same schema, same held-out test split.

Arm A, frontier one-shot: one `FrontierProvider.propose` call. No repair loop.
Invalid output stays invalid. The record is `one_shot_baseline` on the export.

Arm B, GeneAI: the unchanged controller. Finalists are ranked on train, scored
on validation, and the validation winner is frozen. Test is evaluated once
after that. Cloud generation 0 includes the four hand-written DSL baselines.
Cloud fitness is `-(completion_ratio + 0.25*sla_miss_rate + 0.50*hard_tardiness_ratio + 0.15*energy_ratio)`.

The comparison label is `single attempt vs evolutionary search`.
`compute_matched` is always false. GeneAI scores more candidates, so a higher
test fitness is not a compute-matched win.

Cost is the provider's reported USD when a response includes it, otherwise
null. A missing `FRONTIER_API_KEY`, `FRONTIER_BASE_URL`, or `FRONTIER_MODEL`
writes `one_shot_baseline.status=skipped` and sends no request. The runner
does not invent a vendor URL.

`scripts/run_domain.py` attaches that record after the evolutionary export is
signed, then recomputes `integrity_sha256`. Phase 3 can read the same schema
1.0 document.
