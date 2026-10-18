# Benchmark protocol (not implemented)

Phase 2 will compare two arms on the same adapter, schema, and held-out split.
This repository does not run that comparison yet. Exports set
`one_shot_baseline` to null.

Arm A, frontier one-shot: one `FrontierProvider` request, no repair loop, same
validator and evaluator. Invalid output stays invalid.

Arm B, GeneAI: the Phase 1 controller, champion frozen on train and validation,
then one test evaluation.

Report quality, feasibility, call counts, wall time, and cost. Cost is actual
usage when the provider returns it, otherwise null. Do not describe the
comparison as compute-matched: GeneAI evaluates more candidates.

`FrontierProvider` is the client Arm A will call. It does not pick a vendor
URL on its own.
