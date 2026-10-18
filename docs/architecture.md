# Architecture

GeneAI has one evolutionary controller. A domain adapter is a plugin. Switching
domains does not fork the controller, the operators, or the lineage schema.

```text
ObjectiveSpec + EvolutionConfig
        |
        v
DomainRegistry.get(name) -> DomainAdapter
        |
        v
EvolutionController.evolve(adapter, providers, repository)
        |
        +-- model provider: Mock or Veri (seeds, immigrants)
        +-- workflow provider: Mock or CREAO GeneAI Mutator (mutation, crossover, one repair)
        |
        v
RunRepository  events.jsonl + export.json (schema 1.0)
```

The controller does not import toy, shopping, cloud, quant, Veri, or CREAO.
`scripts/run_domain.py` registers `toy_optimization`, `cloud_scheduling`, or
`shopping` and passes that adapter in. Hybrid mode is still CREAO for the
strategist slots and Veri for bulk mutation.

## Loop

Generation 0 inserts any hand-written baselines the adapter declares, labeled
`provider=baseline` and `seeded_baseline=true`, then fills the rest from the
model provider. Generations 1..N:

1. Diversity-aware tournaments of 3, with a small novelty bonus and a family cap.
2. Keep `elite_count` genomes unchanged (same id, same payload hash).
3. Fill the remaining slots with mutation, crossover, and random immigrants.
   Each generation sends at most `strategist_calls_per_generation` (default 3)
   variation calls to the workflow provider, crossover first and then mutations,
   and runs that batch concurrently. The model provider handles bulk mutation
   and immigrants. Genome events record `provider` and `provider_role`
   (`strategist`, `bulk`, `seed`, or `immigrant`).
4. Reject invalid payloads. One repair attempt, then drop the child.
5. Reject duplicate canonical payload hashes.
6. Prune losers out of the living population. They stay in the event log.
7. Checkpoint and emit `generation.completed`.

After the loop, finalists are chosen from train scores with a family cap, scored
once on validation, and the winner is frozen. That champion is evaluated once
on the held-out test split. The test split is not read before `champion.frozen`.
On the toy domain, train, validation, and test are 32 disjoint shifts from the
same [-0.8, 0.8]^2 distribution. Test is not a harder basin.

Stopping conditions: generation limit, model-call cap (Veri and CREAO both
count), evaluation cap, wall clock, estimated USD, or cancel. A stop returns
the partial run and still writes the export.

## Lineage

Every stored genome has `parent_ids`, `operator`, `content_hash`, and metadata
`family` / `species`. Seeds and immigrants have no parents. Crossover has two.
Events use the section 13.2 names. Sequence numbers are a dense 1..N series.
The export's `integrity_sha256` covers every field except itself.

`run.mode` is the mode you asked for (`mock`, `hybrid`, or `live`). It is never
relabeled `recorded`. Provider calls carry `provider_mode` so a mock call
cannot be shown as sponsor execution.

Cloud policies are a JSON score: weights, tie-breaks, and up to two per-job-class
overrides. The v3 pool is heterogeneous (CPU, high-memory, GPU) and the
objective adds hard-deadline tardiness and energy to the blueprint terms.
Shopping train, validation, and test are disjoint shoppers. Every
genome event and the export record `provider` (`mock`, `veri`, `creao`, or
`baseline`). The export schema is still 1.0. `one_shot_baseline` is either a
skipped record or one frontier attempt. Quant is not implemented. The website
is Phase 3.
