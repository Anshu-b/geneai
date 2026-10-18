# Architecture (Phase 1)

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
`scripts/run_toy.py` registers `toy_optimization` and passes it in.

## Loop

Generation 0 seeds a population on the train split. Generations 1..N:

1. Diversity-aware tournaments of 3, with a small novelty bonus and a family cap.
2. Keep `elite_count` genomes unchanged (same id, same payload hash).
3. Fill the remaining slots with mutation, crossover, and random immigrants.
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

`run.mode` is `mock` or `hybrid` in Phase 1. It is never relabeled `recorded`
or `live`. Provider calls carry `provider_mode` so a mock call cannot be shown
as sponsor execution.

## Not in this phase

Shopping, cloud, quant, Browser Use, the one-shot runner, and the website are
not implemented. `FrontierProvider` is only a client for that later baseline.
