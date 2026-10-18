# GeneAI

GeneAI evolves a population of candidate strategies. One controller runs every
domain: a shifted Rastrigin toy, a deterministic cloud scheduler, and an outfit
evaluator. The website is a later phase.

## Run a domain

```bash
uv sync
uv run pytest -q
uv run python scripts/run_domain.py --domain toy --mode mock --seed 42
uv run python scripts/run_domain.py --domain cloud --mode mock --seed 42
uv run python scripts/run_domain.py --domain shopping --mode mock --seed 42
uv run python scripts/tsp_benchmark.py --out tsp_benchmark.json --mode mock
```

The TSP benchmark is a 15-city Euclidean tour. `mock` writes `geneai-tsp-v1`
with Held-Karp, a classic OX/swap GA, an uncalled frontier one-shot, and
GeneAI. `hybrid` uses `VERI_*`, `CREAO_*`, and `FRONTIER_*` and exits 3 if
any of those credentials are missing. Held-Karp is the traditional method and
the optimum. There is no nearest-neighbor arm. GeneAI uses LLM-guided
operators plus 2-opt local search: 20 showcase generations in about 150s, and
8 generations per held-out map in about 60s. The classic GA is unchanged.

Mock mode makes no network calls. The evaluator is real: the cloud simulator,
or the committed fixture catalog for shopping. The process writes
`runs/<run_id>/events.jsonl`, `checkpoint.json`, `export.json`, and
`summary.json`. The export is the Phase 3 replay document (schema 1.0). It
keeps `mode: mock` and attaches `one_shot_baseline`. Without frontier
credentials that arm is `skipped`. The comparison is one attempt versus
evolutionary search, and it is not compute-matched.

The default mock run is population 24 and 12 generations. Cloud generation 0
includes the four hand-written scheduling baselines. The scheduler is
`cloud-scheduler-v3`: CPU, high-memory, and GPU machines, hard deadlines, and
energy charged while a machine is busy. Shopping train, validation,
and test are disjoint shoppers, so the three scores are not the same number.
The test split is unread until the champion is frozen. Shopping fitness is a
subjective rubric. It is not comparable to cloud completion time.

The fixture catalog (`data/fixtures/shopping_catalog.json`) is labeled
`source: fixture`. Product URLs are null. Nothing in it is purchasable.

Check an export:

```bash
uv run python scripts/export_replay.py runs/<run_id>/export.json
```

## Sponsor calls

Copy `.env.example` to `.env`.

Veri needs `VERI_API_KEY`. The deployment id and model default to
`bfe84d51e141` and `geneai-qwen7b`
(`https://api.veri.studio/v1/deployments/bfe84d51e141`). That deployment is
Qwen/Qwen2.5-7B-Instruct on vLLM, 1x L4-24GB, concurrency 8, about $1.05/hr.
The client caps in-flight calls at 8 and retries 429/503 while a cold start wakes.

CREAO needs a `capi_` key in `CREAO_API_KEY`. The GeneAI Mutator app id is
`CREAO_APP_ID=0ed64fee-5003-4d56-a83b-c5f369d7a39c`. Calls go to
`https://agent.creao.ai/api/v1/apps/{appId}/runs` with `inputs.task`,
`inputs.parents`, and `inputs.operator` (`mutate` or `crossover`). A `cr_sk_`
key is the wrong path for this account and is refused before any request.

```bash
uv run python scripts/smoke_veri.py
uv run python scripts/smoke_creao.py
uv run python scripts/run_toy.py --mode hybrid --seed 42 --population 8 --generations 3
```

If a variable is missing, hybrid and live exit 3 and print the setup. They do
not substitute a mock response and label it live.

Live shopping also needs `BROWSER_USE_API_KEY` and `--catalog-url` pointing at
a public page you may read. The client uses Browser Use API v2 tasks
(`POST /api/v2/tasks`, header `X-Browser-Use-API-Key`), does not log in or pay,
and drops any row that has no http(s) product URL. A successful collection is
written to `data/catalogs/shopping_recorded.json` with `source: recorded`.
It does not default to a retailer. Quant is not in this tree.

Hybrid runs call the model provider concurrently (default 8, `VERI_CONCURRENCY`).
Seed calls ask for several distinct genomes and skip hashes already kept.
`--max-wall-seconds` defaults to 900. A wall-clock stop still finishes the
current generation's evaluations and scores the champion on validation before
freezing it. Each finished generation prints one progress line.

## Layout

`geneai_core/controller.py` is the engine. Adapters live in
`geneai_core/domains/`. Providers live in `geneai_core/providers/`. See
`AGENTS.md` and `docs/architecture.md`.
