# GeneAI

GeneAI evolves a population of candidate strategies. One controller runs every
domain. Phase 1 is the headless engine: selection, mutation, crossover,
elitism, a lineage log, and the Veri and CREAO clients.

Shopping, cloud scheduling, and the website are later phases. They are not in
this tree.

## Run the toy domain

```bash
uv sync
uv run pytest -q
uv run python scripts/run_toy.py --mode mock --seed 42 --population 12 --generations 5
```

Mock mode makes no network calls. The process writes `runs/<run_id>/events.jsonl`,
`checkpoint.json`, `export.json`, and `summary.json`. The export is the Phase 3
replay document (schema 1.0) and carries `mode: mock`.

Check an export:

```bash
uv run python scripts/export_replay.py runs/<run_id>/export.json
```

## Sponsor calls

Copy `.env.example` to `.env`.

Veri needs `VERI_API_KEY`, `VERI_DEPLOYMENT_ID`, and `VERI_MODEL`. The client
uses `https://api.veri.studio/v1/deployments/{id}` as the OpenAI base URL.

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

If a variable is missing, the script exits 3 and prints the setup. It does not
substitute a mock response and label it live.

## Layout

`geneai_core/controller.py` is the engine. `geneai_core/domains/toy.py` is the
only adapter. Providers live in `geneai_core/providers/`. See `AGENTS.md` and
`docs/architecture.md`.
