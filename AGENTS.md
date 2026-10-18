# AGENTS.md

GeneAI is one domain-agnostic evolutionary controller. Domain demos are
plugins. Do not fork the engine per demo.

## Phase 1 status

Implemented: typed contracts, registry, toy_optimization, tournament selection,
mutation, crossover, elitism, immigrants, repair, dedup, pruning, budgets,
lineage JSONL, section 13.3 export, Mock providers, VeriProvider, CreaoProvider
(app-scoped GeneAI Mutator), FrontierProvider (unused by the loop), smoke
scripts, and tests.

Not implemented: shopping, cloud, quant, Browser Use, one-shot runner, API, UI.

## Rules

- Do not invent sponsor endpoints, prices, metrics, or browser actions.
- CREAO for this account is only `https://agent.creao.ai/api/v1/apps/{CREAO_APP_ID}`
  with a `capi_` key and body field `inputs`. Never send `agent_id` / `input`
  to that host. Never send `inputs` to `developer.creao.ai/v1`.
- The Mutator input is `{task, parents:[{genome, fitness}], operator: mutate|crossover}`.
  The output is `{children:[{genome, rationale}]}`.
- `CREAO_APP_ID` for this project is `0ed64fee-5003-4d56-a83b-c5f369d7a39c`.
- Veri base URL is `https://api.veri.studio/v1/deployments/{VERI_DEPLOYMENT_ID}`.
- Mock output stays `provider_mode=mock`. Do not relabel it live or recorded.
- The controller must not import a concrete domain or sponsor client.
- Secrets stay in the environment. `.env.example` has empty secrets and the
  public app id.

## Commands

```bash
uv sync
uv run pytest -q
uv run python scripts/run_toy.py --mode mock --seed 42 --population 12 --generations 5
uv run python scripts/smoke_veri.py
uv run python scripts/smoke_creao.py
uv run python scripts/run_toy.py --mode hybrid --seed 42 --population 8 --generations 3
```

Smoke scripts exit 3 when credentials are missing and print the setup. They do
not fake a call. Hybrid mode exits 3 rather than silently falling back to mock.
