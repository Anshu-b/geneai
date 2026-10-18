# AGENTS.md

GeneAI is one domain-agnostic evolutionary controller. Domain demos are
plugins. Do not fork the engine per demo.

## Status

Implemented: typed contracts, registry, the shared controller, toy_optimization,
cloud_scheduling, shopping (fixture catalog plus Browser Use v4 client),
tournament selection, mutation, crossover, elitism, immigrants, repair, dedup,
pruning, budgets, lineage JSONL, section 13.3 export, Mock providers,
VeriProvider, CreaoProvider (app-scoped GeneAI Mutator), FrontierProvider
one-shot arm, and tests.

Not implemented: quant, API, UI. Quant stays skipped until it can be added
without crowding the two primary domains.

## Rules

- Do not invent sponsor endpoints, prices, metrics, or browser actions.
- CREAO for this account is only `https://agent.creao.ai/api/v1/apps/{CREAO_APP_ID}`
  with a `capi_` key and body field `inputs`. Never send `agent_id` / `input`
  to that host. Never send `inputs` to `developer.creao.ai/v1`.
- The Mutator input is `{task, parents:[{genome, fitness}], operator: mutate|crossover}`.
  The answer is `result.chatText` (`{children:[{genome, rationale}]}`), never
  `result.text` (that field echoes the inputs).
- CREAO is a strategist: default 3 workflow calls per generation (crossover,
  then a mutation or two), run concurrently. Veri does bulk mutation.
- `CREAO_APP_ID` for this project is `0ed64fee-5003-4d56-a83b-c5f369d7a39c`.
- Veri base URL is `https://api.veri.studio/v1/deployments/bfe84d51e141`.
  The OpenAI model field is `geneai-qwen7b`. Cap in-flight calls at 8 and
  retry 429/503 (`deployment_waking`) with Retry-After.
- Mock output stays `provider_mode=mock`. Do not relabel it live or recorded.
- The controller must not import a concrete domain or sponsor client.
- Secrets stay in the environment. `.env.example` has empty secrets and the
  public CREAO app id and Veri deployment id.

## Commands

```bash
uv sync
uv run pytest -q
uv run python scripts/run_domain.py --domain toy --mode mock --seed 42
uv run python scripts/run_domain.py --domain cloud --mode mock --seed 42
uv run python scripts/run_domain.py --domain shopping --mode mock --seed 42
uv run python scripts/smoke_veri.py
uv run python scripts/smoke_creao.py
uv run python scripts/run_domain.py --domain cloud --mode hybrid --seed 42 --population 8 --generations 3
```

Shopping `--mode live` also needs `BROWSER_USE_API_KEY` and `--catalog-url`.
The fixture catalog is not a stand-in for that call.

Smoke scripts exit 3 when credentials are missing and print the setup. They do
not fake a call. Hybrid mode exits 3 rather than silently falling back to mock.
