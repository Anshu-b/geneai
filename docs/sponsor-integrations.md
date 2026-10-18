# Sponsor integrations (Phase 1)

Checked 2026-10-18. No sponsor response in this environment was invented.
Credentials were not present here, so live smoke tests are blocked until
`scripts/smoke_veri.py` and `scripts/smoke_creao.py` are run where the keys exist.

## veri.studio

Docs: https://docs.veri.studio/quickstart and
https://docs.veri.studio/deployments/openai-compat

`VERI_API_KEY` is the only secret required for a live call. When the other
variables are unset, `VeriProvider.from_env` uses the deployment created
externally (HTTP 201):

```text
base_url = https://api.veri.studio/v1/deployments/bfe84d51e141
model    = geneai-qwen7b
```

`model` is the deployment name, which the OpenAI client sends as the model
field. The served weights are `Qwen/Qwen2.5-7B-Instruct` on vLLM, 1x L4-24GB,
`concurrency_target` 8, about $1.05/hr. That hourly figure is deployment
billing. `cost_usd` on a proposal stays null unless a chat response actually
carries a price.

The SDK appends `/chat/completions`. A base URL that already ends in
`/chat/completions` is rejected. The SDK's own retries are disabled
(`max_retries=0`) because they give up before a GPU cold start.
`response_format: json_object` is requested once; a 400/422 retries without it.
HTTP 429 and 503 retry until `VERI_WAKE_TIMEOUT_SECONDS` (default 180), sleeping
`Retry-After` or the body's `retry_after_seconds` (default 30s on a wake, each
sleep capped at 60s). A 503 with no code is treated as `deployment_waking`.
HTTP 408 retries a few times with a short backoff. In-flight chat completions
are capped at 8 (`VERI_CONCURRENCY` overrides that). The engine's
`max_concurrent_model_calls` default is the same 8. Token usage is stored when
the response includes it.

Phase 1 role: seed genomes and random immigrants. Fitness still comes from the
domain evaluator.

## CREAO — app-scoped API Trigger only

This account's key starts with `capi_` and is rejected by `developer.creao.ai/v1`.
`CreaoProvider` never sends CLI bodies (`agent_id`, `input`, `webhook_url`).

Public routes (https://docs.creao.ai/developer-api/api-paths and
https://docs.creao.ai/pro/api-trigger):

```text
POST https://agent.creao.ai/api/v1/apps/{appId}/runs
GET  https://agent.creao.ai/api/v1/apps/{appId}/runs/{runId}
GET  https://agent.creao.ai/api/v1/apps/{appId}/runs
Authorization: Bearer capi_...
```

Create body: `{"inputs": {...}}` plus optional `webhookUrl`.
Terminal poll statuses: `completed`, `failed`, `cancelled`.
Create payloads use `id`. The answer is `result.chatText` (a fenced JSON
block, then prose). `result.text` echoes the inputs, including a stringified
`input` field, and is not parsed. If chat text has no valid child, the client
GETs a `children.json` artifact URL on `agent.creao.ai` with the same Bearer
token. Failures read `errorCode`. A live completed run
(`0897c113-58b0-43bf-8a7c-e56f705efa24`) took about 44s
(`startedAt`/`completedAt`). Provenance stores `threadId`, the run id, and
that server latency.

`CREAO_API_KEY` is the only secret required for a live call. When
`CREAO_APP_ID` is unset, `CreaoProvider.from_env` uses
`0ed64fee-5003-4d56-a83b-c5f369d7a39c`.

Live list observation, made outside this environment, for that app:

```json
{"runs": [], "limit": 20, "offset": 0}
```

### GeneAI Mutator contract

That app is the GeneAI Mutator. The wire `inputs` object is exactly:

```json
{
  "task": "instruction text",
  "parents": [{"genome": {}, "fitness": 0.0}],
  "operator": "mutate"
}
```

`operator` is `mutate` or `crossover` only. Engine repairs are sent as
`mutate` with the invalid genome as the parent. The expected `result.chatText` is:

```json
{"children": [{"genome": {}, "rationale": "why"}]}
```

Parsing extracts fenced JSON and the first balanced object, accepts `genome`
or `payload`, validates against the domain schema, clamps out-of-range numbers,
case-folds enums, and drops keys the schema forbids. It does not invent missing
required fields. If the child is still illegal, the provider raises and the
controller may spend one repair call. Credits are recorded when present and are
not converted to USD.

The mutator has no seed operator. Initial genomes and random immigrants go
through the model provider. Mutation and crossover go through CREAO in hybrid
mode. If `result.chatText` is empty, an `application/json` artifact is read
instead (`children` in the name is preferred; `geneai_mutate_result.json` is
accepted). Artifact downloads follow redirects (the URL 302s to object storage)
and treat any non-2xx response as a failure. `result.text` is still ignored.
The parser accepts `{children:[{genome}]}`, `{payload: ...}`, a bare list, or
one genome object.

Veri completions use `VERI_TIMEOUT_SECONDS` (default 120). The L4 deployment
generates about 16 tokens per second, so a 30 second cap timed out. Each call
asks for at most two genomes and about 400 tokens per genome. A repair call
uses about 1400 tokens and `response_format=json_object` when the deployment
accepts it. Replies are parsed from the first balanced JSON object.

`UnverifiedCreaoCliProvider` raises on construction. It is not wired to the
controller or the smoke script.

## Frontier

`FrontierProvider` is the one-shot client (`FRONTIER_API_KEY`, `FRONTIER_BASE_URL`,
`FRONTIER_MODEL`). gpt-5.5 and other gpt-5 / o-series models use the Responses
API (`max_output_tokens`, `store: false`, no temperature). Other models use
Chat Completions with `max_completion_tokens`. Evolution does not call it.
`scripts/run_domain.py` makes one request after the champion is frozen.
Missing variables skip the arm. The comparison is not compute-matched.
Verified settings: `FRONTIER_BASE_URL=https://api.openai.com/v1`,
`FRONTIER_MODEL=gpt-5.5`.

## Browser Use

Live shopping catalog collection uses Browser Use Cloud API v2 tasks. The key
was verified with `GET /api/v2/billing/account` and header `X-Browser-Use-API-Key`.

```text
POST https://api.browser-use.com/api/v2/tasks
GET  https://api.browser-use.com/api/v2/tasks/{task_id}/status
GET  https://api.browser-use.com/api/v2/tasks/{task_id}
Header: X-Browser-Use-API-Key
```

`structuredOutput` is a stringified JSON schema asking for title, price,
product URL, image URL, retailer, and availability. `startUrl` is the
`--catalog-url`. A finished task's `output` string is parsed. A successful
collection is cached at `data/catalogs/shopping_recorded.json` with
`source`, `label`, and `mode` set to `recorded`. That file is not a fixture
and is not committed.

The task forbids login, cart, and payment, and asks for 30 to 40 products
covering tops, bottoms, shoes, and accessories. It stays on the given site,
opens each product page, and copies `product_url` and `image_url` from that
page. There is no default host.
`BROWSER_USE_TIMEOUT_SECONDS` defaults to 420. On timeout the client PATCHes
`/api/v2/tasks/{task_id}` with `{"action":"stop"}` and logs a stop failure.
A product row is kept only when the run returns an `http` or `https`
`product_url` and the image URL answers 200 with `Content-Type: image/*`.
Rows that fail either check are dropped. Counts are `image_verified_count`
and `image_dropped_count`. An empty collection fails. The fixture catalog is
not substituted and then called live. `--catalog fixture` still loads the
fixture and the split note keeps the fixture badge.
`BROWSER_USE_API_KEY` plus `--catalog-url` are both required.

The committed file `data/fixtures/shopping_catalog.json` has `source: fixture`,
null `product_url` and `image_url`, and `availability: unknown`. Those rows
are not purchasable and are not retailer listings.

This is outfit search, not a Visa payment integration. Do not describe a
product link as secured by Visa.

## Modes

| mode | providers | evaluator | label |
| --- | --- | --- | --- |
| mock | deterministic local | domain simulator or fixture catalog | `provider_mode=mock` |
| hybrid | real Veri bulk + real CREAO strategist | same evaluator | run mode `hybrid`, calls `live` |
| live | same providers as hybrid | shopping also requires a Browser Use catalog | not a silent fixture fallback |
| replay | none | none | Phase 3 playback of an export |
