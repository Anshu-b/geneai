# Sponsor integrations (Phase 1)

Checked 2026-10-18. No sponsor response in this environment was invented.
Credentials were not present here, so live smoke tests are blocked until
`scripts/smoke_veri.py` and `scripts/smoke_creao.py` are run where the keys exist.

## veri.studio

Docs: https://docs.veri.studio/quickstart and
https://docs.veri.studio/deployments/openai-compat

`VeriProvider` constructs an OpenAI client with:

```text
base_url = https://api.veri.studio/v1/deployments/{VERI_DEPLOYMENT_ID}
api_key  = VERI_API_KEY
model    = VERI_MODEL
```

The SDK appends `/chat/completions`. A base URL that already ends in
`/chat/completions` is rejected. `response_format: json_object` is requested
once; a 400/422 retries without it. HTTP 429/503/408 retry with `Retry-After`
when present, bounded. Token usage is stored when the response includes it.
`cost_usd` stays null unless a response actually carries a price. Scaled-to-zero
deployments return 503 `deployment_waking`.

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
Create payloads use `id`. Text is read from `result.text` (`result.chatText`
when that differs). Failures read `errorCode`.

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
`mutate` with the invalid genome as the parent. The expected `result.text` is:

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
mode.

`UnverifiedCreaoCliProvider` raises on construction. It is not wired to the
controller or the smoke script.

## Frontier

`FrontierProvider` is an optional OpenAI-compatible client
(`FRONTIER_API_KEY`, `FRONTIER_BASE_URL`, `FRONTIER_MODEL`) for the Phase 2
one-shot arm. Phase 1 does not call it during evolution.

## Modes

| mode | providers | evaluator | label |
| --- | --- | --- | --- |
| mock | deterministic local | toy | `provider_mode=mock` |
| hybrid | real Veri + real CREAO | toy | run mode `hybrid`, calls `live` |
| live | Phase 2 | real domain | not implemented |
| replay | none | none | Phase 3 playback of an export |
