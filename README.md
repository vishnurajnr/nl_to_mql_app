# Conversational NL → MongoDB Atlas Query System

A web chat interface that turns natural language into **safe, deterministic**
MongoDB queries against an `orders` collection, plus a **RAG** route for
documentation questions — using an agentic **three-path routing** architecture.

- **Path A — Direct Lookup:** exact-identifier questions (e.g. *"status of order
  37126471"*) are answered with LLM tool/function calling that extracts the
  `order_id` and runs `findOne({ order_id })`.
- **Path B — Analytical / Root-Cause:** aggregated, comparative, or diagnostic
  questions (e.g. *"Why is revenue low today?"*) are translated into a
  read-only aggregation pipeline, **validated against an allow-list**, executed
  with a `maxTimeMS` ceiling, and summarised back in natural language.
- **Path C — Knowledge Base (RAG):** general/conceptual questions (e.g. *"What
  is MongoDB Atlas Vector Search?"*) trigger a **vector search** over a docs
  collection; the retrieved passages are fed to the LLM to produce a grounded,
  source-cited answer.

```
User text ─▶ AI Intent Router ─┬─▶ Path A: findOne(order_id) ──────▶ synthesise
                               ├─▶ Path B: text→MQL ▶ validate ▶ run ▶ synthesise
                               └─▶ Path C: vector search ▶ retrieve ▶ augment (RAG)
```

## Stack

| Layer        | Choice                                              |
|--------------|-----------------------------------------------------|
| Backend      | Python + **FastAPI**                                |
| Agent        | **LangChain** structured output on **LangGraph** stack |
| Database     | **MongoDB Atlas** via **pymongo**                   |
| LLM          | **Azure OpenAI** or **Google Gemini** (switchable) with tool calling |

> The spec lists OpenAI GPT-4o / Claude 3.5 and a generic `LLM_API_KEY`. This
> app supports two providers behind one `llm` object (see `config.py`):
> **Azure OpenAI** (the DA680 default) and **Google Gemini** (free-tier
> friendly). Pick one with `LLM_PROVIDER`.

### Using Google Gemini (free tier)

Gemini is a good zero-cost option. Get a free key at
[Google AI Studio](https://aistudio.google.com/apikey) (no credit card
required) and set:

```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=<your-gemini-api-key>
GEMINI_MODEL=gemini-2.5-flash   # optional; this is the default
```

**Recommended model: `gemini-2.5-flash`** — supports function/tool calling and
structured output (both required here), 1M-token context, and a free daily
quota (~10 requests/min, ~250/day). If you hit rate limits, switch to
`gemini-2.5-flash-lite` for a higher daily quota (~15 RPM, ~1,000/day) at
slightly lower quality. Avoid Gemini 2.5 **Pro** — it is largely off the free
tier. Free-tier limits change periodically; check Google's
[rate-limit docs](https://ai.google.dev/gemini-api/docs/rate-limits) for
current values.

No code changes are needed to switch providers — `agent.py` only uses
`.with_structured_output(...)` and `.bind_tools(...)`, which both clients
support.

## Files

| File                | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `config.py`         | Env loading + shared Azure OpenAI LLM client.                  |
| `db.py`             | Restricted Atlas client, `orders` schema, index setup, safe read wrappers. |
| `agent.py`          | Router, Path A lookup, Path B MQL generation + **validator** + execution, synthesis. |
| `server.py`         | FastAPI app: `POST /api/chat`, `GET /api/health`, static UI.   |
| `static/index.html` | Single-page chat UI (shows response + generated MQL + raw JSON). |
| `seed_data.py`      | Populate `orders` with demo data (the only writer).           |

## Setup

From the repo root (uses the shared `venv` and `requirements.txt`):

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Environment variables

Set these in the repo-root `.env` (already present in this workspace):

| Variable                | Required | Notes                                             |
|-------------------------|----------|---------------------------------------------------|
| `MONGODB_URI`           | yes      | Atlas connection string.                          |
| `LLM_PROVIDER`          | no       | `azure` or `google`. Auto-detected if unset.      |
| `AZURE_OPENAI_API_KEY`  | if azure | Azure LLM key (the spec's `LLM_API_KEY`).         |
| `AZURE_OPENAI_ENDPOINT` | if azure | Azure OpenAI endpoint URL.                        |
| `GOOGLE_API_KEY`        | if google| Gemini API key (`GEMINI_API_KEY` also accepted).  |
| `GEMINI_MODEL`          | no       | Gemini model. Default `gemini-2.5-flash`.         |
| `MONGODB_DB`            | no       | Target database. Default `da680_shop`.            |
| `MQL_MAX_TIME_MS`       | no       | Per-query timeout. Default `5000`.                |
| `MQL_MAX_RESULT_DOCS`   | no       | Result-size cap. Default `200`.                   |
| `SKIP_INDEX_SETUP`      | no       | Set `1` to skip index creation on startup.        |
| `KB_DB`                 | no       | Knowledge-base database. Default `rag_demo`.      |
| `KB_EMBED_MODE`         | no       | `auto` (Atlas auto-embed, default, no key) or `manual` (client-side Voyage). |
| `KB_TOP_K`              | no       | Passages retrieved per KB question. Default `4`.  |
| `VOYAGE_API_KEY`        | if manual| Required only when `KB_EMBED_MODE=manual`.        |

### Knowledge Base (RAG) route

The third route answers conceptual/documentation questions by vector search
over `rag_demo`. Two collections are supported:

| Mode (`KB_EMBED_MODE`) | Collection            | How the query is embedded                     | Voyage key |
|------------------------|-----------------------|-----------------------------------------------|------------|
| `auto` (default)       | `knowledge_base_auto` | Atlas **auto-embed** index embeds server-side | not needed |
| `manual`               | `knowledge_base`      | app embeds the query with **Voyage** (`voyage-3-lite`, 768-dim) | required |

`auto` mode works out of the box with no extra keys. Use `manual` only if you
want to query the precomputed-embedding collection and have a valid
`VOYAGE_API_KEY`. Override collection/index names via `KB_AUTO_COLLECTION`,
`KB_AUTO_INDEX`, `KB_MANUAL_COLLECTION`, `KB_MANUAL_INDEX`,
`KB_MANUAL_PATH`, `KB_VOYAGE_MODEL` if your setup differs.

## Seed sample data

```bash
cd 8_nl_to_mql_app
python seed_data.py --drop      # drops + reseeds ~14 days of orders
```

This also ensures the required indexes and inserts a demo order `37126471`
for Path A testing. Today's data is intentionally skewed (more returns,
lower revenue) so root-cause questions have something to diagnose.

## Run

```bash
cd 8_nl_to_mql_app
uvicorn server:app --reload --port 8000
# open http://localhost:8000
```

## API

### `POST /api/chat`

Request:
```json
{ "message": "Why are returns high today?", "history": [] }
```

Response:
```json
{
  "response": "Returns today are running well above the 7-day baseline…",
  "path": "analytical",
  "router_reason": "Diagnostic question about a trend.",
  "generated_mql": [ { "$facet": { "today": [ ... ], "baseline": [ ... ] } } ],
  "raw_results": [ { "today": [ ... ], "baseline": [ ... ] } ],
  "error": null,
  "meta": { "pipeline_explanation": "...", "result_count": 1 }
}
```

`generated_mql` and `raw_results` are always included for **audit logs and
debugging**.

## Safety guardrails (Path B)

Enforced in `agent.validate_pipeline` **before** any query runs:

- **Allow-list only:** `$match`, `$group`, `$project`, `$sort`, `$limit`,
  `$unwind`, `$facet`, `$lookup`.
- **Rejects** write/index stages (`$out`, `$merge`, `$indexStats`, …) and
  JavaScript execution (`$where`, `$function`, `$accumulator`) — recursively,
  including inside `$facet` and `$lookup` sub-pipelines.
- **`maxTimeMS`** ceiling on every read (default 5000ms).
- **Result cap** appended to every pipeline (default 200 docs).
- Connection client is read-oriented: `retryWrites=False`,
  `readPreference=secondaryPreferred`, short timeouts.

## Indexes provisioned

Created idempotently on startup (and by `seed_data.py`):

1. `{ order_id: 1 }` — **unique** single-field (fast lookups).
2. `{ created_at: -1, status: 1 }` — compound for time-ranged analytics.
3. `{ "items.category": 1, status: 1 }` — multikey compound for nested item analysis.
