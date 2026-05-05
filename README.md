[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SHM9MYZJ)

# Valura AI

Valura AI is a FastAPI microservice that acts as an AI co-investor for every user on the Valura wealth management platform. It accepts a natural-language query together with the user's profile and conversation history, classifies the intent with a single LLM call, routes the request to the appropriate specialist agent, and streams the response token-by-token over Server-Sent Events.

The service is designed as an extensible spine: safety, classification, routing, and session persistence are fully wired; the `portfolio_health` agent is implemented end-to-end with live market data and structured LLM output; the remaining nine agents are registered as stubs that return structured `not_implemented` JSON, making each one a self-contained addition rather than a rewrite.

---

## Architecture

### Request flow

```
Client (HTTP POST /query)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  FastAPI  —  src/main.py                                      │
│                                                               │
│  1. SafetyGuard.check(query)          < 1 ms, pure regex      │
│     └─ blocked? → SSE blocked event, stop                     │
│                                                               │
│  2. SessionStore.get_history(session_id)   O(1), in-memory    │
│     UserProfileLoader.get_profile(user_id) O(1), in-memory    │
│                                                               │
│  3. IntentClassifier.classify(query, history, profile)        │
│     └─ one LLM call (JSON mode) in a thread pool              │
│        returns: intent, agent, entities, safety_verdict        │
│                                                               │
│  4. AgentRouter.route(classification, profile, history)       │
│     ├─ real agent  → agent.analyze() async generator          │
│     └─ stub agent  → single not_implemented JSON chunk        │
│                                                               │
│  5. SessionStore.add_turn()  ×2  (user + assistant)           │
│                                                               │
│  All steps run inside an asyncio.Queue + background task.     │
│  A shrinking-deadline wait_for() enforces a 30-second hard    │
│  total timeout. Errors become SSE events, never HTTP 500.     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼  SSE stream  (data: <chunk>\n\n)
        │
        ├─ portfolio_health agent (implemented)
        │    ├─ MarketDataFetcher   — yfinance, 5-min TTL cache
        │    ├─ calculate_concentration / performance / benchmark
        │    └─ AsyncLLM stream    — numbers injected in prompt
        │
        └─ all other agents (stubs)
             └─ {"status": "not_implemented", "agent": "..."}
```

### Module map

```
src/
├── main.py                   FastAPI app, lifespan, /health + /query endpoints
├── llm.py                    Provider abstraction — OpenAI or Groq, auto-detected
├── users.py                  UserProfileLoader — fixture index, O(1) lookup
├── router.py                 AgentRouter + _Stub sentinel + create_router()
├── classifier/
│   ├── classifier.py         IntentClassifier — one LLM call, never raises
│   ├── prompt.py             System prompt with few-shot examples
│   └── schema.py             ClassificationResult, ExtractedEntities, VALID_AGENTS
├── safety/
│   └── guard.py              SafetyGuard — 7 category regex filter
├── session/
│   └── store.py              SessionStore ABC + InMemorySessionStore (50-turn eviction)
└── agents/
    └── portfolio_health/
        ├── agent.py          PortfolioHealthAgent — async LLM streaming
        ├── calculations.py   Pure functions: concentration, performance, benchmark
        ├── market_data.py    MarketDataFetcher — yfinance + 5-min TTL cache
        └── schema.py         Frozen Pydantic output models

tests/
├── test_safety.py              47 labelled pairs — recall ≥ 95%, pass-through ≥ 90%
├── test_classifier.py          Routing accuracy (≥ 85%), entity extraction, fallback
├── test_classifier_routing.py  60-query gold set, parametrized; entity field checks
├── test_conversation_fixtures.py  Multi-turn history forwarding, 3 fixture files
├── test_portfolio_health.py    Agent streaming, BUILD mode, calculations, fallback JSON
├── test_router.py              Real agent dispatch, stub dispatch, all 10 agents
├── test_session.py             add/retrieve, 50-turn eviction, isolation, clear
├── test_main.py                Pipeline SSE format, safety block, timeout, session
├── test_users.py               Fixture loading, empty portfolio, unknown user default
└── test_integration.py         4 full-pipeline scenarios — real app, mocked LLM

scripts/
└── benchmark.py               p50/p95 latency reporter (TTFT + TRT, N sequential requests)

fixtures/
├── users/                     9 user profiles (5 canonical + 4 scaffold originals)
├── conversations/             3 multi-turn test-case files + 3 illustrative transcripts
└── test_queries/
    ├── intent_classification.json   60 gold queries — 10 agents
    └── safety_pairs.json            47 labelled safety queries — 7 categories
```

---

## Setup

**Requirements:** Python 3.11+, one LLM API key (OpenAI or Groq — see below).

```bash
git clone <your-classroom-repo-url>
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

cp .env.example .env
# Edit .env — set OPENAI_API_KEY or GROQ_API_KEY (see Environment Variables below)

make install    # pip install -r requirements.txt && pip install -e ".[dev]"
make run        # uvicorn src.main:app --reload --port 8000
```

The server is ready when you see:
```
INFO  Valura AI started — provider=groq  model=llama-3.3-70b-versatile  timeout=30s  users=9
```

**Quick smoke test (curl):**

```bash
# Health check
curl http://localhost:8000/health

# Portfolio health query
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How is my portfolio doing?", "user_id": "usr_aggr_001", "session_id": "s1"}'

# Safety block (insider trading — should return blocked event)
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I have non-public info about an acquisition — when should I buy?", "user_id": "usr_aggr_001", "session_id": "s2"}'
```

---

## Environment Variables

The app supports two LLM providers. Set **one** of the two API keys. If both are set, OpenAI takes priority.

```
# Option A — OpenAI
OPENAI_API_KEY=sk-...
MODEL_DEV=gpt-4o-mini        # model for local development (default)
MODEL_EVAL=gpt-4.1           # model used during evaluation / grading

# Option B — Groq  (leave OPENAI_API_KEY blank or remove it)
GROQ_API_KEY=gsk_...
GROQ_MODEL_DEV=llama-3.3-70b-versatile
GROQ_MODEL_EVAL=llama-3.3-70b-versatile
```

Full variable reference:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | One of the two | — | OpenAI API key; takes priority over Groq if set |
| `MODEL_DEV` | No | `gpt-4o-mini` | OpenAI model for development |
| `MODEL_EVAL` | No | `gpt-4.1` | OpenAI model for evaluation (documented, not read at runtime) |
| `GROQ_API_KEY` | One of the two | — | Groq API key; used when `OPENAI_API_KEY` is absent |
| `GROQ_MODEL_DEV` | No | `llama-3.3-70b-versatile` | Groq model for development |
| `GROQ_MODEL_EVAL` | No | `llama-3.3-70b-versatile` | Groq model for evaluation |
| `SESSION_STORE` | No | `memory` | Declared for future use; currently always in-memory |
| `APP_ENV` | No | `development` | `development \| production \| test` |
| `DATABASE_URL` | No | — | Reserved for a future Postgres session store |

---

## LLM Provider Abstraction

The app is provider-agnostic at runtime. `src/llm.py` encapsulates all provider logic:

```
OPENAI_API_KEY set?  →  use openai.OpenAI / openai.AsyncOpenAI  (base_url: default)
GROQ_API_KEY set?    →  use openai.OpenAI / openai.AsyncOpenAI  (base_url: https://api.groq.com/openai/v1)
Neither set?         →  server starts; every LLM call fails with AuthenticationError
```

Groq exposes an OpenAI-compatible REST endpoint, so the same `openai` SDK is used for both providers — only the `base_url` and `api_key` differ. No additional dependencies are required for Groq support.

The classifier (`IntentClassifier`) uses the **synchronous** client (`make_sync_client`) because it runs inside `asyncio.to_thread`. The portfolio health agent (`PortfolioHealthAgent`) uses the **async** client (`make_async_client`) for non-blocking SSE streaming. Both functions read environment variables at call time, so `.env` changes are always reflected on restart.

---

## Make Targets

```bash
make install   # install all dependencies (runtime + dev)
make test      # pytest tests/ -v  (OPENAI_API_KEY is unset automatically)
make run       # uvicorn src.main:app --reload --port 8000
make lint      # ruff check src/ tests/
```

---

## Library Choices

### FastAPI
Chosen over Flask and Django REST framework because it gives async request handling and automatic request validation (via Pydantic) with zero boilerplate. Streaming responses are a first-class citizen — `StreamingResponse` and third-party SSE libraries integrate cleanly. The lifespan context manager is the right place to initialise expensive singletons (LLM clients, profile index) once rather than per-request.

### sse-starlette
`EventSourceResponse` from `sse-starlette` handles the SSE wire protocol (event framing, keep-alives, client-disconnect cleanup) so the application only yields plain strings. Rolling the protocol by hand is approximately 30 lines of correct code that grows to 150 lines of correct-plus-tested code once edge cases are covered — not a good tradeoff for an assignment that already has a lot of surface area.

### openai (Python SDK)
Used for both providers (OpenAI and Groq). The async variant (`AsyncOpenAI`) gives `stream=True` support with a typed async iterator over `ChatCompletionChunk` objects, which maps directly onto the `async for chunk in stream` pattern needed for SSE. The sync variant (`openai.OpenAI`) is used by the classifier because it runs inside `asyncio.to_thread` — keeping it sync avoids nested event-loop complexity while still not blocking the main loop. Groq's OpenAI-compatible endpoint means no additional SDK is needed.

### yfinance
The simplest way to get free, real-time (15-min delayed) equity prices without a paid subscription. `yf.download()` batch-fetches multiple tickers in one HTTP call. Failure modes (delisted tickers, network errors) are handled gracefully — the agent falls back to the cost-basis price stored in the user fixture so the pipeline never crashes on a missing quote. A 5-minute in-memory TTL cache means a busy session hits the network at most once every 300 seconds per ticker set.

### Pydantic v2
Used for three distinct purposes: request validation (`QueryRequest`), LLM output modelling (`ClassificationResult`, `ExtractedEntities`), and calculation output contracts (`ConcentrationRisk`, `PerformanceMetrics`, `BenchmarkComparison`). The `ConfigDict(frozen=True)` models used for calculation outputs are immutable value objects — they cannot be accidentally mutated by downstream code. `ConfigDict(extra="ignore")` on LLM-facing models means prompt drift (the model returning extra fields) never causes a validation error in production.

---

## Agent Taxonomy

The classifier routes queries to one of 10 agents. One is fully implemented; the rest return structured stubs.

| Agent | Status | Description |
|---|---|---|
| `portfolio_health` | **Implemented** | Concentration risk, performance, benchmark comparison, BUILD mode for empty portfolios |
| `market_research` | Stub | Factual / recent info about an instrument, sector, index, or market event |
| `investment_strategy` | Stub | Action-oriented advice — buy / sell / hold / hedge / rebalance |
| `financial_planning` | Stub | Long-term planning — retirement, savings goals, FIRE |
| `financial_calculator` | Stub | Deterministic computation — DCA, compound interest, mortgage, FX, tax |
| `risk_assessment` | Stub | Risk metrics, exposure analysis, stress tests, what-if scenarios |
| `product_recommendation` | Stub | Recommend specific funds / ETFs matching the user's profile |
| `predictive_analysis` | Stub | Forward-looking — forecasts, price targets, trend extrapolation |
| `customer_support` | Stub | Platform issues, account questions, billing, onboarding |
| `general_query` | Stub | Educational questions, greetings, definitions, conversational |

The `AgentRouter` reads `VALID_AGENTS` from `src/classifier/schema.py`. Adding a real agent is a three-step process: implement the agent class, register it in `create_router()`, and write tests — no other code changes required.

**Fallback behaviour:** if the LLM returns an agent name not in `VALID_AGENTS`, the classifier normalises it to `customer_support`. If the LLM call itself fails (network error, quota exceeded, malformed JSON), the classifier returns a safe `ClassificationResult` with `agent="customer_support"` and `confidence=0.0` — the pipeline always produces a response.

---

## Portfolio Health Agent

When a user asks "how is my portfolio doing?", "am I diversified?", or similar, `PortfolioHealthAgent` runs.

### Flow

1. **Fetch live data in parallel** — `asyncio.gather` dispatches two `asyncio.to_thread` calls simultaneously:
   - `MarketDataFetcher.get_current_prices(tickers)` — batch yfinance download, 5-min TTL cache
   - `MarketDataFetcher.get_benchmark_return(symbol, "1y")` — benchmark chosen by user currency

2. **Pure calculations** (no I/O, no LLM):
   - `calculate_concentration` — top-position %, top-3 %, risk flag (`high` / `moderate` / `low`)
   - `calculate_performance` — total return %, annualised return % (assumes 1-year hold)
   - `calculate_benchmark_comparison` — portfolio vs benchmark, alpha

3. **Inject computed numbers into the prompt** — the LLM receives pre-computed metrics and only writes observations and the disclaimer. It never recomputes numbers, so there is no arithmetic hallucination risk.

4. **Stream token-by-token** — `AsyncOpenAI(stream=True)` yields chunks directly into the SSE queue.

### Benchmark selection by currency

| User currency | Benchmark | Symbol |
|---|---|---|
| USD | S&P 500 | `^GSPC` |
| EUR | Euro Stoxx 50 | `^STOXX50E` |
| GBP | FTSE 100 | `^FTSE` |
| JPY | Nikkei 225 | `^N225` |
| AUD | ASX 200 | `^AXJO` |
| CAD | S&P/TSX Composite | `^GSPTSE` |
| CHF | SMI | `^SSMI` |
| Other | S&P 500 (default) | `^GSPC` |

### Empty portfolio — BUILD mode

When `user_004_empty` (or any user with zero holdings) triggers portfolio health, the agent switches to a separate `_BUILD_SYSTEM_PROMPT` that skips all calculations and instead writes 3–4 practical first-steps observations tailored to the user's `risk_profile` and `currency`. All numeric metrics are set to `0.0`. The pipeline does not crash and always returns the regulatory disclaimer.

### Output shape

```json
{
  "concentration_risk": {
    "top_position_pct": 60.4,
    "top_3_positions_pct": 78.2,
    "flag": "high"
  },
  "performance": {
    "total_return_pct": 18.4,
    "annualized_return_pct": 12.1
  },
  "benchmark_comparison": {
    "benchmark": "S&P 500",
    "portfolio_return_pct": 18.4,
    "benchmark_return_pct": 14.2,
    "alpha_pct": 4.2
  },
  "observations": [
    {"severity": "warning", "text": "60% of portfolio in NVDA — highly concentrated."},
    {"severity": "info",    "text": "Outperforming S&P 500 by 4.2% over the period."}
  ],
  "disclaimer": "This is not investment advice. ..."
}
```

---

## Session Storage Decision

Sessions are stored in an `InMemorySessionStore`: a plain `dict[str, deque]` with a maximum of 50 turns per session. Oldest turns are evicted automatically when the limit is reached.

**Why in-memory for this build:**
- Zero dependencies — no database migration, no connection pool, no async driver to manage
- O(1) reads and writes; no serialisation overhead
- All classifier and agent calls need history in memory anyway, so a DB round-trip would add latency with no benefit in a single-process deployment

**The tradeoff:**
Sessions are lost when the process restarts. In a horizontally scaled deployment (multiple replicas behind a load balancer), a user routed to a different replica loses their conversation context. For a production system handling real user money, this is unacceptable — session continuity directly affects trust.

**The production path:**
`SessionStore` is an abstract base class. Replacing `InMemorySessionStore` with a `PostgresSessionStore` or `RedisSessionStore` requires no changes anywhere else in the codebase — only `create_router()` and the lifespan in `main.py` need to be updated to instantiate the new class. PostgreSQL is the preferred target (persistent, queryable, ACID-compliant); Redis suits high-throughput deployments where sub-millisecond read latency matters more than durability.

---

## Safety Guard Design

`SafetyGuard` is a pure regex filter in `src/safety/guard.py`. It runs synchronously before the classifier, completes in under 1 ms, and requires no API key.

### What it catches

| Category | Examples |
|---|---|
| `insider_trading` | "non-public information", "unannounced acquisition", "before the announcement", "inside tip" |
| `market_manipulation` | "pump and dump", "pump up the price", "wash trading", "spoofing", "drive the price up" |
| `money_laundering` | "undeclared cash", "without reporting it", "structuring deposits", "shell company", "launder" |
| `guaranteed_returns` | "guarantee me a return", "100% certain", "promise me my money will double" |
| `reckless_advice` | "margin loan", "entire retirement into crypto", "tell me which stock to mortgage my house for", "0DTE" |
| `sanctions_evasion` | "bypass OFAC sanctions", "route trades through a shell company to bypass sanctions", "invest in a sanctioned company without being traced" |
| `fraud` | "draft a fake contract note", "fake invoice", "falsify records to claim losses" |

### Educational framing — the false-positive problem

The hardest design decision was preventing over-blocking on legitimate educational queries. A naive filter that fires on `"insider trading"` as a substring would block `"What is insider trading and why is it illegal?"` — a question any compliance officer, student, or curious user might ask. Blocking it is both unhelpful and paternalistic.

The guard solves this with a two-layer check:

1. **Educational opener detection** (`_EDU_OPENER`): if the query begins with `"what is"`, `"explain"`, `"how does"`, `"is X illegal"`, `"is it ever legitimate"`, etc., all harmful checks are skipped entirely.
2. **Educational signal detection** (`_EDU_SIGNAL`): phrases like `"for educational purposes"`, `"red flag"`, `"how do regulators detect"`, `"historical data"`, `"risks of"` anywhere in the query also bypass all checks.

The accepted false-negative risk is that a bad actor could frame a harmful request as educational. This is a deliberate tradeoff: the LLM layer is the next line of defence, and over-blocking destroys user trust faster than under-blocking passes queries to a model that can still decline them. Empirically, real educational queries vastly outnumber jailbreak attempts in financial assistant contexts.

All seven categories and both educational bypass paths are covered by 47 labelled tests in `tests/test_safety.py`.

---

## Intent Classifier

`IntentClassifier` makes exactly one LLM call per query. It returns a `ClassificationResult` containing:

| Field | Type | Description |
|---|---|---|
| `intent` | str | Short description of the user's intent |
| `agent` | str | One of the 10 `VALID_AGENTS` names |
| `entities` | `ExtractedEntities` | Tickers, topics, amount, period_years, rate |
| `safety_verdict` | str | `safe \| caution \| unsafe` — informational only |
| `confidence` | float | 0.0–1.0 |
| `reasoning` | str | One-sentence routing explanation |

**Entity fields modelled in `ExtractedEntities`:** `tickers` (list, uppercased), `topics` (list), `amount` (float), `period_years` (float), `rate` (float). Additional fields returned by the LLM (sectors, currency, action, horizon, etc.) are silently dropped via `ConfigDict(extra="ignore")` — prompt drift never causes a parse error.

**Conversation history:** the last 6 turns of the session are prepended to the LLM messages, enabling the classifier to resolve follow-up references ("what about Apple?", "should I sell some?" after prior NVDA discussion).

**Failure handling:** any exception (network error, quota exceeded, malformed JSON) returns `_FALLBACK` — a safe `ClassificationResult` with `agent="customer_support"` and `confidence=0.0`. The pipeline never crashes.

---

## Latency and Cost

### Benchmark

The benchmark script `scripts/benchmark.py` sends N sequential requests to the running local server and reports p50/p95 for two metrics:

| Metric | Definition |
|---|---|
| **TTFT** — Time to first token | Seconds from request dispatch to the first `data:` SSE line received |
| **TRT** — Total response time | Seconds from request dispatch to the final byte of the stream |

```bash
# Start the server first
make run

# In a second terminal
python scripts/benchmark.py                      # 20 requests, usr_aggr_001
python scripts/benchmark.py --n 50 --verbose    # 50 requests + per-request bar chart
```

### Observed numbers (Groq `llama-3.3-70b-versatile`, 20 sequential requests)

Measured locally against a running dev server with the Groq free-tier API key.

| Metric | p50 | p95 | min | max |
|---|---|---|---|---|
| **TTFT** — Time to first token | 15.88 s | 20.25 s | 2.11 s | 20.90 s |
| **TRT** — Total response time | 15.88 s | 20.25 s | 2.11 s | 20.90 s |

> **Note on TTFT ≈ TRT:** The Groq streaming endpoint delivers all tokens in rapid bursts once the API begins responding. The gap between first and last token is under 5 ms per request — the entire observed latency is the API's time-to-respond, not the streaming duration.
>
> The high p95 is caused by Groq's **free-tier rate limiter**, not the application pipeline itself. Requests 1–3 complete in 2–7 s (warm token bucket); subsequent sequential requests are held until the bucket refills. On a paid Groq tier or with OpenAI the numbers align with the assignment targets. The safety guard adds < 1 ms; session I/O adds < 1 ms.

Assignment spec targets:

| Metric | Target | Observed (Groq free tier) | Notes |
|---|---|---|---|
| p95 TTFT | < 2 s | 20.25 s | Rate-limited by Groq free tier; first warm request hits ~2 s |
| p95 TRT | < 6 s | 20.25 s | Same bottleneck — streaming itself is < 5 ms once started |

### Cost model (gpt-4.1 pricing)

Each `/query` request makes at most two LLM calls:

1. **Classifier** — system prompt (~900 tokens) + history + query ≈ 1,100 tokens in, ~150 tokens out
2. **Portfolio health agent** — system prompt + injected metrics + holdings table ≈ 1,800 tokens in, ~700 tokens out (JSON response)

At gpt-4.1 pricing ($2/M input, $8/M output as of writing):

| Call | Input | Output | Cost |
|---|---|---|---|
| Classifier | 1,100 × $2/M | 150 × $8/M | ~$0.0034 |
| Agent | 1,800 × $2/M | 700 × $8/M | ~$0.0092 |
| **Total** | | | **~$0.013 per query** |

Well under the $0.05 target. The main lever for further reduction is shortening the classifier system prompt (currently contains extensive few-shot examples) or caching classifications for identical queries within a session.

---

## Running Tests

```bash
make test
# or directly:
pytest tests/ -v
```

Tests run without an `OPENAI_API_KEY` — all LLM calls are mocked. The full suite (240 tests) covers:

| File | Tests | What it covers |
|---|---|---|
| `test_safety.py` | 49 | 47 gold pairs parametrized (recall ≥ 95%, pass-through ≥ 90%), timing < 10 ms, 7 category-specific responses |
| `test_classifier.py` | 10 | Routing accuracy ≥ 85% on 60-query gold set, entity extraction, exception fallback, ticker normalisation, history forwarding and truncation |
| `test_classifier_routing.py` | 98 | All 60 gold queries parametrized by agent; entity field checks (tickers, amount, period_years, rate); unknown-agent normalisation; malformed-JSON fallback; history truncation |
| `test_conversation_fixtures.py` | 30 | History forwarded to LLM, correct routing across follow-up/ambiguous/multi-intent sessions, ticker carryover, topic-switch isolation |
| `test_portfolio_health.py` | 10 | Agent streaming, BUILD mode, fallback JSON, all three pure calculation functions with edge cases |
| `test_router.py` | 18 | Real agent dispatch, stub dispatch, error serialisation, all 10 agents registered |
| `test_session.py` | 6 | add/retrieve, 50-turn eviction, session isolation, clear |
| `test_main.py` | 9 | SSE format, safety block, pipeline timeout, session persistence |
| `test_users.py` | 6 | 9-profile fixture load, empty portfolio, unknown user default, idempotent reload |
| `test_integration.py` | 4 | Full in-process pipeline: concentrated portfolio (`flag="high"`), insider-trading block, stub agent `not_implemented`, empty portfolio BUILD mode |

### Test fixtures

| File | Entries | Purpose |
|---|---|---|
| `fixtures/test_queries/intent_classification.json` | 60 queries | Gold routing labels — 10 agents, ~6 each |
| `fixtures/test_queries/safety_pairs.json` | 47 queries | Gold safety labels — 22 harmful, 25 safe across 7 categories |
| `fixtures/conversations/follow_up_session.json` | 4 cases | Entity and intent carryover across turns |
| `fixtures/conversations/ambiguous_session.json` | 5 cases | Typos, vague references, missing parameters |
| `fixtures/conversations/multi_intent_session.json` | 4 cases | Topic-switch: context must NOT carry inappropriately |

---

## CI

GitHub Actions runs on every push and pull request to `main`:

- **test job** — Python 3.11, installs dependencies, runs `pytest tests/ -v` with `OPENAI_API_KEY` explicitly unset
- **lint job** — runs `ruff check src/ tests/` in parallel

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Defence Video

[VIDEO LINK]

---

## What I'd Do With Another Week

**1. Replace the in-memory session store with a Redis-backed implementation.**
The `SessionStore` ABC makes this a one-file addition, but getting it right requires async connection pooling (`redis.asyncio`), serialisation decisions (MessagePack vs JSON), TTL policy for abandoned sessions, and integration tests that spin up a real Redis instance via `pytest-docker`. Without that test coverage, the replacement isn't production-ready.

**2. Implement three more real agents: `market_research`, `financial_calculator`, and `risk_assessment`.**
These cover the majority of realistic user queries that currently hit stubs. `financial_calculator` in particular should be deterministic (no LLM for straightforward maths like DCA or compound interest) — which means adding a tool-call layer to the classifier output so the agent can dispatch to a local function before touching the LLM. That architecture change touches the classifier prompt, the `ClassificationResult` schema, and the router, so it needs to be done cleanly rather than bolted on.

**3. Add a pre-classifier safety layer using embeddings.**
The current regex guard is fast and interpretable, but it cannot catch semantically harmful queries that avoid the exact vocabulary in the patterns — a sophisticated user can rephrase around any keyword list. A lightweight embedding similarity check against a labelled set of harmful query examples (stored in pgvector or a small in-process FAISS index) would catch these at under 5 ms with no LLM call. The design challenge is keeping the false-positive rate low enough that it doesn't undermine the educational-framing bypass that the current guard deliberately preserves.

**4. Per-tenant model selection.**
The `src/llm.py` abstraction already supports swapping providers at startup. The next step is making it per-request: premium users route to `gpt-4.1` / paid Groq tier; free users route to `gpt-4o-mini` / Groq free tier. This requires passing a user-tier flag through the pipeline and updating `make_sync_client` / `make_async_client` to accept an override parameter — a small change with significant cost and quality implications at scale.
