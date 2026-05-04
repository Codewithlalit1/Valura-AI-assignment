[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SHM9MYZJ)

# Valura AI
        
Valura AI is a FastAPI microservice that acts as an AI co-investor for every user on the Valura wealth management platform. It accepts a natural-language query together with the user's profile and conversation history, classifies the intent with a single LLM call, routes the request to the appropriate specialist agent, and streams the response token-by-token over Server-Sent Events. The service is designed as an extensible spine: safety, classification, routing, and session persistence are fully wired; the `portfolio_health` agent is implemented end-to-end with live market data and structured LLM output; the remaining seven agents are registered as stubs that return structured `not_implemented` JSON, making each one a self-contained addition rather than a rewrite.

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
│     └─ one OpenAI call (JSON mode) in a thread pool           │
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
        │    └─ AsyncOpenAI stream  — numbers injected in prompt
        │
        └─ all other agents (stubs)
             └─ {"status": "not_implemented", "agent": "..."}
```

### Module map

```
src/
├── main.py                   FastAPI app, lifespan, /query endpoint
├── users.py                  UserProfileLoader — fixture index, O(1) lookup
├── router.py                 AgentRouter + _Stub sentinel + create_router()
├── classifier/
│   ├── classifier.py         IntentClassifier — one LLM call, never raises
│   ├── prompt.py             System prompt with few-shot examples
│   └── schema.py             ClassificationResult, ExtractedEntities
├── safety/
│   └── guard.py              SafetyGuard — 5 category regex filter
├── session/
│   └── store.py              SessionStore ABC + InMemorySessionStore
└── agents/
    └── portfolio_health/
        ├── agent.py          PortfolioHealthAgent — async LLM streaming
        ├── calculations.py   Pure functions: concentration, perf, benchmark
        ├── market_data.py    MarketDataFetcher — yfinance + TTL cache
        └── schema.py         Pydantic output models
```

---

## Setup

**Requirements:** Python 3.11+, an OpenAI API key.

```bash
git clone <your-classroom-repo-url>
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum

make install    # pip install -r requirements.txt && pip install -e ".[dev]"
make run        # uvicorn src.main:app --reload --port 8000
```

The server is ready when you see:
```
INFO  Valura AI started — model=gpt-4o-mini  timeout=30s  users=9
```

**Quick smoke test (curl):**

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How is my portfolio?", "user_id": "usr_aggr_001", "session_id": "s1"}'
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key used by the classifier and portfolio health agent |
| `MODEL_DEV` | No | `gpt-4o-mini` | Model used at runtime (set in `.env`); controls classifier + agent model |
| `MODEL_EVAL` | No | `gpt-4.1` | Reference model name for evaluation; not read by the app directly |
| `SESSION_STORE` | No | `memory` | Declared in `.env.example` for future use; currently always in-memory |
| `APP_ENV` | No | `development` | `development \| production \| test`; not yet consumed by the app |
| `DATABASE_URL` | No | — | Reserved for a future Postgres session store |

The app reads only `MODEL_DEV` at startup (via `os.getenv`). All other variables are wired into `.env.example` for documentation and future extension.

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
The async variant (`AsyncOpenAI`) gives `stream=True` support with a typed async iterator over `ChatCompletionChunk` objects, which maps directly onto the `async for chunk in stream` pattern needed for SSE. The sync variant (`openai.OpenAI`) is used by the classifier because it runs inside `asyncio.to_thread` — keeping it sync avoids nested event-loop complexity while still not blocking the main loop.

### yfinance
The simplest way to get free, real-time (15-min delayed) equity prices without an API key or rate-limit subscription. `yf.download()` batch-fetches multiple tickers in one HTTP call. Failure modes (delisted tickers, network errors) are handled gracefully — the agent falls back to the cost-basis price stored in the user fixture so the pipeline never crashes on a missing quote. A 5-minute in-memory TTL cache means a busy session hits the network at most once every 300 seconds per ticker set.

### Pydantic v2
Used for three distinct purposes: request validation (`QueryRequest`), LLM output modelling (`ClassificationResult`, `ExtractedEntities`), and calculation output contracts (`ConcentrationRisk`, `PerformanceMetrics`, `BenchmarkComparison`). The `ConfigDict(frozen=True)` models used for calculation outputs are immutable value objects — they cannot be accidentally mutated by downstream code. `ConfigDict(extra="ignore")` on LLM-facing models means prompt drift (the model returning extra fields) never causes a validation error in production.

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
| `insider_trading` | Queries containing "non-public information", "unannounced acquisition", "before the announcement", "inside tip" |
| `market_manipulation` | "pump and dump", "wash trading", "spoofing", "drive the price up", "artificial volume" |
| `money_laundering` | "undeclared cash", "without triggering reporting", "structuring deposits", "shell company", "launder" |
| `guaranteed_returns` | "guarantee me a return", "100% certain", "risk-free and 15%", "certain to go up" |
| `reckless_advice` | "margin loan", "put my entire retirement into crypto", "meme stock", "0DTE", "mortgage my house to invest" |

### Educational framing — the false-positive problem

The hardest design decision was preventing over-blocking on legitimate educational queries. A naive filter that fires on `"insider trading"` as a substring would block `"What is insider trading and why is it illegal?"` — a question any compliance officer, student, or curious user might ask. Blocking it is both unhelpful and paternalistic.

The guard solves this with a two-layer check:

1. **Educational opener detection** (`_EDU_OPENER`): if the query begins with `"what is"`, `"explain"`, `"how does"`, `"is X illegal"`, etc., all harmful checks are skipped entirely.
2. **Educational signal detection** (`_EDU_SIGNAL`): phrases like `"for educational purposes"`, `"red flag"`, `"how do regulators detect"`, `"historical data"` anywhere in the query also bypass all checks.

The accepted false-negative risk is that a bad actor could frame a harmful request as educational — e.g. `"what is the best way to pump and dump?"`. This is a deliberate tradeoff: the LLM layer is the next line of defence, and over-blocking destroys user trust faster than under-blocking passes queries to a model that can still decline them. Empirically, real educational queries vastly outnumber jailbreak attempts in financial assistant contexts.

All five categories and both educational bypass paths are covered by 45 labelled tests in `tests/test_safety.py`.

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
python scripts/benchmark.py             # 20 requests, usr_aggr_001
python scripts/benchmark.py --n 50 --verbose   # 50 requests + per-request bar chart
```

Target thresholds from the assignment spec:

| Metric | Target | Notes |
|---|---|---|
| p95 TTFT | < 2 s | Dominated by classifier LLM call latency |
| p95 TRT | < 6 s | Dominated by portfolio health agent stream duration |

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

Tests run without an `OPENAI_API_KEY` — all LLM calls are mocked. The full suite covers:

| File | What it tests |
|---|---|
| `test_safety.py` | 47 labelled safe/blocked queries across all five categories |
| `test_classifier.py` | ClassificationResult parsing, entity normalisation, fallback behaviour |
| `test_classifier_routing.py` | End-to-end routing for labelled query fixtures |
| `test_portfolio_health.py` | PortfolioHealthAgent — streaming, BUILD mode, fallback JSON, pure calculation functions |
| `test_router.py` | AgentRouter — real agent dispatch, stub dispatch, error handling, all 8 agents |
| `test_session.py` | InMemorySessionStore — add/retrieve, eviction, isolation, clear |
| `test_main.py` | FastAPI pipeline — safety block, timeout, error event, session persistence, SSE format |
| `test_users.py` | UserProfileLoader — fixture loading, empty portfolio, unknown user default |
| `test_integration.py` | Full pipeline with mocked LLM — concentrated portfolio, blocked query, stub agent, empty portfolio BUILD mode |

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
The `SessionStore` ABC makes this a one-file addition, but getting it right requires async connection pooling (`redis.asyncio`), serialisation decisions (MessagePack vs JSON), TTL policy for abandoned sessions, and integration tests that spin up a real Redis instance via `pytest-docker`. Without that test coverage, the replacement isn't production-ready — it's just code that works on a laptop.

**2. Implement three more real agents: `market_research`, `financial_calculator`, and `risk_assessment`.**
These cover the majority of realistic user queries that currently hit stubs. `financial_calculator` in particular should be deterministic (no LLM for straightforward maths like DCA or compound interest) — which means adding a tool-call layer to the classifier output so the agent can dispatch to a local function before touching the LLM. That architecture change touches the classifier prompt, the `ClassificationResult` schema, and the router, so it needs to be done cleanly rather than bolted on.

**3. Add a pre-classifier safety layer using embeddings.**
The current regex guard is fast and interpretable, but it cannot catch semantically harmful queries that avoid the exact vocabulary in the patterns — a sophisticated user can rephrase around any keyword list. A lightweight embedding similarity check against a labelled set of harmful query examples (stored in pgvector or a small in-process FAISS index) would catch these at under 5 ms with no LLM call. The design challenge is keeping the false-positive rate low enough that it doesn't undermine the educational-framing bypass that the current guard deliberately preserves.
