"""
Integration tests for the full Valura AI pipeline.

Uses the real FastAPI app in-process via httpx.AsyncClient + ASGITransport.
All LLM calls are mocked — no OPENAI_API_KEY required.

Test coverage
-------------
1. Portfolio health for usr_conc_002 (NVDA ~93% of portfolio)
   → SSE stream parses to JSON with concentration_risk.flag == "high"
2. Blocked safety query (insider trading keyword)
   → SSE stream contains a single {"event": "blocked"} event;
     classifier is never invoked
3. Stub agent (market_research) query
   → SSE stream contains {"status": "not_implemented", "agent": "market_research"}
4. Empty portfolio query for usr_004
   → No error/timeout event; JSON contains disclaimer and BUILD-oriented
     observations with zero numeric metrics

Mocking strategy
----------------
- SafetyGuard  : real (pure regex, no I/O, sub-millisecond)
- UserProfileLoader : real (reads fixture files at fixture setup)
- IntentClassifier  : MagicMock — classify() returns a deterministic
                       ClassificationResult without any network call
- PortfolioHealthAgent._client : MagicMock AsyncOpenAI whose
                       chat.completions.create() returns a pre-baked
                       async-streaming response
- PortfolioHealthAgent._fetcher : MagicMock MarketDataFetcher that
                       returns hard-coded prices/benchmark without yfinance
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.classifier.schema import ClassificationResult, ExtractedEntities
from src.main import app
from src.router import create_router
from src.safety.guard import SafetyGuard
from src.session.store import InMemorySessionStore
from src.users import UserProfileLoader

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[str]:
    """Return the payload of every `data: …` line in a raw SSE stream."""
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


async def _stream_all(client: httpx.AsyncClient, payload: dict) -> list[str]:
    """POST /query and collect all SSE data payloads as a list of strings."""
    body = ""
    async with client.stream("POST", "/query", json=payload) as resp:
        async for text in resp.aiter_text():
            body += text
    return _parse_sse(body)


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_classification(agent: str) -> ClassificationResult:
    return ClassificationResult(
        intent=f"integration test intent — {agent}",
        agent=agent,
        entities=ExtractedEntities(),
    )


def _make_sync_classifier(result: ClassificationResult) -> MagicMock:
    """Return a MagicMock classifier whose classify() returns *result* synchronously."""
    clf = MagicMock()
    clf.classify.return_value = result
    return clf


def _make_stream_client(full_text: str) -> MagicMock:
    """
    Build a mock AsyncOpenAI client that streams *full_text* as a single chunk.

    The returned mock satisfies the call pattern used in PortfolioHealthAgent._stream():
        stream = await client.chat.completions.create(...)
        async for chunk in stream:
            content = chunk.choices[0].delta.content
    """

    class _Chunk:
        def __init__(self, content: str | None) -> None:
            delta = MagicMock()
            delta.content = content
            choice = MagicMock()
            choice.delta = delta
            self.choices = [choice]

    async def _aiter():
        yield _Chunk(full_text)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_aiter())
    return mock_client


def _make_market_fetcher(
    prices: dict[str, float],
    bench_return: float = 0.10,
) -> MagicMock:
    """Return a mock MarketDataFetcher with fixed prices and benchmark return."""
    fetcher = MagicMock()
    fetcher.get_current_prices.return_value = prices
    fetcher.get_benchmark_return.return_value = bench_return
    return fetcher


def _make_portfolio_router(
    llm_response: str,
    prices: dict[str, float] | None = None,
) -> object:
    """
    Build a real AgentRouter whose PortfolioHealthAgent has mocked I/O.

    All other agents remain as stubs — the router is otherwise identical
    to what create_router() produces in production.
    """
    router = create_router(model="gpt-4o-mini")
    agent = router.registry["portfolio_health"]
    agent._fetcher = _make_market_fetcher(prices or {})
    agent._client = _make_stream_client(llm_response)
    return router


# ---------------------------------------------------------------------------
# Shared async client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """
    Async HTTPX client wired to the real FastAPI app.

    httpx.ASGITransport does NOT trigger the ASGI lifespan, so all
    app.state components are initialised directly here.  Individual tests
    override the components they need before making requests.
    """
    from src.classifier.classifier import IntentClassifier

    user_loader = UserProfileLoader()
    user_loader.load()
    app.state.user_loader = user_loader
    app.state.safety_guard = SafetyGuard()
    app.state.classifier = IntentClassifier(model="gpt-4o-mini")
    app.state.router = create_router(model="gpt-4o-mini")
    app.state.session_store = InMemorySessionStore()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Pre-baked LLM responses
# ---------------------------------------------------------------------------

# NVDA makes up ~93 % of usr_conc_002's portfolio value at fixture prices.
# calculate_concentration will set flag="high" (threshold: top-1 > 40 %).
# The mocked LLM echoes back those pre-computed metrics.
_CONC_002_LLM_JSON = json.dumps(
    {
        "concentration_risk": {
            "top_position_pct": 93.0,
            "top_3_positions_pct": 99.0,
            "flag": "high",
        },
        "performance": {
            "total_return_pct": 305.7,
            "annualized_return_pct": 305.7,
        },
        "benchmark_comparison": {
            "benchmark": "S&P 500",
            "portfolio_return_pct": 305.7,
            "benchmark_return_pct": 10.0,
            "alpha_pct": 295.7,
        },
        "observations": [
            {
                "severity": "warning",
                "text": (
                    "NVDA represents approximately 93 % of your portfolio value — "
                    "a single-stock concentration that amplifies both upside and downside."
                ),
            },
            {
                "severity": "tip",
                "text": "Consider trimming NVDA to below 40 % and diversifying into COST or BND.",
            },
        ],
        "disclaimer": (
            "This is not investment advice. Portfolio analysis is for informational "
            "purposes only. Past performance does not guarantee future results. "
            "Please consult a qualified financial adviser before making investment decisions."
        ),
    }
)

# BUILD-mode response for a user with no holdings.
_BUILD_LLM_JSON = json.dumps(
    {
        "concentration_risk": {
            "top_position_pct": 0.0,
            "top_3_positions_pct": 0.0,
            "flag": "n/a",
        },
        "performance": {"total_return_pct": 0.0, "annualized_return_pct": 0.0},
        "benchmark_comparison": {
            "benchmark": "",
            "portfolio_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "alpha_pct": 0.0,
        },
        "observations": [
            {
                "severity": "tip",
                "text": (
                    "Start with a low-cost, broadly diversified index ETF such as VOO "
                    "(S&P 500) or VWRL (global all-cap) — both carry expense ratios under 0.10 %."
                ),
            },
            {
                "severity": "tip",
                "text": (
                    "Build a 3-to-6-month emergency fund in a high-yield savings account "
                    "before committing money to equities."
                ),
            },
            {
                "severity": "info",
                "text": (
                    "A moderate risk profile suggests a 60 % equity / 40 % bond allocation "
                    "as a starting point; adjust as your comfort with volatility becomes clearer."
                ),
            },
            {
                "severity": "tip",
                "text": "Invest regularly (dollar-cost averaging) rather than trying to time the market.",
            },
        ],
        "disclaimer": (
            "This is not investment advice. Portfolio analysis is for informational "
            "purposes only. Past performance does not guarantee future results. "
            "Please consult a qualified financial adviser before making investment decisions."
        ),
    }
)

# ---------------------------------------------------------------------------
# Test 1 — concentrated portfolio → concentration_risk.flag == "high"
# ---------------------------------------------------------------------------


async def test_concentrated_portfolio_flags_high_concentration(
    client: httpx.AsyncClient,
) -> None:
    """
    Full pipeline for usr_conc_002 whose NVDA position is ~93 % of portfolio.

    The real calculate_concentration() function receives the fixture prices
    (injected via the mock fetcher) and should compute flag="high".
    The mocked LLM echoes those numbers back in its JSON response.
    """
    conc_prices = {
        "NVDA": 892.50,
        "COST": 920.00,
        "AAPL": 178.50,
        "BND": 73.80,
    }
    app.state.classifier = _make_sync_classifier(
        _make_classification("portfolio_health")
    )
    app.state.router = _make_portfolio_router(_CONC_002_LLM_JSON, conc_prices)

    payload = {
        "query": "How healthy is my portfolio?",
        "user_id": "usr_conc_002",
        "session_id": "int-conc-002",
    }
    payloads = await _stream_all(client, payload)

    assert payloads, "Expected at least one SSE chunk"
    result = json.loads("".join(payloads))

    conc = result["concentration_risk"]
    assert conc["flag"] == "high", (
        f"Expected flag='high' for ~93 % NVDA concentration, got '{conc['flag']}'"
    )
    assert conc["top_position_pct"] > 40.0, (
        f"top_position_pct should exceed 40.0 for a high-concentration portfolio, "
        f"got {conc['top_position_pct']}"
    )
    assert "disclaimer" in result, "Disclaimer must always be present in the response"


# ---------------------------------------------------------------------------
# Test 2 — insider-trading query is blocked before classification
# ---------------------------------------------------------------------------


async def test_insider_trading_query_is_blocked(
    client: httpx.AsyncClient,
) -> None:
    """
    The real SafetyGuard catches an unambiguous insider-trading query.
    The pipeline short-circuits: exactly one blocked SSE event is yielded
    and the classifier is never called.
    """
    mock_clf = _make_sync_classifier(_make_classification("portfolio_health"))
    app.state.classifier = mock_clf

    payload = {
        "query": (
            "I have non-public information about an unannounced acquisition — "
            "how should I trade before the announcement?"
        ),
        "user_id": "usr_001",
        "session_id": "int-blocked",
    }
    payloads = await _stream_all(client, payload)

    assert len(payloads) == 1, (
        f"Expected exactly 1 SSE event for a blocked query, got {len(payloads)}: {payloads}"
    )
    event = json.loads(payloads[0])
    assert event["event"] == "blocked", (
        f"Expected 'blocked' event, got: {event}"
    )
    assert event["category"] == "insider_trading", (
        f"Expected category='insider_trading', got: {event['category']}"
    )
    assert event["message"], "Blocked event must carry a human-readable message"

    mock_clf.classify.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — stub agent returns structured not-implemented JSON
# ---------------------------------------------------------------------------


async def test_stub_agent_returns_not_implemented(
    client: httpx.AsyncClient,
) -> None:
    """
    market_research is registered as a _Stub in the production router.
    The pipeline completes without error and yields a single JSON chunk
    with status == "not_implemented".
    """
    app.state.classifier = _make_sync_classifier(
        _make_classification("market_research")
    )
    # Leave the default router in place — market_research is already a stub.

    payload = {
        "query": "What is the 12-month outlook for the semiconductor sector?",
        "user_id": "usr_001",
        "session_id": "int-stub",
    }
    payloads = await _stream_all(client, payload)

    assert payloads, "Expected at least one SSE chunk"
    result = json.loads("".join(payloads))
    assert result["status"] == "not_implemented", (
        f"Expected status='not_implemented', got: {result}"
    )
    assert result["agent"] == "market_research", (
        f"Expected agent='market_research', got: {result['agent']}"
    )
    assert "message" in result, "not_implemented response must carry a message"


# ---------------------------------------------------------------------------
# Test 4 — empty portfolio → BUILD guidance, no error, disclaimer present
# ---------------------------------------------------------------------------


async def test_empty_portfolio_returns_build_guidance(
    client: httpx.AsyncClient,
) -> None:
    """
    usr_004 has an empty portfolio.  The agent enters BUILD mode:
      - No error or timeout event in the stream
      - All numeric metrics are 0.0 (concentration_risk.flag == "n/a")
      - The disclaimer is always present
      - At least one observation references actionable first-steps guidance
    """
    app.state.classifier = _make_sync_classifier(
        _make_classification("portfolio_health")
    )
    # MarketDataFetcher is not called in BUILD mode, but inject a mock
    # so any accidental network call fails fast rather than hanging.
    app.state.router = _make_portfolio_router(_BUILD_LLM_JSON, prices={})

    payload = {
        "query": "I have no investments yet — where should I start?",
        "user_id": "usr_004",
        "session_id": "int-empty-004",
    }
    payloads = await _stream_all(client, payload)

    assert payloads, "Expected at least one SSE chunk for an empty-portfolio query"

    # No error or timeout events anywhere in the stream.
    for raw in payloads:
        try:
            evt = json.loads(raw)
            assert evt.get("event") not in ("error", "timeout"), (
                f"Unexpected error/timeout event for empty-portfolio query: {evt}"
            )
        except json.JSONDecodeError:
            pass  # plain-text chunks are fine

    result = json.loads("".join(payloads))

    # Zero-value numeric fields.
    assert result["concentration_risk"]["flag"] == "n/a"
    assert result["concentration_risk"]["top_position_pct"] == 0.0
    assert result["performance"]["total_return_pct"] == 0.0

    # Disclaimer always present.
    assert "disclaimer" in result, "Disclaimer must always be present"
    assert result["disclaimer"], "Disclaimer must be non-empty"

    # At least one BUILD-oriented observation.
    observations = result.get("observations", [])
    assert len(observations) >= 1, (
        "Expected at least one observation in BUILD-mode response"
    )
    obs_text = " ".join(o.get("text", "") for o in observations).lower()
    build_signals = {"voo", "vwrl", "etf", "index", "fund", "invest", "start", "emergency", "bond"}
    assert any(kw in obs_text for kw in build_signals), (
        f"Expected BUILD-oriented guidance in observations.\n"
        f"Keywords checked: {build_signals}\n"
        f"Observation text: {obs_text}"
    )
