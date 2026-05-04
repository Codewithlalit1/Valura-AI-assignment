"""
Portfolio Health Agent tests.

All tests mock both the MarketDataFetcher and the OpenAI AsyncClient —
no OPENAI_API_KEY and no network calls required.

Mock strategy
-------------
- MarketDataFetcher is injected via PortfolioHealthAgent(market_fetcher=mock).
- The OpenAI async client is injected by setting agent._client = mock_client
  after construction (same lazy-init backdoor used by the classifier tests).
- The async stream is simulated with an async generator that yields pre-built
  chunks, matching the shape AsyncOpenAI returns with stream=True.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.portfolio_health.agent import (
    PortfolioHealthAgent,
    _build_messages_empty,
    _build_messages_populated,
    _format_holdings_table,
)
from src.agents.portfolio_health.calculations import (
    calculate_benchmark_comparison,
    calculate_concentration,
    calculate_performance,
)
from src.agents.portfolio_health.schema import (
    BenchmarkComparison,
    ConcentrationRisk,
    PerformanceMetrics,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE = json.dumps(
    {
        "concentration_risk": {
            "top_position_pct": 92.9,
            "top_3_positions_pct": 99.2,
            "flag": "high",
        },
        "performance": {
            "total_return_pct": 305.7,
            "annualized_return_pct": 305.7,
        },
        "benchmark_comparison": {
            "benchmark": "S&P 500",
            "portfolio_return_pct": 305.7,
            "benchmark_return_pct": 27.1,
            "alpha_pct": 278.6,
        },
        "observations": [
            {
                "severity": "warning",
                "text": "NVDA makes up 92.9% of your portfolio — dangerously concentrated.",
            },
            {
                "severity": "tip",
                "text": "Consider trimming NVDA and spreading into other sectors.",
            },
        ],
        "disclaimer": (
            "This is not investment advice. Portfolio analysis is for informational "
            "purposes only. Past performance does not guarantee future results. "
            "Please consult a qualified financial adviser before making investment decisions."
        ),
    }
)

_BUILD_RESPONSE = json.dumps(
    {
        "concentration_risk": {"top_position_pct": 0.0, "top_3_positions_pct": 0.0, "flag": "n/a"},
        "performance": {"total_return_pct": 0.0, "annualized_return_pct": 0.0},
        "benchmark_comparison": {
            "benchmark": "",
            "portfolio_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "alpha_pct": 0.0,
        },
        "observations": [
            {"severity": "tip", "text": "Start with a low-cost index fund like VOO."},
            {"severity": "info", "text": "Even $50/month in an S&P 500 ETF compounds over time."},
        ],
        "disclaimer": (
            "This is not investment advice. Portfolio analysis is for informational "
            "purposes only. Past performance does not guarantee future results. "
            "Please consult a qualified financial adviser before making investment decisions."
        ),
    }
)


def _make_async_stream(content: str):
    """Return a mock that behaves like AsyncOpenAI's stream=True response."""

    async def _chunk_iter():
        for char in content:
            chunk = MagicMock()
            chunk.choices[0].delta.content = char
            yield chunk
        # Final chunk with None content (as OpenAI does)
        tail = MagicMock()
        tail.choices[0].delta.content = None
        yield tail

    return _chunk_iter()


def _make_mock_client(content: str) -> MagicMock:
    """Build a mock AsyncOpenAI client whose create() returns a stream."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_async_stream(content)
    )
    return mock_client


def _make_mock_fetcher(
    prices: dict[str, float] | None = None,
    bench_return: float | None = 0.271,
) -> MagicMock:
    """Build a mock MarketDataFetcher with controllable return values."""
    fetcher = MagicMock()
    fetcher.get_current_prices.return_value = prices or {}
    fetcher.get_benchmark_return.return_value = bench_return
    return fetcher


async def _collect(gen: AsyncGenerator[str, None]) -> str:
    """Drain an async generator and concatenate all chunks."""
    chunks: list[str] = []
    async for chunk in gen:
        chunks.append(chunk)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Test 1 — empty portfolio does not crash and includes disclaimer + BUILD hints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_portfolio_does_not_crash(load_user) -> None:
    """
    user_004 has an empty portfolio.  analyze() must complete without raising
    and the collected JSON must contain 'disclaimer'.
    """
    user = load_user("usr_004")
    agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())
    agent._client = _make_mock_client(_BUILD_RESPONSE)

    result_str = await _collect(agent.analyze(user, []))

    assert result_str, "No output yielded for empty portfolio"
    result = json.loads(result_str)
    assert "disclaimer" in result, "disclaimer field missing"
    assert result["disclaimer"], "disclaimer must be non-empty"
    # Numeric metrics must be zero for empty portfolio
    assert result["concentration_risk"]["top_position_pct"] == 0.0
    assert result["performance"]["total_return_pct"] == 0.0


# ---------------------------------------------------------------------------
# Test 2 — concentrated portfolio flags as high
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concentrated_portfolio_flags_high(load_user) -> None:
    """
    user_002 has ~93% in NVDA.  The concentration flag in the response must
    indicate 'high' (or 'warning' to match the skeleton's original assertion).
    """
    user = load_user("usr_conc_002")
    agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())
    agent._client = _make_mock_client(_VALID_RESPONSE)

    result_str = await _collect(agent.analyze(user, []))
    result = json.loads(result_str)

    flag = result["concentration_risk"]["flag"]
    assert flag in {"high", "warning"}, f"Expected high concentration flag, got: {flag!r}"


# ---------------------------------------------------------------------------
# Test 3 — disclaimer is always present (even for populated portfolios)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disclaimer_always_present(load_user) -> None:
    """Every response — populated and empty — must carry the disclaimer."""
    for user_id, mock_response in [
        ("usr_aggr_001", _VALID_RESPONSE),
        ("usr_004",      _BUILD_RESPONSE),
    ]:
        user = load_user(user_id)
        agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())
        agent._client = _make_mock_client(mock_response)

        result = json.loads(await _collect(agent.analyze(user, [])))

        assert "disclaimer" in result, f"disclaimer missing for {user_id}"
        assert "not investment advice" in result["disclaimer"].lower(), (
            f"disclaimer text wrong for {user_id}: {result['disclaimer']!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — response is streamed in multiple chunks (not one big yield)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_is_streamed_in_chunks(load_user) -> None:
    """
    analyze() must yield MORE than one chunk — the stream must not be buffered
    into a single yield internally.
    """
    user = load_user("usr_aggr_001")
    agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())
    agent._client = _make_mock_client(_VALID_RESPONSE)

    chunks: list[str] = []
    async for chunk in agent.analyze(user, []):
        chunks.append(chunk)

    assert len(chunks) > 1, (
        f"Expected multiple chunks from streaming, got {len(chunks)}"
    )


# ---------------------------------------------------------------------------
# Test 5 — LLM exception yields fallback JSON with disclaimer (never raises)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_exception_returns_fallback(load_user) -> None:
    """
    When the OpenAI call raises, analyze() must yield a safe fallback JSON
    containing the disclaimer — it must never propagate the exception.
    """
    user = load_user("usr_aggr_001")
    agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())

    # Client whose create() raises unconditionally
    failing_client = MagicMock()
    failing_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("simulated API failure")
    )
    agent._client = failing_client

    result_str = await _collect(agent.analyze(user, []))
    result = json.loads(result_str)

    assert "disclaimer" in result
    assert result["disclaimer"]


# ---------------------------------------------------------------------------
# Test 6 — conversation history is included in the messages sent to the API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_history_forwarded(load_user) -> None:
    """
    Prior turns must appear in the messages list sent to the LLM, so the
    agent can reference context from earlier in the session.
    """
    history = [
        {"role": "user",      "content": "Am I too concentrated in tech?"},
        {"role": "assistant", "content": "Let me run a health check for you."},
    ]
    user = load_user("usr_aggr_001")
    agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())
    agent._client = _make_mock_client(_VALID_RESPONSE)

    await _collect(agent.analyze(user, history))

    call_kwargs = agent._client.chat.completions.create.call_args.kwargs
    messages: list[dict] = call_kwargs["messages"]
    non_system = [m for m in messages if m["role"] != "system"]
    all_text = " ".join(m["content"] for m in non_system)

    assert "too concentrated" in all_text, (
        "History turn missing from API messages"
    )


# ---------------------------------------------------------------------------
# Test 7 — full JSON shape has all required top-level keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_shape_has_required_keys(load_user) -> None:
    """Collected JSON must contain all five top-level keys from the spec."""
    user = load_user("usr_conc_002")
    agent = PortfolioHealthAgent(market_fetcher=_make_mock_fetcher())
    agent._client = _make_mock_client(_VALID_RESPONSE)

    result = json.loads(await _collect(agent.analyze(user, [])))

    required = {"concentration_risk", "performance", "benchmark_comparison",
                "observations", "disclaimer"}
    missing = required - result.keys()
    assert not missing, f"Missing top-level keys: {missing}"

    # observations must be a non-empty list of dicts with severity + text
    obs = result["observations"]
    assert isinstance(obs, list) and len(obs) > 0
    for o in obs:
        assert "severity" in o and "text" in o


# ---------------------------------------------------------------------------
# Test 8 — prompt builders are pure and testable without I/O
# ---------------------------------------------------------------------------


def test_build_messages_empty_contains_build_prompt() -> None:
    """_build_messages_empty must use the BUILD system prompt and include user info."""
    user = {"name": "Jamie Patel", "risk_profile": "moderate", "currency": "USD", "portfolio": []}
    messages = _build_messages_empty(user, [])

    assert messages[0]["role"] == "system"
    assert "BUILD" in messages[0]["content"] or "no holdings" in messages[0]["content"].lower()
    last = messages[-1]["content"]
    assert "Jamie Patel" in last
    assert "moderate" in last


def test_format_holdings_table_sorted_by_value() -> None:
    """Holdings table must be sorted largest-value-first."""
    holdings = [
        {"ticker": "BND",  "quantity": 10, "avg_cost_usd": 75.0,  "current_price_usd": 74.0},
        {"ticker": "NVDA", "quantity": 10, "avg_cost_usd": 200.0, "current_price_usd": 900.0},
    ]
    table = _format_holdings_table(holdings, {})
    lines = [l for l in table.splitlines() if l.strip() and not l.startswith("-")]
    # Header first, then NVDA (larger value) before BND
    assert lines[1].startswith("NVDA"), f"Expected NVDA first, got: {lines[1]}"


def test_build_messages_populated_injects_numbers() -> None:
    """The user message must contain the pre-computed metric values."""
    holdings = [
        {"ticker": "NVDA", "quantity": 150, "avg_cost_usd": 220.0, "current_price_usd": 892.5},
    ]
    prices = {"NVDA": 892.5}
    conc = calculate_concentration(holdings)
    perf = calculate_performance(holdings, prices)
    bench = calculate_benchmark_comparison(perf.total_return_pct, 27.1, "S&P 500")

    messages = _build_messages_populated(
        {"name": "Sam", "risk_profile": "moderate", "currency": "USD"},
        holdings, prices, conc, perf, bench, [],
    )
    user_msg = messages[-1]["content"]

    assert "NVDA" in user_msg
    assert "S&P 500" in user_msg
    assert "100.0" in user_msg or "100" in user_msg  # concentration ~100%
