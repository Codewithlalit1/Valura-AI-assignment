"""
AgentRouter tests.

All tests mock every outbound dependency (OpenAI, MarketDataFetcher) —
no OPENAI_API_KEY and no network calls required.

Coverage
--------
Test 1  — stub agent yields exactly one not-implemented JSON chunk
Test 2  — stub response has all required JSON keys (status, intent, agent, entities, message)
Test 3  — all seven non-portfolio agents produce stub responses (parametrized)
Test 4  — unknown agent name (not in VALID_AGENTS) also gets stub response
Test 5  — real agent (portfolio_health) output is forwarded chunk-by-chunk
Test 6  — real agent exception is caught; router yields error JSON, never raises
Test 7  — router registry contains all eight VALID_AGENTS
Test 8  — is_real() correctly distinguishes real from stub entries
Test 9  — entities dict in stub response matches the ClassificationResult's entities
Test 10 — router round-trip with full ClassificationResult (intent, entities populated)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.classifier.schema import ClassificationResult, ExtractedEntities, VALID_AGENTS
from src.router import AgentRouter, _Stub, _not_implemented_json, create_router

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cr(
    agent: str = "portfolio_health",
    intent: str = "portfolio health check",
    tickers: list[str] | None = None,
    topics: list[str] | None = None,
    amount: float | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        agent=agent,
        entities=ExtractedEntities(
            tickers=tickers or [],
            topics=topics or [],
            amount=amount,
        ),
    )


async def _collect(gen) -> list[str]:
    """Drain an async generator and return the list of yielded strings."""
    chunks: list[str] = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


def _make_real_agent(content: str = '{"ok": true}') -> MagicMock:
    """
    Build a mock agent whose analyze() async-generates the given content
    one character at a time (mirrors how OpenAI streaming works).
    """
    async def _gen(*_args, **_kwargs):
        for char in content:
            yield char

    agent = MagicMock()
    agent.analyze = _gen
    return agent


def _stub_registry(extra: dict | None = None) -> dict[str, object]:
    """All agents stubbed, with optional overrides (e.g. a real mock agent)."""
    reg: dict[str, object] = {name: _Stub(name) for name in VALID_AGENTS}
    if extra:
        reg.update(extra)
    return reg


# ---------------------------------------------------------------------------
# Test 1 — stub agent yields exactly one chunk
# ---------------------------------------------------------------------------


async def test_stub_yields_single_chunk() -> None:
    router = AgentRouter(_stub_registry())
    cr = _make_cr(agent="market_research")

    chunks = await _collect(router.route(cr, {}, []))

    assert len(chunks) == 1, (
        f"Stub must yield exactly 1 chunk, got {len(chunks)}: {chunks}"
    )


# ---------------------------------------------------------------------------
# Test 2 — stub response has all required keys and correct values
# ---------------------------------------------------------------------------


async def test_stub_response_shape() -> None:
    router = AgentRouter(_stub_registry())
    cr = _make_cr(
        agent="investment_strategy",
        intent="should I buy more NVDA?",
        tickers=["NVDA"],
    )

    chunks = await _collect(router.route(cr, {}, []))
    result = json.loads(chunks[0])

    assert result["status"] == "not_implemented"
    assert result["intent"] == "should I buy more NVDA?"
    assert result["agent"] == "investment_strategy"
    assert "entities" in result
    assert "message" in result
    assert result["message"]  # non-empty


# ---------------------------------------------------------------------------
# Test 3 — every non-portfolio agent is a stub (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_name",
    sorted(VALID_AGENTS - {"portfolio_health"}),
)
async def test_all_stubs_return_not_implemented(agent_name: str) -> None:
    """Every agent other than portfolio_health must return not_implemented."""
    router = create_router(model="gpt-4o-mini")
    cr = _make_cr(agent=agent_name, intent=f"{agent_name} query")

    chunks = await _collect(router.route(cr, {}, []))

    assert len(chunks) == 1
    result = json.loads(chunks[0])
    assert result["status"] == "not_implemented"
    assert result["agent"] == agent_name


# ---------------------------------------------------------------------------
# Test 4 — truly unknown agent name also gets a stub response
# ---------------------------------------------------------------------------


async def test_unknown_agent_gets_stub_response() -> None:
    router = AgentRouter(_stub_registry())
    cr = _make_cr(agent="nonexistent_fantasy_agent")

    chunks = await _collect(router.route(cr, {}, []))

    assert len(chunks) == 1
    result = json.loads(chunks[0])
    assert result["status"] == "not_implemented"


# ---------------------------------------------------------------------------
# Test 5 — real agent output is forwarded chunk-by-chunk
# ---------------------------------------------------------------------------


async def test_real_agent_output_forwarded() -> None:
    content = '{"concentration_risk": {}, "disclaimer": "test"}'
    mock_agent = _make_real_agent(content)
    router = AgentRouter({**_stub_registry(), "portfolio_health": mock_agent})
    cr = _make_cr(agent="portfolio_health")

    chunks = await _collect(router.route(cr, {}, []))

    assert len(chunks) == len(content), (
        "Each character must be a separate chunk (streaming, not buffered)"
    )
    assert "".join(chunks) == content


# ---------------------------------------------------------------------------
# Test 6 — real agent exception is caught; yields error JSON, never raises
# ---------------------------------------------------------------------------


async def test_real_agent_exception_caught() -> None:
    async def _broken_analyze(*_args, **_kwargs):
        raise RuntimeError("disk on fire")
        yield  # pragma: no cover — makes it an async generator

    broken = MagicMock()
    broken.analyze = _broken_analyze
    router = AgentRouter({**_stub_registry(), "portfolio_health": broken})
    cr = _make_cr(agent="portfolio_health")

    # Must not raise
    chunks = await _collect(router.route(cr, {}, []))

    assert len(chunks) == 1
    result = json.loads(chunks[0])
    assert result["status"] == "error"
    assert "disk on fire" in result["message"]


# ---------------------------------------------------------------------------
# Test 7 — registry contains all eight VALID_AGENTS
# ---------------------------------------------------------------------------


def test_create_router_registers_all_agents() -> None:
    router = create_router(model="gpt-4o-mini")
    registered = set(router.registry.keys())

    missing = VALID_AGENTS - registered
    assert not missing, f"These agents are missing from the registry: {missing}"
    assert registered == VALID_AGENTS


# ---------------------------------------------------------------------------
# Test 8 — is_real() correctly distinguishes real from stub
# ---------------------------------------------------------------------------


def test_is_real_distinguishes_real_from_stub() -> None:
    router = create_router(model="gpt-4o-mini")

    assert router.is_real("portfolio_health") is True

    for name in VALID_AGENTS - {"portfolio_health"}:
        assert router.is_real(name) is False, (
            f"{name} should be a stub, but is_real() returned True"
        )

    assert router.is_real("does_not_exist") is False


# ---------------------------------------------------------------------------
# Test 9 — entities in stub response mirror the ClassificationResult
# ---------------------------------------------------------------------------


async def test_stub_entities_mirror_classification_result() -> None:
    router = AgentRouter(_stub_registry())
    cr = _make_cr(
        agent="financial_calculator",
        intent="compound interest calculation",
        tickers=[],
        topics=["compound interest"],
        amount=10000.0,
    )

    chunks = await _collect(router.route(cr, {}, []))
    result = json.loads(chunks[0])

    entities = result["entities"]
    assert entities["amount"] == 10000.0
    assert "compound interest" in entities["topics"]


# ---------------------------------------------------------------------------
# Test 10 — _not_implemented_json helper produces valid JSON
# ---------------------------------------------------------------------------


def test_not_implemented_json_is_valid() -> None:
    cr = _make_cr(
        agent="risk_assessment",
        intent="what is my downside risk?",
        tickers=["AAPL", "NVDA"],
    )
    raw = _not_implemented_json(cr)
    parsed = json.loads(raw)  # must not raise

    assert parsed["status"] == "not_implemented"
    assert parsed["agent"] == "risk_assessment"
    assert "AAPL" in parsed["entities"]["tickers"]
    assert "NVDA" in parsed["entities"]["tickers"]
