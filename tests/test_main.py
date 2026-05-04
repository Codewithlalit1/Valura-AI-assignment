"""
Integration tests for the FastAPI application (src/main.py).

Strategy
--------
All tests use httpx.AsyncClient with ASGITransport so the real ASGI app
runs in-process — no TCP port needed and no OPENAI_API_KEY required.

After the lifespan starts (lazily initialised components, no network calls),
each test overrides the relevant app.state components with mocks so the
pipeline runs end-to-end through the HTTP + SSE layer without touching
external services.

SSE parsing
-----------
The raw wire format is `data: <payload>\n\n`.  _parse_sse() strips the
prefix and returns a list of raw payload strings.  For structured events
(blocked / timeout / error) the payload is JSON.  For agent chunks it may
be a raw JSON fragment or plain text.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.classifier.schema import ClassificationResult, ExtractedEntities
from src.main import app, _PIPELINE_TIMEOUT
from src.safety.guard import SafetyGuard, SafetyResult
from src.session.store import InMemorySessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = "http://test"
_QUERY_PAYLOAD = {"query": "how is my portfolio?", "user_id": "usr_004", "session_id": "s1"}


def _parse_sse(text: str) -> list[str]:
    """Return the payload of every `data: ...` line in an SSE stream."""
    return [
        line[6:]
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


async def _stream_all(client: httpx.AsyncClient, payload: dict) -> list[str]:
    """POST /query and collect all SSE data payloads as a list of strings."""
    body = ""
    async with client.stream("POST", "/query", json=payload) as r:
        async for text in r.aiter_text():
            body += text
    return _parse_sse(body)


def _mock_classification(agent: str = "support") -> ClassificationResult:
    return ClassificationResult(
        intent="test intent",
        agent=agent,
        entities=ExtractedEntities(),
    )


def _mock_classifier(result: ClassificationResult) -> MagicMock:
    clf = MagicMock()
    clf.classify.return_value = result
    return clf


def _mock_router(chunks: list[str]) -> MagicMock:
    """Build a mock AgentRouter whose route() yields the given chunks."""

    async def _gen(*_args, **_kwargs):
        for c in chunks:
            yield c

    router = MagicMock()
    router.route = _gen
    return router


# ---------------------------------------------------------------------------
# Shared async client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """
    Async HTTPX client wired to the FastAPI app.

    httpx.ASGITransport does NOT trigger the ASGI lifespan protocol, so
    app.state is initialised here directly — the same values the lifespan
    would create — giving each test a clean, isolated set of components.
    """
    from src.classifier.classifier import IntentClassifier
    from src.router import create_router

    app.state.safety_guard = SafetyGuard()
    app.state.classifier = IntentClassifier(model="gpt-4o-mini")
    app.state.router = create_router(model="gpt-4o-mini")
    app.state.session_store = InMemorySessionStore()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=_BASE,
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1 — health endpoint
# ---------------------------------------------------------------------------


async def test_health_returns_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 2 — safety guard blocks: SSE blocked event, no further processing
# ---------------------------------------------------------------------------


async def test_safety_blocked_yields_blocked_event(client: httpx.AsyncClient) -> None:
    mock_guard = MagicMock(spec=SafetyGuard)
    mock_guard.check.return_value = SafetyResult(
        blocked=True,
        category="insider_trading",
        response="I cannot help with that.",
    )
    app.state.safety_guard = mock_guard

    payloads = await _stream_all(client, _QUERY_PAYLOAD)

    assert len(payloads) == 1, f"Expected 1 SSE event for a blocked query, got: {payloads}"
    event = json.loads(payloads[0])
    assert event["event"] == "blocked"
    assert event["category"] == "insider_trading"
    assert event["message"]


async def test_safety_blocked_does_not_call_classifier(client: httpx.AsyncClient) -> None:
    mock_guard = MagicMock(spec=SafetyGuard)
    mock_guard.check.return_value = SafetyResult(
        blocked=True, category="reckless_advice", response="Blocked."
    )
    mock_clf = _mock_classifier(_mock_classification())
    app.state.safety_guard = mock_guard
    app.state.classifier = mock_clf

    await _stream_all(client, _QUERY_PAYLOAD)

    mock_clf.classify.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — stub agent returns structured not-implemented JSON via SSE
# ---------------------------------------------------------------------------


async def test_stub_agent_returns_not_implemented(client: httpx.AsyncClient) -> None:
    app.state.safety_guard = MagicMock(spec=SafetyGuard, **{
        "check.return_value": SafetyResult(blocked=False)
    })
    app.state.classifier = _mock_classifier(_mock_classification(agent="market_research"))
    # Leave the real router in place — market_research is a stub there.

    payloads = await _stream_all(client, _QUERY_PAYLOAD)

    full = "".join(payloads)
    result = json.loads(full)
    assert result["status"] == "not_implemented"
    assert result["agent"] == "market_research"


# ---------------------------------------------------------------------------
# Test 4 — real agent chunks are forwarded token-by-token
# ---------------------------------------------------------------------------


async def test_real_agent_chunks_forwarded(client: httpx.AsyncClient) -> None:
    chunks = ['{"concentration_risk"', ': {"flag": "low"}', ', "disclaimer": "test"}']
    app.state.safety_guard = MagicMock(spec=SafetyGuard, **{
        "check.return_value": SafetyResult(blocked=False)
    })
    app.state.classifier = _mock_classifier(
        _mock_classification(agent="portfolio_health")
    )
    app.state.router = _mock_router(chunks)

    payloads = await _stream_all(client, _QUERY_PAYLOAD)

    assert len(payloads) == len(chunks), (
        f"Expected {len(chunks)} SSE events, got {len(payloads)}: {payloads}"
    )
    assert "".join(payloads) == "".join(chunks)


# ---------------------------------------------------------------------------
# Test 5 — session is persisted after a complete pipeline run
# ---------------------------------------------------------------------------


async def test_session_persisted_after_pipeline(client: httpx.AsyncClient) -> None:
    app.state.safety_guard = MagicMock(spec=SafetyGuard, **{
        "check.return_value": SafetyResult(blocked=False)
    })
    app.state.classifier = _mock_classifier(_mock_classification(agent="support"))
    app.state.router = _mock_router(['{"status": "not_implemented"}'])

    payload = {"query": "test query", "user_id": "u1", "session_id": "persist-session"}
    await _stream_all(client, payload)

    history = app.state.session_store.get_history("persist-session")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "test query"
    assert history[1]["role"] == "assistant"
    assert '{"status": "not_implemented"}' in history[1]["content"]


# ---------------------------------------------------------------------------
# Test 6 — session history is forwarded to the classifier on follow-up
# ---------------------------------------------------------------------------


async def test_history_forwarded_to_classifier(client: httpx.AsyncClient) -> None:
    store = InMemorySessionStore()
    store.add_turn("hist-session", "user", "tell me about AAPL")
    store.add_turn("hist-session", "assistant", "AAPL is up 5% today.")
    app.state.session_store = store

    captured: list = []

    def _capture_classify(query, history, user_profile):
        captured.append(history[:])
        return _mock_classification()

    mock_clf = MagicMock()
    mock_clf.classify.side_effect = _capture_classify
    app.state.safety_guard = MagicMock(spec=SafetyGuard, **{
        "check.return_value": SafetyResult(blocked=False)
    })
    app.state.classifier = mock_clf
    app.state.router = _mock_router(["ok"])

    await _stream_all(client, {
        "query": "what about NVDA?",
        "user_id": "u1",
        "session_id": "hist-session",
    })

    assert captured, "classify() was never called"
    forwarded = captured[0]
    assert any("AAPL" in t.get("content", "") for t in forwarded), (
        f"Prior turn not forwarded; got: {forwarded}"
    )


# ---------------------------------------------------------------------------
# Test 7 — timeout yields timeout SSE event, never raises
# ---------------------------------------------------------------------------


async def test_timeout_yields_timeout_event(client: httpx.AsyncClient) -> None:
    """
    Use an agent that sleeps longer than the pipeline timeout.
    Override _PIPELINE_TIMEOUT to 0.05 s so the test completes quickly.
    """
    import src.main as main_module

    original_timeout = main_module._PIPELINE_TIMEOUT
    main_module._PIPELINE_TIMEOUT = 0.05  # 50 ms

    async def _slow_gen(*_args, **_kwargs):
        await asyncio.sleep(5)  # longer than the injected 50 ms timeout
        yield "should never reach here"

    slow_router = MagicMock()
    slow_router.route = _slow_gen

    app.state.safety_guard = MagicMock(spec=SafetyGuard, **{
        "check.return_value": SafetyResult(blocked=False)
    })
    app.state.classifier = _mock_classifier(_mock_classification())
    app.state.router = slow_router

    try:
        payloads = await _stream_all(client, _QUERY_PAYLOAD)
    finally:
        main_module._PIPELINE_TIMEOUT = original_timeout  # always restore

    assert payloads, "No SSE events received"
    event = json.loads(payloads[-1])  # timeout event is the last one
    assert event["event"] == "timeout", f"Expected timeout event, got: {event}"


# ---------------------------------------------------------------------------
# Test 8 — pipeline exception yields error SSE event, never raises HTTP 500
# ---------------------------------------------------------------------------


async def test_pipeline_exception_returns_error_event(client: httpx.AsyncClient) -> None:
    async def _broken_gen(*_args, **_kwargs):
        raise RuntimeError("simulated agent failure")
        yield  # pragma: no cover

    broken_router = MagicMock()
    broken_router.route = _broken_gen

    app.state.safety_guard = MagicMock(spec=SafetyGuard, **{
        "check.return_value": SafetyResult(blocked=False)
    })
    app.state.classifier = _mock_classifier(_mock_classification())
    app.state.router = broken_router

    payloads = await _stream_all(client, _QUERY_PAYLOAD)

    assert payloads, "No SSE events yielded on pipeline error"
    event = json.loads(payloads[-1])
    assert event["event"] == "error"
    # Stack trace must never be exposed
    assert "simulated agent failure" not in event["message"]
    assert "Traceback" not in event["message"]
