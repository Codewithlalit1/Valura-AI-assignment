"""
Routing-accuracy tests for IntentClassifier against the 60-query gold set.

The LLM is mocked so no API key is required in CI.  The mock returns the
expected JSON verbatim — this exercises the classifier's JSON parsing,
agent normalisation, and entity extraction logic, not the LLM's intelligence.

Entity coverage note
--------------------
ExtractedEntities models five fields: tickers, topics, amount, period_years,
rate.  The remaining vocabulary (sectors, currency, frequency, horizon,
time_period, index, action, goal) is present in many gold-set fixtures but
is silently dropped by  ConfigDict(extra="ignore").  Tests only assert on
the five modelled fields.
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock

import pytest

from src.classifier.classifier import IntentClassifier
from src.classifier.schema import VALID_AGENTS

# ---------------------------------------------------------------------------
# Load gold set at collection time
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    pathlib.Path(__file__).parent.parent
    / "fixtures"
    / "test_queries"
    / "intent_classification.json"
)
_GOLD: list[dict] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))["queries"]

# Fields modelled in ExtractedEntities — anything else is dropped.
_SUPPORTED_ENTITY_FIELDS = {"tickers", "topics", "amount", "period_years", "rate"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classifier(expected_agent: str, expected_entities: dict) -> IntentClassifier:
    """Return an IntentClassifier whose underlying LLM call is fully mocked."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(
        {
            "intent": "test intent",
            "agent": expected_agent,
            "entities": expected_entities,
            "safety_verdict": "safe",
            "confidence": 0.9,
            "reasoning": "",
        }
    )
    clf = IntentClassifier(model="test-model")
    clf._client = MagicMock()
    clf._client.chat.completions.create.return_value = mock_response
    return clf


def _has_supported_entities(case: dict) -> bool:
    """True when the gold case has at least one entity in the modelled fields."""
    return bool(
        _SUPPORTED_ENTITY_FIELDS & set(case.get("expected_entities") or {})
    )


# ---------------------------------------------------------------------------
# Test 1 — routing accuracy (all 60 queries)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=f"{c['expected_agent']}::{c['query'][:55]}") for c in _GOLD],
)
def test_routing_accuracy(case: dict) -> None:
    """
    For every gold-set query, the classifier must return the expected agent.

    This verifies parsing and normalisation logic — the mock LLM returns the
    expected agent JSON so the result is deterministic.
    """
    expected_agent = case["expected_agent"]
    expected_entities = case.get("expected_entities") or {}

    clf = _make_classifier(expected_agent, expected_entities)
    result = clf.classify(case["query"], [], {})

    assert result.agent == expected_agent, (
        f"Query   : {case['query']!r}\n"
        f"Expected: {expected_agent!r}\n"
        f"Got     : {result.agent!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — entity extraction (cases with supported entity fields)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"{c['expected_agent']}::{c['query'][:55]}")
        for c in _GOLD
        if _has_supported_entities(c)
    ],
)
def test_entity_extraction(case: dict) -> None:
    """
    For gold queries that carry tickers / amount / period_years / rate,
    the classifier must parse those values out of the LLM response.

    Topics are not asserted — the LLM may paraphrase them.
    """
    expected_agent = case["expected_agent"]
    expected_entities = case.get("expected_entities") or {}

    clf = _make_classifier(expected_agent, expected_entities)
    result = clf.classify(case["query"], [], {})
    ents = result.entities

    if "tickers" in expected_entities:
        for ticker in expected_entities["tickers"]:
            assert ticker.upper() in [t.upper() for t in ents.tickers], (
                f"Ticker {ticker!r} missing from {ents.tickers}  "
                f"(query: {case['query']!r})"
            )

    if "amount" in expected_entities:
        assert ents.amount == pytest.approx(expected_entities["amount"]), (
            f"amount mismatch — expected {expected_entities['amount']}, "
            f"got {ents.amount}  (query: {case['query']!r})"
        )

    if "period_years" in expected_entities:
        assert ents.period_years == pytest.approx(
            expected_entities["period_years"]
        ), (
            f"period_years mismatch — expected {expected_entities['period_years']}, "
            f"got {ents.period_years}  (query: {case['query']!r})"
        )

    if "rate" in expected_entities:
        assert ents.rate == pytest.approx(
            expected_entities["rate"], rel=1e-3
        ), (
            f"rate mismatch — expected {expected_entities['rate']}, "
            f"got {ents.rate}  (query: {case['query']!r})"
        )


# ---------------------------------------------------------------------------
# Test 3 — unknown agent is normalised to "support"
# ---------------------------------------------------------------------------


def test_unknown_agent_normalised_to_customer_support() -> None:
    """If the LLM returns an agent name not in VALID_AGENTS it must be normalised."""
    clf = IntentClassifier(model="test-model")
    raw = json.dumps(
        {
            "intent": "something odd",
            "agent": "definitely_not_a_real_agent",
            "entities": {},
            "safety_verdict": "safe",
            "confidence": 0.5,
            "reasoning": "",
        }
    )
    result = clf._parse(raw)
    assert result.agent == "customer_support"


# ---------------------------------------------------------------------------
# Test 4 — malformed JSON triggers fallback, never raises
# ---------------------------------------------------------------------------


def test_malformed_json_returns_fallback() -> None:
    """A non-JSON LLM response must cause classify() to return the safe fallback."""
    clf = IntentClassifier(model="test-model")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "this is not json {"
    clf._client = MagicMock()
    clf._client.chat.completions.create.return_value = mock_response

    result = clf.classify("any query", [], {})

    assert result.agent == "customer_support"
    assert result.confidence == 0.0
    assert "Classification unavailable" in result.reasoning


# ---------------------------------------------------------------------------
# Test 5 — coverage: all 8 agents appear in the gold set
# ---------------------------------------------------------------------------


def test_gold_set_covers_all_agents() -> None:
    """Sanity check: every VALID_AGENT appears at least once in the gold set."""
    agents_in_gold = {c["expected_agent"] for c in _GOLD}
    missing = VALID_AGENTS - agents_in_gold
    assert not missing, f"Agents missing from gold set: {missing}"


# ---------------------------------------------------------------------------
# Test 6 — history truncation: more than 6 turns are truncated to 6
# ---------------------------------------------------------------------------


def test_history_truncated_to_max_turns() -> None:
    """Classifier must not forward more than 6 prior turns to the LLM."""
    clf = IntentClassifier(model="test-model")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(
        {
            "intent": "test",
            "agent": "support",
            "entities": {},
            "safety_verdict": "safe",
            "confidence": 0.9,
            "reasoning": "",
        }
    )
    clf._client = MagicMock()
    clf._client.chat.completions.create.return_value = mock_response

    long_history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    clf.classify("latest query", long_history, {})

    call_kwargs = clf._client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    # messages = [system] + up to 6 history + [user query]
    # history turns are role=="user" messages that are NOT the final query
    history_messages = [
        m for m in messages[1:-1]  # exclude system and final user turn
        if m["role"] == "user"
    ]
    assert len(history_messages) <= 6, (
        f"Expected at most 6 history turns, got {len(history_messages)}"
    )
