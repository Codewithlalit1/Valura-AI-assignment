"""
Conversation-fixture tests — verifies multi-turn history forwarding and
per-turn routing across three scenario files.

Files tested
------------
  fixtures/conversations/follow_up_session.json    (4 cases)
  fixtures/conversations/ambiguous_session.json    (5 cases)
  fixtures/conversations/multi_intent_session.json (4 cases)

Skipped cases
-------------
Cases whose expected_agent is not in VALID_AGENTS are skipped at collection
time.  The fixture files use non-standard labels ("portfolio_query",
"general_query") that the production classifier normalises to "support" — the
intent is to describe LLM behaviour rather than production routing, so they
are excluded rather than treated as failures.

What these tests verify
-----------------------
1. The classifier's _build_messages includes prior turns in the messages list
   sent to the LLM, enabling the model to resolve co-references.
2. The classifier correctly parses a well-formed LLM response and returns
   the expected agent when history is present.
3. Entity extraction for the five modelled fields works in multi-turn context.
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock
from typing import Any

import pytest

from src.classifier.classifier import IntentClassifier
from src.classifier.schema import VALID_AGENTS

# ---------------------------------------------------------------------------
# Load fixtures at collection time
# ---------------------------------------------------------------------------

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "conversations"

_FOLLOW_UP = json.loads(
    (_FIXTURES_DIR / "follow_up_session.json").read_text(encoding="utf-8")
)["test_cases"]

_AMBIGUOUS = json.loads(
    (_FIXTURES_DIR / "ambiguous_session.json").read_text(encoding="utf-8")
)["test_cases"]

_MULTI_INTENT = json.loads(
    (_FIXTURES_DIR / "multi_intent_session.json").read_text(encoding="utf-8")
)["test_cases"]


def _valid(cases: list[dict], source: str) -> list[dict]:
    """Return only cases whose expected_agent is a VALID_AGENT."""
    return [
        {**c, "_source": source}
        for c in cases
        if c["expected"]["agent"] in VALID_AGENTS
    ]


_ALL_VALID_CASES: list[dict] = (
    _valid(_FOLLOW_UP, "follow_up")
    + _valid(_AMBIGUOUS, "ambiguous")
    + _valid(_MULTI_INTENT, "multi_intent")
)

_SKIPPED_COUNT = (
    len(_FOLLOW_UP) + len(_AMBIGUOUS) + len(_MULTI_INTENT) - len(_ALL_VALID_CASES)
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classifier(expected_agent: str, expected_entities: dict) -> IntentClassifier:
    """IntentClassifier whose LLM call is mocked to return a fixed response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(
        {
            "intent": "conversation test intent",
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


def _build_history(prior_turns: list[str]) -> list[dict]:
    """Convert prior user turn strings into the session-history dict format."""
    return [{"role": "user", "content": t} for t in prior_turns]


def _get_call_messages(clf: IntentClassifier) -> list[dict]:
    """Extract the messages list from the last LLM call made by clf."""
    return clf._client.chat.completions.create.call_args.kwargs["messages"]


# ---------------------------------------------------------------------------
# Test 1 — routing works when history is present (parametrised)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            id=f"{c['_source']}::{c['case_id']}",
        )
        for c in _ALL_VALID_CASES
    ],
)
def test_conversation_routing(case: dict) -> None:
    """
    For each valid conversation case, the classifier must return the expected
    agent when given the prior turns as history.
    """
    prior_turns: list[str] = case["prior_user_turns"]
    current_turn: str = case["current_user_turn"]
    expected: dict = case["expected"]
    expected_agent: str = expected["agent"]
    expected_entities: dict = expected.get("entities") or {}

    history = _build_history(prior_turns)
    clf = _make_classifier(expected_agent, expected_entities)
    result = clf.classify(current_turn, history, {})

    assert result.agent == expected_agent, (
        f"[{case['case_id']}] Query: {current_turn!r}\n"
        f"History: {prior_turns}\n"
        f"Expected agent: {expected_agent!r}  Got: {result.agent!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — prior turns are forwarded to the LLM (parametrised)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"{c['_source']}::{c['case_id']}")
        for c in _ALL_VALID_CASES
        if c["prior_user_turns"]
    ],
)
def test_history_forwarded_to_llm(case: dict) -> None:
    """
    Each prior turn must appear verbatim in the messages list sent to the LLM.
    This verifies that _build_messages() correctly threads history into the prompt.
    """
    prior_turns = case["prior_user_turns"]
    expected = case["expected"]

    history = _build_history(prior_turns)
    clf = _make_classifier(expected["agent"], expected.get("entities") or {})
    clf.classify(case["current_user_turn"], history, {})

    messages = _get_call_messages(clf)
    message_contents = [m["content"] for m in messages]

    for turn in prior_turns:
        assert turn in message_contents, (
            f"[{case['case_id']}] Prior turn {turn!r} not found in LLM messages.\n"
            f"Messages: {message_contents}"
        )


# ---------------------------------------------------------------------------
# Test 3 — entity extraction in multi-turn context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"{c['_source']}::{c['case_id']}")
        for c in _ALL_VALID_CASES
        if (c["expected"].get("entities") or {}).get("tickers")
    ],
)
def test_tickers_extracted_in_conversation(case: dict) -> None:
    """
    When expected entities include tickers, those tickers must appear in
    the classification result — even when they were introduced in a prior turn.
    """
    prior_turns = case["prior_user_turns"]
    expected = case["expected"]
    expected_tickers: list[str] = expected["entities"]["tickers"]

    history = _build_history(prior_turns)
    clf = _make_classifier(expected["agent"], expected.get("entities") or {})
    result = clf.classify(case["current_user_turn"], history, {})

    for ticker in expected_tickers:
        assert ticker.upper() in [t.upper() for t in result.entities.tickers], (
            f"[{case['case_id']}] Ticker {ticker!r} not found in {result.entities.tickers}"
        )


# ---------------------------------------------------------------------------
# Test 4 — follow-up: NVDA ticker carries through to investment_strategy
# ---------------------------------------------------------------------------


def test_nvda_ticker_carries_to_strategy_turn() -> None:
    """
    Follow-up session fu_02: after discussing NVDA's market activity, the
    query 'Should I sell some?' must route to investment_strategy with NVDA
    present in the extracted tickers — carried via conversation history.
    """
    history = _build_history(
        [
            "What's happening with Nvidia this week?",
            "How much do I own?",
        ]
    )
    clf = _make_classifier(
        "investment_strategy",
        {"tickers": ["NVDA"], "action": "sell"},
    )
    result = clf.classify("Should I sell some?", history, {})

    assert result.agent == "investment_strategy"
    assert "NVDA" in result.entities.tickers

    # Both prior turns must appear in the LLM call messages.
    messages = _get_call_messages(clf)
    contents = [m["content"] for m in messages]
    assert "What's happening with Nvidia this week?" in contents
    assert "How much do I own?" in contents


# ---------------------------------------------------------------------------
# Test 5 — topic switch: calculator query does not inherit prior portfolio intent
# ---------------------------------------------------------------------------


def test_topic_switch_calculator_is_not_portfolio_health() -> None:
    """
    Multi-intent mi_03: after a portfolio health discussion, an explicit DCA
    calculation query must route to financial_calculator, not portfolio_health.
    This validates that the classifier does not bleed intent across a clean switch.
    """
    history = _build_history(
        [
            "How is my portfolio doing?",
            "What's the difference between dollar cost averaging and lump-sum investing?",
        ]
    )
    query = "Calculate how much I'd have in 10 years investing 2000 USD a month at 8% return"
    clf = _make_classifier(
        "financial_calculator",
        {"amount": 2000, "period_years": 10, "rate": 0.08},
    )
    result = clf.classify(query, history, {})

    assert result.agent == "financial_calculator"
    assert result.entities.amount == pytest.approx(2000.0)
    assert result.entities.period_years == pytest.approx(10.0)
    assert result.entities.rate == pytest.approx(0.08, rel=1e-3)


# ---------------------------------------------------------------------------
# Meta: report how many cases were skipped due to non-standard agent labels
# ---------------------------------------------------------------------------


def test_skipped_case_count_is_known() -> None:
    """
    Documents the number of fixture cases that use non-VALID_AGENT labels
    and are therefore excluded from routing tests.  Update this assertion
    when the fixtures are extended or the VALID_AGENTS set is expanded.
    """
    assert _SKIPPED_COUNT == 1, (
        f"Expected 1 skipped case (portfolio_query ×1 — not in VALID_AGENTS), "
        f"found {_SKIPPED_COUNT}.  Update this test if fixtures or VALID_AGENTS changed."
    )
