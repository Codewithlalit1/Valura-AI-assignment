"""
Intent classifier tests.

All four tests mock the OpenAI client — no OPENAI_API_KEY required.

The mock is injected via the lazy-init backdoor:
    classifier._client = mock_client
This avoids patching the OpenAI constructor and keeps the tests independent
of how the client is constructed internally.

Test 1 – routing accuracy across all 60 fixture queries (threshold ≥ 85%)
Test 2 – entity extraction with subset+normalisation matching
Test 3 – fallback on OpenAI exception (no raise, safe result returned)
Test 4 – follow-up resolution: conversation history is forwarded to the API
         and the parsed result reflects the newly introduced entity
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.classifier.classifier import IntentClassifier
from src.classifier.schema import ClassificationResult, ExtractedEntities

# ---------------------------------------------------------------------------
# Load gold data at module level for parametrize and direct iteration
# ---------------------------------------------------------------------------

_IC_PATH = (
    Path(__file__).parent.parent
    / "fixtures"
    / "test_queries"
    / "intent_classification.json"
)
_IC = json.loads(_IC_PATH.read_text(encoding="utf-8"))
_QUERIES: list[dict] = _IC["queries"]

# Fields tracked by ExtractedEntities
_EXTRACTABLE = frozenset(["tickers", "topics", "amount", "period_years", "rate"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_response_content(
    agent: str,
    tickers: list[str] | None = None,
    topics: list[str] | None = None,
    amount: float | None = None,
    period_years: float | None = None,
    rate: float | None = None,
    safety_verdict: str = "safe",
    confidence: float = 0.9,
) -> str:
    """Return a JSON string shaped like a real LLM classifier response."""
    return json.dumps(
        {
            "intent": f"{agent} intent",
            "agent": agent,
            "entities": {
                "tickers": tickers or [],
                "topics": topics or [],
                "amount": amount,
                "period_years": period_years,
                "rate": rate,
            },
            "safety_verdict": safety_verdict,
            "confidence": confidence,
            "reasoning": f"Classified as {agent}.",
        }
    )


def _mock_completion(content: str) -> MagicMock:
    """Wrap a JSON string in a mock that looks like an OpenAI ChatCompletion."""
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _make_classifier(mock_client: MagicMock) -> IntentClassifier:
    """Construct a classifier with the mock client injected."""
    clf = IntentClassifier(model="gpt-4o-mini")
    clf._client = mock_client  # bypass lazy OpenAI() construction
    return clf


def _entity_matches(result: ClassificationResult, expected: dict[str, Any]) -> bool:
    """
    Subset match with normalisation for fields present in ExtractedEntities.

    Rules (from fixtures/README.md):
    - tickers:      case-fold + drop exchange suffix before comparing
    - topics:       case-fold, subset match
    - amount/rate:  within ±5% of expected
    - period_years: exact integer
    - other fields (action, horizon, …): not in ExtractedEntities, skip
    """
    entities = result.entities

    for field, exp_val in expected.items():
        if field not in _EXTRACTABLE:
            continue  # skip fields not in our schema

        if field == "tickers":
            exp_set = {t.upper().split(".")[0] for t in exp_val}
            act_set = {t.upper().split(".")[0] for t in entities.tickers}
            if not exp_set.issubset(act_set):
                return False

        elif field == "topics":
            exp_set = {s.lower() for s in exp_val}
            act_set = {s.lower() for s in entities.topics}
            if not exp_set.issubset(act_set):
                return False

        elif field == "amount":
            if entities.amount is None:
                return False
            if abs(entities.amount - exp_val) > abs(exp_val) * 0.05:
                return False

        elif field == "period_years":
            if entities.period_years is None:
                return False
            if int(entities.period_years) != int(exp_val):
                return False

        elif field == "rate":
            if entities.rate is None:
                return False
            if abs(entities.rate - exp_val) > abs(exp_val) * 0.05:
                return False

    return True


# ---------------------------------------------------------------------------
# Test 1 — routing accuracy
# ---------------------------------------------------------------------------


def test_routing_accuracy() -> None:
    """
    Mock the LLM to return the expected agent for each fixture query.
    Verifies that the classifier correctly makes the API call, parses the
    JSON response, and surfaces the right agent in ClassificationResult.
    Threshold: ≥ 85% (allows for any edge-case parse failures).
    """
    mock_client = MagicMock()
    clf = _make_classifier(mock_client)

    correct = 0
    misses: list[str] = []

    for case in _QUERIES:
        expected_agent = case["expected_agent"]
        mock_client.chat.completions.create.return_value = _mock_completion(
            _make_response_content(expected_agent)
        )

        result = clf.classify(case["query"], [], {})

        if result.agent == expected_agent:
            correct += 1
        else:
            misses.append(
                f"  [{expected_agent}] → got [{result.agent}]: {case['query']!r}"
            )

    accuracy = correct / len(_QUERIES)
    print(
        f"\n{'=' * 56}\n"
        f"Routing accuracy: {accuracy:.1%}  ({correct}/{len(_QUERIES)})\n"
        f"{'=' * 56}"
    )
    if misses:
        print("Misses:\n" + "\n".join(misses))

    assert accuracy >= 0.85, (
        f"Routing accuracy {accuracy:.1%} below 85% threshold.\n" + "\n".join(misses)
    )


# ---------------------------------------------------------------------------
# Test 2 — entity extraction with subset + normalisation matching
# ---------------------------------------------------------------------------


def test_entity_extraction() -> None:
    """
    For each fixture case that has extractable entities, mock the LLM to
    return those same entities and verify the classifier parses them correctly
    with normalisation rules applied.
    """
    cases_with_extractable = [
        c
        for c in _QUERIES
        if c["expected_entities"]
        and any(k in _EXTRACTABLE for k in c["expected_entities"])
    ]

    mock_client = MagicMock()
    clf = _make_classifier(mock_client)

    matched = 0
    misses: list[str] = []

    for case in cases_with_extractable:
        exp = case["expected_entities"]
        mock_client.chat.completions.create.return_value = _mock_completion(
            _make_response_content(
                agent=case["expected_agent"],
                tickers=exp.get("tickers"),
                topics=exp.get("topics"),
                amount=exp.get("amount"),
                period_years=exp.get("period_years"),
                rate=exp.get("rate"),
            )
        )

        result = clf.classify(case["query"], [], {})

        if _entity_matches(result, exp):
            matched += 1
        else:
            misses.append(f"  {case['query']!r}\n    expected: {exp}\n    got: {result.entities.model_dump()}")

    rate = matched / len(cases_with_extractable) if cases_with_extractable else 1.0
    print(
        f"\n{'=' * 56}\n"
        f"Entity match rate: {rate:.1%}  ({matched}/{len(cases_with_extractable)})\n"
        f"{'=' * 56}"
    )
    if misses:
        print("Entity mismatches:\n" + "\n".join(misses))

    assert rate >= 0.85, (
        f"Entity match rate {rate:.1%} below 85%.\n" + "\n".join(misses)
    )


# ---------------------------------------------------------------------------
# Test 3 — graceful fallback on OpenAI exception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("network timeout"),
        ValueError("malformed JSON"),
        Exception("unexpected API error"),
    ],
    ids=["runtime_error", "value_error", "generic_exception"],
)
def test_fallback_on_exception(exc: Exception) -> None:
    """
    When the OpenAI call raises any exception, classify() must return the
    safe fallback without re-raising.  The test covers multiple exception types.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = exc

    clf = _make_classifier(mock_client)
    result = clf.classify("how is my portfolio doing?", [], {})

    assert isinstance(result, ClassificationResult), "Fallback must be a ClassificationResult"
    assert result.agent == "support", "Fallback agent must be 'support'"
    assert result.intent == "unknown", "Fallback intent must be 'unknown'"
    assert result.safety_verdict == "safe", "Fallback safety_verdict must be 'safe'"
    assert result.confidence == 0.0, "Fallback confidence must be 0.0"
    assert result.entities.tickers == [], "Fallback tickers must be empty"
    assert result.entities.topics == [], "Fallback topics must be empty"
    assert result.entities.amount is None


def test_fallback_on_malformed_json() -> None:
    """Malformed JSON content (not a valid OpenAI error) also falls back cleanly."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("not-json{{{{")

    clf = _make_classifier(mock_client)
    result = clf.classify("what's AAPL at?", [], {})

    assert result.agent == "support"
    assert result.confidence == 0.0


def test_fallback_on_unknown_agent() -> None:
    """If the LLM returns an agent name not in VALID_AGENTS, it is replaced with 'support'."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(
        _make_response_content(agent="nonexistent_agent")
    )

    clf = _make_classifier(mock_client)
    result = clf.classify("test query", [], {})

    assert result.agent == "support"


# ---------------------------------------------------------------------------
# Test 4 — follow-up resolution: history forwarded, new entity extracted
# ---------------------------------------------------------------------------


def test_followup_history_forwarded_and_new_entity_extracted() -> None:
    """
    Scenario: user discussed Microsoft, then asks "what about Apple?"

    Assertions:
    1. The conversation history is present in the messages sent to the API
       (specifically, the prior user turn mentioning Microsoft is included).
    2. The parsed result contains AAPL in tickers (the newly introduced entity).
    3. The agent is market_research (same intent, new subject).
    """
    history = [
        {"role": "user", "content": "tell me about Microsoft"},
        {"role": "assistant", "content": "Microsoft (MSFT) is a technology company..."},
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(
        _make_response_content(
            agent="market_research",
            tickers=["AAPL"],
            topics=[],
        )
    )

    clf = _make_classifier(mock_client)
    result = clf.classify("what about Apple?", history, {})

    # --- assertion 1: history was sent to the API ---
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages: list[dict] = call_kwargs["messages"]

    non_system = [m for m in messages if m["role"] != "system"]
    all_content = " ".join(m["content"] for m in non_system)
    assert "Microsoft" in all_content, (
        "Conversation history must be included in the API call messages. "
        f"Got messages: {non_system}"
    )

    # The final user message should be the follow-up query
    assert non_system[-1]["content"] == "what about Apple?"

    # --- assertion 2: AAPL extracted ---
    assert "AAPL" in result.entities.tickers, (
        f"Expected AAPL in tickers, got {result.entities.tickers}"
    )

    # --- assertion 3: correct agent ---
    assert result.agent == "market_research"


def test_followup_history_truncated_to_six_turns() -> None:
    """
    History longer than 6 turns is truncated — only the last 6 are forwarded.
    The system message is NOT counted toward the limit.
    """
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(10)  # 10 turns total
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(
        _make_response_content(agent="market_research")
    )

    clf = _make_classifier(mock_client)
    clf.classify("latest query", long_history, {})

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages: list[dict] = call_kwargs["messages"]

    # Subtract system message (1) and the current user query (1) = 2 non-history
    history_in_call = [m for m in messages if m["role"] != "system"][:-1]
    assert len(history_in_call) <= 6, (
        f"Expected at most 6 history turns forwarded, got {len(history_in_call)}"
    )

    # The earliest turn forwarded must be from the last 6 of long_history
    if history_in_call:
        assert "turn 4" in history_in_call[0]["content"] or int(
            history_in_call[0]["content"].split()[-1]
        ) >= 4


# ---------------------------------------------------------------------------
# Test bonus — tickers are always uppercased by the classifier, not the LLM
# ---------------------------------------------------------------------------


def test_tickers_normalised_to_uppercase() -> None:
    """
    Even if the LLM returns lowercase tickers, the classifier normalises them.
    """
    raw = json.dumps(
        {
            "intent": "market research",
            "agent": "market_research",
            "entities": {
                "tickers": ["aapl", "nvda", "asml.as"],
                "topics": [],
                "amount": None,
                "period_years": None,
                "rate": None,
            },
            "safety_verdict": "safe",
            "confidence": 0.9,
            "reasoning": "Normalisation test.",
        }
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(raw)

    clf = _make_classifier(mock_client)
    result = clf.classify("tell me about apple", [], {})

    assert "AAPL" in result.entities.tickers
    assert "NVDA" in result.entities.tickers
    assert "ASML.AS" in result.entities.tickers
    assert "aapl" not in result.entities.tickers
