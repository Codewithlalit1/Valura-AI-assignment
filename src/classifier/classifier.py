"""
IntentClassifier — wraps one OpenAI chat completion call.

Design constraints
------------------
- Exactly ONE API call per classify() invocation.
- Never raises: all OpenAI or parse errors return a safe fallback.
- No network calls in __init__; the OpenAI client is lazily constructed
  so tests can inject a mock by setting instance._client before the first call.
- Conversation history is truncated to the last 6 turns before being sent
  to control token cost.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import openai

from src.llm import make_sync_client
from .prompt import SYSTEM_PROMPT
from .schema import (
    VALID_AGENTS,
    VALID_SAFETY_VERDICTS,
    ClassificationResult,
    ExtractedEntities,
)

logger = logging.getLogger(__name__)

# Maximum number of history turns forwarded to the LLM.
_MAX_HISTORY_TURNS = 6

# Returned on any failure — safe, routes to human-friendly support agent.
_FALLBACK = ClassificationResult(
    intent="unknown",
    agent="customer_support",
    entities=ExtractedEntities(),
    safety_verdict="safe",
    confidence=0.0,
    reasoning="Classification unavailable — defaulting to customer_support.",
)


class IntentClassifier:
    """
    Classifies a user query into one of eight agents and extracts entities.

    Parameters
    ----------
    model:
        OpenAI model identifier (e.g. "gpt-4o-mini", "gpt-4.1").
    session_store:
        Injected session-store dependency (unused in classification itself;
        reserved for agents that need session context later).
    """

    def __init__(self, model: str, session_store: Any = None) -> None:
        self.model = model
        self.session_store = session_store
        # Lazy — avoids importing / instantiating OpenAI at import time, which
        # would fail in CI when OPENAI_API_KEY is absent.  Tests set this
        # attribute directly before calling classify().
        self._client: Optional[openai.OpenAI] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        query: str,
        conversation_history: list[dict],
        user_context: dict,
    ) -> ClassificationResult:
        """
        Make one OpenAI call and return a ClassificationResult.

        Falls back to _FALLBACK on any exception — never raises.
        """
        messages = self._build_messages(query, conversation_history, user_context)
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            raw: str = response.choices[0].message.content or ""
            return self._parse(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Classifier error (%s): %s", type(exc).__name__, exc)
            return _FALLBACK.model_copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = make_sync_client()
        return self._client

    def _format_history(self, history: list[dict]) -> list[dict]:
        """Return the last _MAX_HISTORY_TURNS messages, preserving role/content shape."""
        trimmed = history[-_MAX_HISTORY_TURNS:]
        return [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in trimmed
        ]

    def _build_messages(
        self,
        query: str,
        conversation_history: list[dict],
        user_context: dict,
    ) -> list[dict]:
        system_content = SYSTEM_PROMPT
        if user_context:
            system_content = f"{SYSTEM_PROMPT}\n\n## User context\n{_format_user_context(user_context)}"

        messages: list[dict] = [{"role": "system", "content": system_content}]
        messages.extend(self._format_history(conversation_history))
        messages.append({"role": "user", "content": query})
        return messages

    def _parse(self, raw: str) -> ClassificationResult:
        """
        Parse the LLM JSON string into ClassificationResult.

        Normalises agent and safety_verdict; falls back to defaults on bad values.
        Raises on JSON / Pydantic errors — the caller catches them.
        """
        data: dict = json.loads(raw)

        # Normalise agent
        if data.get("agent") not in VALID_AGENTS:
            data["agent"] = "customer_support"

        # Normalise safety_verdict
        if data.get("safety_verdict") not in VALID_SAFETY_VERDICTS:
            data["safety_verdict"] = "safe"

        # Normalise tickers to uppercase
        entities = data.get("entities", {})
        if isinstance(entities, dict) and "tickers" in entities:
            entities["tickers"] = [
                t.upper() if isinstance(t, str) else t
                for t in entities["tickers"]
            ]

        return ClassificationResult(**data)


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _format_user_context(ctx: dict) -> str:
    """Compact single-line summary of user context for the system message."""
    parts: list[str] = []
    if uid := ctx.get("user_id"):
        parts.append(f"user_id={uid}")
    if risk := ctx.get("risk_profile"):
        parts.append(f"risk={risk}")
    if currency := ctx.get("currency"):
        parts.append(f"currency={currency}")
    if portfolio := ctx.get("portfolio"):
        tickers = [h.get("ticker", "") for h in portfolio if isinstance(h, dict)]
        if tickers:
            parts.append(f"holdings=[{', '.join(tickers[:8])}{'…' if len(tickers) > 8 else ''}]")
    return ", ".join(parts) if parts else "(no context)"
