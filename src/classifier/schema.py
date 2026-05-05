"""
Pydantic models for intent classification output.

ExtractedEntities captures the fields the system prompt instructs the LLM to
extract. Fields beyond these (action, goal, horizon, …) are supported by the
prompt but are not yet represented here — extend as new agents require them.

ClassificationResult is the single return type of IntentClassifier.classify().
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Canonical agent names — used for validation and documentation.
VALID_AGENTS: frozenset[str] = frozenset(
    [
        "portfolio_health",
        "market_research",
        "investment_strategy",
        "financial_planning",
        "financial_calculator",
        "risk_assessment",
        "product_recommendation",
        "predictive_analysis",
        "customer_support",
        "general_query",
    ]
)

VALID_SAFETY_VERDICTS: frozenset[str] = frozenset(["safe", "caution", "unsafe"])


class ExtractedEntities(BaseModel):
    """
    Entities the classifier extracts from the query and conversation context.

    Extra fields from the LLM response (e.g. 'action', 'horizon', 'currency')
    are silently dropped — add them as typed fields here when an agent needs them.
    """

    model_config = ConfigDict(extra="ignore")

    tickers: list[str] = Field(
        default_factory=list,
        description="Uppercase ticker symbols (AAPL, ASML.AS, 7203.T).",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Free-form concepts that don't fit other structured fields.",
    )
    amount: Optional[float] = Field(
        default=None,
        description="Monetary or quantity amount as a plain float (50k → 50000).",
    )
    period_years: Optional[float] = Field(
        default=None,
        description="Time horizon in years (integer-equivalent expected).",
    )
    rate: Optional[float] = Field(
        default=None,
        description="Interest/return rate as a decimal fraction (7% → 0.07).",
    )


class ClassificationResult(BaseModel):
    """
    Single return value of IntentClassifier.classify().

    Extra fields from the LLM response (e.g. 'index', 'action') are dropped
    rather than surfaced as a parse error — this keeps the pipeline resilient
    to prompt drift and model verbosity.
    """

    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="Short description of the user's intent.")
    agent: str = Field(description="One of the VALID_AGENTS names.")
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    safety_verdict: str = Field(
        default="safe",
        description="safe | caution | unsafe",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="One-sentence routing explanation.")
