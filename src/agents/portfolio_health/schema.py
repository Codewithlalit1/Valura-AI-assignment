"""
Pydantic output models for portfolio health calculations.

All fields default to safe zero-values so callers can safely handle an
empty portfolio without branching on None.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ConcentrationRisk(BaseModel):
    """Result of calculate_concentration()."""

    model_config = ConfigDict(frozen=True)

    top_position_ticker: str = Field(
        default="",
        description="Ticker symbol of the largest position by current market value.",
    )
    top_position_pct: float = Field(
        default=0.0,
        description="Largest single position as % of total portfolio market value.",
    )
    top_3_positions_pct: float = Field(
        default=0.0,
        description="Top-3 positions combined as % of total portfolio market value.",
    )
    risk_level: str = Field(
        default="low",
        description=(
            '"high"     — top position > 40 % OR top-3 > 70 %\n'
            '"moderate" — top position > 20 % OR top-3 > 50 %\n'
            '"low"      — otherwise'
        ),
    )


class PerformanceMetrics(BaseModel):
    """Result of calculate_performance()."""

    model_config = ConfigDict(frozen=True)

    cost_basis_usd: float = Field(
        default=0.0,
        description="Total cost basis: sum(quantity × avg_cost_usd).",
    )
    current_value_usd: float = Field(
        default=0.0,
        description="Current market value: sum(quantity × live_price).",
    )
    total_return_pct: float = Field(
        default=0.0,
        description="(current_value − cost_basis) / cost_basis × 100.",
    )
    annualized_return_pct: float = Field(
        default=0.0,
        description=(
            "CAGR assuming a 1-year holding period "
            "(no purchase-date data in the holdings schema)."
        ),
    )
    holding_count: int = Field(
        default=0,
        description="Number of non-zero holdings included in the calculation.",
    )


class BenchmarkComparison(BaseModel):
    """Result of calculate_benchmark_comparison()."""

    model_config = ConfigDict(frozen=True)

    portfolio_return_pct: float = Field(
        default=0.0,
        description="Portfolio total return in percentage points (e.g. 27.1).",
    )
    benchmark_return_pct: float = Field(
        default=0.0,
        description="Benchmark return in percentage points (e.g. 15.3).",
    )
    benchmark_name: str = Field(default="", description="Human-readable benchmark label.")
    alpha_pct: float = Field(
        default=0.0,
        description="portfolio_return_pct − benchmark_return_pct (simple alpha).",
    )
    outperformed: bool = Field(
        default=False,
        description="True when alpha_pct > 0.",
    )
