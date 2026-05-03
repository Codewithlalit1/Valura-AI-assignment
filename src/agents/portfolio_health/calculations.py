"""
Pure portfolio-health calculation functions.

Constraints
-----------
- No I/O, no network, no LLM calls.
- Every function returns a Pydantic model with safe defaults.
- Empty or zero-value portfolios never cause exceptions.

Return-percentage convention
----------------------------
All *_pct fields use percentage points: 27.1 means 27.1 %, not 0.271.
Callers must multiply decimal fractions (e.g. from MarketDataFetcher) by 100
before passing them to calculate_benchmark_comparison().
"""
from __future__ import annotations

import math

from .schema import BenchmarkComparison, ConcentrationRisk, PerformanceMetrics

# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

# Thresholds for risk_level assignment (all in percentage points).
_HIGH_TOP1 = 40.0
_HIGH_TOP3 = 70.0
_MODERATE_TOP1 = 20.0
_MODERATE_TOP3 = 50.0


def calculate_concentration(holdings: list[dict]) -> ConcentrationRisk:
    """
    Compute concentration risk from portfolio holdings.

    Uses `current_price_usd` × `quantity` as the market value for each
    holding.  Holdings with zero quantity or zero price are skipped.

    Parameters
    ----------
    holdings:
        List of holding dicts, each expected to have at least:
        ``ticker``, ``quantity``, ``current_price_usd``.

    Returns
    -------
    ConcentrationRisk with safe zero-defaults for an empty portfolio.
    """
    if not holdings:
        return ConcentrationRisk()

    values: list[tuple[str, float]] = []
    for h in holdings:
        qty = float(h.get("quantity") or 0)
        price = float(h.get("current_price_usd") or 0)
        mv = qty * price
        if mv > 0:
            values.append((str(h.get("ticker", "")), mv))

    if not values:
        return ConcentrationRisk()

    total_mv = sum(mv for _, mv in values)
    values.sort(key=lambda x: x[1], reverse=True)

    top1_pct = values[0][1] / total_mv * 100
    top3_pct = sum(mv for _, mv in values[:3]) / total_mv * 100

    if top1_pct > _HIGH_TOP1 or top3_pct > _HIGH_TOP3:
        risk_level = "high"
    elif top1_pct > _MODERATE_TOP1 or top3_pct > _MODERATE_TOP3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return ConcentrationRisk(
        top_position_ticker=values[0][0],
        top_position_pct=round(top1_pct, 2),
        top_3_positions_pct=round(top3_pct, 2),
        risk_level=risk_level,
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def calculate_performance(
    holdings: list[dict],
    current_prices: dict[str, float],
) -> PerformanceMetrics:
    """
    Compute total and annualized portfolio return.

    Live prices from ``current_prices`` override the stored
    ``current_price_usd`` field in each holding.  If a ticker is absent
    from ``current_prices``, the stored price is used as a fallback.

    Holdings with zero quantity or zero cost basis are excluded from the
    calculation to avoid division-by-zero.

    Annualised return uses a 1-year holding-period assumption (CAGR with
    T = 1) because the holdings schema has no purchase-date field.  When
    total_return_pct > −100 the CAGR formula produces a valid float;
    extreme losses that exceed −100 % are clamped to −100 %.

    Parameters
    ----------
    holdings:
        List of holding dicts with ``ticker``, ``quantity``, ``avg_cost_usd``,
        and optionally ``current_price_usd``.
    current_prices:
        Live prices keyed by ticker (e.g. from MarketDataFetcher).

    Returns
    -------
    PerformanceMetrics with safe zero-defaults for an empty portfolio.
    """
    if not holdings:
        return PerformanceMetrics()

    cost_basis = 0.0
    current_value = 0.0
    included = 0

    for h in holdings:
        ticker = str(h.get("ticker", ""))
        qty = float(h.get("quantity") or 0)
        cost_per_share = float(h.get("avg_cost_usd") or 0)

        if qty <= 0 or cost_per_share <= 0:
            continue

        live_price = current_prices.get(ticker)
        if live_price is None:
            live_price = float(h.get("current_price_usd") or cost_per_share)

        cost_basis += qty * cost_per_share
        current_value += qty * live_price
        included += 1

    if cost_basis == 0.0 or included == 0:
        return PerformanceMetrics(holding_count=len(holdings))

    total_return_pct = (current_value - cost_basis) / cost_basis * 100

    # CAGR with T = 1 year. Guard against returns worse than -100 %.
    gain_factor = max(1 + total_return_pct / 100, 0.0)
    annualized_return_pct = (math.pow(gain_factor, 1.0) - 1) * 100  # T=1 → identity

    return PerformanceMetrics(
        cost_basis_usd=round(cost_basis, 2),
        current_value_usd=round(current_value, 2),
        total_return_pct=round(total_return_pct, 4),
        annualized_return_pct=round(annualized_return_pct, 4),
        holding_count=included,
    )


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


def calculate_benchmark_comparison(
    portfolio_return: float,
    benchmark_return: float,
    benchmark_name: str,
) -> BenchmarkComparison:
    """
    Compute simple alpha between portfolio and benchmark.

    Both ``portfolio_return`` and ``benchmark_return`` must be in the same
    units — percentage points (e.g. 27.1 for 27.1 %).  If benchmark data
    comes from MarketDataFetcher.get_benchmark_return() (which returns a
    decimal fraction), multiply by 100 before calling this function.

    Alpha is the simple arithmetic excess return:
        alpha = portfolio_return − benchmark_return

    Parameters
    ----------
    portfolio_return:
        Portfolio total return in percentage points.
    benchmark_return:
        Benchmark return in percentage points.
    benchmark_name:
        Human-readable label, e.g. "S&P 500" or "^GSPC".

    Returns
    -------
    BenchmarkComparison.
    """
    alpha = portfolio_return - benchmark_return
    return BenchmarkComparison(
        portfolio_return_pct=round(portfolio_return, 4),
        benchmark_return_pct=round(benchmark_return, 4),
        benchmark_name=benchmark_name,
        alpha_pct=round(alpha, 4),
        outperformed=alpha > 0,
    )
