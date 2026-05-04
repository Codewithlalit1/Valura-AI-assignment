"""
PortfolioHealthAgent — streams a structured portfolio health check.

Flow
----
1. Fetch live prices + benchmark return in parallel (asyncio.to_thread wraps
   the sync MarketDataFetcher calls so the event loop is not blocked).
2. Run the three pure calculation functions — zero I/O, zero LLM.
3. Inject the computed numbers into the prompt so the LLM only has to write
   the `observations` and `disclaimer` fields, never recompute metrics.
4. Stream the LLM response token-by-token via AsyncOpenAI(stream=True).
5. Yield each content chunk as it arrives; on any exception yield a safe
   fallback JSON that always contains the disclaimer.

Empty portfolio (BUILD mode)
----------------------------
When the user has no holdings, skip all calculations and use a separate
BUILD-oriented system prompt that guides the LLM to give first-steps advice.
The JSON shape is preserved with zero-value numeric fields.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import openai

from .calculations import (
    calculate_benchmark_comparison,
    calculate_concentration,
    calculate_performance,
)
from .market_data import MarketDataFetcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "This is not investment advice. Portfolio analysis is for informational "
    "purposes only. Past performance does not guarantee future results. "
    "Please consult a qualified financial adviser before making investment decisions."
)

# Currency → (yfinance symbol, display name)
_BENCHMARK_BY_CURRENCY: dict[str, tuple[str, str]] = {
    "USD": ("^GSPC",    "S&P 500"),
    "EUR": ("^STOXX50E","Euro Stoxx 50"),
    "GBP": ("^FTSE",    "FTSE 100"),
    "JPY": ("^N225",    "Nikkei 225"),
    "AUD": ("^AXJO",    "ASX 200"),
    "CAD": ("^GSPTSE",  "S&P/TSX Composite"),
    "CHF": ("^SSMI",    "SMI"),
}
_DEFAULT_BENCHMARK = ("^GSPC", "S&P 500")

# Number of prior conversation turns forwarded to the LLM for context.
_MAX_HISTORY_TURNS = 4

# Pre-built fallback emitted when the LLM call fails entirely.
_FALLBACK_JSON: str = json.dumps(
    {
        "concentration_risk": {
            "top_position_pct": 0.0,
            "top_3_positions_pct": 0.0,
            "flag": "n/a",
        },
        "performance": {"total_return_pct": 0.0, "annualized_return_pct": 0.0},
        "benchmark_comparison": {
            "benchmark": "",
            "portfolio_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "alpha_pct": 0.0,
        },
        "observations": [
            {
                "severity": "warning",
                "text": "Portfolio health check is temporarily unavailable. Please try again shortly.",
            }
        ],
        "disclaimer": _DISCLAIMER,
    }
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Valura AI's portfolio health specialist.

You will receive pre-computed metrics for a user's portfolio. Your job is to:
1. Copy those metrics into the required JSON output shape EXACTLY as given.
2. Write 2–4 observations in plain language that surface the most important \
risks or opportunities for this specific investor.
3. Always include the standard regulatory disclaimer verbatim.

## Required output shape — return ONLY valid JSON, no markdown, no code fences

{
  "concentration_risk": {
    "top_position_pct":  <float — copy from input>,
    "top_3_positions_pct": <float — copy from input>,
    "flag": "<high | moderate | low — copy from input>"
  },
  "performance": {
    "total_return_pct": <float — copy from input>,
    "annualized_return_pct": <float — copy from input>
  },
  "benchmark_comparison": {
    "benchmark":           "<name — copy from input>",
    "portfolio_return_pct": <float — copy from input>,
    "benchmark_return_pct": <float — copy from input>,
    "alpha_pct":            <float — copy from input>
  },
  "observations": [
    {"severity": "<warning | info | tip>", "text": "<plain-language observation>"}
  ],
  "disclaimer": "This is not investment advice. Portfolio analysis is for informational purposes only. Past performance does not guarantee future results. Please consult a qualified financial adviser before making investment decisions."
}

## Observation rules
- 2–4 observations maximum; surface the one or two things that matter most
- "warning" = immediate risk or actionable concern (concentration, drawdown, etc.)
- "info"    = useful context, no urgency
- "tip"     = a small improvement the user could consider
- Plain language. If you use a term like "alpha" or "CAGR", explain it in the same sentence
- Reference specific tickers, percentages, or dollar figures from the holdings data
- Do NOT invent numbers — only reference figures from the input
"""

_BUILD_SYSTEM_PROMPT = """\
You are Valura AI's portfolio health specialist helping an investor who has \
no holdings yet and wants to start.

## Required output shape — return ONLY valid JSON, no markdown, no code fences

{
  "concentration_risk":   {"top_position_pct": 0.0, "top_3_positions_pct": 0.0, "flag": "n/a"},
  "performance":          {"total_return_pct": 0.0, "annualized_return_pct": 0.0},
  "benchmark_comparison": {"benchmark": "", "portfolio_return_pct": 0.0, "benchmark_return_pct": 0.0, "alpha_pct": 0.0},
  "observations": [
    {"severity": "<warning | info | tip>", "text": "<plain-language BUILD guidance>"}
  ],
  "disclaimer": "This is not investment advice. Portfolio analysis is for informational purposes only. Past performance does not guarantee future results. Please consult a qualified financial adviser before making investment decisions."
}

## Guidance rules
- Write exactly 3–4 observations covering: first steps, instrument selection, \
position sizing, and one risk-profile-specific consideration
- Tailor every observation to the user's risk_profile and currency
- Plain language — assume no prior investing knowledge
- Be specific and actionable: name real instruments (e.g. VOO, VWRL, a bond ETF) \
rather than vague categories
- Do NOT invent historical returns or guarantee outcomes
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PortfolioHealthAgent:
    """
    Produces a structured portfolio health check streamed token-by-token.

    Parameters
    ----------
    model:
        OpenAI model identifier.
    market_fetcher:
        Optional injected MarketDataFetcher.  A new instance is created when
        omitted.  Inject a mock for tests.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        market_fetcher: Optional[MarketDataFetcher] = None,
    ) -> None:
        self.model = model
        self._fetcher = market_fetcher or MarketDataFetcher()
        # Lazy — tests inject a mock before the first call.
        self._client: Optional[openai.AsyncOpenAI] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        user_profile: dict,
        conversation_history: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Yield the portfolio health check as a stream of JSON text chunks.

        The caller collects chunks and may parse the final concatenated string
        as JSON, or forward each chunk directly to an SSE connection.
        """
        holdings: list[dict] = user_profile.get("portfolio", [])
        is_empty = not any(float(h.get("quantity") or 0) > 0 for h in holdings)

        if is_empty:
            messages = _build_messages_empty(user_profile, conversation_history)
        else:
            messages = await self._prepare_messages_populated(
                user_profile, holdings, conversation_history
            )

        async for chunk in self._stream(messages):
            yield chunk

    # ------------------------------------------------------------------
    # Internal: data gathering + prompt construction
    # ------------------------------------------------------------------

    async def _prepare_messages_populated(
        self,
        user_profile: dict,
        holdings: list[dict],
        conversation_history: list[dict],
    ) -> list[dict]:
        """Fetch live data in parallel, run calculations, return prompt messages."""
        tickers = [str(h["ticker"]) for h in holdings if h.get("ticker")]
        currency = str(user_profile.get("currency", "USD")).upper()
        bench_symbol, bench_name = _BENCHMARK_BY_CURRENCY.get(
            currency, _DEFAULT_BENCHMARK
        )

        # --- parallel I/O (sync helpers dispatched to thread pool) ---
        try:
            prices, bench_decimal = await asyncio.gather(
                asyncio.to_thread(self._fetcher.get_current_prices, tickers),
                asyncio.to_thread(self._fetcher.get_benchmark_return, bench_symbol, "1y"),
            )
        except Exception as exc:
            logger.warning("Market data fetch failed: %s", exc)
            prices, bench_decimal = {}, None

        bench_pct = (bench_decimal or 0.0) * 100

        # --- pure calculations ---
        conc = calculate_concentration(holdings)
        perf = calculate_performance(holdings, prices)
        bench = calculate_benchmark_comparison(
            perf.total_return_pct, bench_pct, bench_name
        )

        # --- build the prompt with injected numbers ---
        return _build_messages_populated(
            user_profile,
            holdings,
            prices,
            conc,
            perf,
            bench,
            conversation_history,
        )

    # ------------------------------------------------------------------
    # Internal: LLM streaming
    # ------------------------------------------------------------------

    async def _stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Call the LLM with stream=True and yield each content chunk."""
        try:
            stream = await self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1200,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            logger.warning("PortfolioHealthAgent LLM stream error: %s", exc)
            yield _FALLBACK_JSON

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI()
        return self._client


# ---------------------------------------------------------------------------
# Prompt builders (module-level pure functions — easy to unit-test)
# ---------------------------------------------------------------------------


def _build_messages_populated(
    user_profile: dict,
    holdings: list[dict],
    prices: dict[str, float],
    conc,
    perf,
    bench,
    conversation_history: list[dict],
) -> list[dict]:
    """Return the messages list for a user with active holdings."""
    name = user_profile.get("name", "the user")
    risk = user_profile.get("risk_profile", "unknown")
    currency = str(user_profile.get("currency", "USD")).upper()

    holdings_table = _format_holdings_table(holdings, prices)

    user_content = f"""\
Portfolio health check for {name}
Risk profile: {risk} | Currency: {currency}

## Pre-computed metrics (copy these exactly into the JSON output)

Concentration risk:
  Largest position: {conc.top_position_ticker} — {conc.top_position_pct:.1f}% of portfolio
  Top-3 positions combined: {conc.top_3_positions_pct:.1f}% of portfolio
  Flag: {conc.risk_level}

Performance (1-year holding-period assumption):
  Cost basis:    ${perf.cost_basis_usd:>12,.2f}
  Current value: ${perf.current_value_usd:>12,.2f}
  Total return:       {perf.total_return_pct:+.2f}%
  Annualised return:  {perf.annualized_return_pct:+.2f}%

Benchmark comparison ({bench.benchmark_name}):
  Portfolio return:  {bench.portfolio_return_pct:+.2f}%
  Benchmark return:  {bench.benchmark_return_pct:+.2f}%
  Alpha:             {bench.alpha_pct:+.2f}%  \
({'outperforming' if bench.outperformed else 'underperforming'})

## Holdings detail (use for writing observations)

{holdings_table}

Generate the JSON health report now. Copy the metrics above exactly. \
Write observations for a {risk}-risk investor in plain, jargon-free language."""

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(_trim_history(conversation_history))
    messages.append({"role": "user", "content": user_content})
    return messages


def _build_messages_empty(
    user_profile: dict,
    conversation_history: list[dict],
) -> list[dict]:
    """Return the messages list for a user with no holdings (BUILD mode)."""
    name = user_profile.get("name", "this investor")
    risk = user_profile.get("risk_profile", "moderate")
    currency = str(user_profile.get("currency", "USD")).upper()

    user_content = (
        f"{name} has no portfolio holdings yet and wants to start investing.\n"
        f"Risk profile: {risk} | Currency: {currency}\n\n"
        "Generate a BUILD-oriented portfolio health report with 3–4 practical "
        "first-steps observations. All numeric fields must be 0.0. "
        "Tailor instrument suggestions to the user's risk profile and currency."
    )

    messages: list[dict] = [{"role": "system", "content": _BUILD_SYSTEM_PROMPT}]
    messages.extend(_trim_history(conversation_history))
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_holdings_table(
    holdings: list[dict],
    prices: dict[str, float],
) -> str:
    """Return a compact table string suitable for inclusion in a prompt."""
    header = f"{'Ticker':<8} {'Qty':>6}  {'Cost/sh':>10}  {'Current':>10}  {'Return':>8}  {'Value USD':>12}"
    rows = [header, "-" * len(header)]

    for h in sorted(
        holdings,
        key=lambda x: float(x.get("quantity") or 0)
        * float(x.get("current_price_usd") or 0),
        reverse=True,
    ):
        ticker = str(h.get("ticker", ""))
        qty = float(h.get("quantity") or 0)
        cost = float(h.get("avg_cost_usd") or 0)
        live = prices.get(ticker) or float(h.get("current_price_usd") or cost)
        value = qty * live
        ret_pct = ((live - cost) / cost * 100) if cost > 0 else 0.0
        rows.append(
            f"{ticker:<8} {qty:>6.0f}  ${cost:>9,.2f}  ${live:>9,.2f}  "
            f"{ret_pct:>+7.1f}%  ${value:>11,.0f}"
        )

    return "\n".join(rows)


def _trim_history(history: list[dict]) -> list[dict]:
    """Return the last _MAX_HISTORY_TURNS messages with clean role/content."""
    trimmed = history[-_MAX_HISTORY_TURNS:]
    return [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in trimmed
    ]
