"""
MarketDataFetcher — live market data via yfinance with 5-minute in-memory cache.

Cache strategy
--------------
Each logical datum (price, benchmark return, sector) is cached independently
so that a hit on AAPL's price doesn't block fetching MSFT's sector.
Entries expire after TTL_SECONDS (300 s = 5 min).

Failure contract
----------------
All public methods catch every exception and return a safe empty value:
  get_current_prices  → {}       (or partial dict for the tickers that succeeded)
  get_benchmark_return → None
  get_sector_info     → ""
They never raise.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

_TTL_SECONDS: int = 300  # 5-minute TTL


class MarketDataFetcher:
    """
    Thin wrapper around yfinance with per-datum in-memory caching.

    Each instance owns its own cache — share a single instance within a
    request to maximise hit rate, or create one per application for a
    process-wide cache.
    """

    def __init__(self) -> None:
        # {str(cache_key_tuple): (monotonic_timestamp, value)}
        self._cache: dict[str, tuple[float, object]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """
        Return the latest price for each ticker.

        Hits the yfinance batch download endpoint in a single round-trip for
        any tickers not already in cache.  Any ticker whose price cannot be
        fetched is omitted from the returned dict (never set to None).
        """
        if not tickers:
            return {}

        result: dict[str, float] = {}
        to_fetch: list[str] = []

        for ticker in tickers:
            cached = self._get(("price", ticker))
            if cached is not None:
                result[ticker] = float(cached)
            else:
                to_fetch.append(ticker)

        if to_fetch:
            fetched = self._batch_download(to_fetch)
            result.update(fetched)

            # Per-ticker fallback for any that the batch download missed
            for ticker in to_fetch:
                if ticker not in fetched:
                    price = self._single_price(ticker)
                    if price is not None:
                        result[ticker] = price

        return result

    def get_benchmark_return(
        self,
        benchmark: str = "^GSPC",
        period: str = "1y",
    ) -> Optional[float]:
        """
        Return the percentage return of *benchmark* over *period* as a decimal.

        Example: a 27.1 % annual gain → 0.271.
        Returns None when history is unavailable or too short.
        """
        key = ("benchmark", benchmark, period)
        cached = self._get(key)
        if cached is not None:
            return float(cached)

        try:
            hist = yf.Ticker(benchmark).history(period=period, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                logger.warning(
                    "Insufficient history for benchmark %s period=%s", benchmark, period
                )
                return None

            start = float(hist["Close"].iloc[0])
            end = float(hist["Close"].iloc[-1])
            if start == 0.0:
                return None

            pct = (end - start) / start
            self._put(key, pct)
            return pct
        except Exception as exc:
            logger.warning(
                "Benchmark return fetch failed [%s/%s]: %s", benchmark, period, exc
            )
            return None

    def get_sector_info(self, ticker: str) -> str:
        """
        Return the sector string for *ticker* (e.g. "Technology").

        Returns "" when the information is unavailable.
        """
        key = ("sector", ticker)
        cached = self._get(key)
        if cached is not None:
            return str(cached)

        try:
            sector = yf.Ticker(ticker).info.get("sector") or ""
            self._put(key, sector)
            return sector
        except Exception as exc:
            logger.warning("Sector fetch failed for %s: %s", ticker, exc)
            return ""

    # ------------------------------------------------------------------
    # Internal fetch helpers
    # ------------------------------------------------------------------

    def _batch_download(self, tickers: list[str]) -> dict[str, float]:
        """
        Download the last closing price for each ticker in one network call.

        yf.download returns a MultiIndex DataFrame when given a list of
        tickers.  Single-element lists also produce a DataFrame (not a
        Series), so the same extraction path handles both cases.
        """
        try:
            import pandas as pd

            data = yf.download(
                tickers,
                period="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if data.empty:
                return {}

            # Prefer 'Close' over 'Adj Close'
            # Cannot use `or` — truth value of a DataFrame is ambiguous
            close = None
            for col in ("Close", "Adj Close"):
                try:
                    close = data[col]
                    break
                except KeyError:
                    pass
            if close is None:
                return {}

            prices: dict[str, float] = {}

            if isinstance(close, pd.DataFrame):
                last = close.iloc[-1]
                for ticker in tickers:
                    val = last.get(ticker)
                    if val is not None and not pd.isna(val):
                        price = float(val)
                        prices[ticker] = price
                        self._put(("price", ticker), price)
            else:
                # Series — single-ticker edge case from older yfinance builds
                val = close.iloc[-1]
                if not pd.isna(val):
                    price = float(val)
                    prices[tickers[0]] = price
                    self._put(("price", tickers[0]), price)

            return prices
        except Exception as exc:
            logger.warning("Batch download failed for %s: %s", tickers, exc)
            return {}

    def _single_price(self, ticker: str) -> Optional[float]:
        """
        Fallback price fetch via Ticker.fast_info — one call per ticker.

        Used only when the batch download returns NaN or fails for a
        specific ticker (e.g. non-US exchanges, indices).
        """
        try:
            price = yf.Ticker(ticker).fast_info.last_price
            if price is not None:
                price = float(price)
                self._put(("price", ticker), price)
                return price
        except Exception as exc:
            logger.warning("Single price fetch failed for %s: %s", ticker, exc)
        return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get(self, key: tuple) -> object | None:
        entry = self._cache.get(str(key))
        if entry is None:
            return None
        cached_at, value = entry
        if time.monotonic() - cached_at > _TTL_SECONDS:
            del self._cache[str(key)]
            return None
        return value

    def _put(self, key: tuple, value: object) -> None:
        self._cache[str(key)] = (time.monotonic(), value)
