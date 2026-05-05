"""
Latency benchmark for the Valura AI /query endpoint.

Sends N sequential POST requests to a locally running server and reports:
  - Time to first SSE token  (TTFT): wall-clock seconds from request dispatch
    to the moment the first `data: ...` line is received.
  - Total response time       (TRT): wall-clock seconds from request dispatch
    to the final byte of the SSE stream.

Prints p50 and p95 for both metrics, plus a per-request table.

Usage
-----
    # Start the server first:
    uvicorn src.main:app --reload --port 8000

    # Then in a second terminal:
    python scripts/benchmark.py

    # Optional flags:
    python scripts/benchmark.py --n 20 --url http://localhost:8000 --verbose
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import httpx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_URL = "http://localhost:8000"
_DEFAULT_N = 20
_USER_ID = "usr_aggr_001"
_SESSION_PREFIX = "bench"
_QUERY = (
    "Give me a full portfolio health check — concentration, performance, "
    "benchmark comparison, and your top observations."
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Result:
    request_num: int
    ttft: float | None   # seconds; None on error
    trt: float | None    # seconds; None on error
    chunks: int
    error: str | None


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def _measure(
    client: httpx.Client,
    url: str,
    request_num: int,
) -> Result:
    """Send one streaming request and measure TTFT and TRT."""
    payload = {
        "query": _QUERY,
        "user_id": _USER_ID,
        "session_id": f"{_SESSION_PREFIX}-{request_num}",
    }

    ttft: float | None = None
    chunks = 0
    t0 = time.perf_counter()

    try:
        with client.stream("POST", f"{url}/query", json=payload, timeout=60.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
        trt = time.perf_counter() - t0
        return Result(request_num, ttft, trt, chunks, None)

    except httpx.ConnectError:
        return Result(
            request_num, None, None, 0,
            f"Connection refused — is the server running at {url}?",
        )
    except Exception as exc:
        return Result(request_num, None, None, 0, str(exc))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of *values* (0–100) using nearest-rank."""
    if not values:
        return float("nan")
    s = sorted(values)
    # nearest-rank: ceil(p/100 * n) - 1, clamped to valid range
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "  error "
    return f"{seconds:6.3f} s"


def _bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Valura AI /query SSE endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=_DEFAULT_URL, help="Base server URL")
    parser.add_argument("--n", type=int, default=_DEFAULT_N, help="Number of requests")
    parser.add_argument("--verbose", action="store_true", help="Print per-request table")
    args = parser.parse_args()

    print(f"\nValura AI — latency benchmark")
    print(f"  URL   : {args.url}/query")
    print(f"  User  : {_USER_ID}")
    print(f"  Query : {_QUERY[:72]}…")
    print(f"  N     : {args.n} sequential requests\n")

    results: list[Result] = []

    with httpx.Client() as client:
        for i in range(1, args.n + 1):
            print(f"  [{i:>2}/{args.n}] ", end="", flush=True)
            r = _measure(client, args.url, i)
            results.append(r)

            if r.error:
                print(f"ERROR — {r.error}")
                if "Connection refused" in (r.error or ""):
                    sys.exit(1)
            else:
                print(
                    f"ttft={_fmt(r.ttft)}  trt={_fmt(r.trt)}  "
                    f"chunks={r.chunks:>3}"
                )

    # Separate good / bad results.
    good = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    print(f"\n{'-' * 62}")
    print(f"  Completed: {len(good)}/{args.n} requests succeeded"
          + (f", {len(errors)} failed" if errors else ""))

    if not good:
        print("  No successful results to report.")
        sys.exit(1)

    ttfts = [r.ttft for r in good if r.ttft is not None]
    trts  = [r.trt  for r in good if r.trt  is not None]

    # Summary table.
    p50_ttft = _percentile(ttfts, 50)
    p95_ttft = _percentile(ttfts, 95)
    p50_trt  = _percentile(trts,  50)
    p95_trt  = _percentile(trts,  95)

    max_trt = max(trts) if trts else 1.0

    print(f"\n  {'Metric':<26}  {'p50':>8}  {'p95':>8}")
    print(f"  {'-' * 26}  {'-' * 8}  {'-' * 8}")
    print(f"  {'Time to first token':<26}  {p50_ttft:>7.3f}s  {p95_ttft:>7.3f}s")
    print(f"  {'Total response time':<26}  {p50_trt:>7.3f}s  {p95_trt:>7.3f}s")

    if args.verbose and good:
        print(f"\n  {'#':>3}  {'TTFT':>8}  {'TRT':>8}  TRT distribution")
        print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*30}")
        for r in good:
            bar = _bar(r.trt or 0, max_trt)
            print(f"  {r.request_num:>3}  {_fmt(r.ttft)}  {_fmt(r.trt)}  {bar}")

    if errors:
        print(f"\n  Failed requests:")
        for r in errors:
            print(f"    [{r.request_num:>2}] {r.error}")

    print()


if __name__ == "__main__":
    main()
