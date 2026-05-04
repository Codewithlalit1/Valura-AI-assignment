"""
Valura AI — FastAPI application.

Single endpoint: POST /query
Pipeline (in order):
  1. SafetyGuard.check()       — local, sync, < 10 ms
  2. IntentClassifier.classify() — one LLM call, sync (run in thread)
  3. AgentRouter.route()       — async generator; real or stub agent
  4. Session persistence       — append user + assistant turns

All responses are streamed via Server-Sent Events (SSE).
Errors are serialised as SSE data events — stack traces are never exposed.

Timeout design
--------------
A background task runs the pipeline and pushes chunks into an asyncio.Queue.
The SSE generator reads from the queue with a shrinking deadline on each
asyncio.wait_for() call, giving a hard total timeout of _PIPELINE_TIMEOUT
seconds across the entire request.  This avoids the complexities of injecting
TimeoutError into an async generator suspended at a yield point.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.classifier.classifier import IntentClassifier
from src.router import AgentRouter, create_router
from src.safety.guard import SafetyGuard
from src.session.store import InMemorySessionStore, SessionStore
from src.users import UserProfileLoader

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL: str = os.getenv("MODEL_DEV", "gpt-4o-mini")

# Hard upper bound for the full safety → classify → stream pipeline.
# Chosen to be comfortably above the p95 target (< 6 s) while protecting
# against runaway LLM calls or network hangs.
_PIPELINE_TIMEOUT: float = 30.0


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Initialise shared singletons once at startup; teardown on shutdown."""
    user_loader = UserProfileLoader()
    user_loader.load()
    app.state.user_loader = user_loader
    app.state.safety_guard = SafetyGuard()
    app.state.classifier = IntentClassifier(model=MODEL)
    app.state.router = create_router(model=MODEL)
    app.state.session_store = InMemorySessionStore()
    logger.info(
        "Valura AI started — model=%s  timeout=%.0fs  users=%d",
        MODEL, _PIPELINE_TIMEOUT, user_loader.loaded_count,
    )
    yield
    logger.info("Valura AI shutting down")


app = FastAPI(
    title="Valura AI",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    user_id: str
    session_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_user_profile(request: Request, user_id: str) -> dict:
    """Look up a user profile from the startup-loaded index (O(1), no I/O)."""
    return request.app.state.user_loader.get_profile(user_id)


def _blocked_event(category: str | None, message: str | None) -> str:
    return json.dumps(
        {
            "event": "blocked",
            "category": category or "unknown",
            "message": message or "This query cannot be processed.",
        }
    )


def _timeout_event() -> str:
    return json.dumps(
        {
            "event": "timeout",
            "message": (
                "The request took too long to process. "
                "Please try again — complex queries may need a moment."
            ),
        }
    )


def _error_event(detail: str = "An internal error occurred. Please try again.") -> str:
    return json.dumps({"event": "error", "message": detail})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns 200 immediately."""
    return {"status": "ok", "model": MODEL}


@app.post("/query")
async def query_endpoint(body: QueryRequest, request: Request):
    """
    Run the full pipeline and stream each token/chunk as an SSE data event.

    SSE event shapes
    ----------------
    Normal chunk  : data: <raw text or JSON string yielded by the agent>
    Blocked       : data: {"event": "blocked", "category": "...", "message": "..."}
    Timeout       : data: {"event": "timeout", "message": "..."}
    Error         : data: {"event": "error",   "message": "..."}
    """
    safety_guard: SafetyGuard = request.app.state.safety_guard
    classifier: IntentClassifier = request.app.state.classifier
    router: AgentRouter = request.app.state.router
    session_store: SessionStore = request.app.state.session_store

    async def event_stream() -> AsyncGenerator[str, None]:
        # Queue bridges the background pipeline task and the SSE generator.
        # None is the sentinel that signals the pipeline has finished.
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def run_pipeline() -> None:
            """Execute every pipeline step; push chunks into the queue."""
            response_parts: list[str] = []
            try:
                # ── 1. Safety guard (synchronous, fast) ──────────────────
                guard = safety_guard.check(body.query)
                if guard.blocked:
                    await queue.put(_blocked_event(guard.category, guard.response))
                    return

                # ── 2. Load session context ───────────────────────────────
                history = session_store.get_history(body.session_id)
                user_profile = _get_user_profile(request, body.user_id)

                # ── 3. Classify (sync call — run in thread pool) ──────────
                classification = await asyncio.to_thread(
                    classifier.classify,
                    body.query,
                    history,
                    user_profile,
                )

                # ── 4. Route + stream chunks ──────────────────────────────
                async for chunk in router.route(
                    classification, user_profile, history
                ):
                    response_parts.append(chunk)
                    await queue.put(chunk)

                # ── 5. Persist both turns to session store ────────────────
                session_store.add_turn(body.session_id, "user", body.query)
                session_store.add_turn(
                    body.session_id,
                    "assistant",
                    "".join(response_parts),
                )

            except asyncio.CancelledError:
                # Cancelled by the timeout handler — exit silently.
                pass
            except Exception as exc:
                logger.error(
                    "Pipeline error [session=%s user=%s]: %s",
                    body.session_id,
                    body.user_id,
                    exc,
                    exc_info=True,
                )
                await queue.put(_error_event())
            finally:
                await queue.put(None)  # sentinel: always signal completion

        pipeline_task = asyncio.create_task(run_pipeline())

        # Track the hard deadline with a monotonic clock.
        # Each asyncio.wait_for() call uses only the *remaining* time so
        # the total wall-clock budget is never exceeded.
        deadline = time.monotonic() + _PIPELINE_TIMEOUT

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pipeline_task.cancel()
                    yield _timeout_event()
                    return

                try:
                    chunk = await asyncio.wait_for(
                        queue.get(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    pipeline_task.cancel()
                    yield _timeout_event()
                    return

                if chunk is None:  # sentinel: pipeline finished cleanly
                    return

                yield chunk

        finally:
            # Guard against the generator being closed early (client disconnect).
            if not pipeline_task.done():
                pipeline_task.cancel()

    return EventSourceResponse(event_stream())
