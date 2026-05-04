"""
Session store: interface + in-memory implementation.

Each session is a list of conversation turns:
    {"role": "user" | "assistant", "content": str, "timestamp": float}

TODO: Replace InMemorySessionStore with a PostgresSessionStore or
      RedisSessionStore for production use.  Both would implement the
      same SessionStore interface so the rest of the codebase requires
      no changes.  PostgreSQL is the preferred target (persistent,
      queryable); Redis suits high-throughput deployments where history
      can be reconstructed from durable storage if evicted.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque

# Maximum conversation turns kept per session in the in-memory store.
_MAX_TURNS: int = 50


class SessionStore(ABC):
    """
    Abstract interface for session-backed conversation history.

    All implementations must be safe to call concurrently within a
    single event-loop thread (i.e. no async locks needed for in-process
    stores, but a Postgres/Redis implementation would use async I/O).
    """

    @abstractmethod
    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """
        Append one turn to the session's history.

        Parameters
        ----------
        session_id:
            Opaque string that identifies the conversation.
        role:
            ``"user"`` or ``"assistant"``.
        content:
            The raw text of the turn.
        """

    @abstractmethod
    def get_history(self, session_id: str) -> list[dict]:
        """
        Return all turns for *session_id*, oldest first.

        Returns an empty list for sessions that have never received a turn.
        """

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all turns for *session_id*.  No-op for unknown sessions."""


class InMemorySessionStore(SessionStore):
    """
    Process-local session store backed by a plain dict of deques.

    Bounded to ``max_turns`` per session (default 50).  When the limit
    is reached the oldest turn is dropped before the new one is appended,
    so the most recent context is always preserved.

    Not thread-safe across OS threads; safe within a single-threaded
    asyncio event loop.

    Parameters
    ----------
    max_turns:
        Maximum number of turns kept per session.  Oldest are evicted first.
    """

    def __init__(self, max_turns: int = _MAX_TURNS) -> None:
        self._max_turns = max_turns
        # session_id -> deque of turn dicts, capped at max_turns
        self._sessions: dict[str, deque[dict]] = {}

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self._max_turns)
        self._sessions[session_id].append(
            {"role": role, "content": content, "timestamp": time.time()}
        )

    def get_history(self, session_id: str) -> list[dict]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
