"""
Unit tests for InMemorySessionStore.

No I/O, no mocks needed — the store is pure in-memory.
"""
from __future__ import annotations

from src.session.store import InMemorySessionStore


def test_add_and_retrieve_turns() -> None:
    """Turns added via add_turn() are returned by get_history() in order."""
    store = InMemorySessionStore()
    store.add_turn("s1", "user", "How is my portfolio?")
    store.add_turn("s1", "assistant", "Your portfolio is up 12%.")
    store.add_turn("s1", "user", "What about NVDA?")

    history = store.get_history("s1")

    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "How is my portfolio?"
    assert history[1]["role"] == "assistant"
    assert history[2]["content"] == "What about NVDA?"
    # Every turn must carry a numeric timestamp
    for turn in history:
        assert isinstance(turn["timestamp"], float)
        assert turn["timestamp"] > 0


def test_max_turn_eviction() -> None:
    """
    When max_turns is exceeded the oldest turns are dropped, not the newest.
    The store must never hold more than max_turns entries per session.
    """
    store = InMemorySessionStore(max_turns=5)

    for i in range(8):
        store.add_turn("s2", "user", f"message {i}")

    history = store.get_history("s2")

    assert len(history) == 5, (
        f"Expected 5 turns after eviction, got {len(history)}"
    )
    # The five most-recent messages (3-7) must be present; 0-2 must be gone
    contents = [t["content"] for t in history]
    assert contents == [f"message {i}" for i in range(3, 8)], (
        f"Wrong turns retained after eviction: {contents}"
    )


def test_empty_session_returns_empty_list() -> None:
    """get_history() for a session that never received a turn returns []."""
    store = InMemorySessionStore()

    result = store.get_history("nonexistent-session-xyz")

    assert result == []


def test_clear_removes_all_turns() -> None:
    """clear() wipes the session; subsequent get_history() returns []."""
    store = InMemorySessionStore()
    store.add_turn("s3", "user", "first")
    store.add_turn("s3", "assistant", "second")

    store.clear("s3")

    assert store.get_history("s3") == []


def test_clear_unknown_session_is_noop() -> None:
    """clear() on a session that never existed must not raise."""
    store = InMemorySessionStore()
    store.clear("ghost-session")  # must not raise


def test_sessions_are_independent() -> None:
    """Turns added to one session must not appear in another."""
    store = InMemorySessionStore()
    store.add_turn("a", "user", "hello from A")
    store.add_turn("b", "user", "hello from B")

    assert store.get_history("a")[0]["content"] == "hello from A"
    assert store.get_history("b")[0]["content"] == "hello from B"
    assert len(store.get_history("a")) == 1
    assert len(store.get_history("b")) == 1
