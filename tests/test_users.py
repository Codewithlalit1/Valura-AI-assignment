"""
Unit tests for UserProfileLoader (src/users.py).

No mocks needed — reads the real fixtures/users/ directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.users import UserProfileLoader

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "users"


def test_all_fixture_users_load() -> None:
    """loaded_count must equal the number of *.json files in fixtures/users/."""
    expected = len(list(_FIXTURES_DIR.glob("*.json")))
    loader = UserProfileLoader()
    loader.load()
    assert loader.loaded_count == expected, (
        f"Expected {expected} profiles, got {loader.loaded_count}"
    )


def test_usr_004_has_empty_portfolio() -> None:
    """usr_004 is the 'new user' fixture — portfolio must be an empty list."""
    loader = UserProfileLoader()
    loader.load()
    profile = loader.get_profile("usr_004")
    assert profile["portfolio"] == [], (
        f"Expected empty portfolio for usr_004, got: {profile['portfolio']}"
    )


def test_unknown_user_returns_safe_default() -> None:
    """get_profile() for an unknown user_id must not raise and must return a
    usable default with the correct user_id and an empty portfolio."""
    loader = UserProfileLoader()
    loader.load()
    profile = loader.get_profile("nonexistent_user_xyz")
    assert profile["user_id"] == "nonexistent_user_xyz"
    assert profile["portfolio"] == []
    assert profile["kyc_status"] == "unknown"


def test_get_profile_returns_copy() -> None:
    """Mutating the returned profile must not affect subsequent calls."""
    loader = UserProfileLoader()
    loader.load()
    profile_a = loader.get_profile("usr_004")
    profile_a["currency"] = "MUTATED"
    profile_b = loader.get_profile("usr_004")
    assert profile_b["currency"] != "MUTATED", (
        "get_profile() returned a mutable reference to the stored dict"
    )


def test_known_ids_contains_all_fixture_users() -> None:
    """known_ids must include every user_id found in the fixture files."""
    loader = UserProfileLoader()
    loader.load()
    ids = loader.known_ids
    assert "usr_004" in ids
    assert len(ids) == loader.loaded_count


def test_load_is_idempotent() -> None:
    """Calling load() twice must not double the profile count."""
    loader = UserProfileLoader()
    loader.load()
    count_first = loader.loaded_count
    loader.load()
    assert loader.loaded_count == count_first
