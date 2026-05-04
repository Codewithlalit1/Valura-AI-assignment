"""
UserProfileLoader — loads all user profiles from fixtures/users/ once at
startup and serves them from memory for the lifetime of the process.

Why load at startup?
--------------------
Scanning the filesystem and parsing JSON on every request adds latency and
I/O syscalls that are completely unnecessary when the file set is static.
Loading once at boot amortises that cost to zero per request.

Production replacement
----------------------
In a multi-tenant production environment this class would be replaced (or
extended) to load profiles from Postgres or a user-service API, using the
same get_profile() interface.  The rest of the codebase requires no changes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Safe default returned for any unknown user_id.
_EMPTY_PROFILE: dict = {
    "user_id": "",
    "name": "Unknown",
    "kyc_status": "unknown",
    "risk_profile": "moderate",
    "currency": "USD",
    "portfolio": [],
}


class UserProfileLoader:
    """
    Reads every ``*.json`` file in *users_dir* and indexes the results by
    ``user_id``.  Any file that cannot be parsed is skipped with a warning.

    Parameters
    ----------
    users_dir:
        Path to the directory that contains the user JSON fixtures.
        Defaults to ``<repo-root>/fixtures/users/``.
    """

    def __init__(self, users_dir: Path | None = None) -> None:
        if users_dir is None:
            users_dir = Path(__file__).parent.parent / "fixtures" / "users"
        self._users_dir = users_dir
        self._profiles: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Parse all ``*.json`` files in *users_dir* and populate the in-memory
        index.  Safe to call multiple times — each call replaces the index.
        """
        profiles: dict[str, dict] = {}
        for path in sorted(self._users_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    profile = json.load(fh)
                user_id = profile.get("user_id")
                if not user_id:
                    logger.warning("Skipping %s — missing user_id field", path.name)
                    continue
                profiles[user_id] = profile
            except Exception as exc:
                logger.warning("Failed to load user fixture %s: %s", path.name, exc)

        self._profiles = profiles
        logger.info(
            "UserProfileLoader: loaded %d profiles from %s",
            len(profiles),
            self._users_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> dict:
        """
        Return the profile for *user_id*.

        If the user_id is not in the index, returns a safe default profile
        with an empty portfolio so callers never receive ``None`` or raise.
        The returned profile is a shallow copy so callers cannot mutate the
        stored index.
        """
        profile = self._profiles.get(user_id)
        if profile is None:
            default = dict(_EMPTY_PROFILE)
            default["user_id"] = user_id
            return default
        return dict(profile)

    @property
    def loaded_count(self) -> int:
        """Number of profiles currently in the index."""
        return len(self._profiles)

    @property
    def known_ids(self) -> frozenset[str]:
        """All user_ids currently in the index (immutable view)."""
        return frozenset(self._profiles)
