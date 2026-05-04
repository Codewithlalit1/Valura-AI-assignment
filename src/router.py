"""
AgentRouter — dispatches a ClassificationResult to the correct agent.

Registry design
---------------
Every agent name in VALID_AGENTS must be present in the registry.
Real agents are stored as agent instances with an ``analyze()`` method.
Agents not yet built are stored as the module-level ``_STUB`` sentinel.

The ``route()`` method is an async generator:
- Real agent  → delegates to agent.analyze(), yielding each chunk as it arrives.
- Stub / unknown → yields a single structured JSON string and returns.
- Any exception inside a real agent → yields a structured error JSON; never raises.

Factory
-------
``create_router(model)`` is the intended entry point.  It wires up
PortfolioHealthAgent as the sole real implementation and marks the other
seven VALID_AGENTS as stubs.  Pass a different model string to switch the
underlying LLM for all real agents at once.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

from src.agents.portfolio_health.agent import PortfolioHealthAgent
from src.classifier.schema import VALID_AGENTS, ClassificationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub sentinel
# ---------------------------------------------------------------------------


class _Stub:
    """Explicitly registered placeholder for a not-yet-implemented agent."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Stub agent={self.name!r}>"


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _not_implemented_json(cr: ClassificationResult) -> str:
    """Return the structured 'not implemented' JSON string for a stub agent."""
    return json.dumps(
        {
            "status": "not_implemented",
            "intent": cr.intent,
            "agent": cr.agent,
            "entities": cr.entities.model_dump(),
            "message": "This agent is not yet implemented in this build.",
        }
    )


def _error_json(cr: ClassificationResult, detail: str) -> str:
    """Return a structured error JSON when a real agent raises unexpectedly."""
    return json.dumps(
        {
            "status": "error",
            "intent": cr.intent,
            "agent": cr.agent,
            "message": f"Agent encountered an unexpected error: {detail}",
        }
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class AgentRouter:
    """
    Dispatches a ClassificationResult to the appropriate agent.

    Parameters
    ----------
    registry:
        Mapping of agent-name → agent-instance or ``_Stub``.
        Callers should use ``create_router()`` rather than constructing
        this directly.
    """

    def __init__(self, registry: dict[str, object]) -> None:
        self._registry = registry

    async def route(
        self,
        classification_result: ClassificationResult,
        user_profile: dict,
        conversation_history: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Yield the agent's response stream.

        For a stub or unknown agent, yields a single JSON string.
        For a real agent, forwards every chunk as it arrives.
        Never raises — all exceptions are caught and serialised.
        """
        agent_name = classification_result.agent
        entry = self._registry.get(agent_name)

        # ----- stub or unknown agent -----
        if entry is None or isinstance(entry, _Stub):
            yield _not_implemented_json(classification_result)
            return

        # ----- real agent -----
        try:
            async for chunk in entry.analyze(user_profile, conversation_history):
                yield chunk
        except Exception as exc:
            logger.warning(
                "AgentRouter: agent %r raised %s: %s",
                agent_name,
                type(exc).__name__,
                exc,
            )
            yield _error_json(classification_result, str(exc))

    # Convenience: expose the registry for inspection / testing
    @property
    def registry(self) -> dict[str, object]:
        return dict(self._registry)

    def is_real(self, agent_name: str) -> bool:
        """Return True when agent_name maps to a real (non-stub) implementation."""
        entry = self._registry.get(agent_name)
        return entry is not None and not isinstance(entry, _Stub)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_router(model: str = "gpt-4o-mini") -> AgentRouter:
    """
    Build the production router with all agents registered.

    ``portfolio_health`` is wired to ``PortfolioHealthAgent``.
    Every other agent in ``VALID_AGENTS`` is registered as a ``_Stub``.

    Parameters
    ----------
    model:
        OpenAI model identifier forwarded to every real agent.
    """
    registry: dict[str, object] = {
        "portfolio_health": PortfolioHealthAgent(model=model),
    }

    for name in VALID_AGENTS:
        if name not in registry:
            registry[name] = _Stub(name)

    return AgentRouter(registry)
