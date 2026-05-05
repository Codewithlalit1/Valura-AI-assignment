"""
LLM provider abstraction — OpenAI or Groq.

Provider is auto-detected from environment variables at call time:

  Priority:  OPENAI_API_KEY  >  GROQ_API_KEY
  If both are set, OpenAI is used.
  If neither is set, the server starts but every LLM call will raise
  AuthenticationError (caught by each caller's fallback path).

Groq exposes an OpenAI-compatible REST endpoint, so we reuse the openai
SDK for both providers — only the base_url and api_key differ.
No additional dependencies are required.

Groq OpenAI-compatible endpoint:
  https://console.groq.com/docs/openai
"""
from __future__ import annotations

import logging
import os

import openai

logger = logging.getLogger(__name__)

# Groq's OpenAI-compatible base URL
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_provider() -> str:
    """
    Return the active provider name: ``"openai"``, ``"groq"``, or ``"none"``.

    Reads environment variables at call time so .env changes (via load_dotenv)
    are always reflected.
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "none"


def get_model() -> str:
    """
    Return the model name appropriate for the active provider.

    OpenAI : MODEL_DEV       (default: gpt-4o-mini)
    Groq   : GROQ_MODEL_DEV  (default: llama-3.3-70b-versatile)
    """
    provider = get_provider()
    if provider == "groq":
        return os.getenv("GROQ_MODEL_DEV", "llama-3.3-70b-versatile")
    return os.getenv("MODEL_DEV", "gpt-4o-mini")


def make_sync_client() -> openai.OpenAI:
    """
    Return a synchronous OpenAI-compatible client for the active provider.
    Used by IntentClassifier (runs inside asyncio.to_thread).
    """
    provider = get_provider()
    if provider == "groq":
        return openai.OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=_GROQ_BASE_URL,
        )
    return openai.OpenAI()


def make_async_client() -> openai.AsyncOpenAI:
    """
    Return an async OpenAI-compatible client for the active provider.
    Used by PortfolioHealthAgent for streaming responses.
    """
    provider = get_provider()
    if provider == "groq":
        return openai.AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=_GROQ_BASE_URL,
        )
    return openai.AsyncOpenAI()
