"""
LLM provider abstraction — OpenAI or Google Gemini.

Provider is auto-detected from environment variables at call time:

  Priority:  OPENAI_API_KEY  >  GOOGLE_API_KEY
  If both are set, OpenAI is used.
  If neither is set, the server starts but every LLM call will raise
  AuthenticationError (caught by each caller's fallback path).

Google Gemini exposes an OpenAI-compatible REST endpoint, so we reuse
the openai SDK for both providers — only the base_url and api_key differ.
No additional dependencies are required.

Gemini OpenAI-compatible endpoint:
  https://ai.google.dev/gemini-api/docs/openai
"""
from __future__ import annotations

import logging
import os

import openai

logger = logging.getLogger(__name__)

# Gemini's OpenAI-compatible base URL
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_provider() -> str:
    """
    Return the active provider name: ``"openai"``, ``"gemini"``, or ``"none"``.

    Reads environment variables at call time so .env changes (via load_dotenv)
    are always reflected.
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


def get_model() -> str:
    """
    Return the model name appropriate for the active provider.

    OpenAI : MODEL_DEV          (default: gpt-4o-mini)
    Gemini : GEMINI_MODEL_DEV   (default: gemini-2.0-flash)
    """
    provider = get_provider()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL_DEV", "gemini-2.0-flash")
    return os.getenv("MODEL_DEV", "gpt-4o-mini")


def make_sync_client() -> openai.OpenAI:
    """
    Return a synchronous OpenAI-compatible client for the active provider.
    Used by IntentClassifier (runs inside asyncio.to_thread).
    """
    provider = get_provider()
    if provider == "gemini":
        return openai.OpenAI(
            api_key=os.environ["GOOGLE_API_KEY"],
            base_url=_GEMINI_BASE_URL,
        )
    return openai.OpenAI()


def make_async_client() -> openai.AsyncOpenAI:
    """
    Return an async OpenAI-compatible client for the active provider.
    Used by PortfolioHealthAgent for streaming responses.
    """
    provider = get_provider()
    if provider == "gemini":
        return openai.AsyncOpenAI(
            api_key=os.environ["GOOGLE_API_KEY"],
            base_url=_GEMINI_BASE_URL,
        )
    return openai.AsyncOpenAI()
