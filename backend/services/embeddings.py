"""Unified embedding service.

Both OpenAI and Qwen use the OpenAI-compatible API, so we use
``langchain_openai.OpenAIEmbeddings`` with different ``base_url`` values.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from backend.config import PROVIDER_CONFIG

logger = logging.getLogger(__name__)


def get_embedding_function(
    provider: str,
    model: str,
) -> OpenAIEmbeddings:
    """Return an ``OpenAIEmbeddings`` instance for the given provider/model.

    Parameters
    ----------
    provider:
        ``"openai"`` or ``"qwen"``.
    model:
        The embedding model name, e.g. ``"text-embedding-3-small"``.
    """
    cfg = PROVIDER_CONFIG.get(provider)
    if cfg is None:
        logger.error("Unknown embedding provider requested: %s", provider)
        raise ValueError(f"Unknown embedding provider: {provider}")

    logger.info(
        "Creating OpenAIEmbeddings instance: provider=%s, model=%s, base_url=%s",
        provider,
        model,
        cfg["base_url"],
    )
    chunk_size: int = cfg.get("embedding_chunk_size", 100)
    return OpenAIEmbeddings(
        model=model,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        # Disable tiktoken pre-tokenization: OpenAI accepts token-ID arrays,
        # but Qwen and other compatible providers only accept raw strings.
        check_embedding_ctx_length=False,
        # Respect each provider's per-request batch size limit.
        chunk_size=chunk_size,
    )


@lru_cache(maxsize=16)
def get_cached_embedding_function(
    provider: str,
    model: str,
) -> OpenAIEmbeddings:
    """Cached version – avoids re-creating the client on every call."""
    logger.debug(
        "get_cached_embedding_function called: provider=%s, model=%s",
        provider,
        model,
    )
    return get_embedding_function(provider, model)
