"""Unified LLM service.

Both OpenAI and Qwen use the OpenAI-compatible chat API, so we use
``langchain_openai.ChatOpenAI`` with different ``base_url`` values.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from backend.config import PROVIDER_CONFIG

logger = logging.getLogger(__name__)


def get_chat_model(
    provider: str,
    model: str,
    *,
    streaming: bool = True,
    temperature: float = 0.7,
) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance for the given provider/model.

    Parameters
    ----------
    provider:
        ``"openai"`` or ``"qwen"``.
    model:
        The chat model name, e.g. ``"gpt-4o-mini"`` or ``"qwen-plus"``.
    streaming:
        Whether to enable streaming output.
    temperature:
        Sampling temperature.
    """
    cfg = PROVIDER_CONFIG.get(provider)
    if cfg is None:
        logger.error("Unknown LLM provider requested: %s", provider)
        raise ValueError(f"Unknown LLM provider: {provider}")

    logger.info(
        "Creating ChatOpenAI instance: provider=%s, model=%s, "
        "streaming=%s, temperature=%.2f, base_url=%s",
        provider,
        model,
        streaming,
        temperature,
        cfg["base_url"],
    )
    return ChatOpenAI(
        model=model,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        streaming=streaming,
        temperature=temperature,
    )
