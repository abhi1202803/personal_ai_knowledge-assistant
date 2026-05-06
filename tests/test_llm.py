"""Tests for backend.services.llm."""

from __future__ import annotations

import pytest

from backend.services.llm import get_chat_model


class TestGetChatModel:
    """Test LLM factory function."""

    def test_openai_provider(self):
        """Should create a ChatOpenAI for openai provider."""
        model = get_chat_model("openai", "gpt-4o-mini", streaming=False)
        assert model is not None
        assert model.model_name == "gpt-4o-mini"

    def test_qwen_provider(self):
        """Should create a ChatOpenAI for qwen provider."""
        model = get_chat_model("qwen", "qwen-plus", streaming=True)
        assert model is not None
        assert model.model_name == "qwen-plus"
        assert model.streaming is True

    def test_unknown_provider_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_model("unknown", "some-model")

    def test_streaming_flag(self):
        """Streaming flag should be passed through."""
        model_stream = get_chat_model("openai", "gpt-4o-mini", streaming=True)
        model_no_stream = get_chat_model("openai", "gpt-4o-mini", streaming=False)
        assert model_stream.streaming is True
        assert model_no_stream.streaming is False

    def test_temperature(self):
        """Temperature should be configurable."""
        model = get_chat_model("openai", "gpt-4o-mini", temperature=0.0)
        assert model.temperature == 0.0
