"""Tests for backend.config."""

from __future__ import annotations

from backend.config import AVAILABLE_MODELS, PROVIDER_CONFIG

ALL_PROVIDERS = ("openai", "qwen", "gemini")


class TestConfig:
    """Verify configuration structures."""

    def test_provider_config_has_all_providers(self):
        for p in ALL_PROVIDERS:
            assert p in PROVIDER_CONFIG

    def test_provider_config_structure(self):
        for provider in ALL_PROVIDERS:
            cfg = PROVIDER_CONFIG[provider]
            assert "api_key" in cfg
            assert "base_url" in cfg
            assert isinstance(cfg["api_key"], str)
            assert isinstance(cfg["base_url"], str)

    def test_qwen_base_url(self):
        assert "dashscope" in PROVIDER_CONFIG["qwen"]["base_url"]

    def test_gemini_base_url(self):
        assert "googleapis" in PROVIDER_CONFIG["gemini"]["base_url"]

    def test_available_models_has_all_providers(self):
        for p in ALL_PROVIDERS:
            assert p in AVAILABLE_MODELS
            assert "chat" in AVAILABLE_MODELS[p]
            assert "embedding" in AVAILABLE_MODELS[p]

    def test_openai_models(self):
        chat = AVAILABLE_MODELS["openai"]["chat"]
        assert "gpt-4o-mini" in chat
        emb = AVAILABLE_MODELS["openai"]["embedding"]
        assert "text-embedding-3-small" in emb

    def test_qwen_models(self):
        chat = AVAILABLE_MODELS["qwen"]["chat"]
        assert "qwen-plus" in chat
        emb = AVAILABLE_MODELS["qwen"]["embedding"]
        assert "text-embedding-v3" in emb

    def test_gemini_chat_models(self):
        chat = AVAILABLE_MODELS["gemini"]["chat"]
        assert len(chat) > 0
        assert any("gemini" in m for m in chat)

    def test_gemini_has_no_embedding_models(self):
        """Gemini does not expose an OpenAI-compatible embedding endpoint."""
        assert AVAILABLE_MODELS["gemini"]["embedding"] == []
