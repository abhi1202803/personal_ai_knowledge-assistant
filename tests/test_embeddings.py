"""Tests for backend.services.embeddings."""

from __future__ import annotations

import pytest

from backend.services.embeddings import (
    get_cached_embedding_function,
    get_embedding_function,
)


class TestGetEmbeddingFunction:
    """Test embedding function factory."""

    def test_openai_provider(self):
        """Should return an OpenAIEmbeddings instance for openai provider."""
        emb = get_embedding_function("openai", "text-embedding-3-small")
        assert emb is not None
        assert emb.model == "text-embedding-3-small"

    def test_qwen_provider(self):
        """Should return an OpenAIEmbeddings instance for qwen provider."""
        emb = get_embedding_function("qwen", "text-embedding-v3")
        assert emb is not None
        assert emb.model == "text-embedding-v3"

    def test_unknown_provider_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_function("unknown_provider", "some-model")


class TestCachedEmbeddingFunction:
    """Test that caching works correctly."""

    def test_same_params_return_same_instance(self):
        """Same provider+model should return the exact same object."""
        # Clear cache first
        get_cached_embedding_function.cache_clear()

        emb1 = get_cached_embedding_function("openai", "text-embedding-3-small")
        emb2 = get_cached_embedding_function("openai", "text-embedding-3-small")
        assert emb1 is emb2

    def test_different_params_return_different_instances(self):
        """Different params should return distinct objects."""
        get_cached_embedding_function.cache_clear()

        emb_openai = get_cached_embedding_function("openai", "text-embedding-3-small")
        emb_qwen = get_cached_embedding_function("qwen", "text-embedding-v3")
        assert emb_openai is not emb_qwen

    def test_cache_info(self):
        """Cache stats should reflect usage."""
        get_cached_embedding_function.cache_clear()

        get_cached_embedding_function("openai", "text-embedding-3-small")
        get_cached_embedding_function("openai", "text-embedding-3-small")
        get_cached_embedding_function("qwen", "text-embedding-v3")

        info = get_cached_embedding_function.cache_info()
        assert info.hits == 1   # second openai call was a hit
        assert info.misses == 2  # first openai + qwen were misses
