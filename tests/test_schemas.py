"""Tests for backend.models.schemas (Pydantic models)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.models.schemas import (
    AvailableModelsResponse,
    ChatRequest,
    DocumentInfo,
    KnowledgeBaseCreate,
    KnowledgeBaseInfo,
    ProviderModels,
)


class TestKnowledgeBaseCreate:
    def test_valid_ascii(self):
        kb = KnowledgeBaseCreate(
            name="test-kb",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        assert kb.name == "test-kb"

    def test_valid_chinese(self):
        kb = KnowledgeBaseCreate(
            name="欧洲文明15讲",
            embedding_provider="qwen",
            embedding_model="text-embedding-v3",
        )
        assert kb.name == "欧洲文明15讲"

    def test_valid_with_spaces(self):
        kb = KnowledgeBaseCreate(
            name="European Civilization 15 Lectures",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        assert kb.name == "European Civilization 15 Lectures"

    def test_name_empty(self):
        with pytest.raises(ValidationError):
            KnowledgeBaseCreate(
                name="",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
            )

    def test_name_too_long(self):
        with pytest.raises(ValidationError):
            KnowledgeBaseCreate(
                name="x" * 201,
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
            )

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            KnowledgeBaseCreate(name="test")  # type: ignore


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(message="hello")
        assert req.model_provider == "openai"
        assert req.model_name == "gpt-4o-mini"
        assert req.image is None
        assert req.kb_name is None

    def test_full(self):
        req = ChatRequest(
            message="test",
            image="base64data",
            kb_name="my-kb",
            model_provider="qwen",
            model_name="qwen-plus",
        )
        assert req.image == "base64data"
        assert req.kb_name == "my-kb"

    def test_missing_message(self):
        with pytest.raises(ValidationError):
            ChatRequest()  # type: ignore


class TestKnowledgeBaseInfo:
    def test_defaults(self):
        info = KnowledgeBaseInfo(
            name="kb",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        assert info.document_count == 0


class TestDocumentInfo:
    def test_defaults(self):
        doc = DocumentInfo(filename="test.pdf")
        assert doc.page_range is None
        assert doc.chunk_count == 0


class TestProviderModels:
    def test_valid(self):
        pm = ProviderModels(chat=["gpt-4o"], embedding=["text-embedding-3-small"])
        assert len(pm.chat) == 1


class TestAvailableModelsResponse:
    def test_valid(self):
        resp = AvailableModelsResponse(
            openai=ProviderModels(chat=["gpt-4o"], embedding=["text-embedding-3-small"]),
            qwen=ProviderModels(chat=["qwen-plus"], embedding=["text-embedding-v3"]),
            gemini=ProviderModels(chat=["gemini-2.0-flash"], embedding=[]),
        )
        assert resp.openai.chat == ["gpt-4o"]
        assert resp.gemini.chat == ["gemini-2.0-flash"]
        assert resp.gemini.embedding == []
