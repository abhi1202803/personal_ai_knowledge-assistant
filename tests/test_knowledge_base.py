"""Tests for backend.services.knowledge_base.

These tests use an isolated temporary ChromaDB directory (see conftest.py).
Embedding calls are mocked to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.services.knowledge_base import (
    add_documents,
    create_knowledge_base,
    delete_knowledge_base,
    get_collection_embedding_info,
    list_documents,
    list_knowledge_bases,
    query_knowledge_base,
)


class TestKnowledgeBaseCRUD:
    """Create / list / delete knowledge bases."""

    def test_create_and_list(self):
        """Creating a KB should make it appear in list."""
        result = create_knowledge_base("test-kb", "openai", "text-embedding-3-small")
        assert result["name"] == "test-kb"
        assert result["embedding_provider"] == "openai"
        assert result["document_count"] == 0

        kbs = list_knowledge_bases()
        names = [kb["name"] for kb in kbs]
        assert "test-kb" in names

    def test_create_duplicate_raises(self):
        """Creating a KB with a duplicate name should raise ValueError."""
        create_knowledge_base("dup-kb", "openai", "text-embedding-3-small")
        with pytest.raises(ValueError, match="already exists"):
            create_knowledge_base("dup-kb", "openai", "text-embedding-3-small")

    def test_delete(self):
        """Deleting a KB should remove it from list."""
        create_knowledge_base("del-kb", "qwen", "text-embedding-v3")
        kbs = list_knowledge_bases()
        assert any(kb["name"] == "del-kb" for kb in kbs)

        delete_knowledge_base("del-kb")
        kbs = list_knowledge_bases()
        assert not any(kb["name"] == "del-kb" for kb in kbs)

    def test_delete_nonexistent_raises(self):
        """Deleting a non-existent KB should raise an exception."""
        with pytest.raises(Exception):
            delete_knowledge_base("nonexistent-kb-xyz")

    def test_get_embedding_info(self):
        """Should return the correct provider and model."""
        create_knowledge_base("info-kb", "qwen", "text-embedding-v2")
        provider, model = get_collection_embedding_info("info-kb")
        assert provider == "qwen"
        assert model == "text-embedding-v2"

    def test_list_empty(self):
        """With no KBs created, list should return empty."""
        kbs = list_knowledge_bases()
        assert kbs == []

    def test_create_multiple(self):
        """Multiple KBs should all be listed."""
        create_knowledge_base("kb-aaa", "openai", "text-embedding-3-small")
        create_knowledge_base("kb-bbb", "qwen", "text-embedding-v3")
        create_knowledge_base("kb-ccc", "openai", "text-embedding-ada-002")

        kbs = list_knowledge_bases()
        names = {kb["name"] for kb in kbs}
        assert names == {"kb-aaa", "kb-bbb", "kb-ccc"}


class TestDocumentManagement:
    """Add / list / query documents in a knowledge base."""

    def _make_fake_embed_fn(self, dim: int = 128):
        """Return a mock embedding function that produces deterministic vectors."""
        mock_fn = MagicMock()
        # embed_documents: return a list of vectors
        mock_fn.embed_documents.side_effect = lambda texts: [
            [float(i % (dim))] * dim for i, _ in enumerate(texts)
        ]
        # embed_query: return one vector
        mock_fn.embed_query.side_effect = lambda text: [0.5] * dim
        return mock_fn

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_add_and_list_documents(self, mock_get_embed):
        mock_get_embed.return_value = self._make_fake_embed_fn()

        create_knowledge_base("doc-kb", "openai", "text-embedding-3-small")
        docs = [
            Document(
                page_content=f"chunk {i}",
                metadata={"source": "manual.pdf", "page_range": "1-5"},
            )
            for i in range(3)
        ]
        count = add_documents("doc-kb", docs)
        assert count == 3

        doc_list = list_documents("doc-kb")
        assert len(doc_list) == 1
        assert doc_list[0]["filename"] == "manual.pdf"
        assert doc_list[0]["chunk_count"] == 3

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_add_empty_docs(self, mock_get_embed):
        mock_get_embed.return_value = self._make_fake_embed_fn()
        create_knowledge_base("empty-doc-kb", "openai", "text-embedding-3-small")
        count = add_documents("empty-doc-kb", [])
        assert count == 0

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_query_returns_documents(self, mock_get_embed):
        mock_embed = self._make_fake_embed_fn()
        mock_get_embed.return_value = mock_embed

        create_knowledge_base("query-kb", "openai", "text-embedding-3-small")

        docs = [
            Document(
                page_content=f"knowledge base documentation chunk {i}",
                metadata={"source": "user_guide.pdf"},
            )
            for i in range(5)
        ]
        add_documents("query-kb", docs)

        results = query_knowledge_base("query-kb", "How does the system work?", n_results=3)
        assert len(results) <= 3
        assert all(isinstance(d, Document) for d in results)

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_add_multiple_sources(self, mock_get_embed):
        mock_get_embed.return_value = self._make_fake_embed_fn()
        create_knowledge_base("multi-src-kb", "openai", "text-embedding-3-small")

        docs_a = [
            Document(page_content="content A", metadata={"source": "fileA.pdf"})
        ]
        docs_b = [
            Document(page_content="content B1", metadata={"source": "fileB.html"}),
            Document(page_content="content B2", metadata={"source": "fileB.html"}),
        ]
        add_documents("multi-src-kb", docs_a)
        add_documents("multi-src-kb", docs_b)

        doc_list = list_documents("multi-src-kb")
        filenames = {d["filename"] for d in doc_list}
        assert filenames == {"fileA.pdf", "fileB.html"}
        for d in doc_list:
            if d["filename"] == "fileB.html":
                assert d["chunk_count"] == 2
