"""Tests for the knowledge-base API endpoints.

Uses FastAPI's TestClient (synchronous) with mocked services where needed.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestListModels:
    """GET /api/models"""

    def test_returns_models(self):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        for provider in ("openai", "qwen", "gemini"):
            assert provider in data
            assert "chat" in data[provider]
            assert "embedding" in data[provider]
        assert len(data["openai"]["chat"]) > 0
        assert len(data["qwen"]["embedding"]) > 0
        assert any("gemini" in m for m in data["gemini"]["chat"])
        assert data["gemini"]["embedding"] == []


class TestKnowledgeBaseCRUD:
    """POST / GET / DELETE /api/knowledge-bases"""

    def test_create_and_list(self):
        resp = client.post("/api/knowledge-bases", json={
            "name": "api-test-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "api-test-kb"
        assert data["embedding_provider"] == "openai"

        resp = client.get("/api/knowledge-bases")
        assert resp.status_code == 200
        names = [kb["name"] for kb in resp.json()]
        assert "api-test-kb" in names

    def test_create_invalid_provider(self):
        resp = client.post("/api/knowledge-bases", json={
            "name": "bad-provider-kb",
            "embedding_provider": "invalid",
            "embedding_model": "text-embedding-3-small",
        })
        assert resp.status_code == 400

    def test_create_invalid_model(self):
        resp = client.post("/api/knowledge-bases", json={
            "name": "bad-model-kb",
            "embedding_provider": "openai",
            "embedding_model": "nonexistent-model",
        })
        assert resp.status_code == 400

    def test_create_duplicate(self):
        client.post("/api/knowledge-bases", json={
            "name": "dup-api-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        resp = client.post("/api/knowledge-bases", json={
            "name": "dup-api-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        assert resp.status_code == 409

    def test_delete(self):
        client.post("/api/knowledge-bases", json={
            "name": "del-api-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        resp = client.delete("/api/knowledge-bases/del-api-kb")
        assert resp.status_code == 200

        resp = client.get("/api/knowledge-bases")
        names = [kb["name"] for kb in resp.json()]
        assert "del-api-kb" not in names

    def test_delete_nonexistent(self):
        resp = client.delete("/api/knowledge-bases/nonexistent-xyz")
        assert resp.status_code == 404

    def test_name_empty(self):
        resp = client.post("/api/knowledge-bases", json={
            "name": "",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        assert resp.status_code == 422  # Pydantic validation

    def test_create_chinese_name(self):
        """Chinese display names should be accepted and round-trip correctly."""
        resp = client.post("/api/knowledge-bases", json={
            "name": "欧洲文明15讲",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "欧洲文明15讲"

        names = [kb["name"] for kb in client.get("/api/knowledge-bases").json()]
        assert "欧洲文明15讲" in names

    def test_create_name_with_spaces(self):
        """Names with spaces should be accepted and round-trip correctly."""
        resp = client.post("/api/knowledge-bases", json={
            "name": "European Civ 15 Lectures",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "European Civ 15 Lectures"


class TestDocumentUpload:
    """POST /api/knowledge-bases/{name}/upload"""

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_upload_pdf(self, mock_get_embed, sample_pdf_bytes: bytes):
        # Create mock embedding function
        mock_fn = MagicMock()
        mock_fn.embed_documents.side_effect = lambda texts: [
            [0.1] * 128 for _ in texts
        ]
        mock_get_embed.return_value = mock_fn

        # Create KB first
        client.post("/api/knowledge-bases", json={
            "name": "upload-test-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })

        # Upload PDF
        resp = client.post(
            "/api/knowledge-bases/upload-test-kb/upload",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"start_page": "1", "end_page": "3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_count"] > 0
        assert data["filename"] == "test.pdf"

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_upload_html(self, mock_get_embed, sample_html_bytes: bytes):
        mock_fn = MagicMock()
        mock_fn.embed_documents.side_effect = lambda texts: [
            [0.1] * 128 for _ in texts
        ]
        mock_get_embed.return_value = mock_fn

        client.post("/api/knowledge-bases", json={
            "name": "upload-html-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })

        resp = client.post(
            "/api/knowledge-bases/upload-html-kb/upload",
            files={"file": ("manual.html", sample_html_bytes, "text/html")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_count"] > 0

    def test_upload_to_nonexistent_kb(self):
        resp = client.post(
            "/api/knowledge-bases/no-such-kb/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 404

    def test_upload_unsupported_type(self):
        client.post("/api/knowledge-bases", json={
            "name": "bad-file-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        resp = client.post(
            "/api/knowledge-bases/bad-file-kb/upload",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert resp.status_code == 400


class TestListDocuments:
    """GET /api/knowledge-bases/{name}/documents"""

    @patch("backend.services.knowledge_base.get_cached_embedding_function")
    def test_list_after_upload(self, mock_get_embed, sample_pdf_bytes: bytes):
        mock_fn = MagicMock()
        mock_fn.embed_documents.side_effect = lambda texts: [
            [0.1] * 128 for _ in texts
        ]
        mock_get_embed.return_value = mock_fn

        client.post("/api/knowledge-bases", json={
            "name": "list-docs-kb",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        })
        client.post(
            "/api/knowledge-bases/list-docs-kb/upload",
            files={"file": ("manual.pdf", sample_pdf_bytes, "application/pdf")},
        )

        resp = client.get("/api/knowledge-bases/list-docs-kb/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "manual.pdf"
        assert docs[0]["chunk_count"] > 0
