"""Tests for the chat API endpoint.

The LLM streaming is mocked to avoid real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestChatEndpoint:
    """POST /api/chat"""

    @patch("backend.services.graph.get_chat_model")
    @patch("backend.services.graph.query_knowledge_base")
    def test_chat_without_kb(self, mock_query, mock_get_model):
        """Chat without a knowledge base should work (no retrieval)."""
        # Setup mock LLM
        mock_llm = MagicMock()

        async def fake_astream(messages):
            for token in ["Hello", " from", " AI"]:
                chunk = MagicMock()
                chunk.content = token
                yield chunk

        mock_llm.astream = fake_astream
        mock_get_model.return_value = mock_llm

        resp = client.post("/api/chat", json={
            "message": "Hi there",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        tokens = []
        done = False
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("done"):
                    done = True
                elif "token" in data:
                    tokens.append(data["token"])

        assert done is True
        assert "".join(tokens) == "Hello from AI"
        # query_knowledge_base should NOT be called (no KB selected)
        mock_query.assert_not_called()

    @patch("backend.services.graph.get_chat_model")
    @patch("backend.services.graph.query_knowledge_base")
    def test_chat_with_kb(self, mock_query, mock_get_model):
        """Chat with a knowledge base should trigger retrieval."""
        from langchain_core.documents import Document

        mock_query.return_value = [
            Document(page_content="The system supports multiple knowledge bases.", metadata={})
        ]

        mock_llm = MagicMock()

        async def fake_astream(messages):
            chunk = MagicMock()
            chunk.content = "The system is great"
            yield chunk

        mock_llm.astream = fake_astream
        mock_get_model.return_value = mock_llm

        resp = client.post("/api/chat", json={
            "message": "How does it work?",
            "kb_name": "some-kb",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
        })
        assert resp.status_code == 200

        mock_query.assert_called_once_with("some-kb", "How does it work?", n_results=5)

    @patch("backend.services.graph.get_chat_model")
    def test_chat_with_image(self, mock_get_model):
        """Chat with an image should include multimodal content."""
        mock_llm = MagicMock()

        async def fake_astream(messages):
            # Verify multimodal message was built
            user_msg = messages[-1]
            assert isinstance(user_msg.content, list)
            chunk = MagicMock()
            chunk.content = "I see an image"
            yield chunk

        mock_llm.astream = fake_astream
        mock_get_model.return_value = mock_llm

        resp = client.post("/api/chat", json={
            "message": "Describe this",
            "image": "iVBORw0KGgo=",
            "model_provider": "openai",
            "model_name": "gpt-4o",
        })
        assert resp.status_code == 200

    @patch("backend.services.graph.get_chat_model")
    def test_chat_error_in_stream(self, mock_get_model):
        """Errors during streaming should be sent as SSE error events."""
        mock_llm = MagicMock()

        async def fake_astream(messages):
            raise RuntimeError("LLM API error")
            yield  # make it a generator  # noqa: unreachable

        mock_llm.astream = fake_astream
        mock_get_model.return_value = mock_llm

        resp = client.post("/api/chat", json={
            "message": "test",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
        })
        assert resp.status_code == 200

        found_error = False
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "error" in data:
                    found_error = True
                    assert "LLM API error" in data["error"]
        assert found_error
