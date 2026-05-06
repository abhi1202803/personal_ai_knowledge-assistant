"""Tests for backend.services.graph.

These tests mock external dependencies (KB query, LLM) to verify the
graph logic in isolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.services.graph import (
    GraphState,
    _build_messages,
    graph,
    retrieve,
)


class TestRetrieveNode:
    """Test the retrieve node logic."""

    def test_no_kb_returns_empty_context(self):
        """When no KB is selected, context should be empty."""
        state: GraphState = {
            "question": "Hello",
            "image": None,
            "kb_name": None,
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "retrieved_context": "",
            "answer": "",
        }
        result = retrieve(state)
        assert result["retrieved_context"] == ""

    @patch("backend.services.graph.query_knowledge_base")
    def test_with_kb_returns_context(self, mock_query):
        """When KB is selected, context should come from query results."""
        mock_query.return_value = [
            Document(page_content="doc1 content", metadata={}),
            Document(page_content="doc2 content", metadata={}),
        ]
        state: GraphState = {
            "question": "How does this work?",
            "image": None,
            "kb_name": "my-kb",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "retrieved_context": "",
            "answer": "",
        }
        result = retrieve(state)
        assert "doc1 content" in result["retrieved_context"]
        assert "doc2 content" in result["retrieved_context"]
        mock_query.assert_called_once_with("my-kb", "How does this work?", n_results=5)

    @patch("backend.services.graph.query_knowledge_base")
    def test_kb_error_returns_empty_context(self, mock_query):
        """If KB query fails, context should be empty (graceful degradation)."""
        mock_query.side_effect = Exception("DB error")
        state: GraphState = {
            "question": "test",
            "image": None,
            "kb_name": "broken-kb",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "retrieved_context": "",
            "answer": "",
        }
        result = retrieve(state)
        assert result["retrieved_context"] == ""


class TestBuildMessages:
    """Test message construction for the LLM."""

    def test_basic_text_message(self):
        state: GraphState = {
            "question": "Hello",
            "image": None,
            "kb_name": None,
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "retrieved_context": "",
            "answer": "",
        }
        messages = _build_messages(state)
        # System prompt + user message = 2
        assert len(messages) == 2
        assert messages[0].type == "system"
        assert messages[1].type == "human"
        assert messages[1].content == "Hello"

    def test_with_context(self):
        state: GraphState = {
            "question": "How does this work?",
            "image": None,
            "kb_name": "kb1",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "retrieved_context": "This system manages knowledge bases and answers questions.",
            "answer": "",
        }
        messages = _build_messages(state)
        # System prompt + context system message + user message = 3
        assert len(messages) == 3
        assert "manages knowledge bases" in messages[1].content

    def test_with_image(self):
        state: GraphState = {
            "question": "Describe this image",
            "image": "iVBORw0KGgo=",  # fake base64
            "kb_name": None,
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "retrieved_context": "",
            "answer": "",
        }
        messages = _build_messages(state)
        assert len(messages) == 2
        # User message should be multimodal
        user_msg = messages[1]
        assert isinstance(user_msg.content, list)
        assert user_msg.content[0]["type"] == "text"
        assert user_msg.content[1]["type"] == "image_url"
        assert "base64" in user_msg.content[1]["image_url"]["url"]


class TestCompiledGraph:
    """Test that the compiled graph has the correct structure."""

    def test_graph_nodes(self):
        """Graph should have retrieve and generate nodes."""
        node_names = set(graph.get_graph().nodes.keys())
        assert "retrieve" in node_names
        assert "generate" in node_names
