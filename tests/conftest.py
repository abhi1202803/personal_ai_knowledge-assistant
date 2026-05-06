"""Shared fixtures for all tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# ---- Force test-specific env vars BEFORE importing backend modules -------
# This ensures every test run uses an isolated ChromaDB directory and
# dummy API keys (no real API calls unless explicitly mocked).


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate environment for every test.

    - Points CHROMADB_PATH at a fresh temp directory per test
    - Sets dummy API keys so config doesn't fail
    - Resets the ChromaDB singleton before and after each test

    Two patches are required because ``knowledge_base.py`` does:
        from backend.config import CHROMADB_PATH
    which binds the string value at import time. Patching only
    ``backend.config.CHROMADB_PATH`` has no effect on that local binding,
    so we must also patch ``backend.services.knowledge_base.CHROMADB_PATH``.
    """
    import backend.config as cfg
    import backend.services.knowledge_base as kb_module
    from backend.services.knowledge_base import reset_client

    chroma_dir = tmp_path / "chromadb"
    chroma_dir.mkdir()

    monkeypatch.setenv("CHROMADB_PATH", str(chroma_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen-key")

    # Patch both the config module AND the local binding in knowledge_base
    monkeypatch.setattr(cfg, "CHROMADB_PATH", str(chroma_dir))
    monkeypatch.setattr(kb_module, "CHROMADB_PATH", str(chroma_dir))

    # Ensure a clean client before the test starts
    reset_client()

    yield

    # Clean up after the test
    reset_client()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a minimal valid PDF in memory using PyMuPDF."""
    import fitz

    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        text_point = fitz.Point(72, 72)
        page.insert_text(text_point, f"This is page {i + 1} content.\n" * 10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_html_bytes() -> bytes:
    """Create a sample HTML document."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Knowledge Base Manual</title></head>
    <body>
        <header>Site Header</header>
        <nav>Navigation</nav>
        <main>
            <h1>Knowledge Base User Guide</h1>
            <p>This is the introduction to the knowledge base system. It provides
            comprehensive functionality for managing and querying documents.</p>
            <h2>Chapter 1: Getting Started</h2>
            <p>To get started with the system, you need to install it
            first. Follow these steps to complete the installation process
            and configure your environment properly.</p>
            <h2>Chapter 2: Document Management</h2>
            <p>The document manager allows you to upload PDF and HTML files.
            You can create multiple knowledge bases, switch between them, and
            query documents using natural language.</p>
        </main>
        <footer>Footer Content</footer>
    </body>
    </html>
    """
    return html.encode("utf-8")
