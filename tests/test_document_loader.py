"""Tests for backend.services.document_loader."""

from __future__ import annotations

import pytest

from backend.services.document_loader import (
    load_document,
    load_html,
    load_pdf,
)


class TestLoadPdf:
    """Tests for PDF loading and page-range filtering."""

    def test_load_full_pdf(self, sample_pdf_bytes: bytes):
        """Loading a PDF without page range should include all pages."""
        chunks = load_pdf(sample_pdf_bytes, "test.pdf")
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata["source"] == "test.pdf"
            assert c.metadata["file_type"] == "pdf"
            assert c.metadata["total_pages"] == 5
            # Each chunk must carry the exact page it came from
            assert "page" in c.metadata
            assert 1 <= c.metadata["page"] <= 5
            assert "chunk_index" in c.metadata

    def test_load_pdf_page_range(self, sample_pdf_bytes: bytes):
        """Specifying start/end page should filter content and record the range."""
        chunks_full = load_pdf(sample_pdf_bytes, "test.pdf")
        chunks_partial = load_pdf(
            sample_pdf_bytes, "test.pdf", start_page=2, end_page=3
        )
        assert len(chunks_partial) > 0
        # All chunks should come from the selected range
        assert chunks_partial[0].metadata["page_range_selected"] == "2-3"
        for c in chunks_partial:
            assert 2 <= c.metadata["page"] <= 3
        # Partial should have less or equal content
        full_text = "".join(c.page_content for c in chunks_full)
        partial_text = "".join(c.page_content for c in chunks_partial)
        assert len(partial_text) <= len(full_text)

    def test_load_pdf_single_page(self, sample_pdf_bytes: bytes):
        """Loading a single page should work."""
        chunks = load_pdf(
            sample_pdf_bytes, "test.pdf", start_page=1, end_page=1
        )
        assert len(chunks) > 0
        assert chunks[0].metadata["page_range_selected"] == "1-1"
        for c in chunks:
            assert c.metadata["page"] == 1

    def test_load_pdf_out_of_range_clamped(self, sample_pdf_bytes: bytes):
        """Pages beyond total should be clamped."""
        chunks = load_pdf(
            sample_pdf_bytes, "test.pdf", start_page=1, end_page=999
        )
        assert len(chunks) > 0
        assert chunks[0].metadata["page_range_selected"] == "1-5"

    def test_load_pdf_empty_bytes(self):
        """Empty bytes should raise an error from PyMuPDF."""
        with pytest.raises(Exception):
            load_pdf(b"", "empty.pdf")


class TestLoadHtml:
    """Tests for HTML loading."""

    def test_load_html_basic(self, sample_html_bytes: bytes):
        """Basic HTML parsing should extract body text."""
        chunks = load_html(sample_html_bytes, "manual.html")
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata["source"] == "manual.html"
            assert c.metadata["file_type"] == "html"
            # Every chunk must carry section provenance
            assert "section" in c.metadata
            assert "chunk_index" in c.metadata

    def test_load_html_strips_nav_footer(self, sample_html_bytes: bytes):
        """Nav, header, footer content should be stripped."""
        chunks = load_html(sample_html_bytes, "manual.html")
        combined = " ".join(c.page_content for c in chunks)
        assert "Navigation" not in combined
        assert "Footer Content" not in combined
        assert "Site Header" not in combined

    def test_load_html_preserves_main_content(self, sample_html_bytes: bytes):
        """Main content should be preserved, and headings should appear as section metadata."""
        chunks = load_html(sample_html_bytes, "manual.html")
        combined = " ".join(c.page_content for c in chunks)
        sections = {c.metadata["section"] for c in chunks}
        # Main body text should be present
        assert "introduction" in combined.lower() or "getting started" in combined.lower()
        # Headings should be captured as section names
        assert any("Chapter" in s or "Knowledge Base" in s for s in sections)

    def test_load_html_empty(self):
        """Empty HTML should return empty list."""
        chunks = load_html(b"<html><body></body></html>", "empty.html")
        assert chunks == []


class TestLoadDocument:
    """Tests for the load_document dispatcher."""

    def test_dispatch_pdf(self, sample_pdf_bytes: bytes):
        chunks = load_document(sample_pdf_bytes, "test.pdf")
        assert len(chunks) > 0
        assert chunks[0].metadata["file_type"] == "pdf"

    def test_dispatch_html(self, sample_html_bytes: bytes):
        chunks = load_document(sample_html_bytes, "manual.html")
        assert len(chunks) > 0
        assert chunks[0].metadata["file_type"] == "html"

    def test_dispatch_htm(self, sample_html_bytes: bytes):
        chunks = load_document(sample_html_bytes, "manual.HTM")
        assert len(chunks) > 0

    def test_dispatch_pdf_with_page_range(self, sample_pdf_bytes: bytes):
        chunks = load_document(
            sample_pdf_bytes, "test.pdf", start_page=2, end_page=4
        )
        assert len(chunks) > 0
        assert chunks[0].metadata["page_range_selected"] == "2-4"
        for c in chunks:
            assert 2 <= c.metadata["page"] <= 4

    def test_dispatch_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(b"content", "data.csv")
