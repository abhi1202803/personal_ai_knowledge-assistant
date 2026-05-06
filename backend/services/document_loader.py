"""Document loading and chunking utilities.

Supports:
- PDF  (via PyMuPDF / fitz) with optional page-range filtering
- HTML (via BeautifulSoup4)

All loaders return a list of LangChain ``Document`` objects, ready to be
embedded and stored in a vector database.
"""

from __future__ import annotations

import logging
from typing import Optional

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Default chunking parameters
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


def _get_splitter(
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )


# ------------------------------------------------------------------
# PDF
# ------------------------------------------------------------------

def load_pdf(
    file_bytes: bytes,
    filename: str,
    *,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Parse a PDF and return chunked ``Document`` objects.

    Each chunk carries precise provenance metadata:
    - ``page``: the 1-based page number the chunk originates from.
    - ``page_range_selected``: user-selected range (e.g. ``"3-10"``).
    - ``chunk_index``: sequential index among all chunks from this upload.
    - ``chunk_size`` / ``chunk_overlap``: splitter parameters used.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the PDF file.
    filename:
        Original filename (used in metadata).
    start_page:
        1-based start page (inclusive).  ``None`` means page 1.
    end_page:
        1-based end page (inclusive).  ``None`` means last page.
    """
    logger.info(
        "Loading PDF: filename=%s, size=%d bytes, start_page=%s, end_page=%s",
        filename,
        len(file_bytes),
        start_page,
        end_page,
    )

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    logger.info("PDF opened: total_pages=%d", total_pages)

    # Normalise to 0-based indices
    s = max((start_page or 1) - 1, 0)
    e = min((end_page or total_pages) - 1, total_pages - 1)
    page_range_str = f"{s + 1}-{e + 1}"
    logger.debug("Normalised page range: [%d, %d] (0-based)", s, e)

    # Build one Document per page so that split chunks inherit the page number.
    page_docs: list[Document] = []
    for page_num in range(s, e + 1):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            page_docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "file_type": "pdf",
                    "page": page_num + 1,           # 1-based, exact page
                    "page_range_selected": page_range_str,
                    "total_pages": total_pages,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            ))
        else:
            logger.debug("Page %d is empty, skipping", page_num + 1)

    doc.close()

    if not page_docs:
        logger.warning("No text extracted from PDF '%s'", filename)
        return []

    logger.info("Extracted text from %d pages", len(page_docs))

    splitter = _get_splitter(chunk_size, chunk_overlap)
    # split_documents preserves each page's metadata in the resulting chunks.
    chunks = splitter.split_documents(page_docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "PDF chunked: filename=%s, page_range=%s, chunks=%d",
        filename,
        page_range_str,
        len(chunks),
    )
    return chunks


# ------------------------------------------------------------------
# HTML
# ------------------------------------------------------------------

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {"p", "li", "td", "th", "pre", "blockquote", "dd", "dt"}


def load_html(
    file_bytes: bytes,
    filename: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Parse an HTML file and return chunked ``Document`` objects.

    Each chunk carries precise provenance metadata:
    - ``section``: the heading text of the section the chunk belongs to.
    - ``chunk_index``: sequential index among all chunks from this upload.
    - ``chunk_size`` / ``chunk_overlap``: splitter parameters used.
    """
    logger.info(
        "Loading HTML: filename=%s, size=%d bytes",
        filename,
        len(file_bytes),
    )

    html_str = file_bytes.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_str, "lxml")

    # Remove non-content tags
    removed_count = 0
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
        removed_count += 1
    logger.debug("Removed %d non-content tags from HTML", removed_count)

    # Walk all block-level and heading elements in document order.
    # Group content by the most-recently-seen heading so each section
    # becomes its own Document, letting chunks inherit the heading metadata.
    current_section = filename  # fallback when document has no headings
    buffer: list[str] = []
    section_docs: list[Document] = []

    def _flush(section: str) -> None:
        text = "\n".join(buffer).strip()
        if text:
            section_docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "file_type": "html",
                    "section": section,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            ))
        buffer.clear()

    for tag in soup.find_all(_HEADING_TAGS | _BLOCK_TAGS):
        if tag.name in _HEADING_TAGS:
            _flush(current_section)
            current_section = tag.get_text(separator=" ", strip=True)
        else:
            text = tag.get_text(separator=" ", strip=True)
            if text:
                buffer.append(text)

    _flush(current_section)

    if not section_docs:
        logger.warning("No text extracted from HTML '%s'", filename)
        return []

    logger.info(
        "HTML parsed into %d sections from '%s'", len(section_docs), filename
    )

    splitter = _get_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(section_docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info("HTML chunked: filename=%s, chunks=%d", filename, len(chunks))
    return chunks


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

def load_document(
    file_bytes: bytes,
    filename: str,
    *,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> list[Document]:
    """Auto-detect file type and load accordingly."""
    lower = filename.lower()
    logger.info("load_document called: filename=%s", filename)

    if lower.endswith(".pdf"):
        return load_pdf(
            file_bytes, filename,
            start_page=start_page, end_page=end_page,
        )
    elif lower.endswith((".html", ".htm")):
        return load_html(file_bytes, filename)
    else:
        logger.error("Unsupported file type: %s", filename)
        raise ValueError(
            f"Unsupported file type: {filename}. "
            "Only .pdf, .html, and .htm files are supported."
        )
