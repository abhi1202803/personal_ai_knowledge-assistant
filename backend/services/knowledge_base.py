"""Knowledge-base management backed by ChromaDB.

Each knowledge base maps to a ChromaDB *collection*.

Design: the user-facing *display name* (supports any language / spaces) is
stored in collection metadata under the key ``display_name``.  ChromaDB only
allows ``[a-zA-Z0-9._-]`` for collection names, so we derive a stable,
deterministic *internal name* from the display name using a short MD5 hash.
All public API functions accept the **display name**.
"""

from __future__ import annotations

import hashlib
import logging
import re

import chromadb
from langchain_core.documents import Document

from backend.config import CHROMADB_PATH
from backend.services.embeddings import get_cached_embedding_function

logger = logging.getLogger(__name__)

# Singleton ChromaDB client (persistent)
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB PersistentClient at path: %s", CHROMADB_PATH)
        _client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return _client


def reset_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _client
    _client = None
    logger.debug("ChromaDB client reset")


def _make_collection_name(display_name: str) -> str:
    """Derive a ChromaDB-compatible internal collection name from any display name.

    Strategy:
    - Keep ASCII alphanumeric chars and hyphens from the display name as a
      human-readable prefix (up to 40 chars).
    - Append a deterministic 8-char MD5 hex suffix so the full name is unique
      and stable for the same display name.
    - Result always starts and ends with an alphanumeric character.
    """
    short_hash = hashlib.md5(display_name.encode("utf-8")).hexdigest()[:8]
    ascii_part = re.sub(r"[^a-zA-Z0-9]", "-", display_name)
    ascii_part = ascii_part.strip("-")
    ascii_part = re.sub(r"-+", "-", ascii_part)[:40]
    prefix = ascii_part if ascii_part and ascii_part[0].isalnum() else "kb"
    return f"{prefix}-{short_hash}"


def _get_display_name(col: chromadb.Collection) -> str:
    """Return the display name stored in metadata, falling back to collection name."""
    meta = col.metadata or {}
    return meta.get("display_name", col.name)


# ------------------------------------------------------------------
# CRUD helpers
# ------------------------------------------------------------------

def list_knowledge_bases() -> list[dict]:
    """Return metadata dicts for every collection."""
    client = _get_client()
    collections = client.list_collections()
    logger.info("Listing knowledge bases, found %d collections", len(collections))

    result = []
    for item in collections:
        col = client.get_collection(name=item) if isinstance(item, str) else item
        meta = col.metadata or {}
        info = {
            "name": meta.get("display_name", col.name),
            "embedding_provider": meta.get("embedding_provider", "unknown"),
            "embedding_model": meta.get("embedding_model", "unknown"),
            "document_count": col.count(),
        }
        logger.debug(
            "  KB: %r (internal=%s, provider=%s, docs=%d)",
            info["name"],
            col.name,
            info["embedding_provider"],
            info["document_count"],
        )
        result.append(info)
    return result


def create_knowledge_base(
    display_name: str,
    embedding_provider: str,
    embedding_model: str,
) -> dict:
    """Create a new ChromaDB collection for the given display name.

    Raises ``ValueError`` if a collection with the same display name exists.
    """
    collection_name = _make_collection_name(display_name)
    logger.info(
        "Creating knowledge base: display_name=%r, internal=%s, provider=%s, model=%s",
        display_name,
        collection_name,
        embedding_provider,
        embedding_model,
    )
    client = _get_client()

    # Check existence by internal name
    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if collection_name in existing:
        logger.warning("Knowledge base %r already exists (internal=%s)", display_name, collection_name)
        raise ValueError(f"Knowledge base '{display_name}' already exists.")

    metadata = {
        "display_name": display_name,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }
    client.create_collection(name=collection_name, metadata=metadata)
    logger.info("Knowledge base %r created (internal=%s)", display_name, collection_name)
    return {
        "name": display_name,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "document_count": 0,
    }


def delete_knowledge_base(display_name: str) -> None:
    """Delete the collection for the given display name."""
    collection_name = _make_collection_name(display_name)
    logger.info("Deleting knowledge base %r (internal=%s)", display_name, collection_name)
    _get_client().delete_collection(name=collection_name)
    logger.info("Knowledge base %r deleted", display_name)


def get_collection_embedding_info(display_name: str) -> tuple[str, str]:
    """Return ``(provider, model)`` for an existing collection."""
    collection_name = _make_collection_name(display_name)
    col = _get_client().get_collection(name=collection_name)
    meta = col.metadata or {}
    provider = meta.get("embedding_provider", "openai")
    model = meta.get("embedding_model", "text-embedding-3-small")
    logger.debug(
        "Collection %r embedding info: provider=%s, model=%s",
        display_name,
        provider,
        model,
    )
    return provider, model


# ------------------------------------------------------------------
# Document management
# ------------------------------------------------------------------

def add_documents(display_name: str, docs: list[Document]) -> int:
    """Embed and store documents into the named collection.

    The embedding function is determined by the collection metadata so
    that all vectors in a collection come from the same API.

    Returns the number of chunks added.
    """
    if not docs:
        logger.warning("add_documents called with empty doc list for KB %r", display_name)
        return 0

    logger.info("Adding %d document chunks to KB %r", len(docs), display_name)

    provider, model = get_collection_embedding_info(display_name)
    embed_fn = get_cached_embedding_function(provider, model)

    collection_name = _make_collection_name(display_name)
    client = _get_client()
    col = client.get_collection(name=collection_name)

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    logger.info(
        "Generating embeddings for %d texts via %s/%s",
        len(texts),
        provider,
        model,
    )
    embeddings = embed_fn.embed_documents(texts)
    logger.info("Embeddings generated, dimension=%d", len(embeddings[0]) if embeddings else 0)

    base_id = col.count()
    ids = [f"{collection_name}_{base_id + i}" for i in range(len(texts))]

    col.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(
        "Successfully added %d chunks to KB %r (ids: %s .. %s)",
        len(texts),
        display_name,
        ids[0],
        ids[-1],
    )
    return len(texts)


def list_documents(display_name: str) -> list[dict]:
    """List distinct source documents stored in a collection."""
    logger.info("Listing documents in KB %r", display_name)
    collection_name = _make_collection_name(display_name)
    col = _get_client().get_collection(name=collection_name)

    results = col.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    sources: dict[str, dict] = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        if src not in sources:
            sources[src] = {
                "filename": src,
                "page_range": meta.get("page_range"),
                "chunk_count": 0,
            }
        sources[src]["chunk_count"] += 1

    logger.info("KB %r has %d distinct source documents", display_name, len(sources))
    return list(sources.values())


def query_knowledge_base(
    display_name: str,
    query: str,
    n_results: int = 5,
) -> list[Document]:
    """Retrieve the most relevant documents for *query*.

    Uses the same embedding provider/model that was used to populate the
    collection, guaranteeing vector-space consistency.
    """
    logger.info(
        "Querying KB %r: query='%s' (n_results=%d)",
        display_name,
        query[:80] + ("..." if len(query) > 80 else ""),
        n_results,
    )

    provider, model = get_collection_embedding_info(display_name)
    embed_fn = get_cached_embedding_function(provider, model)

    logger.debug("Embedding query via %s/%s", provider, model)
    query_embedding = embed_fn.embed_query(query)

    collection_name = _make_collection_name(display_name)
    col = _get_client().get_collection(name=collection_name)

    count = col.count()
    effective_n = min(n_results, count) if count > 0 else 1
    logger.debug("Collection count=%d, effective n_results=%d", count, effective_n)

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=effective_n,
        include=["documents", "metadatas", "distances"],
    )

    docs: list[Document] = []
    for text, meta in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        docs.append(Document(page_content=text, metadata=meta))

    distances = results.get("distances", [[]])[0]
    logger.info(
        "Query returned %d results from KB %r, distances=%s",
        len(docs),
        display_name,
        [f"{d:.4f}" for d in distances[:5]],
    )
    return docs
