"""Knowledge-base management API endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import AVAILABLE_MODELS
from backend.models.schemas import (
    AvailableModelsResponse,
    DocumentInfo,
    KnowledgeBaseCreate,
    KnowledgeBaseInfo,
    ProviderModels,
)
from backend.services import knowledge_base as kb_service
from backend.services.document_loader import load_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["knowledge"])


# ------------------------------------------------------------------
# Knowledge base CRUD
# ------------------------------------------------------------------

@router.get("/knowledge-bases", response_model=list[KnowledgeBaseInfo])
async def list_knowledge_bases():
    """List all knowledge bases."""
    logger.info("GET /api/knowledge-bases")
    items = kb_service.list_knowledge_bases()
    logger.info("Returning %d knowledge bases", len(items))
    return [KnowledgeBaseInfo(**item) for item in items]


@router.post("/knowledge-bases", response_model=KnowledgeBaseInfo)
async def create_knowledge_base(body: KnowledgeBaseCreate):
    """Create a new knowledge base."""
    logger.info(
        "POST /api/knowledge-bases: name=%r, provider=%s, model=%s",
        body.name,
        body.embedding_provider,
        body.embedding_model,
    )
    if body.embedding_provider not in AVAILABLE_MODELS:
        logger.warning("Invalid embedding provider: %s", body.embedding_provider)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedding provider '{body.embedding_provider}'. "
                   f"Valid options: {list(AVAILABLE_MODELS)}",
        )
    valid_models = AVAILABLE_MODELS.get(body.embedding_provider, {}).get("embedding", [])
    if body.embedding_model not in valid_models:
        logger.warning(
            "Invalid embedding model: %s (valid: %s)",
            body.embedding_model,
            valid_models,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid embedding model '{body.embedding_model}' "
                f"for provider '{body.embedding_provider}'. "
                f"Valid options: {valid_models}"
            ),
        )
    try:
        info = kb_service.create_knowledge_base(
            body.name,
            body.embedding_provider,
            body.embedding_model,
        )
        logger.info("Knowledge base %r created successfully", body.name)
        return KnowledgeBaseInfo(**info)
    except ValueError as e:
        logger.error("Failed to create KB: %s", e)
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/knowledge-bases/{name}")
async def delete_knowledge_base(name: str):
    """Delete a knowledge base."""
    logger.info("DELETE /api/knowledge-bases/%s", name)
    try:
        kb_service.delete_knowledge_base(name)
        logger.info("Knowledge base '%s' deleted", name)
        return {"message": f"Knowledge base '{name}' deleted."}
    except Exception as e:
        logger.error("Failed to delete KB '%s': %s", name, e, exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))


# ------------------------------------------------------------------
# Document upload
# ------------------------------------------------------------------

@router.post("/knowledge-bases/{name}/upload")
async def upload_document(
    name: str,
    file: UploadFile = File(...),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
):
    """Upload a PDF or HTML file into a knowledge base.

    For PDFs, ``start_page`` and ``end_page`` (1-based, inclusive) can be
    provided to select a page range.
    """
    logger.info(
        "POST /api/knowledge-bases/%s/upload: file=%s, start_page=%s, end_page=%s",
        name,
        file.filename,
        start_page,
        end_page,
    )
    # Verify KB exists
    try:
        kb_service.get_collection_embedding_info(name)
    except Exception:
        logger.error("Knowledge base '%s' not found for upload", name)
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge base '{name}' not found.",
        )

    file_bytes = await file.read()
    filename = file.filename or "unknown"
    logger.info("Read %d bytes from uploaded file '%s'", len(file_bytes), filename)

    try:
        docs = load_document(
            file_bytes,
            filename,
            start_page=start_page,
            end_page=end_page,
        )
    except ValueError as e:
        logger.error("Document loading error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    if not docs:
        logger.warning("No text extracted from file '%s'", filename)
        raise HTTPException(
            status_code=400,
            detail="No text content could be extracted from the file.",
        )

    logger.info("Parsed %d chunks from '%s', adding to KB '%s'", len(docs), filename, name)
    chunk_count = kb_service.add_documents(name, docs)
    logger.info("Upload complete: %d chunks added to KB '%s'", chunk_count, name)
    return {
        "message": f"Successfully added {chunk_count} chunks from '{filename}'.",
        "filename": filename,
        "chunk_count": chunk_count,
    }


@router.get(
    "/knowledge-bases/{name}/documents",
    response_model=list[DocumentInfo],
)
async def list_documents(name: str):
    """List documents stored in a knowledge base."""
    logger.info("GET /api/knowledge-bases/%s/documents", name)
    try:
        items = kb_service.list_documents(name)
        logger.info("KB '%s' has %d distinct documents", name, len(items))
        return [DocumentInfo(**item) for item in items]
    except Exception as e:
        logger.error("Failed to list docs for KB '%s': %s", name, e, exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))


# ------------------------------------------------------------------
# Available models
# ------------------------------------------------------------------

@router.get("/models", response_model=AvailableModelsResponse)
async def list_models():
    """Return available chat and embedding models for each provider."""
    logger.info("GET /api/models")
    return AvailableModelsResponse(
        **{provider: ProviderModels(**models) for provider, models in AVAILABLE_MODELS.items()}
    )
