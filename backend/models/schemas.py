from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    """Request body for creating a knowledge base."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Knowledge base name",
    )
    embedding_provider: str = Field(..., description="Embedding provider: openai / qwen")
    embedding_model: str = Field(..., description="Embedding model name")


class KnowledgeBaseInfo(BaseModel):
    """Response model for a knowledge base."""

    name: str
    embedding_provider: str
    embedding_model: str
    document_count: int = 0


class DocumentInfo(BaseModel):
    """Response model for a document in a knowledge base."""

    filename: str
    page_range: Optional[str] = None
    chunk_count: int = 0


class ChatRequest(BaseModel):
    """Request body for chat."""

    message: str = Field(..., description="User message")
    image: Optional[str] = Field(None, description="Base64-encoded image data")
    image_text: Optional[str] = Field(None, description="OCR text extracted from the image")
    kb_name: Optional[str] = Field(None, description="Optional knowledge base name")
    model_provider: str = Field("groq", description="Model provider")
    model_name: str = Field("llama-3.3-70b-versatile", description="Model name")


class ProviderModels(BaseModel):
    """Available models for one provider."""

    chat: list[str]
    embedding: list[str]


class AvailableModelsResponse(BaseModel):
    """Response listing all available models, keyed by provider name."""

    model_config = {"extra": "allow"}

    openai: ProviderModels
    qwen: ProviderModels
    gemini: ProviderModels
    groq: ProviderModels
