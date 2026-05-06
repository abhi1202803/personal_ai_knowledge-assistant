"""FastAPI application entry point."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.chat import router as chat_router
from backend.api.knowledge import router as knowledge_router
from backend.config import (
    AVAILABLE_MODELS,
    CHROMADB_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
)

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


# ------------------------------------------------------------------
# Startup / shutdown events
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    logger.info("=" * 60)
    logger.info("Personal RAG Chatbot starting up")
    logger.info("  ChromaDB path : %s", CHROMADB_PATH)
    logger.info(
        "  OpenAI        : base_url=%s  key_set=%s",
        OPENAI_BASE_URL,
        bool(OPENAI_API_KEY),
    )
    logger.info(
        "  Qwen          : base_url=%s  key_set=%s",
        QWEN_BASE_URL,
        bool(QWEN_API_KEY),
    )
    logger.info(
        "  Groq          : base_url=%s  key_set=%s",
        GROQ_BASE_URL,
        bool(GROQ_API_KEY),
    )
    all_chat = {p: models["chat"] for p, models in AVAILABLE_MODELS.items()}
    logger.info("  Available chat models: %s", all_chat)
    logger.info("  API docs      : http://127.0.0.1:8000/docs")
    logger.info("=" * 60)
    yield
    # ---- shutdown ----
    logger.info("Personal RAG Chatbot shutting down")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="Personal RAG Chatbot",
    description="本地 AI 知识助手 - 基于 LangGraph + ChromaDB 的多知识库 RAG 系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(chat_router)
app.include_router(knowledge_router)


@app.get("/api/health")
async def health():
    return {"message": "Personal AI Assistant API is running."}


frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        requested = frontend_dist / full_path
        if full_path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Personal AI Assistant API is running."}
