"""FastAPI application entry point."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.knowledge import router as knowledge_router
from backend.config import (
    AVAILABLE_MODELS,
    CHROMADB_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
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
    all_chat = {
        p: AVAILABLE_MODELS[p]["chat"] for p in ("openai", "qwen")
    }
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


@app.get("/")
async def root():
    return {"message": "Personal RAG Chatbot API is running."}
