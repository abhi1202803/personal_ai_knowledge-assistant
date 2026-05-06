# Personal AI Assistant

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/abhi1202803/personal_ai_knowledge-assistant)

A local AI knowledge assistant built with **LangGraph + FastAPI + React**. Create multiple independent knowledge bases, upload PDF/HTML documents, and chat with your data using various LLM providers.

一个基于 LangGraph + FastAPI + React 的本地 AI 知识助手。支持创建多个独立知识库，上传 PDF / HTML 文档，结合多家大模型 API 进行问答。

## Features

- **Multi-model support** — switch between OpenAI, Qwen, and Gemini on the fly
- **Knowledge base management** — local ChromaDB vector store with multi-KB CRUD (supports Chinese names)
- **Document parsing** — PDF (with page range selection) and HTML upload
- **Embedding consistency** — each KB is bound to an embedding provider at creation time
- **Multimodal input** — text Q&A with optional image input (for vision-capable models)
- **Streaming responses** — real-time SSE streaming output

## Supported Models

| Provider | Chat Models | Embedding Models |
|----------|-------------|------------------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | text-embedding-3-small/large, ada-002 |
| **Qwen** | qwen-max, qwen-plus, qwen-turbo, qwen-vl-plus/max | text-embedding-v1/v2/v3 |
| **Gemini** | gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro/flash | — |

> Gemini does not currently provide an OpenAI-compatible embedding endpoint. Use OpenAI or Qwen for KB embeddings.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Node.js 18+

### Backend

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# Start the backend server
uv run uvicorn backend.main:app --reload --port 8000
```

Or with pip:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Then visit http://localhost:5173.

### Running Tests

All external calls (LLM, embedding) are mocked — no real API keys required.

```bash
uv run pytest tests/ -v
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API Key | — |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | `https://api.openai.com/v1` |
| `QWEN_API_KEY` | Qwen (DashScope) API Key | — |
| `QWEN_BASE_URL` | Qwen DashScope endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `GEMINI_API_KEY` | Google Gemini API Key | — |
| `GEMINI_BASE_URL` | Gemini OpenAI-compatible endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `CHROMADB_PATH` | ChromaDB persistent storage path | `./data/chromadb` |

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Environment variables & model config
│   ├── api/
│   │   ├── chat.py          # POST /api/chat (SSE streaming)
│   │   └── knowledge.py     # Knowledge base CRUD + file upload
│   ├── services/
│   │   ├── embeddings.py    # Embedding service
│   │   ├── llm.py           # LLM service
│   │   ├── knowledge_base.py# ChromaDB collection management
│   │   ├── document_loader.py# PDF / HTML parsing & chunking
│   │   └── graph.py         # LangGraph RAG workflow
│   └── models/
│       └── schemas.py       # Pydantic request/response models
├── frontend/
│   └── src/
│       ├── components/      # React UI components
│       ├── api/             # API client
│       └── types/           # TypeScript type definitions
├── tests/                   # pytest test suite
├── pyproject.toml           # Project config (uv / pip)
└── requirements.txt         # Python dependencies (pip fallback)
```

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain, ChromaDB
- **Frontend**: React, TypeScript, Vite
- **LLM Providers**: OpenAI, Qwen (DashScope), Gemini

## License

MIT
