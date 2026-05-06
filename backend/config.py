import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- Qwen (DashScope OpenAI-compatible) ---
QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL: str = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- Gemini (Google OpenAI-compatible endpoint) ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL: str = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Groq (OpenAI-compatible endpoint) ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# --- ChromaDB ---
CHROMADB_PATH: str = os.getenv(
    "CHROMADB_PATH", str(_project_root / "data" / "chromadb")
)

# --- Provider configs (lookup tables) ---
PROVIDER_CONFIG = {
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "embedding_chunk_size": 1000,  # OpenAI supports up to 2048 inputs per request
    },
    "qwen": {
        "api_key": QWEN_API_KEY,
        "base_url": QWEN_BASE_URL,
        "embedding_chunk_size": 10,  # DashScope embedding API limit: max 10 texts per request
    },
    "gemini": {
        "api_key": GEMINI_API_KEY,
        "base_url": GEMINI_BASE_URL,
        "embedding_chunk_size": 100,
    },
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": GROQ_BASE_URL,
        "embedding_chunk_size": 100,
    },
}

# --- Available models ---
AVAILABLE_MODELS = {
    "openai": {
        "chat": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "embedding": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
    },
    "qwen": {
        "chat": [
            "qwen-plus",
            "qwen-turbo",
            "qwen-max",
            "qwen-vl-plus",
            "qwen-vl-max",
        ],
        "embedding": [
            "text-embedding-v3",
            "text-embedding-v2",
            "text-embedding-v1",
        ],
    },
    "gemini": {
        "chat": [
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        # Gemini does not expose an OpenAI-compatible embedding endpoint;
        # use OpenAI or Qwen embeddings for knowledge bases.
        "embedding": [],
    },
    "groq": {
        "chat": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        # Groq is used for chat. Use OpenAI or Qwen for knowledge-base embeddings.
        "embedding": [],
    },
}
