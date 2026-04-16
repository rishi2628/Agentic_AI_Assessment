"""
Application settings loaded from environment / .env file.
All values have sensible defaults so the app runs with zero configuration
as long as Ollama is running locally.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Disable third-party telemetry immediately (before any heavy import)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # ChromaDB
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")   # LangSmith

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")       # "ollama" | "groq"
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH: Path = BASE_DIR / os.getenv("VECTORSTORE_PATH", "vectorstore")
POLICIES_PATH: Path = BASE_DIR / os.getenv("POLICIES_PATH", "data/policies")

# ── Agent limits ──────────────────────────────────────────────────────────────
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "10"))
MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "2"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
