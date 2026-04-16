# conftest.py — root pytest configuration
# Disable all external telemetry / network calls so tests run fully offline.
import os

# LangSmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_API_KEY", "none")
os.environ.setdefault("LANGSMITH_API_KEY", "none")

# ChromaDB anonymised telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
