"""
LLM factory — returns a LangChain chat model based on the configured provider.

Supported providers
-------------------
* ollama  (default) — free, 100 % local via Ollama (https://ollama.ai)
* groq    — free-tier cloud API (https://console.groq.com)

Selecting a provider
--------------------
Set ``LLM_PROVIDER`` in your ``.env`` file:

    LLM_PROVIDER=ollama   LLM_MODEL=llama3.2:3b
    LLM_PROVIDER=groq     GROQ_API_KEY=gsk_...   GROQ_MODEL=llama3-8b-8192
"""

from langchain_core.language_models.chat_models import BaseChatModel
import config.settings as settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Return a configured chat-model instance.

    Parameters
    ----------
    temperature:
        Sampling temperature (0 = deterministic, good for RAG answers).

    Raises
    ------
    ValueError
        If an unsupported ``LLM_PROVIDER`` is configured.
    ImportError
        If the required provider package is not installed.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama"
            ) from exc

        return ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file.\n"
                "Get a free key at https://console.groq.com"
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            ) from exc

        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Choose 'ollama' (local) or 'groq' (free cloud)."
    )


def get_embedding_function():
    """
    Return a ChromaDB-compatible embedding function using sentence-transformers.

    The model (``all-MiniLM-L6-v2`` by default) is downloaded automatically
    from HuggingFace on first use (~90 MB, CPU-friendly).
    """
    try:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError as exc:
        raise ImportError(
            "chromadb and/or sentence-transformers are not installed.\n"
            "Run: pip install chromadb sentence-transformers"
        ) from exc

    return SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )
