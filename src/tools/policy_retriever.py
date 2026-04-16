"""
Policy Retriever — RAG tool for the Policy & Incident Copilot.

Responsibilities
----------------
1. Load all ``*.txt`` policy documents from ``data/policies/``.
2. Chunk them with overlap (preserves context across chunk boundaries).
3. Embed and persist in a local ChromaDB vector store.
4. Expose a ``retrieve(query, k)`` method used by both single-agent
   and multi-agent researcher nodes.

The vector store is created automatically on first use; subsequent runs
load the existing store without re-embedding (fast start-up).

ChromaDB stores data at the path configured in ``config.settings.VECTORSTORE_PATH``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import chromadb

import config.settings as settings
from src.utils.llm_factory import get_embedding_function
from src.utils.logger import CopilotLogger

_COLLECTION_NAME = "policy_docs"
_CHUNK_SIZE = 800          # characters per chunk
_CHUNK_OVERLAP = 150       # overlap to avoid boundary information loss
_logger = CopilotLogger("policy_retriever")


# ── Data model ──────────────────────────────────────────────────────────────

class PolicyChunk:
    """A single retrievable chunk from a policy document."""

    def __init__(self, text: str, source: str, doc_version: str, chunk_id: str) -> None:
        self.text = text
        self.source = source           # filename, e.g. "vpn_access_policy.txt"
        self.doc_version = doc_version # parsed from the file header, e.g. "v1.8"
        self.chunk_id = chunk_id

    def as_citation(self) -> str:
        return f"[{self.source} {self.doc_version}]"

    def __repr__(self) -> str:
        return f"PolicyChunk(source={self.source!r}, chunk_id={self.chunk_id!r})"


# ── Chunking ────────────────────────────────────────────────────────────────

def _extract_version(text: str) -> str:
    """
    Parse the document version from the policy header.
    Expects a line of the form:  ``Version:   2.3``
    """
    match = re.search(r"Version:\s+([\w\.]+)", text)
    return match.group(1) if match else "unknown"


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.
    Splitting happens at whitespace boundaries where possible.
    """
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Walk back to nearest whitespace to avoid mid-word splits
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]   # filter near-empty tail chunks


# ── Vector store helpers ─────────────────────────────────────────────────────

def _load_policy_files(policies_dir: Path) -> List[PolicyChunk]:
    """Load and chunk all ``*.txt`` files in *policies_dir*."""
    chunks: List[PolicyChunk] = []
    for policy_file in sorted(policies_dir.glob("*.txt")):
        text = policy_file.read_text(encoding="utf-8")
        version = _extract_version(text)
        file_chunks = _chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
        for idx, chunk_text in enumerate(file_chunks):
            chunk_id = f"{policy_file.stem}__{idx}"
            chunks.append(
                PolicyChunk(
                    text=chunk_text,
                    source=policy_file.name,
                    doc_version=version,
                    chunk_id=chunk_id,
                )
            )
    _logger.info(
        f"Loaded {len(chunks)} chunks from {len(list(policies_dir.glob('*.txt')))} policy files."
    )
    return chunks


# ── PolicyRetriever ──────────────────────────────────────────────────────────

class PolicyRetriever:
    """
    Manages the ChromaDB collection and exposes a simple ``retrieve`` interface.

    Parameters
    ----------
    policies_path:
        Directory containing ``*.txt`` policy documents.
    vectorstore_path:
        Directory where ChromaDB persists its data.
    force_rebuild:
        If ``True``, delete and recreate the vector store from scratch.
        Useful when policy files have been updated.
    """

    def __init__(
        self,
        policies_path: Path | None = None,
        vectorstore_path: Path | None = None,
        force_rebuild: bool = False,
    ) -> None:
        self._policies_path = policies_path or settings.POLICIES_PATH
        self._vectorstore_path = vectorstore_path or settings.VECTORSTORE_PATH
        self._ef = get_embedding_function()
        self._client = chromadb.PersistentClient(path=str(self._vectorstore_path))
        self._collection = self._get_or_create_collection(force_rebuild)

    # ── initialisation ────────────────────────────────────────────────────────

    def _get_or_create_collection(self, force_rebuild: bool) -> chromadb.Collection:
        if force_rebuild:
            try:
                self._client.delete_collection(_COLLECTION_NAME)
                _logger.info("Deleted existing vector store for rebuild.")
            except Exception:
                pass

        collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        if collection.count() == 0:
            _logger.info("Vector store is empty — indexing policy documents…")
            self._index_policies(collection)
        else:
            _logger.info(
                f"Loaded existing vector store ({collection.count()} chunks)."
            )
        return collection

    def _index_policies(self, collection: chromadb.Collection) -> None:
        chunks = _load_policy_files(self._policies_path)
        if not chunks:
            _logger.warning(
                f"No policy files found at {self._policies_path}. "
                "Check the POLICIES_PATH setting."
            )
            return

        collection.add(
            documents=[c.text for c in chunks],
            ids=[c.chunk_id for c in chunks],
            metadatas=[
                {"source": c.source, "version": c.doc_version}
                for c in chunks
            ],
        )
        _logger.info(f"Indexed {len(chunks)} chunks into vector store.")

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> List[dict]:
        """
        Retrieve the top-*k* most relevant policy chunks for *query*.

        Returns
        -------
        List of dicts with keys: ``text``, ``source``, ``version``, ``score``.
        Lower ``score`` = more similar (cosine distance).
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(
                {
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "version": meta.get("version", "unknown"),
                    "score": round(distance, 4),
                }
            )
        return docs

    def format_context(self, docs: List[dict]) -> str:
        """
        Format retrieved docs into a numbered context string with citations,
        ready to inject into an LLM prompt.
        """
        lines = []
        for i, doc in enumerate(docs, start=1):
            citation = f"[Source {i}: {doc['source']} {doc['version']}]"
            lines.append(f"{citation}\n{doc['text']}\n")
        return "\n---\n".join(lines)

    def rebuild_index(self) -> None:
        """Force a full re-index of all policy documents."""
        self._collection = self._get_or_create_collection(force_rebuild=True)
