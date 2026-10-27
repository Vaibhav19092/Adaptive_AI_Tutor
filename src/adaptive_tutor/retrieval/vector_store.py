"""Embeddings and the local Chroma vector store that holds document chunks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from adaptive_tutor.config import Settings

logger = logging.getLogger(__name__)


class PrefixedEmbeddings(Embeddings):
    """Add task prefixes before embedding.

    Some embedding models (like nomic-embed-text) give better search results
    when questions start with "search_query:" and stored passages start with
    "search_document:". The prefix is only used for the numbers; the stored
    text stays unchanged.
    """

    def __init__(self, inner: Embeddings, query_prefix: str = "", document_prefix: str = "") -> None:
        self._inner = inner
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([_add_prefix(self._document_prefix, t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(_add_prefix(self._query_prefix, text))


def _add_prefix(prefix: str, text: str) -> str:
    return f"{prefix} {text}" if prefix else text


def create_embeddings(settings: Settings) -> Embeddings:
    """The local embedding model, served by Ollama. Nothing leaves your computer."""
    base = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.request_timeout_seconds},
    )
    return PrefixedEmbeddings(
        base,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


def create_vector_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    """Open (or create) the Chroma collection saved on disk."""
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        # Chroma can send anonymous usage statistics; we switch that off.
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info("Opening Chroma collection '%s' at %s", settings.chroma_collection, settings.chroma_persist_dir)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        # Measure closeness with cosine distance: 0 = same direction, 1 = unrelated.
        collection_metadata={"hnsw:space": "cosine"},
    )


@dataclass(frozen=True)
class StoredDocument:
    """A summary of one file in the library."""

    document_id: str
    source: str
    chunks: int
    pages: int


ProgressCallback = Callable[[int, int], None]


class DocumentStore:
    """A small, beginner-friendly layer over Chroma for our app's needs."""

    def __init__(self, vector_store: Chroma, batch_size: int = 32) -> None:
        self._store = vector_store
        self._batch_size = batch_size

    def add_chunks(self, chunks: list[Document], on_progress: ProgressCallback | None = None) -> int:
        """Embed and save chunks in batches, reporting progress after each batch."""
        total = len(chunks)
        for start in range(0, total, self._batch_size):
            batch = chunks[start : start + self._batch_size]
            ids = [str(chunk.metadata["chunk_id"]) for chunk in batch]
            # Using our own ids means adding the same chunk again replaces it
            # instead of creating a duplicate.
            self._store.add_documents(batch, ids=ids)
            if on_progress is not None:
                on_progress(min(start + len(batch), total), total)
        logger.info("Stored %d chunks", total)
        return total

    def has_document(self, document_id: str) -> bool:
        result = self._store.get(where={"document_id": document_id}, limit=1, include=["metadatas"])
        return bool(result["ids"])

    def count(self) -> int:
        """How many chunks are stored in total."""
        return len(self._store.get(include=["metadatas"])["ids"])

    def list_documents(self) -> list[StoredDocument]:
        """One summary per file, sorted by name."""
        result = self._store.get(include=["metadatas"])
        summary: dict[str, dict] = {}
        for metadata in result["metadatas"]:
            if not metadata or "document_id" not in metadata:
                continue
            entry = summary.setdefault(
                str(metadata["document_id"]),
                {"source": str(metadata.get("source", "unknown")), "chunks": 0, "pages": set()},
            )
            entry["chunks"] += 1
            if "page" in metadata:
                entry["pages"].add(metadata["page"])

        documents = [
            StoredDocument(document_id=doc_id, source=e["source"], chunks=e["chunks"], pages=len(e["pages"]))
            for doc_id, e in summary.items()
        ]
        return sorted(documents, key=lambda d: d.source.lower())

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk of one file. Returns how many chunks were removed."""
        ids = self._store.get(where={"document_id": document_id}, include=["metadatas"])["ids"]
        if ids:
            self._store.delete(ids=ids)
        logger.info("Deleted document %s (%d chunks)", document_id, len(ids))
        return len(ids)

    def clear(self) -> int:
        """Remove every chunk of every file. Your original files are not touched."""
        ids = self._store.get(include=["metadatas"])["ids"]
        if ids:
            self._store.delete(ids=ids)
        logger.info("Cleared the library (%d chunks)", len(ids))
        return len(ids)

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """The k closest chunks, each with its cosine distance (smaller = closer)."""
        return self._store.similarity_search_with_score(query, k=k)