"""Search the document library and package each result with its citation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from langchain_core.documents import Document

if TYPE_CHECKING:  # only needed for type hints; avoids loading Chroma here
    from adaptive_tutor.retrieval.vector_store import DocumentStore


@dataclass(frozen=True)
class RetrievedChunk:
    """One passage found for a question, with where it came from and how relevant it is."""

    text: str
    source: str
    page: int | None
    chunk_index: int
    document_id: str
    score: float  # 0.0 = unrelated, 1.0 = a perfect match

    @property
    def citation(self) -> str:
        """A human-readable reference, like 'biology.pdf, page 4'."""
        return f"{self.source}, page {self.page}" if self.page is not None else self.source

    @classmethod
    def from_document(cls, document: Document, score: float) -> RetrievedChunk:
        metadata = document.metadata or {}
        page = metadata.get("page")
        return cls(
            text=document.page_content,
            source=str(metadata.get("source", "unknown")),
            page=int(page) if page is not None else None,
            chunk_index=int(metadata.get("chunk_index", 0)),
            document_id=str(metadata.get("document_id", "")),
            score=score,
        )


class Retriever(Protocol):
    """What the graph needs from any retriever. Tests can pass a simple fake."""

    def document_count(self) -> int: ...

    def search(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


def distance_to_relevance(distance: float) -> float:
    """Turn cosine distance into a 0-1 relevance score (cosine similarity)."""
    return max(0.0, min(1.0, 1.0 - distance))


class DocumentRetriever:
    """Finds the passages most similar to a question."""

    def __init__(self, store: DocumentStore, top_k: int) -> None:
        self._store = store
        self._top_k = top_k

    def document_count(self) -> int:
        return self._store.count()

    def search(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """top_k=None uses the normal RETRIEVAL_TOP_K. Corrective retries ask for more."""
        results = self._store.similarity_search(question, k=top_k or self._top_k)
        chunks = [RetrievedChunk.from_document(doc, distance_to_relevance(distance)) for doc, distance in results]
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the passages so the model can cite them as [1], [2], ..."""
    if not chunks:
        return "(no passages)"
    return "\n\n".join(
        f"[{number}] ({chunk.citation})\n{chunk.text}" for number, chunk in enumerate(chunks, start=1)
    )