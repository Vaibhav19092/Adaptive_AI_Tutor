import pytest
from langchain_core.documents import Document

from adaptive_tutor.retrieval.retriever import (
    DocumentRetriever,
    RetrievedChunk,
    distance_to_relevance,
    format_context,
)


def make_chunk(source: str, page: int | None, text: str = "Some text.") -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, page=page, chunk_index=0, document_id="d", score=0.7)


def test_citation_with_and_without_page():
    assert make_chunk("biology.pdf", 4).citation == "biology.pdf, page 4"
    assert make_chunk("notes.txt", None).citation == "notes.txt"


def test_distance_to_relevance_stays_between_0_and_1():
    assert distance_to_relevance(0.0) == 1.0
    assert distance_to_relevance(0.25) == pytest.approx(0.75)
    assert distance_to_relevance(1.6) == 0.0  # opposite meaning is clamped to 0


def test_from_document_reads_metadata():
    document = Document(
        page_content="Cells are small.",
        metadata={"source": "bio.pdf", "page": 3, "chunk_index": 5, "document_id": "abc"},
    )
    chunk = RetrievedChunk.from_document(document, score=0.6)
    assert (chunk.source, chunk.page, chunk.chunk_index, chunk.document_id) == ("bio.pdf", 3, 5, "abc")


def test_retriever_converts_distances_and_sorts_best_first():
    class FakeStore:
        def count(self):
            return 2

        def similarity_search(self, query, k):
            return [
                (Document(page_content="far", metadata={"source": "a.txt"}), 0.6),
                (Document(page_content="close", metadata={"source": "b.txt"}), 0.1),
            ]

    chunks = DocumentRetriever(FakeStore(), top_k=2).search("question")

    assert [chunk.text for chunk in chunks] == ["close", "far"]
    assert chunks[0].score == pytest.approx(0.9)


def test_format_context_numbers_each_passage():
    context = format_context([make_chunk("a.pdf", 1, "First."), make_chunk("b.txt", None, "Second.")])
    assert context == "[1] (a.pdf, page 1)\nFirst.\n\n[2] (b.txt)\nSecond."