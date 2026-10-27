"""These tests use a real local Chroma database (in a temporary folder) with fake embeddings."""

from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from adaptive_tutor.retrieval.retriever import DocumentRetriever
from adaptive_tutor.retrieval.vector_store import DocumentStore, PrefixedEmbeddings, create_vector_store


class RecordingEmbeddings(Embeddings):
    """Remembers the exact texts it was asked to embed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.seen.append(text)
        return [1.0, 0.0]


class MappedEmbeddings(Embeddings):
    """Hand-picked vectors, so we know exactly what the similarity should be."""

    VECTORS = {
        "apples": [1.0, 0.0, 0.0],
        "bananas": [0.0, 1.0, 0.0],
        "fruit question": [1.0, 1.0, 0.0],  # 45 degrees from both: cosine similarity 0.707
    }

    def embed_documents(self, texts):
        return [self.VECTORS[text] for text in texts]

    def embed_query(self, text):
        return self.VECTORS[text]


def make_chunk(text: str, document_id: str = "doc1", index: int = 0, page: int | None = None) -> Document:
    metadata = {"source": f"{document_id}.txt", "document_id": document_id, "chunk_index": index,
                "chunk_id": f"{document_id}-{index}"}
    if page is not None:
        metadata["page"] = page
    return Document(page_content=text, metadata=metadata)


@pytest.fixture
def make_store(tmp_path, settings):
    def _make(embeddings: Embeddings) -> DocumentStore:
        test_settings = replace(settings, chroma_persist_dir=tmp_path / "chroma", chroma_collection="test")
        return DocumentStore(create_vector_store(test_settings, embeddings), batch_size=2)

    return _make


def test_prefixes_are_added_before_embedding():
    inner = RecordingEmbeddings()
    embeddings = PrefixedEmbeddings(inner, query_prefix="search_query:", document_prefix="search_document:")
    embeddings.embed_documents(["Cells are small."])
    embeddings.embed_query("What is a cell?")
    assert inner.seen == ["search_document: Cells are small.", "search_query: What is a cell?"]


def test_no_prefix_leaves_text_unchanged():
    inner = RecordingEmbeddings()
    PrefixedEmbeddings(inner).embed_query("Hello")
    assert inner.seen == ["Hello"]


def test_add_count_and_list_documents(make_store):
    store = make_store(DeterministicFakeEmbedding(size=16))
    progress = []
    chunks = [make_chunk("a", "doc1", 0, page=1), make_chunk("b", "doc1", 1, page=2), make_chunk("c", "doc2", 0)]

    store.add_chunks(chunks, lambda done, total: progress.append((done, total)))

    assert store.count() == 3
    assert progress == [(2, 3), (3, 3)]  # two batches of up to 2
    documents = store.list_documents()
    assert [(d.source, d.chunks, d.pages) for d in documents] == [("doc1.txt", 2, 2), ("doc2.txt", 1, 0)]


def test_adding_the_same_chunks_again_does_not_duplicate(make_store):
    store = make_store(DeterministicFakeEmbedding(size=16))
    store.add_chunks([make_chunk("a")])
    store.add_chunks([make_chunk("a")])
    assert store.count() == 1


def test_has_document(make_store):
    store = make_store(DeterministicFakeEmbedding(size=16))
    store.add_chunks([make_chunk("a", "doc1")])
    assert store.has_document("doc1") is True
    assert store.has_document("doc2") is False


def test_delete_document_removes_only_that_document(make_store):
    store = make_store(DeterministicFakeEmbedding(size=16))
    store.add_chunks([make_chunk("a", "doc1", 0), make_chunk("b", "doc1", 1), make_chunk("c", "doc2", 0)])

    removed = store.delete_document("doc1")

    assert removed == 2
    assert [d.document_id for d in store.list_documents()] == ["doc2"]


def test_clear_removes_everything(make_store):
    store = make_store(DeterministicFakeEmbedding(size=16))
    store.add_chunks([make_chunk("a", "doc1"), make_chunk("b", "doc2")])
    assert store.clear() == 2
    assert store.count() == 0


def test_scores_are_cosine_similarity(make_store):
    """Proves the store measures cosine distance, so our 0-1 scores mean what we say."""
    store = make_store(MappedEmbeddings())
    store.add_chunks([make_chunk("apples", "doc1", 0), make_chunk("bananas", "doc1", 1)])

    chunks = DocumentRetriever(store, top_k=2).search("fruit question")

    assert len(chunks) == 2
    assert all(chunk.score == pytest.approx(0.7071, abs=0.01) for chunk in chunks)