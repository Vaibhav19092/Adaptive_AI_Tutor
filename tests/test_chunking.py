from adaptive_tutor.documents.chunking import chunk_pages, make_document_id
from adaptive_tutor.documents.ingestion import PageText


def test_document_id_is_based_on_content():
    assert make_document_id(b"hello") == make_document_id(b"hello")
    assert make_document_id(b"hello") != make_document_id(b"world")
    assert len(make_document_id(b"hello")) == 16


def test_chunks_carry_source_page_and_unique_ids():
    pages = [
        PageText("biology.pdf", 1, "Cells divide. " * 40),
        PageText("biology.pdf", 2, "A short second page."),
    ]
    chunks = chunk_pages(pages, "doc123", chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 2
    assert all(chunk.metadata["source"] == "biology.pdf" for chunk in chunks)
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_chunks_never_mix_two_pages():
    pages = [PageText("a.pdf", 1, "alpha " * 60), PageText("a.pdf", 2, "omega " * 60)]
    chunks = chunk_pages(pages, "doc1", chunk_size=120, chunk_overlap=30)
    for chunk in chunks:
        expected_word = "alpha" if chunk.metadata["page"] == 1 else "omega"
        assert set(chunk.page_content.split()) == {expected_word}


def test_text_files_have_no_page_key():
    chunks = chunk_pages([PageText("notes.txt", None, "Some notes.")], "doc1", 200, 20)
    assert "page" not in chunks[0].metadata