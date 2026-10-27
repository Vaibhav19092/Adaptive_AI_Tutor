from dataclasses import replace

import pytest

from adaptive_tutor.documents.chunking import make_document_id
from adaptive_tutor.documents.ingestion import DocumentReadError
from adaptive_tutor.documents.pipeline import ingest_file


class FakeStore:
    """Records what would be stored, without Chroma or embeddings."""

    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self.existing_ids = existing_ids or set()
        self.added = []

    def has_document(self, document_id: str) -> bool:
        return document_id in self.existing_ids

    def add_chunks(self, chunks, on_progress=None) -> int:
        self.added.extend(chunks)
        if on_progress is not None:
            on_progress(len(chunks), len(chunks))
        return len(chunks)


def test_new_file_is_read_chunked_and_stored(settings):
    store = FakeStore()
    progress = []
    data = b"Photosynthesis turns light into sugar. " * 50

    report = ingest_file("notes.txt", data, store, settings, lambda done, total: progress.append((done, total)))

    assert report.already_present is False
    assert report.pages == 1
    assert report.chunks == len(store.added) > 1
    assert report.document_id == make_document_id(data)
    assert progress[-1] == (report.chunks, report.chunks)


def test_same_file_twice_is_not_stored_again(settings):
    data = b"Some notes about cells."
    store = FakeStore(existing_ids={make_document_id(data)})

    report = ingest_file("notes.txt", data, store, settings)

    assert report.already_present is True
    assert store.added == []


def test_file_over_the_size_limit_is_rejected(settings):
    small_limit = replace(settings, max_upload_mb=0.001)  # about 1 KB
    with pytest.raises(DocumentReadError, match="limit"):
        ingest_file("big.txt", b"x" * 5000, FakeStore(), small_limit)