"""Turn an uploaded file into searchable chunks: check -> read -> split -> store."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from adaptive_tutor.config import Settings
from adaptive_tutor.documents.chunking import chunk_pages, make_document_id
from adaptive_tutor.documents.ingestion import DocumentReadError, extract_pages

if TYPE_CHECKING:  # only needed for type hints; avoids loading Chroma here
    from adaptive_tutor.retrieval.vector_store import DocumentStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class IngestionReport:
    """What happened when a file was added."""

    source: str
    document_id: str
    pages: int
    chunks: int
    already_present: bool = False


def ingest_file(
    filename: str,
    data: bytes,
    store: DocumentStore,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
) -> IngestionReport:
    """Add one file to the library, skipping it if the same file is already there."""
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise DocumentReadError(
            f"{filename} is {size_mb:.1f} MB; the limit is {settings.max_upload_mb:g} MB "
            "(MAX_UPLOAD_MB in .env)."
        )

    document_id = make_document_id(data)
    if store.has_document(document_id):
        logger.info("%s is already in the library; skipping", filename)
        return IngestionReport(filename, document_id, pages=0, chunks=0, already_present=True)

    pages = extract_pages(filename, data)
    chunks = chunk_pages(pages, document_id, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise DocumentReadError(f"{filename} produced no text to store.")

    store.add_chunks(chunks, on_progress)
    logger.info("Added %s: %d page(s), %d chunks", filename, len(pages), len(chunks))
    return IngestionReport(filename, document_id, pages=len(pages), chunks=len(chunks))