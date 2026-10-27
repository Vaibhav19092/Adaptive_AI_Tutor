"""Split page text into small, overlapping chunks that remember where they came from."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from adaptive_tutor.documents.ingestion import PageText


def make_document_id(data: bytes) -> str:
    """A short fingerprint of the file's bytes: the same file always gets the same id."""
    return hashlib.sha256(data).hexdigest()[:16]


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split each page separately, so every chunk belongs to exactly one page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split at paragraphs first, then lines, sentences, and words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            text = piece.strip()
            if not text:
                continue
            index = len(chunks)
            metadata: dict[str, str | int] = {
                "source": page.source,
                "document_id": document_id,
                "chunk_index": index,
                "chunk_id": f"{document_id}-{index}",
            }
            # Chroma does not accept None as a metadata value, so text files
            # simply have no "page" key.
            if page.page is not None:
                metadata["page"] = page.page
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks