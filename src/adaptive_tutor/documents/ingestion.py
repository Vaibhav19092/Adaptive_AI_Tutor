"""Read uploaded files (PDF, TXT, MD) and extract their text page by page."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class PageText:
    """The text of one page. Text files count as a single page with no number."""

    source: str
    page: int | None
    text: str


class DocumentError(Exception):
    """Base class for problems with an uploaded file."""


class UnsupportedFileTypeError(DocumentError):
    """The file type is not one we can read."""


class DocumentReadError(DocumentError):
    """The file could not be read, or contains no usable text."""


def extract_pages(filename: str, data: bytes) -> list[PageText]:
    """Return the readable text of a file, one PageText per page."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"{filename}: '{extension or 'no extension'}' files are not supported. Use {allowed}."
        )

    if not data:
        raise DocumentReadError(f"{filename} is empty.")

    if extension == ".pdf":
        pages = _extract_pdf(filename, data)
    else:
        pages = _extract_text_file(filename, data)

    if not pages:
        raise DocumentReadError(
            f"No readable text found in {filename}. Scanned PDFs (pictures of pages) "
            "need OCR, which this tutor does not support yet."
        )

    logger.info("Extracted %d page(s) of text from %s", len(pages), filename)
    return pages


def _extract_pdf(filename: str, data: bytes) -> list[PageText]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentReadError(f"{filename} is password-protected. Remove the password and try again.")
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises many different error types
        raise DocumentReadError(f"{filename} could not be opened as a PDF ({type(exc).__name__}).") from exc

    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - skip one broken page, keep the rest
            logger.warning("Could not read page %d of %s; skipping it", number, filename)
            continue
        if text:
            pages.append(PageText(source=filename, page=number, text=text))
    return pages


def _extract_text_file(filename: str, data: bytes) -> list[PageText]:
    try:
        text = data.decode("utf-8-sig")  # "-sig" also removes a hidden BOM marker
    except UnicodeDecodeError:
        text = data.decode("latin-1")  # never fails; good enough for old Windows files
    text = _clean_text(text)
    return [PageText(source=filename, page=None, text=text)] if text else []


def _clean_text(text: str) -> str:
    """Remove junk characters, trailing spaces, and runs of blank lines."""
    text = text.replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()