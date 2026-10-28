"""Shared test fixtures."""

from collections.abc import Callable
from datetime import date

import pytest

from adaptive_tutor.config import Settings

# Phase 5-9 tests use a fixed "today", so review dates are predictable.
TODAY = date(2026, 10, 28)


@pytest.fixture
def settings() -> Settings:
    # Phase 2 fields keep their defaults (e.g. min_relevance_score=0.5).
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model:latest",
        temperature=0.0,
        request_timeout_seconds=10.0,
        max_question_length=100,
        log_level="INFO",
    )


def build_pdf(page_texts: list[str]) -> bytes:
    """Build a tiny but valid PDF with one line of text per page.

    This lets PDF tests run without any sample files or extra packages.
    """
    page_ids = [4 + 2 * i for i in range(len(page_texts))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, page_texts):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_position = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(pdf)


@pytest.fixture
def make_pdf() -> Callable[[list[str]], bytes]:
    return build_pdf


@pytest.fixture
def repository(tmp_path):
    """Phase 7: a real SQLite progress database in a temporary folder."""
    from adaptive_tutor.memory.repository import LearnerRepository

    return LearnerRepository(tmp_path / "tutor.sqlite")


@pytest.fixture
def today() -> date:
    return TODAY