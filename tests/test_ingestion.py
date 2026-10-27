import pytest

from adaptive_tutor.documents.ingestion import (
    DocumentReadError,
    UnsupportedFileTypeError,
    extract_pages,
)


def test_text_file_becomes_one_page_without_a_number():
    pages = extract_pages("notes.txt", b"Plants make sugar.   \n\n\n\n\nThey need light.")
    assert len(pages) == 1
    assert pages[0].source == "notes.txt"
    assert pages[0].page is None
    assert pages[0].text == "Plants make sugar.\n\nThey need light."  # spaces and blank lines tidied


def test_markdown_with_hidden_bom_is_decoded():
    pages = extract_pages("notes.md", "\ufeff# Cells\nCells are small.".encode("utf-8"))
    assert pages[0].text.startswith("# Cells")


def test_non_utf8_text_falls_back_to_latin1():
    pages = extract_pages("old.txt", "café au lait".encode("latin-1"))
    assert pages[0].text == "café au lait"


def test_unsupported_file_type_raises():
    with pytest.raises(UnsupportedFileTypeError, match="not supported"):
        extract_pages("slides.pptx", b"data")


def test_empty_file_raises():
    with pytest.raises(DocumentReadError, match="empty"):
        extract_pages("notes.txt", b"")


def test_pdf_pages_keep_their_page_numbers(make_pdf):
    data = make_pdf(["Cells are small.", "Mitochondria make energy."])
    pages = extract_pages("biology.pdf", data)
    assert [page.page for page in pages] == [1, 2]
    assert "Cells are small." in pages[0].text
    assert "Mitochondria make energy." in pages[1].text


def test_pdf_without_any_text_raises(make_pdf):
    with pytest.raises(DocumentReadError, match="No readable text"):
        extract_pages("scan.pdf", make_pdf([""]))


def test_broken_pdf_raises():
    with pytest.raises(DocumentReadError, match="could not be opened"):
        extract_pages("broken.pdf", b"this is not really a pdf")