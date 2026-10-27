"""Streamlit user interface for the Adaptive AI Tutor (Phase 2)."""

import logging
import sys
import textwrap
from pathlib import Path

import streamlit as st

# Make the "src" folder importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adaptive_tutor.config import Settings, load_settings  # noqa: E402
from adaptive_tutor.documents.ingestion import DocumentError  # noqa: E402
from adaptive_tutor.documents.pipeline import ingest_file  # noqa: E402
from adaptive_tutor.graph.workflow import ask_tutor  # noqa: E402
from adaptive_tutor.llm.provider import (  # noqa: E402
    LLMError,
    check_embedding_model_available,
    check_model_available,
    describe_llm_error,
)
from adaptive_tutor.logging_setup import setup_logging  # noqa: E402
from adaptive_tutor.retrieval.vector_store import DocumentStore  # noqa: E402
from adaptive_tutor.services import TutorServices, build_services  # noqa: E402

st.set_page_config(page_title="Adaptive AI Tutor", page_icon="🎓")

try:
    settings = load_settings()
except ValueError as exc:
    st.error(f"Configuration problem: {exc}. Check your .env file.")
    st.stop()

setup_logging(settings.log_level)
logger = logging.getLogger("adaptive_tutor.app")


@st.cache_resource
def get_services(_settings: Settings) -> TutorServices:
    """Build the model, document library, and graph once. Restart after editing .env.

    The leading underscore tells Streamlit not to try hashing this argument.
    """
    return build_services(_settings)


def show_model_status(label: str, model_name: str, check) -> None:
    """Show a green or red box saying whether a model is ready."""
    try:
        check(settings)
        st.success(f"{label} `{model_name}` is ready.")
    except LLMError as exc:
        st.error(f"{label}: {exc}")


def add_uploaded_files(uploaded_files, store: DocumentStore) -> None:
    """Read, chunk, embed, and store each uploaded file, with a progress bar."""
    for uploaded in uploaded_files:
        progress = st.progress(0.0, text=f"Reading {uploaded.name}...")

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Embedding {uploaded.name}: {done}/{total} chunks")

        try:
            report = ingest_file(uploaded.name, uploaded.getvalue(), store, settings, on_progress)
        except DocumentError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - embedding or database failures
            logger.exception("Failed to add %s", uploaded.name)
            message = describe_llm_error(exc, settings, model_name=settings.embedding_model)
            st.error(f"{uploaded.name}: {message}")
        else:
            if report.already_present:
                st.info(f"{report.source} is already in your library.")
            else:
                st.success(f"Added {report.source}: {report.pages} page(s), {report.chunks} chunks.")
        finally:
            progress.empty()


try:
    services = get_services(settings)
except Exception:  # noqa: BLE001 - e.g. the data folder cannot be created
    logger.exception("Could not start the tutor services")
    st.error("Could not open the document library. Check the terminal logs for details.")
    st.stop()

# ---------- Sidebar: models and documents ----------
with st.sidebar:
    st.header("Local models")
    show_model_status("Chat model", settings.ollama_model, check_model_available)
    show_model_status("Embedding model", settings.embedding_model, check_embedding_model_available)
    if st.button("Re-check models"):
        st.rerun()

    st.header("Your documents")
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Add PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        add_clicked = st.form_submit_button("Add to library")

    if add_clicked:
        if uploaded_files:
            add_uploaded_files(uploaded_files, services.store)
        else:
            st.warning("Choose at least one file first.")

    try:
        documents = services.store.list_documents()
    except Exception:  # noqa: BLE001
        logger.exception("Could not list documents")
        st.error("Could not read the document library. Check the terminal logs.")
        documents = []

    if not documents:
        st.caption("No documents yet. Add a file above to ask questions about it. Files stay on this computer.")

    for document in documents:
        info_column, delete_column = st.columns([4, 1])
        details = f"{document.chunks} chunks"
        if document.pages:
            details = f"{document.pages} pages, {details}"
        info_column.markdown(f"**{document.source}**  \n{details}")
        if delete_column.button("🗑️", key=f"delete-{document.document_id}", help=f"Remove {document.source}"):
            services.store.delete_document(document.document_id)
            st.rerun()

    if documents:
        confirm_clear = st.checkbox("I want to remove every document")
        if st.button("Clear library", disabled=not confirm_clear):
            services.store.clear()
            st.rerun()

# ---------- Main area ----------
st.title("🎓 Adaptive AI Tutor")

mode = st.radio(
    "Answer from",
    ["My documents", "General knowledge"],
    horizontal=True,
    help="My documents: answers only from your uploaded files, with sources. "
    "General knowledge: the model's own training, which can be wrong.",
)
use_documents = mode == "My documents"

if use_documents:
    st.caption("Answers come only from your uploaded documents, with sources.")
else:
    st.caption("Answers come from the model's own knowledge, not your documents, so they can be wrong.")

with st.form("question_form"):
    question = st.text_area(
        "Ask a question",
        placeholder="e.g. What is an embedding, in simple words?",
        height=120,
    )
    submitted = st.form_submit_button("Ask the tutor")

if submitted:
    with st.spinner("Thinking... local models on a CPU can take a while."):
        try:
            result = ask_tutor(services.graph, question, use_documents=use_documents)
        except Exception:  # noqa: BLE001 - safety net; nodes already handle known errors
            logger.exception("Unexpected workflow failure")
            st.error("Something unexpected went wrong. Check the terminal logs.")
            st.stop()

    if result.get("error"):
        st.error(result["error"])
    elif result.get("insufficient_evidence"):
        st.warning(result["answer"])
        best_score = result.get("best_score")
        if best_score is not None:
            st.caption(
                f"The closest passage scored {best_score:.2f}; "
                f"answers need at least {settings.min_relevance_score:.2f} (MIN_RELEVANCE_SCORE)."
            )
    else:
        st.markdown(result["answer"])

    sources = result.get("sources") or []
    if sources and not result.get("error"):
        with st.expander(f"Sources ({len(sources)})", expanded=True):
            for number, chunk in enumerate(sources, start=1):
                st.markdown(f"**[{number}] {chunk.citation}** · relevance {chunk.score:.2f}")
                st.caption(textwrap.shorten(chunk.text, width=300, placeholder=" ..."))

    with st.expander("Show LangGraph workflow steps"):
        st.write(" → ".join(result.get("steps", [])))