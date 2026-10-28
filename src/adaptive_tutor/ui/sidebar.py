"""The sidebar: learner profiles (Phase 6-7), local models, documents, and web research."""

from __future__ import annotations

import logging

import streamlit as st

from adaptive_tutor.config import Settings
from adaptive_tutor.documents.ingestion import DocumentError
from adaptive_tutor.documents.pipeline import ingest_file
from adaptive_tutor.learner.profile import LearnerProfile
from adaptive_tutor.llm.provider import LLMError, check_embedding_model_available, check_model_available, describe_llm_error
from adaptive_tutor.memory.checkpointing import delete_threads
from adaptive_tutor.retrieval.vector_store import DocumentStore
from adaptive_tutor.services import TutorServices

logger = logging.getLogger("adaptive_tutor.app")

LEARNER_KEY = "learner_select"
NEW_LEARNER_KEY = "new_learner_id"
LAST_RESULT_PREFIX = "last_learning_result:"


def _show_model_status(settings: Settings, label: str, model_name: str, check) -> None:
    try:
        check(settings)
        st.success(f"{label} `{model_name}` is ready.")
    except LLMError as exc:
        st.error(f"{label}: {exc}")


def _add_uploaded_files(uploaded_files, store: DocumentStore, settings: Settings) -> None:
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
            st.error(f"{uploaded.name}: {describe_llm_error(exc, settings, model_name=settings.embedding_model)}")
        else:
            if report.already_present:
                st.info(f"{report.source} is already in your library.")
            else:
                st.success(f"Added {report.source}: {report.pages} page(s), {report.chunks} chunks.")
        finally:
            progress.empty()


def _learner_section(services: TutorServices) -> LearnerProfile | None:
    repository = services.repository
    st.header("Learner")
    try:
        learners = repository.list_learners()
    except Exception:  # noqa: BLE001
        logger.exception("Could not list learners")
        st.error("Could not open the progress database. Check the terminal logs.")
        return None

    by_id = {learner.learner_id: learner for learner in learners}
    # A newly created learner is selected on the next run (before the widget is drawn).
    if NEW_LEARNER_KEY in st.session_state:
        st.session_state[LEARNER_KEY] = st.session_state.pop(NEW_LEARNER_KEY)
    if st.session_state.get(LEARNER_KEY) not in by_id:
        st.session_state.pop(LEARNER_KEY, None)

    selected = None
    if learners:
        learner_id = st.selectbox(
            "Who is learning?", list(by_id), format_func=lambda key: by_id[key].name, key=LEARNER_KEY
        )
        selected = by_id.get(learner_id)

    with st.expander("Add a learner", expanded=not learners):
        with st.form("new_learner_form", clear_on_submit=True):
            name = st.text_input("Name", max_chars=40)
            if st.form_submit_button("Create profile"):
                try:
                    profile = repository.create_learner(name)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[NEW_LEARNER_KEY] = profile.learner_id
                    st.rerun()

    if selected is not None:
        with st.expander("Reset or delete progress"):
            st.caption("Reset keeps the profile but removes all lessons, quizzes, and mastery. Delete removes everything.")
            confirm = st.checkbox(f"I'm sure, for {selected.name}", key=f"confirm-{selected.learner_id}")
            reset_column, delete_column = st.columns(2)
            if reset_column.button("Reset", disabled=not confirm):
                threads = repository.reset_progress(selected.learner_id)
                delete_threads(services.checkpointer, threads)
                st.session_state.pop(LAST_RESULT_PREFIX + selected.learner_id, None)
                st.rerun()
            if delete_column.button("Delete", disabled=not confirm, type="primary"):
                threads = repository.delete_learner(selected.learner_id)
                delete_threads(services.checkpointer, threads)
                st.session_state.pop(LAST_RESULT_PREFIX + selected.learner_id, None)
                st.rerun()
    st.caption("Profiles keep each learner's progress apart on this computer. They are not password-protected.")
    return selected


def render_sidebar(services: TutorServices, settings: Settings) -> LearnerProfile | None:
    """Draw the sidebar and return the selected learner (or None)."""
    with st.sidebar:
        learner = _learner_section(services)

        st.header("Local models")
        _show_model_status(settings, "Chat model", settings.ollama_model, check_model_available)
        _show_model_status(settings, "Embedding model", settings.embedding_model, check_embedding_model_available)
        if st.button("Re-check models"):
            st.rerun()

        st.header("Your documents")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_files = st.file_uploader("Add PDF, TXT, or MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
            add_clicked = st.form_submit_button("Add to library")
        if add_clicked:
            if uploaded_files:
                _add_uploaded_files(uploaded_files, services.store, settings)
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

        st.header("Web research")
        provider = services.search_provider
        if provider is None:
            st.caption(
                "Off. Nothing is sent to the internet. To turn it on, set RESEARCH_ENABLED=true "
                "in .env (see the README) and restart the app."
            )
        else:
            st.info(f"On, using **{provider.name}**. Search queries go to `{provider.destination}`. Your documents are never sent.")
            if settings.research_fetch_pages:
                st.caption("Page reading is on: result pages are opened only where their robots.txt allows it.")
    return learner