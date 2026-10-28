"""Streamlit user interface for the Adaptive AI Tutor (Phase 4)."""

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
from adaptive_tutor.research.search_provider import WebResult  # noqa: E402
from adaptive_tutor.retrieval.vector_store import DocumentStore  # noqa: E402
from adaptive_tutor.services import TutorServices, build_services  # noqa: E402

st.set_page_config(page_title="Adaptive AI Tutor", page_icon="🎓")

MODE_DOCUMENTS = "My documents"
MODE_WEB = "Web research"
MODE_GENERAL = "General knowledge"

MODE_CAPTIONS = {
    MODE_DOCUMENTS: "Answers come only from your uploaded documents, with sources.",
    MODE_WEB: (
        "Answers come only from web search results, with links. A short search query made "
        "from your question is sent to the search service; your documents are never sent."
    ),
    MODE_GENERAL: "Answers come from the model's own knowledge, not your documents, so they can be wrong.",
}

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


def safe_link(result: WebResult) -> str:
    """A Markdown link that a strange title or URL can't break."""
    title = result.title.replace("[", "(").replace("]", ")")
    url = result.url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{title}]({url})"


def show_web_sources(web_sources: list[WebResult], provider_name: str) -> None:
    """List the web results the answer was built from, clearly marked as internet sources."""
    with st.expander(f"Web sources ({len(web_sources)}) · from the internet", expanded=True):
        st.caption("Internet sources can be wrong or out of date. Check important facts.")
        for number, result in enumerate(web_sources, start=1):
            st.markdown(
                f"**[W{number}]** {safe_link(result)} · {result.domain} · "
                f"{result.content_source} · quality {result.quality_score:.2f}"
            )
            st.caption(" · ".join(result.quality_notes))
            st.caption(textwrap.shorten(result.content, width=300, placeholder=" ..."))
        if provider_name == "Wikipedia":
            st.caption("Wikipedia text is shared under the CC BY-SA licence; the links above credit each article.")


def show_rejected_sources(rejected: list[WebResult]) -> None:
    """Show which results were filtered out, and why, so the checks are easy to understand."""
    with st.expander(f"Filtered out ({len(rejected)})"):
        for result in rejected:
            st.markdown(f"**{result.title}** · {result.domain or result.url}")
            st.caption(" · ".join(result.quality_notes))


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

    st.header("Web research")
    provider = services.search_provider
    if provider is None:
        st.caption(
            "Off. Nothing is sent to the internet. To turn it on, set RESEARCH_ENABLED=true "
            "in .env (see the README) and restart the app."
        )
    else:
        # No automatic test search here: it would use up free quota on every click.
        st.info(
            f"On, using **{provider.name}**. Search queries go to `{provider.destination}`. "
            "Your documents are never sent."
        )
        if settings.research_fetch_pages:
            st.caption("Page reading is on: result pages are opened only where their robots.txt allows it.")

# ---------- Main area ----------
st.title("🎓 Adaptive AI Tutor")

modes = [MODE_DOCUMENTS, MODE_GENERAL]
if services.search_provider is not None:
    modes.insert(1, MODE_WEB)

mode = st.radio(
    "Answer from",
    modes,
    horizontal=True,
    help="My documents: only your uploaded files, with sources. "
    "Web research: only web search results, with links (when turned on). "
    "General knowledge: the model's own training, which can be wrong.",
)
use_documents = mode == MODE_DOCUMENTS
use_web_research = mode == MODE_WEB
st.caption(MODE_CAPTIONS[mode])

# Phase 4: Self-RAG checks (on by default) and the optional web fallback (off by default).
check_evidence = False
allow_web_fallback = False
if mode != MODE_GENERAL:
    check_evidence = st.checkbox(
        "Check evidence and answers carefully (Self-RAG)",
        value=True,
        help="The tutor grades each passage, checks the evidence is enough, searches again "
        "with a better query if needed, and checks the answer against the sources. "
        "Safer but slower: several extra calls to the local model.",
    )
    if use_documents and check_evidence and services.search_provider is not None:
        allow_web_fallback = st.checkbox(
            f"If my documents don't cover it, search {services.search_provider.name}",
            value=False,
            help="Only a short search query is sent. Your documents are never sent.",
        )

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
            result = ask_tutor(
                services.graph,
                question,
                use_documents=use_documents,
                use_web_research=use_web_research,
                check_evidence=check_evidence,
                allow_web_fallback=allow_web_fallback,
            )
        except Exception:  # noqa: BLE001 - safety net; nodes already handle known errors
            logger.exception("Unexpected workflow failure")
            st.error("Something unexpected went wrong. Check the terminal logs.")
            st.stop()

    # Show exactly what left the computer, so web research is never hidden.
    if result.get("search_query") and services.search_provider is not None:
        sent_query = result["search_query"].replace("`", "'")
        st.caption(f"Search query sent to {services.search_provider.name}: `{sent_query}`")

    if result.get("error"):
        st.error(result["error"])
    elif result.get("insufficient_evidence"):
        st.warning(result["answer"])
        best_score = result.get("best_score")
        # The score hint only helps when no passage was usable at all.
        if best_score is not None and not result.get("sources") and not result.get("used_web_fallback"):
            st.caption(
                f"The closest passage scored {best_score:.2f}; "
                f"answers need at least {settings.min_relevance_score:.2f} (MIN_RELEVANCE_SCORE)."
            )
            if services.search_provider is not None:
                st.info("Tip: choose **Web research** above to look this up online instead.")
        if "web_results_found" in result:
            st.caption(
                f"The search returned {result['web_results_found']} result(s); none passed the "
                f"quality checks (RESEARCH_MIN_QUALITY = {settings.research_min_quality:.2f})."
            )
    else:
        if result.get("used_web_fallback"):
            st.info("Your documents didn't cover this, so this answer comes from a web search.")
        if result.get("evidence_verdict") == "contradictory":
            st.warning("Your sources disagree about this. The answer describes each view.")
        st.markdown(result["answer"])
        if result.get("grounded"):
            st.success(
                "✓ Checked: the local model judged every claim to be supported by the sources. "
                "This check helps, but it is not a guarantee."
            )

    sources = result.get("sources") or []
    if sources and not result.get("error"):
        if result.get("insufficient_evidence"):
            sources_label = f"Related passages ({len(sources)}) · from your files, not enough for a reliable answer"
        else:
            sources_label = f"Document sources ({len(sources)}) · from your files"
        with st.expander(sources_label, expanded=True):
            for number, chunk in enumerate(sources, start=1):
                st.markdown(f"**[{number}] {chunk.citation}** · relevance {chunk.score:.2f}")
                st.caption(textwrap.shorten(chunk.text, width=300, placeholder=" ..."))

    web_sources = result.get("web_sources") or []
    if web_sources and not result.get("error") and services.search_provider is not None:
        show_web_sources(web_sources, services.search_provider.name)

    rejected = result.get("rejected_web_sources") or []
    if rejected and not result.get("error"):
        show_rejected_sources(rejected)

    rejected_passages = result.get("rejected_sources") or []
    if rejected_passages and not result.get("error"):
        with st.expander(f"Passages judged not relevant ({len(rejected_passages)})"):
            for chunk in rejected_passages:
                st.markdown(f"**{chunk.citation}** · similarity {chunk.score:.2f}")
                st.caption(textwrap.shorten(chunk.text, width=200, placeholder=" ..."))

    corrections = result.get("corrections") or []
    if corrections:
        with st.expander("How the answer was checked (Self-RAG and Corrective RAG)"):
            for number, note in enumerate(corrections, start=1):
                st.markdown(f"{number}. {note}")

    with st.expander("Show LangGraph workflow steps"):
        st.write(" → ".join(result.get("steps", [])))