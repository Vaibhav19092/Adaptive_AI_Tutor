"""The "Ask a question" tab: the Phase 1-4 question-answering workflow (unchanged behaviour).

New in Phase 9: each run records its timing and a few quality flags (never the text).
"""

from __future__ import annotations

import logging
import textwrap
import time

import streamlit as st

from adaptive_tutor.config import Settings
from adaptive_tutor.evaluation.metrics import question_event_data
from adaptive_tutor.graph.workflow import ask_tutor
from adaptive_tutor.learner.profile import LearnerProfile
from adaptive_tutor.research.search_provider import WebResult
from adaptive_tutor.services import TutorServices

logger = logging.getLogger("adaptive_tutor.app")

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


def safe_link(title: str, url: str) -> str:
    """A Markdown link that a strange title or URL can't break."""
    title = title.replace("[", "(").replace("]", ")")
    url = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{title}]({url})"


def show_web_sources(web_sources: list[WebResult], provider_name: str) -> None:
    with st.expander(f"Web sources ({len(web_sources)}) · from the internet", expanded=True):
        st.caption("Internet sources can be wrong or out of date. Check important facts.")
        for number, result in enumerate(web_sources, start=1):
            st.markdown(
                f"**[W{number}]** {safe_link(result.title, result.url)} · {result.domain} · "
                f"{result.content_source} · quality {result.quality_score:.2f}"
            )
            st.caption(" · ".join(result.quality_notes))
            st.caption(textwrap.shorten(result.content, width=300, placeholder=" ..."))
        if provider_name == "Wikipedia":
            st.caption("Wikipedia text is shared under the CC BY-SA licence; the links above credit each article.")


def show_rejected_sources(rejected: list[WebResult]) -> None:
    with st.expander(f"Filtered out ({len(rejected)})"):
        for result in rejected:
            st.markdown(f"**{result.title}** · {result.domain or result.url}")
            st.caption(" · ".join(result.quality_notes))


def _record_question(services: TutorServices, learner: LearnerProfile | None, started: float, result: dict) -> None:
    try:
        services.repository.record_event(
            learner.learner_id if learner else None,
            "question",
            int((time.perf_counter() - started) * 1000),
            question_event_data(result),
        )
    except Exception:  # noqa: BLE001 - metrics must never break the app
        logger.exception("Could not record a metric event")


def render_ask_page(services: TutorServices, settings: Settings, learner: LearnerProfile | None) -> None:
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
        question = st.text_area("Ask a question", placeholder="e.g. What is an embedding, in simple words?", height=120)
        submitted = st.form_submit_button("Ask the tutor")

    if not submitted:
        return

    with st.spinner("Thinking... local models on a CPU can take a while."):
        started = time.perf_counter()
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
            return
    _record_question(services, learner, started, result)

    if result.get("search_query") and services.search_provider is not None:
        sent_query = result["search_query"].replace("`", "'")
        st.caption(f"Search query sent to {services.search_provider.name}: `{sent_query}`")

    if result.get("error"):
        st.error(result["error"])
    elif result.get("insufficient_evidence"):
        st.warning(result["answer"])
        best_score = result.get("best_score")
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
            label = f"Related passages ({len(sources)}) · from your files, not enough for a reliable answer"
        else:
            label = f"Document sources ({len(sources)}) · from your files"
        with st.expander(label, expanded=True):
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