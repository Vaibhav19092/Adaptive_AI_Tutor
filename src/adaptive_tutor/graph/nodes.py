"""The functions (nodes) that LangGraph runs, plus the routing decisions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.state import TutorState
from adaptive_tutor.llm.provider import describe_llm_error
from adaptive_tutor.research.content_extraction import PageFetcher
from adaptive_tutor.research.search_provider import SearchError, SearchProvider, WebResult, format_web_context
from adaptive_tutor.research.search_query import clean_search_query
from adaptive_tutor.research.source_verification import apply_hard_rules, score_results
from adaptive_tutor.retrieval.corrective_rag import plan_retrieval
from adaptive_tutor.retrieval.retriever import Retriever, format_context

logger = logging.getLogger(__name__)

NodeFunction = Callable[[TutorState], dict[str, Any]]

# ---------- Prompts ----------

TUTOR_SYSTEM_PROMPT = (
    "You are a patient, friendly tutor for beginners.\n"
    "Explain things in simple language and avoid unnecessary jargon.\n"
    "Structure every answer as:\n"
    "1. A short, direct answer (2-3 sentences).\n"
    "2. A simple explanation.\n"
    "3. One concrete example or everyday analogy.\n"
    "If you are not sure about something, say so clearly instead of guessing."
)

TUTOR_PROMPT = ChatPromptTemplate.from_messages(
    [("system", TUTOR_SYSTEM_PROMPT), ("human", "{question}")]
)

GROUNDED_SYSTEM_PROMPT = (
    "You are a patient, friendly tutor for beginners.\n"
    "Answer the question using ONLY the numbered context passages you are given.\n"
    "After each sentence that uses a passage, cite it like [1] or [2].\n"
    "Explain in simple language, and add a short example or analogy when the passages support it.\n"
    "If the passages do not contain enough information to answer, say so clearly. "
    "Do not fill gaps with outside knowledge, and never invent citations."
)

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GROUNDED_SYSTEM_PROMPT),
        ("human", "Context passages:\n\n{context}\n\nQuestion: {question}{extra_instructions}"),
    ]
)

NO_EVIDENCE_MESSAGE = (
    "I couldn't find information about this in your uploaded documents, so I can't give "
    "a reliable answer from them. Try rephrasing the question, adding a document that "
    "covers this topic, or switching to General knowledge."
)

PARTIAL_EVIDENCE_MESSAGE = (
    "I found passages in your documents that are related, but they don't contain enough "
    "information to answer this reliably, so I won't guess. Open \"Related passages\" below "
    "to read what I found, or add a document that covers this topic."
)

NO_DOCUMENTS_MESSAGE = (
    "You haven't added any documents yet. Upload a PDF or text file in the sidebar, "
    "or switch to General knowledge."
)

NO_RETRIEVER_MESSAGE = "Document search is not set up in this app, so only General knowledge works."

# ----- Phase 3: web research -----

SEARCH_QUERY_SYSTEM_PROMPT = (
    "You turn a learner's question into a short web search query.\n"
    "Reply with ONLY the query: 3 to 8 important keywords on one line.\n"
    "No quotes, no explanation, and do not answer the question."
)

SEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SEARCH_QUERY_SYSTEM_PROMPT), ("human", "Question: {question}")]
)

WEB_GROUNDED_SYSTEM_PROMPT = (
    "You are a patient, friendly tutor for beginners.\n"
    "Answer the question using ONLY the numbered web search results you are given.\n"
    "After each sentence that uses a result, cite it like [W1] or [W2].\n"
    "These results come from the internet, so they may be incomplete, outdated, or wrong. "
    "If they disagree with each other, say so and cite both.\n"
    "Explain in simple language, and add a short example or analogy when the results support it.\n"
    "If the results do not contain enough information to answer, say so clearly. "
    "Do not fill gaps with outside knowledge, and never invent citations."
)

WEB_GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", WEB_GROUNDED_SYSTEM_PROMPT),
        ("human", "Web search results:\n\n{context}\n\nQuestion: {question}{extra_instructions}"),
    ]
)

NO_WEB_EVIDENCE_MESSAGE = (
    "The web search didn't find any results good enough to answer from, so I can't give "
    "a reliable answer. Try rephrasing the question with more specific words, or open "
    "\"Filtered out\" below to see what was rejected and why."
)

RESEARCH_DISABLED_MESSAGE = (
    "Web research is turned off. To use it, set RESEARCH_ENABLED=true in .env "
    "(see the README), then restart the app."
)

UNEXPECTED_SEARCH_ERROR_MESSAGE = "Something unexpected went wrong during the web search. Check the terminal logs."

NO_WEB_FALLBACK_EVIDENCE_MESSAGE = (
    "Your documents didn't cover this, and the web search didn't find results good enough "
    "to answer from either, so I can't give a reliable answer."
)

# ----- Phase 4: extra instructions for the answer writers -----

CONTRADICTION_INSTRUCTION = (
    "Important: the sources disagree with each other about this. Do not pick a side. "
    "Describe each view, cite the source for each one, and tell the learner that the sources disagree."
)

RETRY_INSTRUCTION = (
    "Important: your previous answer was rejected because {problem}. Write a new answer that "
    "uses ONLY facts stated in the sources, cites a source after every sentence, and leaves out "
    "anything the sources don't say."
)


# ---------- Shared helpers ----------

def _extra_instructions(state: TutorState) -> str:
    """Phase 4 adds instructions after a failed check. Phase 2/3 answers get none ("")."""
    parts = []
    if state.get("evidence_verdict") == "contradictory":
        parts.append(CONTRADICTION_INSTRUCTION)
    if state.get("grounding_problem"):
        parts.append(RETRY_INSTRUCTION.format(problem=state["grounding_problem"]))
    return "\n\n" + "\n".join(parts) if parts else ""


def _invoke_chain(chain: Runnable, inputs: dict[str, str], settings: Settings, step: str) -> dict[str, Any]:
    """Run a prompt -> model -> text chain and turn any failure into a friendly error."""
    try:
        answer = chain.invoke(inputs)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a friendly message
        logger.exception("Model call failed in %s", step)
        return {"error": describe_llm_error(exc, settings), "steps": [step]}

    answer = answer.strip()
    if not answer:
        return {
            "error": "The model returned an empty answer. Try rephrasing your question.",
            "steps": [step],
        }

    logger.info("Answer generated in %s (%d characters)", step, len(answer))
    return {"answer": answer, "steps": [step]}


# ---------- Nodes ----------

def make_validate_question_node(max_length: int) -> NodeFunction:
    """Build a node that cleans and checks the learner's question."""

    def validate_question(state: TutorState) -> dict[str, Any]:
        question = state.get("question", "").strip()

        if not question:
            return {"error": "Please type a question first.", "steps": ["validate_question"]}

        if len(question) > max_length:
            return {
                "error": (
                    f"Your question is {len(question)} characters long. "
                    f"Please keep it under {max_length} characters."
                ),
                "steps": ["validate_question"],
            }

        # Log only the length, not the text, to keep learner questions private.
        logger.info("Question accepted (%d characters)", len(question))
        return {"question": question, "steps": ["validate_question"]}

    return validate_question


def make_generate_answer_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Build a node that answers from the model's general knowledge (Phase 1)."""
    chain = TUTOR_PROMPT | llm | StrOutputParser()

    def generate_answer(state: TutorState) -> dict[str, Any]:
        logger.info("Asking model '%s' (general knowledge)", settings.ollama_model)
        return _invoke_chain(chain, {"question": state["question"]}, settings, "generate_answer")

    return generate_answer


def make_retrieve_node(retriever: Retriever | None, settings: Settings) -> NodeFunction:
    """Build a node that finds relevant passages in the learner's documents."""

    def retrieve_documents(state: TutorState) -> dict[str, Any]:
        step = ["retrieve_documents"]
        if retriever is None:
            return {"error": NO_RETRIEVER_MESSAGE, "steps": step}

        # Phase 4: the first search is the normal one; retries use the corrective plan.
        attempts = state.get("retrieval_attempts", 0)
        plan = plan_retrieval(attempts, state["question"], state.get("retrieval_query"), settings)
        try:
            if retriever.document_count() == 0:
                return {"error": NO_DOCUMENTS_MESSAGE, "steps": step}
            if attempts == 0:
                chunks = retriever.search(plan.query)
            else:
                chunks = retriever.search(plan.query, top_k=plan.top_k)
        except Exception as exc:  # noqa: BLE001 - embedding or database failures
            logger.exception("Document retrieval failed")
            return {"error": describe_llm_error(exc, settings, model_name=settings.embedding_model), "steps": step}

        # Keep only passages that are similar enough to the question.
        relevant = [chunk for chunk in chunks if chunk.score >= plan.min_score]
        best_score = chunks[0].score if chunks else 0.0
        logger.info(
            "Search attempt %d: %d passages; %d passed the %.2f threshold (best score %.2f)",
            plan.attempt, len(chunks), len(relevant), plan.min_score, best_score,
        )
        result: dict[str, Any] = {
            "sources": relevant,
            "best_score": best_score,
            "retrieval_attempts": attempts + 1,
            "steps": step,
        }
        if state.get("check_evidence"):
            result["corrections"] = [plan.describe(found=len(chunks), kept=len(relevant))]
        return result

    return retrieve_documents


def make_generate_grounded_answer_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Build a node that answers ONLY from the retrieved passages, with citations."""
    chain = GROUNDED_PROMPT | llm | StrOutputParser()

    def generate_grounded_answer(state: TutorState) -> dict[str, Any]:
        sources = state.get("sources", [])
        logger.info("Asking model '%s' with %d passages", settings.ollama_model, len(sources))
        inputs = {
            "question": state["question"],
            "context": format_context(sources),
            "extra_instructions": _extra_instructions(state),
        }
        result = _invoke_chain(chain, inputs, settings, "generate_grounded_answer")
        result["answer_source"] = "documents"
        result["answer_attempts"] = state.get("answer_attempts", 0) + 1
        return result

    return generate_grounded_answer


def report_insufficient_evidence(state: TutorState) -> dict[str, Any]:
    """The documents don't cover the question: say so honestly instead of guessing."""
    logger.info("Not enough evidence in the documents; not writing an answer")
    # Phase 4 can find related passages that still aren't enough to answer.
    message = PARTIAL_EVIDENCE_MESSAGE if state.get("sources") else NO_EVIDENCE_MESSAGE
    return {
        "answer": message,
        "insufficient_evidence": True,
        "steps": ["report_insufficient_evidence"],
    }


# ----- Phase 3: web research nodes -----

def make_generate_search_query_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Build a node that turns the question into a short search query.

    Only this query leaves the computer, never your documents.
    """
    chain = SEARCH_QUERY_PROMPT | llm | StrOutputParser()

    def generate_search_query(state: TutorState) -> dict[str, Any]:
        step = ["generate_search_query"]
        question = state["question"]
        try:
            raw_query = chain.invoke({"question": question})
        except Exception as exc:  # noqa: BLE001 - Ollama down, model missing, ...
            logger.exception("Search query generation failed")
            return {"error": describe_llm_error(exc, settings), "steps": step}

        # If the model returns nothing useful, fall back to the question itself.
        query = clean_search_query(raw_query, fallback=question)
        logger.info("Search query ready (%d characters)", len(query))
        return {"search_query": query, "steps": step}

    return generate_search_query


def _add_page_text(result: WebResult, page_fetcher: PageFetcher) -> WebResult:
    """Replace a short snippet with the page's text, if robots.txt allows reading it."""
    if result.content_source != "snippet":
        return result  # e.g. Wikipedia already gave us the article intro
    text = page_fetcher.fetch_text(result.url)
    if text and len(text) > len(result.content):
        return replace(result, content=text, content_source="web page")
    return result


def make_search_web_node(
    search_provider: SearchProvider | None,
    settings: Settings,
    page_fetcher: PageFetcher | None = None,
) -> NodeFunction:
    """Build a node that searches the web, then filters the results with quality checks."""

    def search_web(state: TutorState) -> dict[str, Any]:
        step = ["search_web"]
        if search_provider is None:
            return {"error": RESEARCH_DISABLED_MESSAGE, "steps": step}

        query = state.get("search_query") or state["question"]
        try:
            raw_results = search_provider.search(query, settings.search_max_results)
        except SearchError as exc:
            logger.warning("Web search via %s failed: %s", search_provider.name, type(exc).__name__)
            return {"error": str(exc), "steps": step}
        except Exception:  # noqa: BLE001 - a bug in a provider must not crash the app
            logger.exception("Unexpected web search failure")
            return {"error": UNEXPECTED_SEARCH_ERROR_MESSAGE, "steps": step}

        # 1. Cheap rules first, so blocked or private sites are never contacted.
        kept, rejected = apply_hard_rules(raw_results, settings.research_blocked_domains)
        # 2. Optionally read the full pages (only where robots.txt allows).
        if page_fetcher is not None:
            kept = [_add_page_text(result, page_fetcher) for result in kept]
        # 3. Score what is left and drop low-quality results.
        accepted, low_quality = score_results(kept, query, settings.research_min_quality)

        logger.info(
            "Web search via %s: %d results, %d accepted", search_provider.name, len(raw_results), len(accepted)
        )
        return {
            "web_sources": accepted,
            "rejected_web_sources": rejected + low_quality,
            "web_results_found": len(raw_results),
            "steps": step,
        }

    return search_web


def make_generate_web_answer_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Build a node that answers ONLY from the web results, citing them as [W1], [W2], ..."""
    chain = WEB_GROUNDED_PROMPT | llm | StrOutputParser()

    def generate_web_answer(state: TutorState) -> dict[str, Any]:
        web_sources = state.get("web_sources", [])
        logger.info("Asking model '%s' with %d web results", settings.ollama_model, len(web_sources))
        inputs = {
            "question": state["question"],
            "context": format_web_context(web_sources),
            "extra_instructions": _extra_instructions(state),
        }
        result = _invoke_chain(chain, inputs, settings, "generate_web_answer")
        result["answer_source"] = "web"
        result["answer_attempts"] = state.get("answer_attempts", 0) + 1
        return result

    return generate_web_answer


def report_no_web_results(state: TutorState) -> dict[str, Any]:
    """Nothing usable was found online: say so honestly instead of guessing."""
    logger.info("No web result passed the quality checks; not calling the model")
    message = NO_WEB_FALLBACK_EVIDENCE_MESSAGE if state.get("used_web_fallback") else NO_WEB_EVIDENCE_MESSAGE
    return {
        "answer": message,
        "insufficient_evidence": True,
        "steps": ["report_no_web_results"],
    }


def handle_error(state: TutorState) -> dict[str, Any]:
    """Final stop for any failure. The error message is already in the state."""
    logger.warning("Workflow stopped with an error: %s", state.get("error"))
    return {"steps": ["handle_error"]}


# ---------- Routing decisions (conditional edges) ----------

def route_after_validation(
    state: TutorState,
) -> Literal["generate_answer", "retrieve_documents", "generate_search_query", "handle_error"]:
    """Bad question -> error. Web mode -> research. Document mode -> search documents.
    Otherwise answer from general knowledge."""
    if state.get("error"):
        return "handle_error"
    if state.get("use_web_research"):
        return "generate_search_query"
    if state.get("use_documents"):
        return "retrieve_documents"
    return "generate_answer"


def route_after_retrieval(
    state: TutorState,
) -> Literal["grade_documents", "generate_grounded_answer", "report_insufficient_evidence", "handle_error"]:
    """Error -> error. Checking on -> grade the passages (Phase 4).
    No relevant passages -> say so. Otherwise answer from them."""
    if state.get("error"):
        return "handle_error"
    if state.get("check_evidence"):
        return "grade_documents"
    if not state.get("sources"):
        return "report_insufficient_evidence"
    return "generate_grounded_answer"


def route_after_search_query(state: TutorState) -> Literal["search_web", "handle_error"]:
    """Only search once we have a query."""
    return "handle_error" if state.get("error") else "search_web"


def route_after_web_search(
    state: TutorState,
) -> Literal["grade_web_results", "generate_web_answer", "report_no_web_results", "handle_error"]:
    """Error -> error. No good results -> say so. Checking on -> grade them (Phase 4).
    Otherwise answer from them."""
    if state.get("error"):
        return "handle_error"
    if not state.get("web_sources"):
        return "report_no_web_results"
    if state.get("check_evidence"):
        return "grade_web_results"
    return "generate_web_answer"


def route_after_generation(state: TutorState) -> str:
    """Error -> error. Checking on -> check the answer (Phase 4). Otherwise finish."""
    if state.get("error"):
        return "handle_error"
    # Only answers built from sources can be checked against sources.
    if state.get("check_evidence") and state.get("answer_source"):
        return "check_answer"
    return END