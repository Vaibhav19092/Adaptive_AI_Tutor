"""Phase 4: Self-RAG checks and Corrective RAG steps, as LangGraph nodes and routers.

Self-RAG = the tutor checks its own work:
  grade each passage -> check the evidence is enough -> check the answer is supported.
Corrective RAG = when a check fails, the tutor fixes it:
  rewrite the query and search wider -> (optionally) search the web -> rewrite the answer.
Every loop has a hard limit (MAX_RETRIEVAL_RETRIES, MAX_ANSWER_RETRIES), and when the
limits are used up the tutor says it can't answer reliably instead of guessing.
"""

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
from adaptive_tutor.research.search_provider import WEB_CONTEXT_MAX_CHARS, format_web_context
from adaptive_tutor.research.search_query import clean_search_query
from adaptive_tutor.retrieval.retriever import format_context
from adaptive_tutor.retrieval.self_rag import chunk_key, citation_problem, parse_evidence_verdict, parse_yes_no

logger = logging.getLogger(__name__)

NodeFunction = Callable[[TutorState], dict[str, Any]]
Router = Callable[[TutorState], str]

# ---------- Prompts for the checker ----------
# Each check asks a small, single question with a one-word answer.
# Small local models are much more reliable at that than at long structured replies.

GRADE_SYSTEM_PROMPT = (
    "You check whether a passage is useful for answering a learner's question.\n"
    "Reply with only one word: yes or no."
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\nPassage:\n{passage}\n\n"
            "Does this passage contain information that helps answer the question?",
        ),
    ]
)

EVIDENCE_SYSTEM_PROMPT = (
    "You check whether some sources are enough to answer a learner's question.\n"
    "Reply with exactly one word:\n"
    "SUFFICIENT if the sources contain enough information to answer the question,\n"
    "INSUFFICIENT if important information is missing,\n"
    "CONTRADICTORY if the sources disagree with each other about the answer."
)

EVIDENCE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", EVIDENCE_SYSTEM_PROMPT), ("human", "Sources:\n\n{context}\n\nQuestion: {question}")]
)

GROUNDING_SYSTEM_PROMPT = (
    "You check whether a tutor's answer is supported by its sources.\n"
    "Reply with only one word: yes or no.\n"
    "yes means every factual claim in the answer is stated in the sources.\n"
    "no means the answer contains at least one claim the sources do not support."
)

GROUNDING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", GROUNDING_SYSTEM_PROMPT), ("human", "Sources:\n\n{context}\n\nAnswer to check:\n{answer}")]
)

REWRITE_SYSTEM_PROMPT = (
    "A search of the learner's own notes did not find good passages.\n"
    "Write ONE better search query for those notes: use different words, synonyms, "
    "and the key technical terms.\n"
    "Reply with ONLY the new query on one line."
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        ("human", "Original question: {question}\nPrevious search query: {previous_query}"),
    ]
)

UNRELIABLE_ANSWER_MESSAGE = (
    "I wrote an answer, but my check found claims that your sources don't support, even after "
    "trying again. So I won't show it, because it could be wrong. Try a narrower question, or "
    "read the sources below yourself."
)


# ---------- Small helpers ----------

def _ask(chain: Runnable, inputs: dict[str, str]) -> str:
    """Run one checker call and return its text reply."""
    return str(chain.invoke(inputs))


def _error(exc: Exception, settings: Settings, step: str) -> dict[str, Any]:
    logger.exception("Model call failed in %s", step)
    return {"error": describe_llm_error(exc, settings), "steps": [step]}


def _evidence_context(state: TutorState, kind: str) -> str:
    """The numbered sources for one kind of evidence, exactly as the answer writer saw them."""
    if kind == "web":
        return format_web_context(state.get("web_sources", []))
    return format_context(state.get("sources", []))


# ---------- Self-RAG step 1: grade each passage ----------

def make_grade_documents_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Ask the model, passage by passage: does this help answer the question?"""
    chain = GRADE_PROMPT | llm | StrOutputParser()

    def grade_documents(state: TutorState) -> dict[str, Any]:
        step = "grade_documents"
        approved = list(state.get("approved_sources", []))
        rejected = list(state.get("rejected_sources", []))
        already_graded = {chunk_key(chunk) for chunk in approved + rejected}
        # On a retry, only NEW passages are graded; earlier verdicts are kept.
        candidates = [chunk for chunk in state.get("sources", []) if chunk_key(chunk) not in already_graded]

        newly_approved, unclear = [], 0
        for chunk in candidates:
            try:
                verdict = parse_yes_no(_ask(chain, {"question": state["question"], "passage": chunk.text}))
            except Exception as exc:  # noqa: BLE001
                return _error(exc, settings, step)
            if verdict is None:
                unclear += 1  # an unclear reply is treated as "not relevant", to be safe
            if verdict:
                newly_approved.append(chunk)
            else:
                rejected.append(chunk)

        approved += newly_approved
        if candidates:
            note = f"Relevance check: {len(newly_approved)} of {len(candidates)} new passage(s) judged relevant"
            if unclear:
                note += f" ({unclear} unclear reply(ies) counted as not relevant)"
        else:
            note = "Relevance check: no new passages to check"
        logger.info("Graded %d passages: %d relevant", len(candidates), len(newly_approved))
        return {
            "sources": approved,
            "approved_sources": approved,
            "rejected_sources": rejected,
            "corrections": [note],
            "steps": [step],
        }

    return grade_documents


def make_grade_web_results_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """The same relevance check for web results (after the Phase 3 quality rules)."""
    chain = GRADE_PROMPT | llm | StrOutputParser()

    def grade_web_results(state: TutorState) -> dict[str, Any]:
        step = "grade_web_results"
        results = state.get("web_sources", [])
        rejected = list(state.get("rejected_web_sources", []))

        relevant, unclear = [], 0
        for result in results:
            passage = result.content[:WEB_CONTEXT_MAX_CHARS]
            try:
                verdict = parse_yes_no(_ask(chain, {"question": state["question"], "passage": passage}))
            except Exception as exc:  # noqa: BLE001
                return _error(exc, settings, step)
            if verdict is None:
                unclear += 1
            if verdict:
                relevant.append(result)
            else:
                reason = "Rejected: the relevance check judged it not useful for this question"
                rejected.append(replace(result, quality_notes=result.quality_notes + (reason,)))

        note = f"Relevance check (web): {len(relevant)} of {len(results)} result(s) judged relevant"
        if unclear:
            note += f" ({unclear} unclear reply(ies) counted as not relevant)"
        return {"web_sources": relevant, "rejected_web_sources": rejected, "corrections": [note], "steps": [step]}

    return grade_web_results


# ---------- Self-RAG step 2: is the evidence enough? ----------

def make_check_evidence_node(llm: Runnable, settings: Settings, kind: Literal["documents", "web"]) -> NodeFunction:
    """Ask: are these sources SUFFICIENT, INSUFFICIENT, or CONTRADICTORY?"""
    chain = EVIDENCE_PROMPT | llm | StrOutputParser()
    step = "check_document_evidence" if kind == "documents" else "check_web_evidence"
    label = "your documents" if kind == "documents" else "web results"

    def check_evidence(state: TutorState) -> dict[str, Any]:
        try:
            reply = _ask(chain, {"context": _evidence_context(state, kind), "question": state["question"]})
        except Exception as exc:  # noqa: BLE001
            return _error(exc, settings, step)

        verdict = parse_evidence_verdict(reply)
        note = f"Evidence check ({label}): {verdict}"
        if verdict is None:
            verdict = "insufficient"  # to be safe, an unclear reply means "not enough"
            note = f"Evidence check ({label}): unclear reply, counted as insufficient"
        logger.info("Evidence check (%s): %s", kind, verdict)
        return {"evidence_verdict": verdict, "corrections": [note], "steps": [step]}

    return check_evidence


# ---------- Corrective RAG: rewrite the query, or fall back to the web ----------

def make_rewrite_query_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Ask the model for a better search query for the learner's own documents."""
    chain = REWRITE_PROMPT | llm | StrOutputParser()

    def rewrite_query(state: TutorState) -> dict[str, Any]:
        step = "rewrite_query"
        previous = state.get("retrieval_query") or state["question"]
        try:
            reply = _ask(chain, {"question": state["question"], "previous_query": previous})
        except Exception as exc:  # noqa: BLE001
            return _error(exc, settings, step)

        new_query = clean_search_query(reply, fallback=previous)
        note = f"Rewrote the document search query: '{new_query}'"
        if new_query.lower() == previous.lower():
            note += " (same as before, so the wider search does the work)"
        logger.info("Query rewritten (%d characters)", len(new_query))
        return {"retrieval_query": new_query, "corrections": [note], "steps": [step]}

    return rewrite_query


def start_web_fallback(state: TutorState) -> dict[str, Any]:
    """Documents fell short and the learner allowed web search: switch evidence source.

    The document passages are cleared so the answer is built from web results only,
    keeping the two kinds of evidence separate.
    """
    logger.info("Falling back to web research")
    return {
        "used_web_fallback": True,
        "sources": [],
        "corrections": ["Your documents didn't have enough evidence, so the tutor searched the web (you allowed this)"],
        "steps": ["start_web_fallback"],
    }


# ---------- Self-RAG step 3: is the answer supported? ----------

def make_check_answer_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Check the answer: first with simple citation rules, then by asking the model."""
    chain = GROUNDING_PROMPT | llm | StrOutputParser()

    def check_answer(state: TutorState) -> dict[str, Any]:
        step = "check_answer"
        kind = state.get("answer_source", "documents")
        source_count = len(state.get("web_sources" if kind == "web" else "sources", []))
        answer = state.get("answer", "")

        # 1. Rule-based check: no citations, or citations to sources that don't exist.
        problem = citation_problem(answer, source_count, web=(kind == "web"))

        # 2. Model-based check, only if the citations look fine.
        if problem is None:
            try:
                verdict = parse_yes_no(_ask(chain, {"context": _evidence_context(state, kind), "answer": answer}))
            except Exception as exc:  # noqa: BLE001
                return _error(exc, settings, step)
            if verdict is True:
                logger.info("Answer check passed")
                return {
                    "grounded": True,
                    "grounding_problem": "",
                    "corrections": ["Answer check: every claim was judged supported by the sources"],
                    "steps": [step],
                }
            problem = (
                "the checker found claims the sources don't support"
                if verdict is False
                else "the checker's reply was unclear"
            )

        logger.info("Answer check failed")
        return {
            "grounded": False,
            "grounding_problem": problem,
            "corrections": [f"Answer check failed: {problem}"],
            "steps": [step],
        }

    return check_answer


def report_unreliable_answer(state: TutorState) -> dict[str, Any]:
    """No answer passed the check within the retry limit: don't show an unsupported answer."""
    logger.info("No answer passed the grounding check; not showing it")
    return {
        "answer": UNRELIABLE_ANSWER_MESSAGE,
        "insufficient_evidence": True,
        "unreliable_answer": True,
        "steps": ["report_unreliable_answer"],
    }


# ---------- Routers ----------

def _corrective_step(state: TutorState, settings: Settings, web_available: bool) -> str:
    """What to do when the documents fall short. Every option is bounded."""
    if state.get("retrieval_attempts", 0) <= settings.max_retrieval_retries:
        return "rewrite_query"  # try searching the documents again
    if web_available and state.get("allow_web_fallback") and not state.get("used_web_fallback"):
        return "start_web_fallback"  # the learner allowed a web search
    return "report_insufficient_evidence"  # out of options: say so honestly


def make_route_after_grading(settings: Settings, web_available: bool) -> Router:
    def route_after_grading(state: TutorState) -> str:
        """Relevant passages -> check if they're enough. None -> correct the search."""
        if state.get("error"):
            return "handle_error"
        if state.get("sources"):
            return "check_document_evidence"
        return _corrective_step(state, settings, web_available)

    return route_after_grading


def make_route_after_document_check(settings: Settings, web_available: bool) -> Router:
    def route_after_document_check(state: TutorState) -> str:
        """Enough (or disagreeing) evidence -> write the answer. Not enough -> correct the search."""
        if state.get("error"):
            return "handle_error"
        if state.get("evidence_verdict") in ("sufficient", "contradictory"):
            return "generate_grounded_answer"
        return _corrective_step(state, settings, web_available)

    return route_after_document_check


def route_after_rewrite(state: TutorState) -> Literal["retrieve_documents", "handle_error"]:
    """After rewriting the query, search the documents again."""
    return "handle_error" if state.get("error") else "retrieve_documents"


def route_after_web_grading(
    state: TutorState,
) -> Literal["check_web_evidence", "report_no_web_results", "handle_error"]:
    """Relevant web results -> check if they're enough. None -> say so (no web retries: saves quota)."""
    if state.get("error"):
        return "handle_error"
    return "check_web_evidence" if state.get("web_sources") else "report_no_web_results"


def route_after_web_check(
    state: TutorState,
) -> Literal["generate_web_answer", "report_no_web_results", "handle_error"]:
    if state.get("error"):
        return "handle_error"
    if state.get("evidence_verdict") in ("sufficient", "contradictory"):
        return "generate_web_answer"
    return "report_no_web_results"


def make_route_after_answer_check(settings: Settings) -> Router:
    def route_after_answer_check(state: TutorState) -> str:
        """Supported -> done. Not supported -> write again (bounded), then refuse."""
        if state.get("error"):
            return "handle_error"
        if state.get("grounded"):
            return END
        if state.get("answer_attempts", 0) < 1 + settings.max_answer_retries:
            return "generate_web_answer" if state.get("answer_source") == "web" else "generate_grounded_answer"
        return "report_unreliable_answer"

    return route_after_answer_check