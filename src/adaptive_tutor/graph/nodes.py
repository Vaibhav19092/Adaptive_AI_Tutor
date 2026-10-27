"""The functions (nodes) that LangGraph runs, plus the routing decisions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.state import TutorState
from adaptive_tutor.llm.provider import describe_llm_error
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
        ("human", "Context passages:\n\n{context}\n\nQuestion: {question}"),
    ]
)

NO_EVIDENCE_MESSAGE = (
    "I couldn't find information about this in your uploaded documents, so I can't give "
    "a reliable answer from them. Try rephrasing the question, adding a document that "
    "covers this topic, or switching to General knowledge."
)

NO_DOCUMENTS_MESSAGE = (
    "You haven't added any documents yet. Upload a PDF or text file in the sidebar, "
    "or switch to General knowledge."
)

NO_RETRIEVER_MESSAGE = "Document search is not set up in this app, so only General knowledge works."


# ---------- Shared helper ----------

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

        try:
            if retriever.document_count() == 0:
                return {"error": NO_DOCUMENTS_MESSAGE, "steps": step}
            chunks = retriever.search(state["question"])
        except Exception as exc:  # noqa: BLE001 - embedding or database failures
            logger.exception("Document retrieval failed")
            return {"error": describe_llm_error(exc, settings, model_name=settings.embedding_model), "steps": step}

        # Keep only passages that are similar enough to the question.
        relevant = [chunk for chunk in chunks if chunk.score >= settings.min_relevance_score]
        best_score = chunks[0].score if chunks else 0.0
        logger.info(
            "Retrieved %d passages; %d passed the %.2f threshold (best score %.2f)",
            len(chunks), len(relevant), settings.min_relevance_score, best_score,
        )
        return {"sources": relevant, "best_score": best_score, "steps": step}

    return retrieve_documents


def make_generate_grounded_answer_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Build a node that answers ONLY from the retrieved passages, with citations."""
    chain = GROUNDED_PROMPT | llm | StrOutputParser()

    def generate_grounded_answer(state: TutorState) -> dict[str, Any]:
        sources = state.get("sources", [])
        logger.info("Asking model '%s' with %d passages", settings.ollama_model, len(sources))
        inputs = {"question": state["question"], "context": format_context(sources)}
        return _invoke_chain(chain, inputs, settings, "generate_grounded_answer")

    return generate_grounded_answer


def report_insufficient_evidence(state: TutorState) -> dict[str, Any]:
    """The documents don't cover the question: say so honestly instead of guessing."""
    logger.info("No passage was relevant enough; not calling the model")
    return {
        "answer": NO_EVIDENCE_MESSAGE,
        "insufficient_evidence": True,
        "steps": ["report_insufficient_evidence"],
    }


def handle_error(state: TutorState) -> dict[str, Any]:
    """Final stop for any failure. The error message is already in the state."""
    logger.warning("Workflow stopped with an error: %s", state.get("error"))
    return {"steps": ["handle_error"]}


# ---------- Routing decisions (conditional edges) ----------

def route_after_validation(
    state: TutorState,
) -> Literal["generate_answer", "retrieve_documents", "handle_error"]:
    """Bad question -> error. Document mode -> search first. Otherwise answer directly."""
    if state.get("error"):
        return "handle_error"
    if state.get("use_documents"):
        return "retrieve_documents"
    return "generate_answer"


def route_after_retrieval(
    state: TutorState,
) -> Literal["generate_grounded_answer", "report_insufficient_evidence", "handle_error"]:
    """Error -> error. No relevant passages -> say so. Otherwise answer from them."""
    if state.get("error"):
        return "handle_error"
    if not state.get("sources"):
        return "report_insufficient_evidence"
    return "generate_grounded_answer"


def route_after_generation(state: TutorState) -> str:
    """Finish normally, or go to error handling."""
    return "handle_error" if state.get("error") else END