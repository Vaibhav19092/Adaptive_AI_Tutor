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

logger = logging.getLogger(__name__)

NodeFunction = Callable[[TutorState], dict[str, Any]]

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
    """Build a node that asks the model to answer the question."""
    # LCEL "pipe": fill the prompt -> send to the model -> extract plain text.
    chain = TUTOR_PROMPT | llm | StrOutputParser()

    def generate_answer(state: TutorState) -> dict[str, Any]:
        logger.info("Asking model '%s'", settings.ollama_model)
        try:
            answer = chain.invoke({"question": state["question"]})
        except Exception as exc:  # noqa: BLE001 - every failure becomes a friendly message
            logger.exception("Model call failed")
            return {"error": describe_llm_error(exc, settings), "steps": ["generate_answer"]}

        answer = answer.strip()
        if not answer:
            return {
                "error": "The model returned an empty answer. Try rephrasing your question.",
                "steps": ["generate_answer"],
            }

        logger.info("Answer generated (%d characters)", len(answer))
        return {"answer": answer, "steps": ["generate_answer"]}

    return generate_answer


def handle_error(state: TutorState) -> dict[str, Any]:
    """Final stop for any failure. The error message is already in the state."""
    logger.warning("Workflow stopped with an error: %s", state.get("error"))
    return {"steps": ["handle_error"]}


def route_after_validation(state: TutorState) -> Literal["generate_answer", "handle_error"]:
    """Conditional edge: only call the model if the question is valid."""
    return "handle_error" if state.get("error") else "generate_answer"


def route_after_generation(state: TutorState) -> str:
    """Conditional edge: finish normally, or go to error handling."""
    return "handle_error" if state.get("error") else END