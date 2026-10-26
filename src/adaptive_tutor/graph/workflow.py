"""Wire the nodes together into a LangGraph workflow."""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.nodes import (
    handle_error,
    make_generate_answer_node,
    make_validate_question_node,
    route_after_generation,
    route_after_validation,
)
from adaptive_tutor.graph.state import TutorState


def build_tutor_graph(llm: Runnable, settings: Settings) -> CompiledStateGraph:
    """Create the Phase 1 workflow.

    START -> validate_question -> generate_answer -> END
                    |                   |
                    +--> handle_error <-+ --> END
    """
    graph = StateGraph(TutorState)

    graph.add_node("validate_question", make_validate_question_node(settings.max_question_length))
    graph.add_node("generate_answer", make_generate_answer_node(llm, settings))
    graph.add_node("handle_error", handle_error)

    graph.add_edge(START, "validate_question")
    graph.add_conditional_edges(
        "validate_question",
        route_after_validation,
        {"generate_answer": "generate_answer", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {"handle_error": "handle_error", END: END},
    )
    graph.add_edge("handle_error", END)

    return graph.compile()


def ask_tutor(graph: CompiledStateGraph, question: str) -> TutorState:
    """Run the workflow once and return the final state."""
    return graph.invoke({"question": question, "steps": []})