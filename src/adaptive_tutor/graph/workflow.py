"""Wire the nodes together into a LangGraph workflow."""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.nodes import (
    handle_error,
    make_generate_answer_node,
    make_generate_grounded_answer_node,
    make_retrieve_node,
    make_validate_question_node,
    report_insufficient_evidence,
    route_after_generation,
    route_after_retrieval,
    route_after_validation,
)
from adaptive_tutor.graph.state import TutorState
from adaptive_tutor.retrieval.retriever import Retriever


def build_tutor_graph(
    llm: Runnable,
    settings: Settings,
    retriever: Retriever | None = None,
) -> CompiledStateGraph:
    """Create the Phase 2 workflow.

    START -> validate_question --(general)--> generate_answer ----------> END
                      |
                      +--(documents)--> retrieve_documents
                                           |-- relevant --> generate_grounded_answer --> END
                                           +-- none -----> report_insufficient_evidence --> END
    Any error at any step goes to handle_error -> END.
    """
    graph = StateGraph(TutorState)

    graph.add_node("validate_question", make_validate_question_node(settings.max_question_length))
    graph.add_node("generate_answer", make_generate_answer_node(llm, settings))
    graph.add_node("retrieve_documents", make_retrieve_node(retriever, settings))
    graph.add_node("generate_grounded_answer", make_generate_grounded_answer_node(llm, settings))
    graph.add_node("report_insufficient_evidence", report_insufficient_evidence)
    graph.add_node("handle_error", handle_error)

    graph.add_edge(START, "validate_question")
    graph.add_conditional_edges(
        "validate_question",
        route_after_validation,
        {
            "generate_answer": "generate_answer",
            "retrieve_documents": "retrieve_documents",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieval,
        {
            "generate_grounded_answer": "generate_grounded_answer",
            "report_insufficient_evidence": "report_insufficient_evidence",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {"handle_error": "handle_error", END: END},
    )
    graph.add_conditional_edges(
        "generate_grounded_answer",
        route_after_generation,
        {"handle_error": "handle_error", END: END},
    )
    graph.add_edge("report_insufficient_evidence", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


def ask_tutor(graph: CompiledStateGraph, question: str, use_documents: bool = False) -> TutorState:
    """Run the workflow once and return the final state."""
    return graph.invoke({"question": question, "use_documents": use_documents, "steps": []})