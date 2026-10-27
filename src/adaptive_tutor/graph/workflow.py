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
    make_generate_search_query_node,
    make_generate_web_answer_node,
    make_retrieve_node,
    make_search_web_node,
    make_validate_question_node,
    report_insufficient_evidence,
    report_no_web_results,
    route_after_generation,
    route_after_retrieval,
    route_after_search_query,
    route_after_validation,
    route_after_web_search,
)
from adaptive_tutor.graph.state import TutorState
from adaptive_tutor.research.content_extraction import PageFetcher
from adaptive_tutor.research.search_provider import SearchProvider
from adaptive_tutor.retrieval.retriever import Retriever


def build_tutor_graph(
    llm: Runnable,
    settings: Settings,
    retriever: Retriever | None = None,
    search_provider: SearchProvider | None = None,
    page_fetcher: PageFetcher | None = None,
) -> CompiledStateGraph:
    """Create the Phase 3 workflow.

    START -> validate_question
       |-- general ---> generate_answer ---------------------------------------> END
       |-- documents -> retrieve_documents
       |                  |-- relevant --> generate_grounded_answer -----------> END
       |                  +-- none ------> report_insufficient_evidence -------> END
       +-- web -------> generate_search_query -> search_web
                                                  |-- good --> generate_web_answer -> END
                                                  +-- none --> report_no_web_results -> END
    Any error at any step goes to handle_error -> END.
    """
    graph = StateGraph(TutorState)

    # Phase 1 and 2 nodes
    graph.add_node("validate_question", make_validate_question_node(settings.max_question_length))
    graph.add_node("generate_answer", make_generate_answer_node(llm, settings))
    graph.add_node("retrieve_documents", make_retrieve_node(retriever, settings))
    graph.add_node("generate_grounded_answer", make_generate_grounded_answer_node(llm, settings))
    graph.add_node("report_insufficient_evidence", report_insufficient_evidence)
    graph.add_node("handle_error", handle_error)

    # Phase 3 nodes
    graph.add_node("generate_search_query", make_generate_search_query_node(llm, settings))
    graph.add_node("search_web", make_search_web_node(search_provider, settings, page_fetcher))
    graph.add_node("generate_web_answer", make_generate_web_answer_node(llm, settings))
    graph.add_node("report_no_web_results", report_no_web_results)

    graph.add_edge(START, "validate_question")
    graph.add_conditional_edges(
        "validate_question",
        route_after_validation,
        {
            "generate_answer": "generate_answer",
            "retrieve_documents": "retrieve_documents",
            "generate_search_query": "generate_search_query",
            "handle_error": "handle_error",
        },
    )

    # Document branch (Phase 2)
    graph.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieval,
        {
            "generate_grounded_answer": "generate_grounded_answer",
            "report_insufficient_evidence": "report_insufficient_evidence",
            "handle_error": "handle_error",
        },
    )

    # Web branch (Phase 3)
    graph.add_conditional_edges(
        "generate_search_query",
        route_after_search_query,
        {"search_web": "search_web", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "search_web",
        route_after_web_search,
        {
            "generate_web_answer": "generate_web_answer",
            "report_no_web_results": "report_no_web_results",
            "handle_error": "handle_error",
        },
    )

    # Every answer-writing node either finishes or reports an error.
    for answer_node in ("generate_answer", "generate_grounded_answer", "generate_web_answer"):
        graph.add_conditional_edges(
            answer_node,
            route_after_generation,
            {"handle_error": "handle_error", END: END},
        )

    graph.add_edge("report_insufficient_evidence", END)
    graph.add_edge("report_no_web_results", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


def ask_tutor(
    graph: CompiledStateGraph,
    question: str,
    use_documents: bool = False,
    use_web_research: bool = False,
) -> TutorState:
    """Run the workflow once and return the final state.

    Choose at most one evidence source per question, so document evidence
    and web evidence never get mixed up.
    """
    if use_documents and use_web_research:
        raise ValueError("Choose either documents or web research for a question, not both.")
    return graph.invoke(
        {
            "question": question,
            "use_documents": use_documents,
            "use_web_research": use_web_research,
            "steps": [],
        }
    )