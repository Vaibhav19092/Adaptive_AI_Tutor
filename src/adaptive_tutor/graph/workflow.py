"""Wire the nodes together into a LangGraph workflow."""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.checking_nodes import (
    make_check_answer_node,
    make_check_evidence_node,
    make_grade_documents_node,
    make_grade_web_results_node,
    make_rewrite_query_node,
    make_route_after_answer_check,
    make_route_after_document_check,
    make_route_after_grading,
    report_unreliable_answer,
    route_after_rewrite,
    route_after_web_check,
    route_after_web_grading,
    start_web_fallback,
)
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

# LangGraph stops any run after this many steps, as a last safety net against loops.
# The longest possible run with the largest retry settings is about 25 steps.
RECURSION_LIMIT = 50


def build_tutor_graph(
    llm: Runnable,
    settings: Settings,
    retriever: Retriever | None = None,
    search_provider: SearchProvider | None = None,
    page_fetcher: PageFetcher | None = None,
) -> CompiledStateGraph:
    """Create the Phase 4 workflow.

    Basic mode (check_evidence=False) is exactly the Phase 3 graph.
    Checked mode adds Self-RAG checks and Corrective RAG loops:

    documents: retrieve_documents -> grade_documents -> check_document_evidence
                  -> generate_grounded_answer -> check_answer -> END
               if passages are irrelevant or not enough:
                  rewrite_query -> retrieve_documents (wider)        [max MAX_RETRIEVAL_RETRIES]
                  then, if allowed: start_web_fallback -> web branch
                  else: report_insufficient_evidence
    web:       search_web -> grade_web_results -> check_web_evidence
                  -> generate_web_answer -> check_answer -> END
    answers:   if check_answer fails, write again                    [max MAX_ANSWER_RETRIES]
                  then: report_unreliable_answer
    Any error at any step goes to handle_error -> END.
    """
    web_available = search_provider is not None
    graph = StateGraph(TutorState)

    # Phase 1-3 nodes
    graph.add_node("validate_question", make_validate_question_node(settings.max_question_length))
    graph.add_node("generate_answer", make_generate_answer_node(llm, settings))
    graph.add_node("retrieve_documents", make_retrieve_node(retriever, settings))
    graph.add_node("generate_grounded_answer", make_generate_grounded_answer_node(llm, settings))
    graph.add_node("report_insufficient_evidence", report_insufficient_evidence)
    graph.add_node("handle_error", handle_error)
    graph.add_node("generate_search_query", make_generate_search_query_node(llm, settings))
    graph.add_node("search_web", make_search_web_node(search_provider, settings, page_fetcher))
    graph.add_node("generate_web_answer", make_generate_web_answer_node(llm, settings))
    graph.add_node("report_no_web_results", report_no_web_results)

    # Phase 4 nodes
    graph.add_node("grade_documents", make_grade_documents_node(llm, settings))
    graph.add_node("check_document_evidence", make_check_evidence_node(llm, settings, "documents"))
    graph.add_node("rewrite_query", make_rewrite_query_node(llm, settings))
    graph.add_node("start_web_fallback", start_web_fallback)
    graph.add_node("grade_web_results", make_grade_web_results_node(llm, settings))
    graph.add_node("check_web_evidence", make_check_evidence_node(llm, settings, "web"))
    graph.add_node("check_answer", make_check_answer_node(llm, settings))
    graph.add_node("report_unreliable_answer", report_unreliable_answer)

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

    # ----- Document branch -----
    graph.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieval,
        {
            "grade_documents": "grade_documents",
            "generate_grounded_answer": "generate_grounded_answer",
            "report_insufficient_evidence": "report_insufficient_evidence",
            "handle_error": "handle_error",
        },
    )
    # What to do when documents fall short (shared by the two routers below).
    corrective_targets = {
        "rewrite_query": "rewrite_query",
        "start_web_fallback": "start_web_fallback",
        "report_insufficient_evidence": "report_insufficient_evidence",
        "handle_error": "handle_error",
    }
    graph.add_conditional_edges(
        "grade_documents",
        make_route_after_grading(settings, web_available),
        {**corrective_targets, "check_document_evidence": "check_document_evidence"},
    )
    graph.add_conditional_edges(
        "check_document_evidence",
        make_route_after_document_check(settings, web_available),
        {**corrective_targets, "generate_grounded_answer": "generate_grounded_answer"},
    )
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"retrieve_documents": "retrieve_documents", "handle_error": "handle_error"},
    )
    graph.add_edge("start_web_fallback", "generate_search_query")

    # ----- Web branch -----
    graph.add_conditional_edges(
        "generate_search_query",
        route_after_search_query,
        {"search_web": "search_web", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "search_web",
        route_after_web_search,
        {
            "grade_web_results": "grade_web_results",
            "generate_web_answer": "generate_web_answer",
            "report_no_web_results": "report_no_web_results",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "grade_web_results",
        route_after_web_grading,
        {
            "check_web_evidence": "check_web_evidence",
            "report_no_web_results": "report_no_web_results",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "check_web_evidence",
        route_after_web_check,
        {
            "generate_web_answer": "generate_web_answer",
            "report_no_web_results": "report_no_web_results",
            "handle_error": "handle_error",
        },
    )

    # ----- Answers -----
    # Every answer writer either finishes, goes to the answer check, or reports an error.
    for answer_node in ("generate_answer", "generate_grounded_answer", "generate_web_answer"):
        graph.add_conditional_edges(
            answer_node,
            route_after_generation,
            {"check_answer": "check_answer", "handle_error": "handle_error", END: END},
        )
    graph.add_conditional_edges(
        "check_answer",
        make_route_after_answer_check(settings),
        {
            "generate_grounded_answer": "generate_grounded_answer",
            "generate_web_answer": "generate_web_answer",
            "report_unreliable_answer": "report_unreliable_answer",
            "handle_error": "handle_error",
            END: END,
        },
    )

    for final_node in ("report_insufficient_evidence", "report_no_web_results", "report_unreliable_answer", "handle_error"):
        graph.add_edge(final_node, END)

    return graph.compile()


def ask_tutor(
    graph: CompiledStateGraph,
    question: str,
    use_documents: bool = False,
    use_web_research: bool = False,
    check_evidence: bool = False,
    allow_web_fallback: bool = False,
) -> TutorState:
    """Run the workflow once and return the final state.

    check_evidence=True turns on the Phase 4 checks (only for documents or web research).
    allow_web_fallback=True lets a document question search the web if documents fall short.
    """
    if use_documents and use_web_research:
        raise ValueError("Choose either documents or web research for a question, not both.")
    uses_sources = use_documents or use_web_research
    return graph.invoke(
        {
            "question": question,
            "use_documents": use_documents,
            "use_web_research": use_web_research,
            "check_evidence": check_evidence and uses_sources,
            "allow_web_fallback": allow_web_fallback and check_evidence and use_documents,
            "steps": [],
            "corrections": [],
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )