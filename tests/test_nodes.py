from langgraph.graph import END

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.checking_nodes import (
    make_route_after_answer_check,
    make_route_after_document_check,
    make_route_after_grading,
    route_after_rewrite,
    route_after_web_check,
    route_after_web_grading,
)
from adaptive_tutor.graph.nodes import (
    make_validate_question_node,
    route_after_generation,
    route_after_retrieval,
    route_after_search_query,
    route_after_validation,
    route_after_web_search,
)
from adaptive_tutor.research.search_provider import WebResult
from adaptive_tutor.retrieval.retriever import RetrievedChunk


def make_chunk(score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        text="Cells are the basic unit of life.",
        source="biology.pdf",
        page=1,
        chunk_index=0,
        document_id="doc1",
        score=score,
    )


# ---------- Phase 1 ----------

def test_validate_question_strips_whitespace():
    validate = make_validate_question_node(max_length=100)
    result = validate({"question": "   What is RAG?   "})
    assert result == {"question": "What is RAG?", "steps": ["validate_question"]}


def test_validate_question_rejects_empty_question():
    validate = make_validate_question_node(max_length=100)
    result = validate({"question": "   "})
    assert result["error"] == "Please type a question first."


def test_validate_question_rejects_long_question():
    validate = make_validate_question_node(max_length=10)
    result = validate({"question": "x" * 11})
    assert "under 10 characters" in result["error"]


def test_route_after_validation():
    assert route_after_validation({"question": "ok"}) == "generate_answer"
    assert route_after_validation({"error": "bad"}) == "handle_error"


def test_route_after_generation():
    assert route_after_generation({"answer": "hi"}) == END
    assert route_after_generation({"error": "bad"}) == "handle_error"


# ---------- Phase 2 ----------

def test_route_after_validation_in_document_mode():
    assert route_after_validation({"question": "ok", "use_documents": True}) == "retrieve_documents"
    # An invalid question never reaches retrieval, even in document mode.
    assert route_after_validation({"error": "bad", "use_documents": True}) == "handle_error"


def test_route_after_retrieval():
    assert route_after_retrieval({"sources": [make_chunk()]}) == "generate_grounded_answer"
    assert route_after_retrieval({"sources": []}) == "report_insufficient_evidence"
    assert route_after_retrieval({}) == "report_insufficient_evidence"
    assert route_after_retrieval({"error": "bad"}) == "handle_error"


# ---------- Phase 3 ----------

def make_web_result() -> WebResult:
    return WebResult(title="Cell", url="https://en.wikipedia.org/wiki/Cell", content="Cells...", provider="Wikipedia", rank=1)


def test_route_after_validation_in_web_mode():
    assert route_after_validation({"question": "ok", "use_web_research": True}) == "generate_search_query"
    assert route_after_validation({"error": "bad", "use_web_research": True}) == "handle_error"


def test_route_after_search_query():
    assert route_after_search_query({"search_query": "cells"}) == "search_web"
    assert route_after_search_query({"error": "bad"}) == "handle_error"


def test_route_after_web_search():
    assert route_after_web_search({"web_sources": [make_web_result()]}) == "generate_web_answer"
    assert route_after_web_search({"web_sources": []}) == "report_no_web_results"
    assert route_after_web_search({}) == "report_no_web_results"
    assert route_after_web_search({"error": "bad"}) == "handle_error"


# ---------- Phase 4 ----------

SETTINGS = Settings(
    ollama_base_url="http://localhost:11434",
    ollama_model="m",
    temperature=0.0,
    request_timeout_seconds=10.0,
    max_question_length=100,
    log_level="INFO",
)  # max_retrieval_retries=1, max_answer_retries=1


def test_checking_adds_branches_to_the_old_routers():
    assert route_after_retrieval({"check_evidence": True, "sources": []}) == "grade_documents"
    assert route_after_web_search({"check_evidence": True, "web_sources": [make_web_result()]}) == "grade_web_results"
    assert route_after_web_search({"check_evidence": True, "web_sources": []}) == "report_no_web_results"
    assert route_after_generation({"check_evidence": True, "answer_source": "documents"}) == "check_answer"
    # A general-knowledge answer has no answer_source, so it is never checked.
    assert route_after_generation({"check_evidence": True, "answer": "hi"}) == END


def test_route_after_grading_corrects_the_search():
    route = make_route_after_grading(SETTINGS, web_available=True)
    assert route({"sources": [make_chunk()]}) == "check_document_evidence"
    assert route({"sources": [], "retrieval_attempts": 1}) == "rewrite_query"
    assert route({"sources": [], "retrieval_attempts": 2}) == "report_insufficient_evidence"
    assert route({"sources": [], "retrieval_attempts": 2, "allow_web_fallback": True}) == "start_web_fallback"
    assert route({"error": "bad"}) == "handle_error"


def test_web_fallback_needs_a_provider_and_happens_only_once():
    no_web = make_route_after_grading(SETTINGS, web_available=False)
    assert no_web({"sources": [], "retrieval_attempts": 2, "allow_web_fallback": True}) == "report_insufficient_evidence"
    route = make_route_after_grading(SETTINGS, web_available=True)
    state = {"sources": [], "retrieval_attempts": 2, "allow_web_fallback": True, "used_web_fallback": True}
    assert route(state) == "report_insufficient_evidence"


def test_route_after_document_check():
    route = make_route_after_document_check(SETTINGS, web_available=False)
    assert route({"evidence_verdict": "sufficient"}) == "generate_grounded_answer"
    assert route({"evidence_verdict": "contradictory"}) == "generate_grounded_answer"
    assert route({"evidence_verdict": "insufficient", "retrieval_attempts": 1}) == "rewrite_query"
    assert route({"evidence_verdict": "insufficient", "retrieval_attempts": 2}) == "report_insufficient_evidence"


def test_web_check_routers():
    assert route_after_rewrite({"retrieval_query": "q"}) == "retrieve_documents"
    assert route_after_rewrite({"error": "bad"}) == "handle_error"
    assert route_after_web_grading({"web_sources": [make_web_result()]}) == "check_web_evidence"
    assert route_after_web_grading({"web_sources": []}) == "report_no_web_results"
    assert route_after_web_check({"evidence_verdict": "sufficient"}) == "generate_web_answer"
    assert route_after_web_check({"evidence_verdict": "insufficient"}) == "report_no_web_results"


def test_route_after_answer_check_is_bounded():
    route = make_route_after_answer_check(SETTINGS)
    assert route({"grounded": True}) == END
    assert route({"grounded": False, "answer_attempts": 1, "answer_source": "documents"}) == "generate_grounded_answer"
    assert route({"grounded": False, "answer_attempts": 1, "answer_source": "web"}) == "generate_web_answer"
    assert route({"grounded": False, "answer_attempts": 2, "answer_source": "documents"}) == "report_unreliable_answer"
    assert route({"error": "bad"}) == "handle_error"