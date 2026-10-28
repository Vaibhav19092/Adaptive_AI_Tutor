"""Phase 9: evaluation metrics, computed from local events."""

from adaptive_tutor.evaluation.metrics import (
    activity_event_data,
    check_adaptation,
    grading_consistency,
    percentile,
    question_event_data,
    rate,
    summarize_events,
)
from adaptive_tutor.retrieval.retriever import RetrievedChunk


def chunk(n):
    return RetrievedChunk("text", "bio.pdf", n, n, "doc", 0.8)


def test_rate_and_percentile():
    assert rate(1, 4) == 0.25 and rate(0, 0) is None
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) is None


def test_question_event_contains_no_text():
    state = {
        "question": "secret question", "answer": "Mitochondria make energy [1].", "use_documents": True,
        "check_evidence": True, "answer_source": "documents", "grounded": True,
        "sources": [chunk(1)], "approved_sources": [chunk(1)], "rejected_sources": [chunk(2)], "retrieval_attempts": 2,
    }
    data = question_event_data(state)
    assert data == {
        "mode": "documents", "checked": True, "error": False, "refused": False, "relevant": 1, "graded": 2,
        "retrieval_attempts": 2, "grounded": True, "citation_ok": True,
    }
    assert "secret" not in repr(data)


def test_bad_citation_is_counted():
    state = {"answer": "Energy [3].", "use_documents": True, "answer_source": "documents", "sources": [chunk(1)]}
    assert question_event_data(state)["citation_ok"] is False


def test_summary_of_events():
    events = [
        {"kind": "question", "duration_ms": 2000, "data": {"relevant": 2, "graded": 4, "grounded": True, "citation_ok": True}},
        {"kind": "question", "duration_ms": 4000, "data": {"relevant": 1, "graded": 1, "grounded": False, "citation_ok": None}},
        {"kind": "grading", "duration_ms": 1000, "data": {}},
    ]
    summary = summarize_events(events)
    assert summary["retrieval_relevance"] == 0.6
    assert summary["groundedness"] == 0.5
    assert summary["citation_correctness"] == 1.0
    assert summary["latency"]["question"] == {"runs": 2, "median_s": 3.0, "p90_s": 3.8}


def test_grading_consistency():
    assert grading_consistency([0.5, 0.5, 1.0]) == {"runs": 3, "agreement": 0.667, "spread": 0.5}
    assert grading_consistency([1.0, 1.0])["agreement"] == 1.0
    assert grading_consistency([])["agreement"] is None


def test_activity_event_data():
    state = {"chosen_action": "lesson", "evidence_kind": "documents", "evidence": [{}, {}], "rejected_evidence": [{}],
             "lesson_grounded": True, "quiz": {"questions": [1, 2, 3]}, "assessment": {"score": 0.75}, "check_evidence": True}
    data = activity_event_data(state)
    assert (data["relevant"], data["graded"], data["grounded"], data["questions"], data["score"]) == (2, 3, True, 3, 0.75)


def test_every_adaptation_scenario_behaves_as_the_rules_say(today):
    results = check_adaptation(today)
    assert len(results) == 5
    failed = [(r.name, r.expected, r.got) for r in results if not r.passed]
    assert failed == []