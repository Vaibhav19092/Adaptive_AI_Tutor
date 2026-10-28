"""Phase 9: local evaluation metrics. No paid services; everything stays in data/tutor.sqlite.

Each run of a workflow stores one small "event": how long it took plus a few
numbers and yes/no flags (never the text of questions, answers, or documents).
The dashboard turns those events into these metrics:

- Retrieval relevance:  passages the checker judged relevant / passages it graded
- Answer groundedness:  checked answers or lessons that passed / all that were checked
- Citation correctness: answers whose citations all point to real sources / answers with sources
- Response latency:     median and 90th-percentile time per kind of run
- Quiz grading consistency: grade the same answer several times; how often do the grades agree?
- Adaptation correctness: fixed example profiles; does the planner recommend what the rules say?

These are simple, honest indicators for a learning project, not validated benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from adaptive_tutor.learner.learning_planner import recommend
from adaptive_tutor.learner.mastery import ConceptProgress
from adaptive_tutor.retrieval.self_rag import citation_problem


def rate(numerator: int, denominator: int) -> float | None:
    """numerator / denominator, or None when there is nothing to measure yet."""
    return round(numerator / denominator, 3) if denominator else None


def percentile(values: list[float], fraction: float) -> float | None:
    """A simple percentile: percentile([1, 2, 3, 4], 0.5) -> 2.5."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)


# ---------- What one run records (numbers and flags only) ----------

def question_event_data(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize one Phase 1-4 question run from its final state."""
    answer_source = result.get("answer_source")
    data: dict[str, Any] = {
        "mode": "web" if result.get("use_web_research") else "documents" if result.get("use_documents") else "general",
        "checked": bool(result.get("check_evidence")),
        "error": bool(result.get("error")),
        "refused": bool(result.get("insufficient_evidence")),
        "relevant": len(result.get("approved_sources") or []),
        "graded": len(result.get("approved_sources") or []) + len(result.get("rejected_sources") or []),
        "retrieval_attempts": int(result.get("retrieval_attempts") or 0),
        "grounded": result.get("grounded") if result.get("check_evidence") and answer_source else None,
        "citation_ok": None,
    }
    if answer_source and not data["error"] and not data["refused"]:
        web = answer_source == "web"
        count = len(result.get("web_sources" if web else "sources") or [])
        data["citation_ok"] = citation_problem(result.get("answer", ""), count, web=web) is None
    return data


def activity_event_data(state: dict[str, Any]) -> dict[str, Any]:
    """Summarize one learning-workflow run from its state."""
    evidence = state.get("evidence") or []
    rejected = state.get("rejected_evidence") or []
    assessment = state.get("assessment") or {}
    checked_lesson = "lesson_grounded" in state
    return {
        "action": state.get("chosen_action", ""),
        "evidence_kind": state.get("evidence_kind", ""),
        "checked": bool(state.get("check_evidence")),
        "error": bool(state.get("error")),
        "refused": bool(state.get("refused")),
        "relevant": len(evidence) if state.get("evidence_kind") == "documents" else 0,
        "graded": (len(evidence) + len(rejected)) if state.get("evidence_kind") == "documents" else 0,
        "grounded": bool(state.get("lesson_grounded")) if checked_lesson else None,
        "questions": len((state.get("quiz") or {}).get("questions", [])),
        "score": assessment.get("score"),
    }


# ---------- Summaries for the dashboard ----------

def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = graded = grounded_yes = grounded_checked = citations_ok = citations_checked = 0
    latency: dict[str, list[float]] = {}
    for event in events:
        data = event.get("data") or {}
        latency.setdefault(event["kind"], []).append(event["duration_ms"] / 1000)
        relevant += int(data.get("relevant") or 0)
        graded += int(data.get("graded") or 0)
        if data.get("grounded") is not None:
            grounded_checked += 1
            grounded_yes += 1 if data["grounded"] else 0
        if data.get("citation_ok") is not None:
            citations_checked += 1
            citations_ok += 1 if data["citation_ok"] else 0
    return {
        "runs": len(events),
        "retrieval_relevance": rate(relevant, graded),
        "passages_graded": graded,
        "groundedness": rate(grounded_yes, grounded_checked),
        "grounding_checks": grounded_checked,
        "citation_correctness": rate(citations_ok, citations_checked),
        "citation_checks": citations_checked,
        "latency": {
            kind: {"runs": len(times), "median_s": percentile(times, 0.5), "p90_s": percentile(times, 0.9)}
            for kind, times in sorted(latency.items())
        },
    }


def grading_consistency(score_runs: list[float]) -> dict[str, Any]:
    """Grades of the SAME answer from several runs, e.g. [0.5, 0.5, 1.0].

    agreement = share of runs that gave the most common grade (1.0 = perfectly consistent).
    """
    if not score_runs:
        return {"runs": 0, "agreement": None, "spread": None}
    rounded = [round(score, 2) for score in score_runs]
    most_common = max(set(rounded), key=rounded.count)
    return {
        "runs": len(rounded),
        "agreement": round(rounded.count(most_common) / len(rounded), 3),
        "spread": round(max(rounded) - min(rounded), 3),
    }


# ---------- Adaptation correctness: does the planner follow its own rules? ----------

@dataclass(frozen=True)
class ScenarioResult:
    name: str
    expected: str
    got: str

    @property
    def passed(self) -> bool:
        return self.expected == self.got


def _scenarios(today: date) -> list[tuple[str, list[ConceptProgress], dict[str, list[str]], str]]:
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=8)
    return [
        (
            "Struggling learner gets remediation",
            [ConceptProgress("photosynthesis", "biology", mastery=0.2, attempts=2, last_reviewed=yesterday, next_review=today)],
            {},
            "remediate photosynthesis",
        ),
        (
            "Weak prerequisite is taught first",
            [
                ConceptProgress("backpropagation", "neural networks", mastery=0.1, attempts=2, last_reviewed=yesterday, next_review=today),
                ConceptProgress("derivatives", "neural networks", mastery=0.0, attempts=0),
            ],
            {"backpropagation": ["derivatives"]},
            "prerequisite derivatives",
        ),
        (
            "Half-learned concept gets practice",
            [ConceptProgress("embeddings", "rag", mastery=0.6, attempts=3, last_reviewed=yesterday, next_review=today)],
            {},
            "practice embeddings",
        ),
        (
            "Mastered concept is revised when due",
            [ConceptProgress("chunking", "rag", mastery=0.9, attempts=5, streak=3, last_reviewed=week_ago, next_review=yesterday)],
            {},
            "revise chunking",
        ),
        (
            "Fully mastered topic leads to advancement",
            [ConceptProgress("cells", "biology", mastery=0.9, attempts=5, streak=3, last_reviewed=yesterday,
                             next_review=today + timedelta(days=6))],
            {},
            "advance biology",
        ),
    ]


def check_adaptation(today: date) -> list[ScenarioResult]:
    """Run the planner on fixed example profiles and compare with the expected first suggestion."""
    results = []
    for name, progress, prerequisites, expected in _scenarios(today):
        suggestions = recommend(progress, prerequisites, today)
        got = f"{suggestions[0].action} {suggestions[0].concept}" if suggestions else "nothing"
        results.append(ScenarioResult(name, expected, got))
    return results