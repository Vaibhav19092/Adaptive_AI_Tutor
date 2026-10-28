"""The "My progress" tab: the learning dashboard and local evaluation metrics (Phase 9)."""

from __future__ import annotations

import json
import logging
from datetime import date

import httpx
import streamlit as st

from adaptive_tutor.assessment.schemas import Quiz, ShortAnswerQuestion
from adaptive_tutor.config import Settings
from adaptive_tutor.evaluation.metrics import check_adaptation, grading_consistency, summarize_events
from adaptive_tutor.evaluation.resources import ollama_loaded_models, process_memory_mb
from adaptive_tutor.learner.learning_planner import recommend
from adaptive_tutor.learner.mastery import concept_status
from adaptive_tutor.learner.profile import LearnerProfile
from adaptive_tutor.services import TutorServices

logger = logging.getLogger("adaptive_tutor.app")

STATUS_LABELS = {
    "new": "🆕 not quizzed yet",
    "remediate": "🔁 needs re-teaching",
    "practice": "✍️ practising",
    "mastered": "✅ mastered",
    "revise": "🗓️ due for revision",
}
CONSISTENCY_RUNS = 3


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _latest_short_answer(services: TutorServices, learner_id: str, history: list[dict]):
    """The most recent graded short answer, with its question, for the consistency check."""
    for item in history:
        if item["status"] != "graded" or not item.get("result"):
            continue
        quiz_data = services.repository.get_quiz(learner_id, item["quiz_id"])
        if not quiz_data:
            continue
        questions = {q.question_id: q for q in Quiz.from_dict(quiz_data).questions}
        for result in item["result"]["results"]:
            question = questions.get(result["question_id"])
            if isinstance(question, ShortAnswerQuestion) and result["learner_answer"] and result["graded"]:
                return question, result["learner_answer"]
    return None


def _evaluation_section(services: TutorServices, settings: Settings, learner: LearnerProfile, history: list[dict]) -> None:
    st.subheader("Evaluation (measured on this computer)")
    st.caption("Simple indicators for a learning project, not validated benchmarks. Only numbers are stored, never your text.")
    summary = summarize_events(services.repository.list_events(learner_id=learner.learner_id))
    columns = st.columns(3)
    columns[0].metric("Retrieval relevance", _percent(summary["retrieval_relevance"]),
                      help=f"Passages the checker judged relevant, out of {summary['passages_graded']} graded.")
    columns[1].metric("Groundedness", _percent(summary["groundedness"]),
                      help=f"Checked answers and lessons that passed, out of {summary['grounding_checks']}.")
    columns[2].metric("Citation correctness", _percent(summary["citation_correctness"]),
                      help=f"Answers whose citations all point to real sources, out of {summary['citation_checks']}.")

    if summary["latency"]:
        st.markdown("**Response time**")
        st.dataframe(
            [{"Kind of run": kind, "Runs": data["runs"], "Median (s)": data["median_s"], "90th percentile (s)": data["p90_s"]}
             for kind, data in summary["latency"].items()],
            hide_index=True,
        )

    st.markdown("**Adaptation self-check** (does the planner follow its own rules?)")
    for result in check_adaptation(date.today()):
        st.markdown(f"{'✅' if result.passed else '❌'} {result.name}: expected *{result.expected}*, got *{result.got}*")

    st.markdown("**Quiz grading consistency**")
    sample = _latest_short_answer(services, learner.learner_id, history)
    if sample is None:
        st.caption("Answer a short-answer question first; then the same answer can be re-graded to see if the grades agree.")
    elif st.button(f"Re-grade my latest short answer {CONSISTENCY_RUNS} times"):
        question, answer = sample
        with st.spinner("Re-grading..."):
            try:
                scores = [services.short_answer_grader.grade(question, answer).score for _ in range(CONSISTENCY_RUNS)]
            except Exception:  # noqa: BLE001
                logger.exception("Consistency check failed")
                st.error("The local model couldn't grade right now. Check that Ollama is running.")
                scores = []
        if scores:
            result = grading_consistency(scores)
            st.write(f"Scores: {', '.join(f'{s:.0%}' for s in scores)} · agreement {result['agreement']:.0%} · spread {result['spread']:.2f}")
            st.caption("Multiple-choice grading needs no model, so it is always 100% consistent.")

    st.markdown("**Local resource usage**")
    memory = process_memory_mb()
    st.write(f"This app (Python process): {memory:.0f} MB" if memory else "This app: memory could not be measured")
    try:
        models = ollama_loaded_models(settings.ollama_base_url)
    except httpx.HTTPError:
        st.caption("Ollama isn't reachable, so its memory use is unknown.")
    else:
        if models:
            st.dataframe(models, hide_index=True)
        else:
            st.caption("No model is loaded in Ollama right now (models unload after a few idle minutes).")


def render_progress_page(services: TutorServices, settings: Settings, learner: LearnerProfile | None) -> None:
    if learner is None:
        st.info("Create or choose a learner profile in the sidebar to see progress.")
        return
    repository = services.repository
    learner_id = learner.learner_id
    today = date.today()
    try:
        progress = repository.get_progress(learner_id)
        history = repository.list_assessments(learner_id)
        lessons = repository.list_lessons(learner_id)
        misconceptions = repository.get_misconceptions(learner_id)
        prerequisites = repository.get_prerequisites(learner_id)
        levels = repository.topic_levels(learner_id)
    except Exception:  # noqa: BLE001
        logger.exception("Could not read progress")
        st.error("Could not read your saved progress. Check the terminal logs.")
        return

    graded = [item for item in history if item["status"] == "graded"]
    scores = [item["score"] for item in graded if item["score"] is not None]
    topics = sorted({item.topic for item in progress} | {item["topic"] for item in history})

    st.subheader(f"{learner.name}'s progress")
    columns = st.columns(4)
    columns[0].metric("Topics studied", len(topics))
    columns[1].metric("Concepts tracked", len(progress))
    columns[2].metric("Quizzes graded", len(graded))
    columns[3].metric("Average score", _percent(sum(scores) / len(scores) if scores else None))

    if not progress and not history:
        st.info("Nothing here yet. Open **Learn & practise** and start a lesson.")
        return

    st.markdown("#### Suggested next")
    for item in recommend(progress, prerequisites, today, levels):
        st.markdown(f"- **{item.action.capitalize()}: {item.concept}** ({item.topic}): {item.reason}")

    st.markdown("#### Mastery by concept")
    rows = []
    for item in sorted(progress, key=lambda p: (p.topic, p.mastery)):
        rows.append({
            "Topic": item.topic,
            "Concept": item.concept,
            "Status": STATUS_LABELS[concept_status(item, today)],
            "Mastery": item.mastery,
            "Answered": item.attempts,
            "Correct": item.correct,
            "Next review": item.next_review.isoformat() if item.next_review else "—",
        })
    st.dataframe(rows, hide_index=True, column_config={
        "Mastery": st.column_config.ProgressColumn("Mastery", min_value=0.0, max_value=1.0, format="%.2f"),
    })
    st.caption("Mastery moves 40% of the way toward each new score. It's a rule of thumb, not a validated measure.")

    weak = [item.concept for item in progress if concept_status(item, today) == "remediate"]
    due = [item for item in progress if concept_status(item, today) == "revise"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Weak concepts")
        st.write(", ".join(weak) if weak else "None right now.")
    with right:
        st.markdown("#### Revision due")
        st.write(", ".join(f"{item.concept} (since {item.next_review})" for item in due) if due else "Nothing due today.")

    if misconceptions:
        with st.expander(f"Possible misconceptions ({len(misconceptions)})"):
            st.caption("Guesses from the grader about what might be confusing, not diagnoses.")
            for item in misconceptions:
                st.markdown(f"- **{item['concept']}**: {item['note']} ({item['created_at'][:10]})")

    st.markdown("#### Quiz history")
    st.dataframe(
        [{"Date": item["created_at"][:16].replace("T", " "), "Topic": item["topic"], "Level": item["level"],
          "Status": item["status"], "Score": _percent(item["score"])} for item in history],
        hide_index=True,
    )

    if lessons:
        st.markdown("#### Lessons and their sources")
        for lesson in lessons:
            with st.expander(f"{lesson['lesson']['title']} · {lesson['level']} · {lesson['created_at'][:10]}"):
                st.markdown(lesson["lesson"]["summary"])
                if lesson["evidence_kind"] == "general":
                    st.caption("From general knowledge (no sources).")
                for number, source in enumerate(lesson["sources"], start=1):
                    tag = f"W{number}" if source.get("kind") == "web" else str(number)
                    url = f" · {source['url']}" if source.get("url") else ""
                    st.caption(f"[{tag}] {source['citation']}{url}")

    export = {
        "learner": learner.name,
        "progress": [item.to_dict() for item in progress],
        "assessments": history,
        "misconceptions": misconceptions,
    }
    st.download_button("Download my progress (JSON)", json.dumps(export, indent=2),
                       file_name=f"progress-{learner.name}.json", mime="application/json")

    st.divider()
    _evaluation_section(services, settings, learner, history)