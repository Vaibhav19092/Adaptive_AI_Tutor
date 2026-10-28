"""The "Learn & practise" tab: lessons, quizzes, feedback, and what to do next (Phases 5-8).

Answers stay hidden: before submission this page only reads ActivityOutcome.questions,
the public view of the quiz, which has no answers, explanations, or rubric.
"""

from __future__ import annotations

import logging

import streamlit as st

from adaptive_tutor.graph.learning_workflow import (
    ActivityOutcome,
    discard_activity,
    pending_activity,
    start_activity,
    submit_answers,
)
from adaptive_tutor.learner.learning_planner import Recommendation, activity_request
from adaptive_tutor.learner.profile import LearnerProfile
from adaptive_tutor.services import TutorServices
from adaptive_tutor.ui.ask_page import safe_link
from adaptive_tutor.ui.sidebar import LAST_RESULT_PREFIX

logger = logging.getLogger("adaptive_tutor.app")

ACTIVITY_CHOICES = {
    "Let the tutor decide": "auto",
    "Lesson + quiz": "lesson",
    "Quiz only": "quiz",
    "What should I study next?": "next",
}
SOURCE_CHOICES = {"My documents": "documents", "General knowledge": "general"}
LEVEL_LABELS = {"beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced"}
ACTION_ICONS = {"prerequisite": "🧱", "remediate": "🔁", "revise": "🗓️", "practice": "✍️", "start": "▶️", "advance": "🚀"}
SPINNER_TEXT = "Working... a checked lesson and quiz take about 10 calls to the local model, so this can take minutes on a CPU."


def _run_safely(function, *args, **kwargs) -> ActivityOutcome | None:
    """Run a workflow helper; show a friendly message instead of crashing the page."""
    try:
        return function(*args, **kwargs)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - safety net; nodes already handle known errors
        logger.exception("Unexpected learning workflow failure")
        st.error("Something unexpected went wrong. Check the terminal logs.")
    return None


def _show_decisions(state: dict) -> None:
    decisions = state.get("decisions") or []
    if decisions:
        with st.expander("How the tutor decided (agent log)"):
            for number, note in enumerate(decisions, start=1):
                st.markdown(f"{number}. {note}")
    with st.expander("Show LangGraph workflow steps"):
        st.write(" → ".join(state.get("steps", [])))


def _show_sources(state: dict) -> None:
    evidence = state.get("evidence") or []
    if not evidence or state.get("evidence_kind") == "general":
        return
    web = evidence[0].get("kind") == "web"
    label = "Web sources · from the internet" if web else "Document sources · from your files"
    with st.expander(f"{label} ({len(evidence)})"):
        for number, item in enumerate(evidence, start=1):
            tag = f"W{number}" if web else str(number)
            citation = safe_link(item["citation"], item["url"]) if item.get("url") else item["citation"]
            st.markdown(f"**[{tag}]** {citation}")
            st.caption(item["text"][:300] + (" ..." if len(item["text"]) > 300 else ""))


def _show_lesson(state: dict, services: TutorServices) -> None:
    lesson = state.get("lesson") or {}
    if not lesson:
        return
    if state.get("search_query") and services.search_provider is not None:
        query = state["search_query"].replace("`", "'")
        st.caption(f"Search query sent to {services.search_provider.name}: `{query}`")
    kind = state.get("evidence_kind")
    if kind == "general":
        st.warning("This lesson comes from the model's general knowledge (no sources), so it can contain mistakes.")
    elif kind == "web":
        st.info("Your documents didn't cover this, so the lesson is built from web sources.")
    if state.get("partial_evidence"):
        st.info("Your documents only partly cover this topic, so the lesson sticks to what they say.")
    if state.get("evidence_verdict") == "contradictory":
        st.warning("Your sources disagree about part of this topic. The lesson describes each view.")

    with st.container(border=True):
        st.markdown(f"### {lesson['title']}")
        st.caption(f"{LEVEL_LABELS.get(lesson['level'], lesson['level'])} · concepts: {', '.join(lesson['concepts'])}")
        st.markdown(f"*{lesson['summary']}*")
        st.markdown(lesson["explanation"])
        st.markdown(f"**💡 Analogy** (the tutor's own comparison): {lesson['analogy']}")
        st.markdown(f"**🔎 Example:** {lesson['example']}")
        st.markdown("**Key points**")
        st.markdown("\n".join(f"- {point}" for point in lesson["key_points"]))
        if lesson.get("prerequisites"):
            st.caption("Helpful to know first: " + ", ".join(lesson["prerequisites"]))
    if state.get("lesson_grounded"):
        st.success("✓ Checked: the local model judged every fact in the explanation and key points to be supported by the sources.")
    _show_sources(state)


def _render_quiz(outcome: ActivityOutcome, services: TutorServices, learner: LearnerProfile) -> None:
    quiz_id = outcome.state.get("quiz_id", "quiz")
    st.markdown("#### Quiz")
    st.caption("Answers and explanations appear after you submit.")
    if outcome.state.get("error"):
        st.error(outcome.state["error"])
    with st.form(f"quiz-{quiz_id}"):
        widgets = {}  # unique keys: widgets in different tabs must never share an id
        for number, question in enumerate(outcome.questions, start=1):
            st.markdown(f"**Q{number}.** {question.prompt}")
            st.caption(f"Concept: {question.concept}")
            key = f"{quiz_id}-{question.question_id}"
            if question.options:
                options = question.options
                widgets[question.question_id] = st.radio(
                    "Choose one", list(range(len(options))), format_func=lambda i, o=options: o[i],
                    index=None, key=key, label_visibility="collapsed",
                )
            else:
                widgets[question.question_id] = st.text_area("Your answer", key=key, height=100)
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        with st.spinner("Grading... short answers are checked by the local model."):
            done = _run_safely(submit_answers, services.learning_graph, services.repository,
                               learner.learner_id, outcome.thread_id, widgets)
        if done is not None:
            if not done.waiting:
                st.session_state[LAST_RESULT_PREFIX + learner.learner_id] = done.state
            st.rerun()
    if st.button("Discard this quiz", help="It stays in your history as discarded and doesn't change your mastery."):
        discard_activity(services.repository, learner.learner_id, outcome.thread_id)
        st.rerun()


def _show_recommendations(state: dict, services: TutorServices, learner: LearnerProfile, options: dict) -> None:
    items = [Recommendation.from_dict(item) for item in state.get("recommendations") or []]
    if not items:
        return
    st.markdown("#### What to do next")
    for index, item in enumerate(items):
        text_column, button_column = st.columns([5, 1])
        text_column.markdown(f"{ACTION_ICONS.get(item.action, '•')} **{item.action.capitalize()}: {item.concept}**  \n{item.reason}")
        if button_column.button("Start", key=f"start-rec-{index}-{item.action}-{item.concept}"):
            request, action, level = activity_request(item)
            _start(services, learner, request, action, level, options)


def _show_results(state: dict, services: TutorServices, learner: LearnerProfile, options: dict) -> None:
    if state.get("error"):
        st.error(state["error"])
    elif state.get("refused"):
        st.warning(state["message"])
    elif state.get("chosen_action") == "next":
        st.info(state.get("message", ""))

    assessment = state.get("assessment")
    if assessment:
        _show_lesson(state, services)
        score = assessment.get("score")
        st.markdown("#### Your results")
        st.metric("Score", f"{score:.0%}" if score is not None else "not graded",
                  help="Average of the graded questions.")
        for number, item in enumerate(assessment["results"], start=1):
            mark = "✅" if item["correct"] else ("⚠️" if not item["graded"] else "❌")
            with st.container(border=True):
                st.markdown(f"{mark} **Q{number}.** {item['prompt']}  \n*Concept: {item['concept']} · score {item['score']:.0%}*")
                st.markdown(f"**Your answer:** {item['learner_answer'] or '(no answer)'}")
                st.markdown(item["feedback"])
                for point in item.get("point_results") or []:
                    st.markdown(f"{'✔️' if point['covered'] else '✖️'} {point['point']}" + (f" · {point['reason']}" if point["reason"] else ""))
                if item["kind"] == "short_answer":
                    st.caption(f"Model answer: {item['correct_answer']}")
                if item.get("misconception"):
                    st.caption(f"Possible misconception (a guess, not a diagnosis): {item['misconception']}")
        notes = state.get("mastery_notes") or []
        if notes:
            st.markdown("**Mastery updates**")
            st.markdown("\n".join(f"- {note}" for note in notes))
            st.caption("Mastery is a simple rule of thumb (see the README), not a validated measure.")
    _show_recommendations(state, services, learner, options)
    _show_decisions(state)


def _start(services: TutorServices, learner: LearnerProfile, request: str, action: str, level: str, options: dict) -> None:
    with st.spinner(SPINNER_TEXT):
        outcome = _run_safely(
            start_activity, services.learning_graph, services.repository, learner.learner_id, request,
            action=action, level=level, source_mode=options["source_mode"],
            allow_web=options["allow_web"], check_evidence=options["check_evidence"],
        )
    if outcome is None:
        return
    key = LAST_RESULT_PREFIX + learner.learner_id
    if outcome.waiting:
        st.session_state.pop(key, None)
    else:
        st.session_state[key] = outcome.state
    st.rerun()


def render_learn_page(services: TutorServices, learner: LearnerProfile | None) -> None:
    if learner is None:
        st.info("Create or choose a learner profile in the sidebar to start learning.")
        return

    try:
        pending = pending_activity(services.learning_graph, services.repository, learner.learner_id)
    except Exception:  # noqa: BLE001
        logger.exception("Could not read the saved activity")
        st.error("Could not read your saved quiz. Check the terminal logs.")
        pending = None

    if pending is not None:
        st.caption(f"Welcome back, {learner.name}. Your quiz is saved, even if you close the app.")
        _show_lesson(pending.state, services)
        _render_quiz(pending, services, learner)
        _show_decisions(pending.state)
        return

    # Every option has its own key: the Ask tab has a checkbox with the same label, and
    # Streamlit refuses two identical widgets on one page (DuplicateWidgetID).
    activity_label = st.radio("Activity", list(ACTIVITY_CHOICES), horizontal=True, key="learn_activity")
    level_column, source_column = st.columns(2)
    level = level_column.selectbox("Level", list(LEVEL_LABELS), format_func=LEVEL_LABELS.get, key="learn_level")
    source_label = source_column.radio("Teach from", list(SOURCE_CHOICES), horizontal=True, key="learn_source")
    source_mode = SOURCE_CHOICES[source_label]
    check_evidence = st.checkbox(
        "Check the material, lesson, and answer keys carefully (Self-RAG)", value=True, key="learn_check",
        help="Grades each passage, retries with a better search, checks the lesson against the sources, "
        "and double-checks each multiple-choice answer key. Safer but slower.",
    )
    allow_web = False
    if source_mode == "documents" and services.search_provider is not None:
        allow_web = st.checkbox(
            f"If my documents don't cover it, search {services.search_provider.name}", value=False, key="learn_allow_web",
            help="Only a short search query is sent. Your documents are never sent.",
        )
    options = {"source_mode": source_mode, "check_evidence": check_evidence, "allow_web": allow_web}

    with st.form("learn_form", clear_on_submit=False):
        request = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. Teach me photosynthesis · Quiz me on embeddings · What should I study next?",
            max_chars=300,
        )
        submitted = st.form_submit_button("Start", type="primary")
    if submitted:
        _start(services, learner, request, ACTIVITY_CHOICES[activity_label], level, options)

    last = st.session_state.get(LAST_RESULT_PREFIX + learner.learner_id)
    if last:
        st.divider()
        _show_results(last, services, learner, options)