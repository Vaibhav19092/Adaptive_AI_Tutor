"""Phase 8: wire the learning workflow, and the helpers the app uses to run it.

START -> load_profile -> understand_request
  |-- next ------------------------------------------------> recommend_next -> END
  |-- general knowledge -> generate_lesson / create_quiz
  +-- documents -> retrieve_material -> check_material
        |-- good material ------------------------------> generate_lesson / create_quiz
        |-- weak, retries left --> rewrite_material_query -> retrieve_material   [bounded]
        |-- weak, web allowed ---> research_web --> generate_lesson / create_quiz
        |                                      +-> report_no_material -> END
        |-- some material -------> use_partial_material -> generate_lesson / create_quiz
        +-- nothing -------------> report_no_material -> END
generate_lesson -> check_lesson (checked mode, with sources)
        |-- supported ------------> create_quiz
        |-- not, retries left ----> generate_lesson (told why)                  [bounded]
        +-- not, no retries ------> report_unreliable_lesson -> END
create_quiz -> wait_for_answers  (PAUSE: interrupt, state saved by the checkpointer)
            -> grade_answers -> update_profile -> recommend_next -> save_progress -> END
               (a grading failure goes back to wait_for_answers)
Any error -> handle_error -> END
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from adaptive_tutor.assessment.schemas import LEVELS, Quiz, QuestionView, public_view
from adaptive_tutor.config import Settings
from adaptive_tutor.evaluation.metrics import activity_event_data
from adaptive_tutor.graph.learning_nodes import (
    NO_LEARNER_MESSAGE,
    handle_error,
    make_check_lesson_node,
    make_check_material_node,
    make_create_quiz_node,
    make_generate_lesson_node,
    make_grade_answers_node,
    make_load_profile_node,
    make_recommend_next_node,
    make_research_web_node,
    make_retrieve_material_node,
    make_rewrite_material_query_node,
    make_route_after_lesson_check,
    make_route_after_material_check,
    make_save_progress_node,
    make_understand_request_node,
    make_update_profile_node,
    report_no_material,
    report_unreliable_lesson,
    route_after_grading,
    route_after_lesson,
    route_after_partial,
    route_after_profile,
    route_after_quiz,
    route_after_recommendation,
    route_after_request,
    route_after_research,
    route_after_retrieval,
    route_after_rewrite,
    route_after_save,
    route_after_update,
    use_partial_material,
    wait_for_answers,
)
from adaptive_tutor.graph.learning_state import LearningState
from adaptive_tutor.memory.checkpointing import learner_of_thread, make_thread_id, thread_config
from adaptive_tutor.memory.repository import LearnerRepository
from adaptive_tutor.research.search_provider import SearchProvider
from adaptive_tutor.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

ACTIONS = ("auto", "lesson", "quiz", "next")
SOURCE_MODES = ("documents", "general")
WAITING_NODE = "wait_for_answers"


def build_learning_graph(
    llm: Runnable,
    json_llm: Runnable,
    settings: Settings,
    repository: LearnerRepository,
    retriever: Retriever | None = None,
    search_provider: SearchProvider | None = None,
    checkpointer: Any = None,
    clock: Callable[[], date] = date.today,
) -> CompiledStateGraph:
    """Build the learning workflow. A checkpointer is required for the quiz pause."""
    graph = StateGraph(LearningState)

    graph.add_node("load_profile", make_load_profile_node(repository))
    graph.add_node("understand_request", make_understand_request_node(llm, settings, clock))
    graph.add_node("retrieve_material", make_retrieve_material_node(retriever, settings))
    graph.add_node("check_material", make_check_material_node(llm, settings))
    graph.add_node("rewrite_material_query", make_rewrite_material_query_node(llm, settings))
    graph.add_node("research_web", make_research_web_node(llm, settings, search_provider))
    graph.add_node("use_partial_material", use_partial_material)
    graph.add_node("report_no_material", report_no_material)
    graph.add_node("generate_lesson", make_generate_lesson_node(json_llm, settings))
    graph.add_node("check_lesson", make_check_lesson_node(llm, settings))
    graph.add_node("report_unreliable_lesson", report_unreliable_lesson)
    graph.add_node("create_quiz", make_create_quiz_node(llm, json_llm, settings, repository))
    graph.add_node("wait_for_answers", wait_for_answers)
    graph.add_node("grade_answers", make_grade_answers_node(json_llm, settings))
    graph.add_node("update_profile", make_update_profile_node(repository, clock))
    graph.add_node("recommend_next", make_recommend_next_node(repository, clock))
    graph.add_node("save_progress", make_save_progress_node(repository))
    graph.add_node("handle_error", handle_error)

    teach = {"generate_lesson": "generate_lesson", "create_quiz": "create_quiz"}
    error = {"handle_error": "handle_error"}

    graph.add_edge(START, "load_profile")
    graph.add_conditional_edges("load_profile", route_after_profile, {"understand_request": "understand_request", **error})
    graph.add_conditional_edges(
        "understand_request",
        route_after_request,
        {"recommend_next": "recommend_next", "retrieve_material": "retrieve_material", **teach, **error},
    )
    graph.add_conditional_edges("retrieve_material", route_after_retrieval, {"check_material": "check_material", **error})
    graph.add_conditional_edges(
        "check_material",
        make_route_after_material_check(settings, web_available=search_provider is not None),
        {
            **teach,
            "rewrite_material_query": "rewrite_material_query",
            "research_web": "research_web",
            "use_partial_material": "use_partial_material",
            "report_no_material": "report_no_material",
            **error,
        },
    )
    graph.add_conditional_edges("rewrite_material_query", route_after_rewrite, {"retrieve_material": "retrieve_material", **error})
    graph.add_conditional_edges("research_web", route_after_research, {**teach, "report_no_material": "report_no_material", **error})
    graph.add_conditional_edges("use_partial_material", route_after_partial, teach)
    graph.add_conditional_edges(
        "generate_lesson", route_after_lesson, {"check_lesson": "check_lesson", "create_quiz": "create_quiz", **error}
    )
    graph.add_conditional_edges(
        "check_lesson",
        make_route_after_lesson_check(settings),
        {
            "create_quiz": "create_quiz",
            "generate_lesson": "generate_lesson",
            "report_unreliable_lesson": "report_unreliable_lesson",
            **error,
        },
    )
    graph.add_conditional_edges("create_quiz", route_after_quiz, {"wait_for_answers": "wait_for_answers", **error})
    graph.add_edge("wait_for_answers", "grade_answers")
    graph.add_conditional_edges(
        "grade_answers", route_after_grading, {"wait_for_answers": "wait_for_answers", "update_profile": "update_profile"}
    )
    graph.add_conditional_edges("update_profile", route_after_update, {"recommend_next": "recommend_next", **error})
    graph.add_conditional_edges(
        "recommend_next", route_after_recommendation, {"save_progress": "save_progress", **error, END: END}
    )
    graph.add_conditional_edges("save_progress", route_after_save, {**error, END: END})
    for final_node in ("report_no_material", "report_unreliable_lesson", "handle_error"):
        graph.add_edge(final_node, END)

    return graph.compile(checkpointer=checkpointer)


# ---------- Helpers used by the app ----------

@dataclass(frozen=True)
class ActivityOutcome:
    """The result of starting or continuing an activity."""

    thread_id: str
    state: dict[str, Any]
    waiting: bool  # True = paused, waiting for quiz answers

    @property
    def questions(self) -> list[QuestionView]:
        """The quiz WITHOUT answers, safe to show before submission."""
        quiz = self.state.get("quiz")
        return public_view(Quiz.from_dict(quiz)) if quiz else []


def _is_waiting(graph: CompiledStateGraph, thread_id: str, result: dict[str, Any]) -> bool:
    if "__interrupt__" in result:
        return True
    snapshot = graph.get_state(thread_config(thread_id))
    return WAITING_NODE in tuple(snapshot.next or ())


def _record(repository: LearnerRepository, learner_id: str, kind: str, started: float, state: dict[str, Any]) -> None:
    try:
        repository.record_event(learner_id, kind, int((time.perf_counter() - started) * 1000), activity_event_data(state))
    except Exception:  # noqa: BLE001 - metrics must never break learning
        logger.exception("Could not record a metric event")


def start_activity(
    graph: CompiledStateGraph,
    repository: LearnerRepository,
    learner_id: str,
    request: str,
    action: str = "auto",
    level: str = "beginner",
    source_mode: str = "documents",
    allow_web: bool = False,
    check_evidence: bool = True,
) -> ActivityOutcome:
    """Start a new activity (lesson + quiz, quiz, or "what next") in its own thread."""
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {SOURCE_MODES}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")

    if repository.get_learner(learner_id) is None:
        return ActivityOutcome("", {"error": NO_LEARNER_MESSAGE, "steps": []}, False)

    thread_id = make_thread_id(learner_id)
    repository.start_activity(learner_id, thread_id)
    started = time.perf_counter()
    result = graph.invoke(
        {
            "learner_id": learner_id,
            "thread_id": thread_id,
            "request": request,
            "action": action,
            "level": level,
            "source_mode": source_mode,
            "allow_web": allow_web and source_mode == "documents",
            "check_evidence": check_evidence,
            "steps": [],
            "decisions": [],
        },
        thread_config(thread_id),
    )
    waiting = _is_waiting(graph, thread_id, result)
    if not waiting:
        repository.close_activity(learner_id, thread_id, "failed" if result.get("error") else "done")
    _record(repository, learner_id, "learning_run", started, result)
    return ActivityOutcome(thread_id, dict(result), waiting)


def _check_owner(learner_id: str, thread_id: str) -> None:
    if learner_of_thread(thread_id) != learner_id:
        raise ValueError("This activity belongs to a different learner.")


def submit_answers(
    graph: CompiledStateGraph,
    repository: LearnerRepository,
    learner_id: str,
    thread_id: str,
    answers: dict[str, Any],
) -> ActivityOutcome:
    """Resume the paused graph with the learner's answers: grade, update, plan, save."""
    _check_owner(learner_id, thread_id)
    started = time.perf_counter()
    result = graph.invoke(Command(resume=answers), thread_config(thread_id))
    waiting = _is_waiting(graph, thread_id, result)
    if not waiting:
        repository.close_activity(learner_id, thread_id, "failed" if result.get("error") else "done")
    _record(repository, learner_id, "grading", started, result)
    return ActivityOutcome(thread_id, dict(result), waiting)


def pending_activity(graph: CompiledStateGraph, repository: LearnerRepository, learner_id: str) -> ActivityOutcome | None:
    """The learner's paused quiz, if any, read back from the checkpoint (works after a restart)."""
    thread_id = repository.open_activity(learner_id)
    if thread_id is None:
        return None
    snapshot = graph.get_state(thread_config(thread_id))
    if WAITING_NODE in tuple(snapshot.next or ()):
        return ActivityOutcome(thread_id, dict(snapshot.values), True)
    # Nothing to resume (e.g. the app stopped mid-run, or the checkpoint was deleted).
    repository.close_activity(learner_id, thread_id, "abandoned")
    return None


def discard_activity(repository: LearnerRepository, learner_id: str, thread_id: str) -> None:
    """Give up a paused quiz. It stays in the history as 'discarded' and changes no mastery."""
    _check_owner(learner_id, thread_id)
    repository.close_activity(learner_id, thread_id, "discarded")