"""Phase 8: the shared state of the learning workflow (lesson -> quiz -> grading -> plan).

This is a SEPARATE state from TutorState: the Phase 1-4 question-answering graph
is unchanged. Everything here is plain data (strings, numbers, lists, dicts),
because the checkpointer saves this state to disk while the learner answers.
"""

import operator
from typing import Annotated, Any, TypedDict


class LearningState(TypedDict, total=False):
    # ----- Input (set by start_activity) -----
    learner_id: str
    thread_id: str  # this activity's checkpoint thread
    request: str  # what the learner typed, e.g. "Teach me photosynthesis"
    action: str  # "auto", "lesson", "quiz", or "next"
    level: str  # "beginner", "intermediate", or "advanced"
    source_mode: str  # "documents" or "general"
    allow_web: bool  # may research the web if documents fall short
    check_evidence: bool  # run the Self-RAG checks

    # ----- 1. load_profile -----
    profile: list[dict[str, Any]]  # ConceptProgress rows as dicts
    prerequisite_map: dict[str, list[str]]

    # ----- 2. understand_request -----
    chosen_action: str  # "lesson", "quiz", or "next"
    topic: str
    focus_concepts: list[str]  # weak concepts to target (adaptation)

    # ----- 3-6. material: retrieve, check, correct, research -----
    material_query: str  # rewritten query for the next document search
    material_attempts: int
    candidates: list[dict[str, Any]]  # newest search results, not yet graded
    evidence: list[dict[str, Any]]  # approved evidence (documents OR web, never both)
    rejected_evidence: list[dict[str, Any]]
    evidence_kind: str  # "documents", "web", or "general"
    evidence_verdict: str  # "sufficient", "insufficient", or "contradictory"
    partial_evidence: bool  # the sources only partly cover the topic
    used_web: bool
    search_query: str  # the exact text sent to the search service

    # ----- 7. lesson -----
    lesson: dict[str, Any]
    lesson_id: str
    lesson_attempts: int
    lesson_problem: str
    lesson_grounded: bool

    # ----- 8. quiz (contains the answers: never shown before submission) -----
    quiz: dict[str, Any]
    quiz_id: str
    quiz_notes: list[str]

    # ----- 9. answers and grading -----
    answers: dict[str, Any]  # {question_id: option index or text}
    assessment: dict[str, Any]

    # ----- 10-12. profile update, plan, save -----
    progress_updates: list[dict[str, Any]]
    new_misconceptions: list[dict[str, str]]
    mastery_notes: list[str]
    recommendations: list[dict[str, Any]]
    saved: bool

    # ----- Endings -----
    message: str  # a final message (refusal or "what next" summary)
    refused: bool  # True when the tutor declined to teach from weak evidence
    error: str

    # Appended by every node (reducer operator.add), like `steps` in Phase 1-4.
    steps: Annotated[list[str], operator.add]
    decisions: Annotated[list[str], operator.add]  # plain-language log of every choice