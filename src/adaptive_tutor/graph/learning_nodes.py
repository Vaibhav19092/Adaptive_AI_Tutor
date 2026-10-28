"""Phase 8: the nodes (steps) and routers (decisions) of the learning workflow.

Load profile -> understand request -> retrieve material -> check it (Self-RAG)
-> correct the search / research the web (Corrective RAG) -> teach (lesson)
-> check the lesson -> assess (quiz) -> PAUSE for the learner's answers
-> grade -> update the profile -> plan what's next -> save.

Like Phases 1-4, every node comes from a factory (so tests can pass fakes),
returns only the keys it changes plus its name in "steps", and turns every
failure into a friendly message in "error".
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END
from langgraph.types import interrupt

from adaptive_tutor.assessment.answer_evaluator import RUBRIC_PROMPT, ShortAnswerGrader, grade_quiz
from adaptive_tutor.assessment.quiz_generator import KEY_CHECK_PROMPT, QUIZ_PROMPT, generate_quiz, verify_answer_keys
from adaptive_tutor.assessment.schemas import (
    LEVELS,
    AssessmentResult,
    Lesson,
    MultipleChoiceQuestion,
    Quiz,
    normalize_concept,
    normalize_topic,
)
from adaptive_tutor.config import Settings
from adaptive_tutor.graph.checking_nodes import EVIDENCE_PROMPT, GRADE_PROMPT, GROUNDING_PROMPT, REWRITE_PROMPT
from adaptive_tutor.graph.learning_state import LearningState
from adaptive_tutor.graph.nodes import NO_DOCUMENTS_MESSAGE, NO_RETRIEVER_MESSAGE, SEARCH_QUERY_PROMPT
from adaptive_tutor.learner.intent import clean_topic, detect_intent, parse_intent
from adaptive_tutor.learner.learning_planner import focus_concepts, recommend
from adaptive_tutor.learner.mastery import ConceptProgress, describe_change, update_progress
from adaptive_tutor.llm.provider import describe_llm_error
from adaptive_tutor.llm.structured import StructuredOutputError
from adaptive_tutor.memory.repository import LearnerRepository
from adaptive_tutor.research.search_provider import SearchError, SearchProvider
from adaptive_tutor.research.search_query import clean_search_query
from adaptive_tutor.research.source_verification import apply_hard_rules, score_results
from adaptive_tutor.retrieval.corrective_rag import plan_retrieval
from adaptive_tutor.retrieval.retriever import Retriever
from adaptive_tutor.retrieval.self_rag import parse_evidence_verdict, parse_yes_no
from adaptive_tutor.teaching.evidence import chunk_to_evidence, format_evidence, web_to_evidence
from adaptive_tutor.teaching.lesson_generator import LESSON_PROMPT, generate_lesson, lesson_citation_problem

logger = logging.getLogger(__name__)

NodeFunction = Callable[[LearningState], dict[str, Any]]
Router = Callable[[LearningState], str]
Clock = Callable[[], date]

MAX_REQUEST_CHARS = 300

# ---------- Prompts and messages ----------

INTENT_SYSTEM_PROMPT = (
    "You sort a learner's request into one of three kinds. Reply with only one word:\n"
    "LESSON if they want to learn about or understand a topic,\n"
    "QUIZ if they want to be tested or to practise,\n"
    "NEXT if they ask what to study next or for a recommendation."
)

INTENT_PROMPT = ChatPromptTemplate.from_messages([("system", INTENT_SYSTEM_PROMPT), ("human", "Request: {request}")])

NO_LEARNER_MESSAGE = "Choose or create a learner profile in the sidebar first."
EMPTY_REQUEST_MESSAGE = "Please type what you want to learn, for example “Teach me photosynthesis”."
NO_TOPIC_MESSAGE = (
    "Tell me a topic first, for example “Teach me photosynthesis” or “Quiz me on embeddings”. "
    "Once you have quiz results, I can also pick a topic for you."
)
NO_MATERIAL_MESSAGE = (
    "Your documents don't cover this topic well enough for a reliable lesson, so I won't guess. "
    "Add a document about it, tick “Allow web research” (when it's turned on), or choose "
    "“General knowledge” as the source."
)
NO_WEB_MATERIAL_MESSAGE = (
    "Neither your documents nor the web search had good enough material on this topic, "
    "so I won't teach it from guesses. Try a more specific topic, or choose “General knowledge”."
)
UNRELIABLE_LESSON_MESSAGE = (
    "I wrote a lesson, but my check found claims your sources don't support, even after trying again. "
    "I won't show it, because it could teach you something wrong. Try a narrower topic."
)
NO_QUESTIONS_MESSAGE = (
    "I couldn't write quiz questions I'm confident are correct for this material. "
    "Try again, or try a narrower topic."
)
NO_RECOMMENDATIONS_MESSAGE = (
    "There's nothing to recommend yet: take a lesson and a quiz first, then I can suggest "
    "what to practise, revise, or learn next."
)


# ---------- Shared helpers ----------

def _error(exc: Exception, settings: Settings, step: str) -> dict[str, Any]:
    logger.exception("Model call failed in %s", step)
    return {"error": describe_llm_error(exc, settings), "steps": [step]}


def _ask(chain: Runnable, inputs: dict[str, Any]) -> str:
    return str(chain.invoke(inputs))


def _topic_question(state: LearningState) -> str:
    """The 'question' the material must answer, used by the Phase 4 checker prompts."""
    focus = state.get("focus_concepts") or []
    extra = f", especially {', '.join(focus)}" if focus else ""
    return f"What should a learner know about {state.get('topic', '')}{extra}?"


def _progress_rows(state: LearningState) -> list[ConceptProgress]:
    return [ConceptProgress.from_dict(row) for row in state.get("profile", [])]


# ---------- 1. Load the learner profile ----------

def make_load_profile_node(repository: LearnerRepository) -> NodeFunction:
    def load_profile(state: LearningState) -> dict[str, Any]:
        step = ["load_profile"]
        learner_id = state.get("learner_id", "")
        try:
            if not learner_id or repository.get_learner(learner_id) is None:
                return {"error": NO_LEARNER_MESSAGE, "steps": step}
            progress = repository.get_progress(learner_id)
            prerequisites = repository.get_prerequisites(learner_id)
        except Exception:  # noqa: BLE001 - database problems
            logger.exception("Could not load the learner profile")
            return {"error": "Could not read your saved progress. Check the terminal logs.", "steps": step}
        note = f"Loaded the learner profile: {len(progress)} concept(s) tracked"
        return {
            "profile": [item.to_dict() for item in progress],
            "prerequisite_map": prerequisites,
            "decisions": [note],
            "steps": step,
        }

    return load_profile


# ---------- 2. Understand the request ----------

def make_understand_request_node(llm: Runnable, settings: Settings, clock: Clock) -> NodeFunction:
    chain = INTENT_PROMPT | llm | StrOutputParser()

    def understand_request(state: LearningState) -> dict[str, Any]:
        step = ["understand_request"]
        request = " ".join((state.get("request") or "").split())
        action = state.get("action") or "auto"
        if not request and action in ("auto", "lesson"):
            return {"error": EMPTY_REQUEST_MESSAGE, "steps": step}
        if len(request) > MAX_REQUEST_CHARS:
            return {"error": f"Please keep the request under {MAX_REQUEST_CHARS} characters.", "steps": step}

        decisions: list[str] = []
        if action in ("lesson", "quiz", "next"):
            chosen = action
            decisions.append(f"Activity chosen by you: {chosen}")
        else:
            chosen = detect_intent(request)
            if chosen:
                decisions.append(f"Understood the request as: {chosen} (from its wording)")
            else:
                try:
                    chosen = parse_intent(_ask(chain, {"request": request}))
                except Exception as exc:  # noqa: BLE001
                    return _error(exc, settings, "understand_request")
                if chosen:
                    decisions.append(f"Understood the request as: {chosen} (the model decided)")
                else:
                    chosen = "lesson"
                    decisions.append("The request was unclear, so the tutor chose a lesson")

        if chosen == "next":
            return {"chosen_action": "next", "topic": "", "focus_concepts": [], "decisions": decisions, "steps": step}

        today = clock()
        progress = _progress_rows(state)
        topic = normalize_topic(clean_topic(request))
        focus: list[str] = []
        if not topic:
            # No topic given ("quiz me"): let the planner pick the most useful one.
            suggestions = recommend(progress, state.get("prerequisite_map", {}), today, limit=1)
            if not suggestions:
                return {"error": NO_TOPIC_MESSAGE, "steps": step}
            topic, focus = suggestions[0].topic, [suggestions[0].concept]
            decisions.append(f"No topic given: the planner chose “{topic}” (focus: {focus[0]}) because: {suggestions[0].reason}")
        else:
            focus = focus_concepts(progress, topic, today)
            if focus:
                decisions.append(f"Adapting to your profile: extra focus on {', '.join(focus)}")
        return {"chosen_action": chosen, "topic": topic, "focus_concepts": focus, "decisions": decisions, "steps": step}

    return understand_request


# ---------- 3. Retrieve material from the learner's documents ----------

def make_retrieve_material_node(retriever: Retriever | None, settings: Settings) -> NodeFunction:
    def retrieve_material(state: LearningState) -> dict[str, Any]:
        step = ["retrieve_material"]
        if retriever is None:
            return {"error": NO_RETRIEVER_MESSAGE, "steps": step}
        attempts = state.get("material_attempts", 0)
        base_query = " ".join([state.get("topic", "")] + list(state.get("focus_concepts") or []))
        plan = plan_retrieval(attempts, base_query, state.get("material_query"), settings)
        try:
            if retriever.document_count() == 0:
                return {"error": NO_DOCUMENTS_MESSAGE, "steps": step}
            chunks = retriever.search(plan.query) if attempts == 0 else retriever.search(plan.query, top_k=plan.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retrieval for a lesson failed")
            return {"error": describe_llm_error(exc, settings, model_name=settings.embedding_model), "steps": step}
        relevant = [chunk for chunk in chunks if chunk.score >= plan.min_score]
        return {
            "candidates": [chunk_to_evidence(chunk) for chunk in relevant],
            "material_attempts": attempts + 1,
            "evidence_kind": "documents",
            "decisions": [plan.describe(found=len(chunks), kept=len(relevant))],
            "steps": step,
        }

    return retrieve_material


# ---------- 4. Check the material (Self-RAG) ----------

def make_check_material_node(llm: Runnable, settings: Settings) -> NodeFunction:
    grade_chain = GRADE_PROMPT | llm | StrOutputParser()
    evidence_chain = EVIDENCE_PROMPT | llm | StrOutputParser()

    def check_material(state: LearningState) -> dict[str, Any]:
        step = "check_material"
        approved = list(state.get("evidence", []))
        rejected = list(state.get("rejected_evidence", []))
        known = {item["key"] for item in approved + rejected}
        candidates = [item for item in state.get("candidates", []) if item["key"] not in known]
        question = _topic_question(state)

        if not state.get("check_evidence"):
            # Basic mode: like Phase 2, every passage above the similarity threshold is used.
            approved += candidates
            verdict = "sufficient" if approved else "insufficient"
            return {
                "evidence": approved,
                "candidates": [],
                "evidence_verdict": verdict,
                "decisions": [f"Basic mode: using {len(approved)} passage(s) without the relevance check"],
                "steps": [step],
            }

        newly_approved, unclear = 0, 0
        for item in candidates:
            try:
                verdict = parse_yes_no(_ask(grade_chain, {"question": question, "passage": item["text"]}))
            except Exception as exc:  # noqa: BLE001
                return _error(exc, settings, step)
            if verdict is None:
                unclear += 1
            if verdict:
                approved.append(item)
                newly_approved += 1
            else:
                rejected.append(item)
        notes = [f"Relevance check: {newly_approved} of {len(candidates)} new passage(s) judged relevant"
                 + (f" ({unclear} unclear, counted as not relevant)" if unclear else "")]

        verdict_value = "insufficient"
        if approved:
            try:
                reply = _ask(evidence_chain, {"context": format_evidence(approved), "question": question})
            except Exception as exc:  # noqa: BLE001
                return _error(exc, settings, step)
            verdict_value = parse_evidence_verdict(reply) or "insufficient"
            notes.append(f"Evidence check: {verdict_value}")
        return {
            "evidence": approved,
            "rejected_evidence": rejected,
            "candidates": [],
            "evidence_verdict": verdict_value,
            "decisions": notes,
            "steps": [step],
        }

    return check_material


# ---------- 5. Corrective RAG: rewrite the query ----------

def make_rewrite_material_query_node(llm: Runnable, settings: Settings) -> NodeFunction:
    chain = REWRITE_PROMPT | llm | StrOutputParser()

    def rewrite_material_query(state: LearningState) -> dict[str, Any]:
        step = "rewrite_material_query"
        previous = state.get("material_query") or state.get("topic", "")
        try:
            reply = _ask(chain, {"question": _topic_question(state), "previous_query": previous})
        except Exception as exc:  # noqa: BLE001
            return _error(exc, settings, step)
        query = clean_search_query(reply, fallback=previous)
        return {"material_query": query, "decisions": [f"Rewrote the document search query: '{query}'"], "steps": [step]}

    return rewrite_material_query


# ---------- 6. Research the web (only when allowed) ----------

def make_research_web_node(llm: Runnable, settings: Settings, search_provider: SearchProvider | None) -> NodeFunction:
    query_chain = SEARCH_QUERY_PROMPT | llm | StrOutputParser()
    grade_chain = GRADE_PROMPT | llm | StrOutputParser()

    def research_web(state: LearningState) -> dict[str, Any]:
        step = "research_web"
        if search_provider is None:
            return {"error": "Web research is turned off.", "steps": [step]}
        question = _topic_question(state)
        try:
            query = clean_search_query(_ask(query_chain, {"question": question}), fallback=state.get("topic", ""))
        except Exception as exc:  # noqa: BLE001
            return _error(exc, settings, step)
        try:
            raw = search_provider.search(query, settings.search_max_results)
        except SearchError as exc:
            return {"error": str(exc), "search_query": query, "steps": [step]}
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected web search failure")
            return {"error": "Something unexpected went wrong during the web search.", "steps": [step]}

        kept, _ = apply_hard_rules(raw, settings.research_blocked_domains)
        accepted, _ = score_results(kept, query, settings.research_min_quality)
        relevant = accepted
        if state.get("check_evidence"):
            relevant = []
            for result in accepted:
                try:
                    reply = _ask(grade_chain, {"question": question, "passage": web_to_evidence(result)["text"]})
                except Exception as exc:  # noqa: BLE001
                    return _error(exc, settings, step)
                if parse_yes_no(reply):
                    relevant.append(result)
        note = (f"Searched {search_provider.name} for '{query}': {len(raw)} result(s), "
                f"{len(accepted)} passed the quality rules, {len(relevant)} used")
        # Document passages are dropped: a lesson uses ONE kind of evidence, never a mix.
        return {
            "evidence": [web_to_evidence(result) for result in relevant],
            "evidence_kind": "web",
            "evidence_verdict": "sufficient" if relevant else "insufficient",
            "partial_evidence": False,
            "used_web": True,
            "search_query": query,
            "decisions": [note],
            "steps": [step],
        }

    return research_web


def report_no_material(state: LearningState) -> dict[str, Any]:
    message = NO_WEB_MATERIAL_MESSAGE if state.get("used_web") else NO_MATERIAL_MESSAGE
    return {"message": message, "refused": True, "steps": ["report_no_material"]}


# ---------- 7. Teach: write the lesson ----------

def make_generate_lesson_node(json_llm: Runnable, settings: Settings) -> NodeFunction:
    chain = LESSON_PROMPT | json_llm | StrOutputParser()

    def generate_lesson_node(state: LearningState) -> dict[str, Any]:
        step = "generate_lesson"
        general = state.get("source_mode") == "general"
        evidence = [] if general else list(state.get("evidence", []))
        try:
            lesson, tries = generate_lesson(
                chain,
                state["topic"],
                state.get("level", "beginner"),
                evidence,
                settings.structured_output_retries,
                focus_concepts=state.get("focus_concepts"),
                contradiction=state.get("evidence_verdict") == "contradictory",
                problem=state.get("lesson_problem", ""),
            )
        except StructuredOutputError as exc:
            return {"error": str(exc), "steps": [step]}
        except Exception as exc:  # noqa: BLE001
            return _error(exc, settings, step)
        note = f"Wrote a {lesson.level} lesson with {len(lesson.concepts)} concept(s)"
        if tries > 1:
            note += f" (the JSON needed {tries} tries)"
        return {
            "lesson": lesson.to_dict(),
            "lesson_id": state.get("lesson_id") or uuid.uuid4().hex[:12],
            "lesson_attempts": state.get("lesson_attempts", 0) + 1,
            "evidence_kind": "general" if general else state.get("evidence_kind", "documents"),
            "decisions": [note],
            "steps": [step],
        }

    return generate_lesson_node


def make_check_lesson_node(llm: Runnable, settings: Settings) -> NodeFunction:
    """Self-RAG for lessons: citation rules first, then the Phase 4 grounding question."""
    chain = GROUNDING_PROMPT | llm | StrOutputParser()

    def check_lesson(state: LearningState) -> dict[str, Any]:
        step = "check_lesson"
        lesson = Lesson.from_dict(state["lesson"])
        evidence = state.get("evidence", [])
        problem = lesson_citation_problem(lesson, evidence)
        if problem is None:
            try:
                verdict = parse_yes_no(_ask(chain, {"context": format_evidence(evidence), "answer": lesson.fact_text()}))
            except Exception as exc:  # noqa: BLE001
                return _error(exc, settings, step)
            if verdict:
                return {
                    "lesson_grounded": True,
                    "lesson_problem": "",
                    "decisions": ["Lesson check: every fact was judged supported by the sources"],
                    "steps": [step],
                }
            problem = "the checker found facts the sources don't support" if verdict is False else "the checker's reply was unclear"
        return {"lesson_grounded": False, "lesson_problem": problem, "decisions": [f"Lesson check failed: {problem}"], "steps": [step]}

    return check_lesson


def report_unreliable_lesson(state: LearningState) -> dict[str, Any]:
    # The unsupported lesson is removed from the state, so it can never be shown.
    return {"message": UNRELIABLE_LESSON_MESSAGE, "refused": True, "lesson": {}, "steps": ["report_unreliable_lesson"]}


# ---------- 8. Assess: write the quiz ----------

def make_create_quiz_node(llm: Runnable, json_llm: Runnable, settings: Settings, repository: LearnerRepository) -> NodeFunction:
    quiz_chain = QUIZ_PROMPT | json_llm | StrOutputParser()
    key_chain = KEY_CHECK_PROMPT | llm | StrOutputParser()

    def create_quiz(state: LearningState) -> dict[str, Any]:
        step = "create_quiz"
        topic = state["topic"]
        lesson = Lesson.from_dict(state["lesson"]) if state.get("lesson") else None
        evidence = [] if state.get("source_mode") == "general" else state.get("evidence", [])
        if lesson is not None:
            material = lesson.as_material()
            concepts = list(lesson.concepts)
        elif evidence:
            material = format_evidence(evidence)
            concepts = list(state.get("focus_concepts") or [])
        else:
            material = f"(No study material: use well-established general knowledge about {topic}.)"
            concepts = list(state.get("focus_concepts") or [])
        concepts = concepts or [normalize_concept(topic)]

        quiz_id = uuid.uuid4().hex[:12]
        try:
            quiz, notes, _ = generate_quiz(
                quiz_chain, quiz_id, topic, state.get("level", "beginner"), material, concepts,
                settings.quiz_multiple_choice, settings.quiz_short_answer,
                settings.structured_output_retries, focus_concepts=state.get("focus_concepts"),
            )
            if state.get("check_evidence") and any(isinstance(q, MultipleChoiceQuestion) for q in quiz.questions):
                quiz, key_notes = verify_answer_keys(key_chain, quiz, material)
                notes += key_notes
        except StructuredOutputError as exc:
            return {"error": str(exc), "steps": [step]}
        except Exception as exc:  # noqa: BLE001
            return _error(exc, settings, step)
        if not quiz.questions:
            return {"error": NO_QUESTIONS_MESSAGE, "steps": [step]}

        try:
            learner_id = state["learner_id"]
            lesson_concepts = list(lesson.concepts) if lesson else []
            repository.add_concepts(learner_id, topic, list(dict.fromkeys(lesson_concepts + quiz.concepts())))
            if lesson and lesson.prerequisites:
                repository.add_prerequisites(learner_id, lesson_concepts, list(lesson.prerequisites))
            repository.save_activity_material(
                learner_id, state["thread_id"], quiz.to_dict(), lesson.to_dict() if lesson else None,
                state.get("lesson_id") if lesson else None, evidence, state.get("evidence_kind", "general"),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Could not save the lesson and quiz")
            return {"error": "Could not save the quiz. Check the terminal logs.", "steps": [step]}

        count = len(quiz.questions)
        return {
            "quiz": quiz.to_dict(),
            "quiz_id": quiz.quiz_id,
            "quiz_notes": notes,
            "decisions": [f"Wrote a quiz with {count} question(s) on: {', '.join(quiz.concepts())}"],
            "steps": [step],
        }

    return create_quiz


# ---------- 9. Pause for the learner, then grade ----------

def wait_for_answers(state: LearningState) -> dict[str, Any]:
    """PAUSE here. interrupt() saves the state (checkpoint) and ends this run.

    The app shows the quiz WITHOUT answers. When the learner submits, the app resumes
    the graph with Command(resume=answers), and interrupt() returns those answers.
    Only the quiz id is passed out, never the answer key.
    """
    answers = interrupt({"quiz_id": state.get("quiz_id", "")})
    if not isinstance(answers, dict):
        answers = {}
    return {"answers": answers, "error": "", "steps": ["wait_for_answers"]}


def make_grade_answers_node(json_llm: Runnable, settings: Settings) -> NodeFunction:
    grader = ShortAnswerGrader(RUBRIC_PROMPT | json_llm | StrOutputParser(), settings.structured_output_retries)

    def grade_answers(state: LearningState) -> dict[str, Any]:
        step = "grade_answers"
        quiz = Quiz.from_dict(state["quiz"])
        try:
            result = grade_quiz(quiz, state.get("answers", {}), grader)
        except Exception as exc:  # noqa: BLE001 - e.g. Ollama stopped: the learner can submit again
            failure = _error(exc, settings, step)
            failure["error"] += " Your quiz is still open: submit it again when the model is ready."
            return failure
        ungraded = sum(1 for item in result.results if not item.graded)
        note = f"Graded {len(result.results)} answer(s): {result.correct_count} correct"
        if ungraded:
            note += f", {ungraded} could not be graded reliably (left out of mastery)"
        return {"assessment": result.to_dict(), "decisions": [note], "steps": [step]}

    return grade_answers


# ---------- 10. Update the learner profile (computed, not yet saved) ----------

def make_update_profile_node(repository: LearnerRepository, clock: Clock) -> NodeFunction:
    def update_profile(state: LearningState) -> dict[str, Any]:
        step = ["update_profile"]
        today = clock()
        result = AssessmentResult.from_dict(state["assessment"])
        topic = state.get("topic", "")
        try:
            # Read fresh progress: the learner may resume this quiz days later.
            current = {item.concept: item for item in repository.get_progress(state["learner_id"])}
        except Exception:  # noqa: BLE001
            logger.exception("Could not read progress")
            return {"error": "Could not read your saved progress. Check the terminal logs.", "steps": step}

        before = dict(current)
        misconceptions = []
        for item in result.results:
            if not item.graded:
                continue
            progress = current.get(item.concept) or ConceptProgress(concept=item.concept, topic=topic)
            current[item.concept] = update_progress(progress, item.score, today)
            if item.misconception:
                misconceptions.append({"concept": item.concept, "note": item.misconception})

        changed = [concept for concept in current if before.get(concept) != current[concept]]
        notes = [
            describe_change(before.get(c) or ConceptProgress(concept=c, topic=topic), current[c]) for c in changed
        ]
        return {
            "progress_updates": [current[c].to_dict() for c in changed],
            "profile": [item.to_dict() for item in current.values()],
            "new_misconceptions": misconceptions,
            "mastery_notes": notes,
            "decisions": [f"Updated mastery for {len(changed)} concept(s)"],
            "steps": step,
        }

    return update_profile


# ---------- 11. Plan: remediation, practice, revision, or advancement ----------

def make_recommend_next_node(repository: LearnerRepository, clock: Clock) -> NodeFunction:
    def recommend_next(state: LearningState) -> dict[str, Any]:
        step = ["recommend_next"]
        learner_id = state["learner_id"]
        try:
            levels = repository.topic_levels(learner_id)
            prerequisites = repository.get_prerequisites(learner_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not read progress for planning")
            return {"error": "Could not read your saved progress. Check the terminal logs.", "steps": step}
        if state.get("quiz") and state.get("topic"):
            current = levels.get(state["topic"], "beginner")
            level = state.get("level", "beginner")
            if LEVELS.index(level) >= LEVELS.index(current):
                levels[state["topic"]] = level
        suggestions = recommend(_progress_rows(state), prerequisites, clock(), levels)
        result: dict[str, Any] = {"recommendations": [item.to_dict() for item in suggestions], "steps": step}
        if suggestions:
            top = suggestions[0]
            result["decisions"] = [f"Planner: next, {top.action} “{top.concept}”, because {top.reason[0].lower()}{top.reason[1:]}"]
        else:
            result["decisions"] = ["Planner: nothing to recommend yet"]
        if state.get("chosen_action") == "next":
            result["message"] = (
                "Here's what I suggest next, most important first." if suggestions else NO_RECOMMENDATIONS_MESSAGE
            )
        return result

    return recommend_next


# ---------- 12. Save the updated progress ----------

def make_save_progress_node(repository: LearnerRepository) -> NodeFunction:
    def save_progress(state: LearningState) -> dict[str, Any]:
        step = ["save_progress"]
        try:
            repository.record_assessment(
                state["learner_id"],
                state["quiz_id"],
                state["assessment"],
                [ConceptProgress.from_dict(row) for row in state.get("progress_updates", [])],
                state.get("new_misconceptions", []),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Could not save progress")
            return {"error": "Could not save your results. Check the terminal logs.", "steps": step}
        return {"saved": True, "decisions": ["Saved your results and updated mastery"], "steps": step}

    return save_progress


def handle_error(state: LearningState) -> dict[str, Any]:
    logger.warning("Learning workflow stopped with an error: %s", state.get("error"))
    return {"steps": ["handle_error"]}


# ---------- Routers ----------

def _teach_or_assess(state: LearningState) -> str:
    return "generate_lesson" if state.get("chosen_action") == "lesson" else "create_quiz"


def route_after_profile(state: LearningState) -> str:
    return "handle_error" if state.get("error") else "understand_request"


def route_after_request(state: LearningState) -> str:
    """'next' -> plan. General knowledge -> teach/assess directly. Documents -> retrieve."""
    if state.get("error"):
        return "handle_error"
    if state.get("chosen_action") == "next":
        return "recommend_next"
    if state.get("source_mode") == "general":
        return _teach_or_assess(state)
    return "retrieve_material"


def route_after_retrieval(state: LearningState) -> str:
    return "handle_error" if state.get("error") else "check_material"


def make_route_after_material_check(settings: Settings, web_available: bool) -> Router:
    def route_after_material_check(state: LearningState) -> str:
        """Good material -> teach. Weak -> rewrite and search again (bounded),
        then the web if allowed, then partial material, then an honest refusal."""
        if state.get("error"):
            return "handle_error"
        evidence = state.get("evidence", [])
        if evidence and state.get("evidence_verdict") in ("sufficient", "contradictory"):
            return _teach_or_assess(state)
        # Basic mode (checks off) is like Phase 2: no rewriting, no second search.
        if state.get("check_evidence") and state.get("material_attempts", 0) <= settings.max_retrieval_retries:
            return "rewrite_material_query"
        if web_available and state.get("allow_web") and not state.get("used_web"):
            return "research_web"
        if evidence:
            return "use_partial_material"
        return "report_no_material"

    return route_after_material_check


def use_partial_material(state: LearningState) -> dict[str, Any]:
    """The documents cover the topic only partly: teach what they say, and say so."""
    return {
        "partial_evidence": True,
        "decisions": ["Your documents only partly cover this topic: the lesson sticks to what they say"],
        "steps": ["use_partial_material"],
    }


def route_after_partial(state: LearningState) -> str:
    return _teach_or_assess(state)


def route_after_rewrite(state: LearningState) -> str:
    return "handle_error" if state.get("error") else "retrieve_material"


def route_after_research(state: LearningState) -> str:
    if state.get("error"):
        return "handle_error"
    return _teach_or_assess(state) if state.get("evidence") else "report_no_material"


def route_after_lesson(state: LearningState) -> str:
    if state.get("error"):
        return "handle_error"
    if state.get("check_evidence") and state.get("evidence_kind") in ("documents", "web") and state.get("evidence"):
        return "check_lesson"
    return "create_quiz"


def make_route_after_lesson_check(settings: Settings) -> Router:
    def route_after_lesson_check(state: LearningState) -> str:
        if state.get("error"):
            return "handle_error"
        if state.get("lesson_grounded"):
            return "create_quiz"
        if state.get("lesson_attempts", 0) < 1 + settings.max_answer_retries:
            return "generate_lesson"
        return "report_unreliable_lesson"

    return route_after_lesson_check


def route_after_quiz(state: LearningState) -> str:
    return "handle_error" if state.get("error") else "wait_for_answers"


def route_after_grading(state: LearningState) -> str:
    """A grading failure goes back to waiting, so the learner can submit again."""
    return "wait_for_answers" if state.get("error") else "update_profile"


def route_after_update(state: LearningState) -> str:
    return "handle_error" if state.get("error") else "recommend_next"


def route_after_recommendation(state: LearningState) -> str:
    if state.get("error"):
        return "handle_error"
    return "save_progress" if state.get("assessment") else END


def route_after_save(state: LearningState) -> str:
    return "handle_error" if state.get("error") else END