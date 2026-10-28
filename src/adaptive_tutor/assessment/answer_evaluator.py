"""Phase 5: grade the learner's answers and explain the mistakes.

- Multiple choice: deterministic. The chosen option is compared with the key;
  no model call, so the same answer always gets the same grade.
- Short answer: rubric-based. The model checks the answer against each key
  point and gives a reason per point, as JSON. The score is the share of key
  points covered. If the model's reply is unusable even after a retry, the answer
  is marked "not graded" (it doesn't count toward mastery) instead of guessing.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from adaptive_tutor.assessment.schemas import (
    AssessmentResult,
    MultipleChoiceQuestion,
    PointResult,
    Question,
    QuestionResult,
    Quiz,
    SchemaError,
    ShortAnswerQuestion,
)
from adaptive_tutor.llm.structured import Invokable, StructuredOutputError, invoke_structured

logger = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 1500  # longer answers are cut before grading (keeps the prompt small)

RUBRIC_SYSTEM_PROMPT = (
    "You grade a learner's short answer against a list of numbered key points.\n"
    "For each key point decide if the learner's answer states it (in any wording).\n"
    "Reply with ONLY one JSON object in this format:\n"
    '{{"points": [{{"point": 1, "covered": true, "reason": "short reason"}}], '
    '"misconception": "a misunderstanding the answer shows, or an empty string", '
    '"feedback": "one or two kind sentences for the learner"}}\n'
    "Include every key point exactly once. Be fair but strict: vague answers do not cover a point."
)

RUBRIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RUBRIC_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nKey points:\n{key_points}\n\nLearner's answer:\n{answer}{format_feedback}"),
    ]
)

NO_ANSWER_FEEDBACK = "No answer was given, so this counts as incorrect."
UNGRADED_FEEDBACK = (
    "I couldn't grade this answer reliably, so it doesn't count toward your mastery. "
    "Compare it with the model answer below."
)

_TRUE_WORDS = {"true", "yes", "covered", "1"}
_FALSE_WORDS = {"false", "no", "missing", "not covered", "0"}


# ---------- Multiple choice (no model) ----------

def grade_multiple_choice(question: MultipleChoiceQuestion, answer: Any) -> QuestionResult:
    """Compare the chosen option index with the answer key."""
    correct_text = question.correct_option
    valid = isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < len(question.options)
    chosen = answer if valid else None

    if chosen is None:
        score, feedback, misconception, learner_answer = 0.0, NO_ANSWER_FEEDBACK, "", ""
    elif chosen == question.correct_index:
        learner_answer = question.options[chosen]
        score, feedback, misconception = 1.0, f"Correct! {question.explanation}", ""
    else:
        learner_answer = question.options[chosen]
        score = 0.0
        feedback = f"Not quite. You chose “{learner_answer}”, but the answer is “{correct_text}”. {question.explanation}"
        misconception = f"Possibly confused “{learner_answer}” with “{correct_text}”"

    return QuestionResult(
        question_id=question.question_id,
        kind=question.kind,
        concept=question.concept,
        prompt=question.prompt,
        learner_answer=learner_answer,
        score=score,
        graded=True,
        feedback=feedback,
        correct_answer=correct_text,
        misconception=misconception,
    )


# ---------- Short answer (model + rubric) ----------

def _read_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    raise SchemaError(f"'covered' must be true or false, got {value!r}")


def parse_rubric_reply(data: Any, key_points: tuple[str, ...]) -> tuple[tuple[PointResult, ...], str, str]:
    """Check the grader's JSON: one verdict per key point. Returns (points, misconception, feedback)."""
    if not isinstance(data, dict):
        raise SchemaError("the grading reply must be a JSON object")
    items = data.get("points")
    if not isinstance(items, list):
        raise SchemaError("the grading reply has no 'points' list")

    verdicts: dict[int, tuple[bool, str]] = {}
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise SchemaError("each item in 'points' must be an object")
        try:
            number = int(item.get("point", position))
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"point number {item.get('point')!r} is not a number") from exc
        if 1 <= number <= len(key_points) and number not in verdicts:
            verdicts[number] = (_read_bool(item.get("covered")), str(item.get("reason") or "").strip())

    missing = [n for n in range(1, len(key_points) + 1) if n not in verdicts]
    if missing:
        raise SchemaError(f"key point(s) {', '.join(map(str, missing))} were not graded")

    points = tuple(
        PointResult(point=key_points[n - 1], covered=verdicts[n][0], reason=verdicts[n][1])
        for n in range(1, len(key_points) + 1)
    )
    misconception = str(data.get("misconception") or "").strip()
    if misconception.lower() in {"none", "n/a", "no", "null"}:
        misconception = ""
    return points, misconception, str(data.get("feedback") or "").strip()


class ShortAnswerGrader:
    """Grades one short answer with a rubric. `chain` is RUBRIC_PROMPT | json model | parser."""

    def __init__(self, chain: Invokable, retries: int) -> None:
        self._chain = chain
        self._retries = retries

    def grade(self, question: ShortAnswerQuestion, answer: Any) -> QuestionResult:
        text = str(answer or "").strip()[:MAX_ANSWER_CHARS]
        base = {
            "question_id": question.question_id,
            "kind": question.kind,
            "concept": question.concept,
            "prompt": question.prompt,
            "learner_answer": text,
            "correct_answer": question.model_answer,
        }
        if not text:  # nothing to grade: no model call needed
            missed = tuple(PointResult(point, False, "not mentioned") for point in question.key_points)
            return QuestionResult(**base, score=0.0, graded=True, feedback=NO_ANSWER_FEEDBACK, point_results=missed)

        inputs = {
            "question": question.prompt,
            "key_points": "\n".join(f"{n}. {point}" for n, point in enumerate(question.key_points, start=1)),
            "answer": text,
        }
        try:
            (points, misconception, feedback), _ = invoke_structured(
                self._chain, inputs, lambda data: parse_rubric_reply(data, question.key_points), self._retries, "grading"
            )
        except StructuredOutputError:
            # Cautious fallback: don't invent a grade.
            logger.warning("Short answer %s could not be graded reliably", question.question_id)
            return QuestionResult(**base, score=0.0, graded=False, feedback=UNGRADED_FEEDBACK)

        covered = sum(1 for point in points if point.covered)
        score = round(covered / len(points), 3)
        summary = f"You covered {covered} of {len(points)} key point(s)."
        missing = [point.point for point in points if not point.covered]
        if missing:
            summary += " Missing: " + "; ".join(missing) + "."
        return QuestionResult(
            **base,
            score=score,
            graded=True,
            feedback=f"{summary} {feedback}".strip(),
            point_results=points,
            misconception=misconception,
        )


def grade_question(question: Question, answer: Any, short_grader: ShortAnswerGrader) -> QuestionResult:
    if isinstance(question, MultipleChoiceQuestion):
        return grade_multiple_choice(question, answer)
    return short_grader.grade(question, answer)


def grade_quiz(quiz: Quiz, answers: dict[str, Any], short_grader: ShortAnswerGrader) -> AssessmentResult:
    """Grade every question. Answers are {question_id: option index or text}."""
    results = tuple(grade_question(q, answers.get(q.question_id), short_grader) for q in quiz.questions)
    return AssessmentResult(quiz_id=quiz.quiz_id, results=results)