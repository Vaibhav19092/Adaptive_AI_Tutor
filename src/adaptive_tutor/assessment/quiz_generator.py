"""Phase 5: write a quiz (multiple choice + short answer) from a lesson or from evidence.

Every question gets a concept label, so results can update mastery per concept.
Two safety steps follow generation:

1. Shuffle the options. Small models like to put the right answer first; shuffling
   (with a fixed seed per question, so it is repeatable) removes that pattern.
2. Answer-key check (when careful checking is on): the model answers each
   multiple-choice question again, separately, from the material. If its answer
   disagrees with the key, the key may be wrong, so the question is dropped.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from adaptive_tutor.assessment.schemas import (
    LEVELS,
    MultipleChoiceQuestion,
    Quiz,
    parse_option_letter,
    parse_quiz,
)
from adaptive_tutor.llm.structured import Invokable, invoke_structured

logger = logging.getLogger(__name__)

QUIZ_SYSTEM_PROMPT = (
    "You write quiz questions that check a learner's understanding of some study material.\n"
    "Reply with ONLY one JSON object in this format:\n"
    '{{"multiple_choice": [{{"concept": "concept name", "question": "...", '
    '"options": ["...", "...", "...", "..."], "answer": "B", '
    '"explanation": "why the right option is right"}}], '
    '"short_answer": [{{"concept": "concept name", "question": "...", '
    '"key_points": ["a point a good answer must make"], "model_answer": "a short good answer"}}]}}\n'
    "Rules: use only facts from the material; give each multiple-choice question 4 options with "
    "exactly one correct option; \"answer\" is the letter of the correct option (A, B, C or D); "
    "don't mention option letters in the explanation; give each short-answer question 2 or 3 key points."
)

QUIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QUIZ_SYSTEM_PROMPT),
        (
            "human",
            "Topic: {topic}\nLevel: {level}\n"
            "Write {multiple_choice} multiple-choice and {short_answer} short-answer question(s).\n"
            "Use these concept names: {concepts}.{focus}\n\nMaterial:\n{material}{format_feedback}",
        ),
    ]
)

FOCUS_INSTRUCTION = " Ask at least one question about each concept the learner finds hard: {concepts}."

KEY_CHECK_SYSTEM_PROMPT = (
    "You answer a multiple-choice question using only the study material given.\n"
    "Reply with only the letter of the correct option."
)

KEY_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [("system", KEY_CHECK_SYSTEM_PROMPT), ("human", "Material:\n{material}\n\nQuestion: {question}\n{options}")]
)


def build_quiz_inputs(
    topic: str,
    level: str,
    material: str,
    concepts: list[str],
    multiple_choice: int,
    short_answer: int,
    focus_concepts: list[str] | None = None,
) -> dict[str, Any]:
    focus = FOCUS_INSTRUCTION.format(concepts=", ".join(focus_concepts)) if focus_concepts else ""
    return {
        "topic": topic,
        "level": level if level in LEVELS else "beginner",
        "material": material,
        "concepts": ", ".join(concepts) if concepts else topic,
        "multiple_choice": multiple_choice,
        "short_answer": short_answer,
        "focus": focus,
    }


def shuffle_options(question: MultipleChoiceQuestion, seed: str) -> MultipleChoiceQuestion:
    """Mix up the options and move the correct index with them. Same seed -> same order."""
    order = list(range(len(question.options)))
    random.Random(seed).shuffle(order)
    options = tuple(question.options[i] for i in order)
    return replace(question, options=options, correct_index=order.index(question.correct_index))


def generate_quiz(
    chain: Invokable,
    quiz_id: str,
    topic: str,
    level: str,
    material: str,
    concepts: list[str],
    multiple_choice: int,
    short_answer: int,
    retries: int,
    focus_concepts: list[str] | None = None,
) -> tuple[Quiz, list[str], int]:
    """Ask for a quiz, check it, and shuffle the options. Returns (quiz, notes, tries used)."""
    inputs = build_quiz_inputs(topic, level, material, concepts, multiple_choice, short_answer, focus_concepts)
    allowed = tuple(concepts)

    def parse(data: Any) -> tuple[Quiz, list[str]]:
        return parse_quiz(data, quiz_id, topic, inputs["level"], allowed, multiple_choice, short_answer)

    (quiz, notes), tries = invoke_structured(chain, inputs, parse, retries, "quiz")
    questions = tuple(
        shuffle_options(question, f"{quiz_id}-{question.question_id}")
        if isinstance(question, MultipleChoiceQuestion)
        else question
        for question in quiz.questions
    )
    return replace(quiz, questions=questions), notes, tries


def format_options(question: MultipleChoiceQuestion) -> str:
    return "\n".join(f"{'ABCDE'[i]}) {option}" for i, option in enumerate(question.options))


def verify_answer_keys(chain: Invokable, quiz: Quiz, material: str) -> tuple[Quiz, list[str]]:
    """Self-check each multiple-choice answer key. Disagreements are dropped; unclear replies kept.

    Model connection errors are not caught here (the graph node handles them).
    """
    kept, notes = [], []
    for question in quiz.questions:
        if not isinstance(question, MultipleChoiceQuestion):
            kept.append(question)
            continue
        reply = str(
            chain.invoke({"material": material, "question": question.prompt, "options": format_options(question)})
        )
        answer = parse_option_letter(reply, len(question.options))
        if answer is None:
            notes.append(f"Answer-key check for {question.question_id}: unclear reply, question kept")
            kept.append(question)
        elif answer == question.correct_index:
            notes.append(f"Answer-key check for {question.question_id}: confirmed")
            kept.append(question)
        else:
            notes.append(f"Answer-key check for {question.question_id}: the check disagreed with the key, question dropped")
    logger.info("Answer-key check kept %d of %d questions", len(kept), len(quiz.questions))
    return replace(quiz, questions=tuple(kept)), notes