"""Phase 5: write a short structured lesson as JSON, grounded in evidence when there is some.

A lesson has a title, a summary, a simple explanation, an everyday analogy, one
concrete example, key points, and concept labels. Concept labels are what quiz
questions and mastery estimates are attached to later.

Lessons from documents or the web must cite their sources ([1] or [W1]) in the
explanation and key points; the graph checks this (Self-RAG) before the lesson is shown.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from adaptive_tutor.assessment.schemas import LEVELS, Lesson, parse_lesson
from adaptive_tutor.llm.structured import Invokable, invoke_structured
from adaptive_tutor.retrieval.self_rag import citation_problem
from adaptive_tutor.teaching.evidence import Evidence, format_evidence, is_web_evidence

# Double braces {{ }} are literal braces: single braces are prompt variables.
LESSON_SYSTEM_PROMPT = (
    "You are a patient, friendly tutor who writes short structured lessons.\n"
    "Reply with ONLY one JSON object with exactly these keys:\n"
    '{{"title": "a short title", '
    '"summary": "one or two sentences", '
    '"explanation": "a clear explanation in simple words", '
    '"analogy": "an everyday comparison that makes the idea easier", '
    '"example": "one concrete example", '
    '"key_points": ["3 to 5 short facts to remember"], '
    '"concepts": ["1 to 4 concept names, 1 to 4 words each"], '
    '"prerequisites": ["0 to 2 concepts the learner should know first"]}}\n'
    "Write for the learner's level, avoid unexplained jargon, and keep the lesson under 350 words."
)

LESSON_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", LESSON_SYSTEM_PROMPT),
        (
            "human",
            "Topic: {topic}\nLevel: {level}. {level_guidance}\n\n{source_instructions}\n\n"
            "{context}{focus}{extra_instructions}{format_feedback}",
        ),
    ]
)

LEVEL_GUIDANCE = {
    "beginner": "Assume no background knowledge. Use short sentences and define every new term.",
    "intermediate": "Assume the basics are known. Explain how and why, not only what.",
    "advanced": "Be precise and include important details, limits, and trade-offs.",
}

SOURCE_INSTRUCTIONS = (
    "Use ONLY facts from the numbered sources below. In the explanation and in every key point, "
    "cite the source after each fact, like {example}. The analogy and the example may be your own "
    "teaching aids; do not cite them. Never invent citations."
)

GENERAL_INSTRUCTIONS = (
    "No sources are available, so use your general knowledge. Do not add citations. "
    "Leave out any fact you are not sure about."
)

FOCUS_INSTRUCTION = (
    "\n\nThe learner has struggled with: {concepts}. Explain these especially simply, step by step, "
    "and include them in the concepts list."
)

CONTRADICTION_INSTRUCTION = (
    "\n\nImportant: the sources disagree about part of this topic. Do not pick a side: "
    "describe each view and cite its source."
)

RETRY_INSTRUCTION = (
    "\n\nImportant: your previous lesson was rejected because {problem}. Write it again using ONLY "
    "facts stated in the sources, with a citation after every fact in the explanation and key points."
)


def build_lesson_inputs(
    topic: str,
    level: str,
    evidence: list[Evidence],
    focus_concepts: list[str] | None = None,
    contradiction: bool = False,
    problem: str = "",
) -> dict[str, Any]:
    """Fill in the prompt variables (everything except format_feedback)."""
    level = level if level in LEVELS else "beginner"
    if evidence:
        example = "[W1]" if is_web_evidence(evidence) else "[1]"
        source_instructions = SOURCE_INSTRUCTIONS.format(example=example)
        context = "Sources:\n\n" + format_evidence(evidence)
    else:
        source_instructions = GENERAL_INSTRUCTIONS
        context = ""

    extra = ""
    if contradiction:
        extra += CONTRADICTION_INSTRUCTION
    if problem:
        extra += RETRY_INSTRUCTION.format(problem=problem)
    focus = FOCUS_INSTRUCTION.format(concepts=", ".join(focus_concepts)) if focus_concepts else ""

    return {
        "topic": topic,
        "level": level,
        "level_guidance": LEVEL_GUIDANCE[level],
        "source_instructions": source_instructions,
        "context": context,
        "focus": focus,
        "extra_instructions": extra,
    }


def generate_lesson(
    chain: Invokable,
    topic: str,
    level: str,
    evidence: list[Evidence],
    retries: int,
    focus_concepts: list[str] | None = None,
    contradiction: bool = False,
    problem: str = "",
) -> tuple[Lesson, int]:
    """Ask the model for a lesson and check its JSON. Returns (lesson, tries used)."""
    inputs = build_lesson_inputs(topic, level, evidence, focus_concepts, contradiction, problem)
    return invoke_structured(chain, inputs, lambda data: parse_lesson(data, topic, inputs["level"]), retries, "lesson")


def lesson_citation_problem(lesson: Lesson, evidence: list[Evidence]) -> str | None:
    """Rule-based check: does the lesson cite sources that exist? (Same rules as Phase 4.)"""
    return citation_problem(lesson.fact_text(), len(evidence), web=is_web_evidence(evidence))