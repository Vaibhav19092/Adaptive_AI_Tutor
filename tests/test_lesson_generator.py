"""Phase 5: lesson prompts, grounding instructions, and the citation rule."""

import json

import pytest

from adaptive_tutor.assessment.schemas import Lesson
from adaptive_tutor.llm.structured import StructuredOutputError
from adaptive_tutor.teaching.evidence import format_evidence
from adaptive_tutor.teaching.lesson_generator import (
    GENERAL_INSTRUCTIONS,
    LESSON_PROMPT,
    build_lesson_inputs,
    generate_lesson,
    lesson_citation_problem,
)

DOC = {"kind": "document", "key": "d-1", "citation": "bio.pdf, page 2", "url": "", "text": "Plants make sugar.", "score": 0.8}
WEB = {"kind": "web", "key": "u", "citation": "Leaf (en.wikipedia.org)", "url": "u", "text": "Leaves are green.", "score": 0.7}
LESSON = {
    "title": "T", "summary": "S", "explanation": "Plants make sugar [1].", "analogy": "A", "example": "E",
    "key_points": ["sugar [1]", "light [1]"], "concepts": ["photosynthesis"],
}


class Chain:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs = []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_evidence_is_numbered_like_phases_2_and_3():
    assert format_evidence([DOC]).startswith("[1] (bio.pdf, page 2)")
    assert format_evidence([WEB]).startswith("[W1] (Leaf (en.wikipedia.org))")


def test_grounded_inputs_include_sources_focus_and_retry_reason():
    inputs = build_lesson_inputs("photosynthesis", "beginner", [DOC], ["chlorophyll"], contradiction=True, problem="no citations")
    assert "[1]" in inputs["source_instructions"]
    assert "Plants make sugar." in inputs["context"]
    assert "chlorophyll" in inputs["focus"]
    assert "disagree" in inputs["extra_instructions"] and "no citations" in inputs["extra_instructions"]
    LESSON_PROMPT.invoke(dict(inputs, format_feedback=""))  # every variable is filled in


def test_general_lessons_have_no_sources_or_citations():
    inputs = build_lesson_inputs("photosynthesis", "advanced", [])
    assert inputs["source_instructions"] == GENERAL_INSTRUCTIONS
    assert inputs["context"] == ""


def test_generate_lesson_retries_bad_json_then_succeeds():
    chain = Chain("not json", json.dumps(LESSON))
    lesson, tries = generate_lesson(chain, "photosynthesis", "beginner", [DOC], retries=1)
    assert tries == 2 and lesson.topic == "photosynthesis"
    assert chain.inputs[1]["format_feedback"]


def test_generate_lesson_gives_up_after_the_limit():
    with pytest.raises(StructuredOutputError):
        generate_lesson(Chain("nope"), "t", "beginner", [], retries=0)


def test_citation_rule_for_lessons():
    lesson = Lesson.from_dict({**LESSON, "topic": "t", "level": "beginner", "prerequisites": []})
    assert lesson_citation_problem(lesson, [DOC]) is None
    assert "only 0 source" in lesson_citation_problem(lesson, [])
    uncited = Lesson.from_dict({**LESSON, "explanation": "No cites.", "key_points": ["a", "b"],
                                "topic": "t", "level": "beginner", "prerequisites": []})
    assert lesson_citation_problem(uncited, [DOC]) == "the answer has no citations"