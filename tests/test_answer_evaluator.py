"""Phase 5: deterministic multiple-choice grading and rubric-based short-answer grading."""

import json

from adaptive_tutor.assessment.answer_evaluator import (
    NO_ANSWER_FEEDBACK,
    RUBRIC_PROMPT,
    UNGRADED_FEEDBACK,
    ShortAnswerGrader,
    grade_multiple_choice,
    grade_quiz,
    parse_rubric_reply,
)
from adaptive_tutor.assessment.schemas import MultipleChoiceQuestion, Quiz, SchemaError, ShortAnswerQuestion

QUESTION = MultipleChoiceQuestion("q1", "chlorophyll", "What absorbs light?", ("Water", "Chlorophyll", "Salt"), 1, "It is green.")
SHORT = ShortAnswerQuestion("q2", "photosynthesis", "What is made?", ("sugar is made", "oxygen is released"), "Sugar and oxygen.")


class Chain:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.inputs = []

    def invoke(self, inputs):
        self.calls += 1
        self.inputs.append(inputs)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def rubric(*covered, misconception=""):
    return json.dumps({
        "points": [{"point": n, "covered": c, "reason": f"reason {n}"} for n, c in enumerate(covered, start=1)],
        "misconception": misconception,
        "feedback": "Good effort.",
    })


def test_multiple_choice_correct_answer():
    result = grade_multiple_choice(QUESTION, 1)
    assert result.score == 1.0 and result.correct
    assert result.feedback.startswith("Correct!")


def test_multiple_choice_wrong_answer_explains_the_mistake():
    result = grade_multiple_choice(QUESTION, 0)
    assert result.score == 0.0 and not result.correct
    assert "You chose “Water”" in result.feedback and "“Chlorophyll”" in result.feedback
    assert "It is green." in result.feedback
    assert result.misconception.startswith("Possibly")


def test_multiple_choice_grading_is_deterministic_and_handles_blanks():
    assert grade_multiple_choice(QUESTION, 0) == grade_multiple_choice(QUESTION, 0)
    for blank in (None, 7, True, "1"):
        result = grade_multiple_choice(QUESTION, blank)
        assert result.score == 0.0 and result.feedback == NO_ANSWER_FEEDBACK


def test_short_answer_is_scored_per_key_point():
    chain = Chain(rubric(True, False, misconception="Thinks plants eat soil"))
    result = ShortAnswerGrader(chain, retries=1).grade(SHORT, "Plants make sugar.")
    assert result.score == 0.5 and result.correct and result.graded
    assert [p.covered for p in result.point_results] == [True, False]
    assert "Missing: oxygen is released" in result.feedback
    assert result.misconception == "Thinks plants eat soil"
    RUBRIC_PROMPT.invoke(chain.inputs[0])  # every prompt variable is filled in


def test_blank_short_answer_needs_no_model_call():
    chain = Chain(rubric(True, True))
    result = ShortAnswerGrader(chain, retries=1).grade(SHORT, "   ")
    assert chain.calls == 0
    assert result.score == 0.0 and result.graded


def test_unusable_grading_reply_is_marked_ungraded_not_guessed():
    chain = Chain("I think it's fine", '{"points": [{"point": 1, "covered": true}]}')
    result = ShortAnswerGrader(chain, retries=1).grade(SHORT, "Sugar.")
    assert chain.calls == 2  # one try + one retry
    assert not result.graded and result.feedback == UNGRADED_FEEDBACK
    assert not result.correct


def test_parse_rubric_reply_needs_every_point():
    points, misconception, _ = parse_rubric_reply(
        {"points": [{"point": "2", "covered": "no"}, {"point": 1, "covered": "yes"}], "misconception": "none"},
        ("a", "b"),
    )
    assert [p.covered for p in points] == [True, False]
    assert misconception == ""
    try:
        parse_rubric_reply({"points": [{"point": 1, "covered": True}]}, ("a", "b"))
    except SchemaError as exc:
        assert "2" in str(exc)
    else:
        raise AssertionError("a missing point must be rejected")


def test_grade_quiz_combines_both_kinds():
    quiz = Quiz("quiz", "t", "beginner", (QUESTION, SHORT))
    result = grade_quiz(quiz, {"q1": 1, "q2": "Sugar and oxygen"}, ShortAnswerGrader(Chain(rubric(True, True)), 1))
    assert result.score == 1.0 and result.correct_count == 2