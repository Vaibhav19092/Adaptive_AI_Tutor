"""Phase 5: checking the model's lesson and quiz JSON, and hiding answers."""

from dataclasses import fields

import pytest

from adaptive_tutor.assessment.schemas import (
    AssessmentResult,
    Lesson,
    MultipleChoiceQuestion,
    PointResult,
    QuestionResult,
    QuestionView,
    Quiz,
    SchemaError,
    ShortAnswerQuestion,
    match_concept,
    normalize_concept,
    parse_lesson,
    parse_option_letter,
    parse_quiz,
    public_view,
)

LESSON_JSON = {
    "title": "Photosynthesis",
    "summary": "How plants make food.",
    "explanation": "Plants use light to make sugar [1].",
    "analogy": "Like a solar-powered kitchen.",
    "example": "A leaf in sunlight.",
    "key_points": ["Light is needed [1]", "Sugar is made [1]", "Oxygen is released [1]"],
    "concepts": ["Photosynthesis", "Chlorophyll."],
    "prerequisites": ["cells", "photosynthesis"],
}

MC = {"concept": "chlorophyll", "question": "What absorbs light?", "options": ["Chlorophyll", "Water", "Salt", "Air"],
      "answer": "A", "explanation": "Chlorophyll is the green pigment."}
SA = {"concept": "photosynthesis", "question": "What does photosynthesis make?",
      "key_points": ["sugar", "oxygen"], "model_answer": "Sugar and oxygen."}


def test_parse_lesson_normalizes_concepts_and_drops_self_prerequisites():
    lesson = parse_lesson(LESSON_JSON, "photosynthesis", "beginner")
    assert lesson.concepts == ("photosynthesis", "chlorophyll")
    assert lesson.prerequisites == ("cells",)  # "photosynthesis" is the lesson's own concept
    assert "Plants use light" in lesson.fact_text()
    assert "solar-powered" not in lesson.fact_text()  # the analogy isn't a checked fact


@pytest.mark.parametrize("missing", ["title", "explanation", "analogy", "example"])
def test_lesson_without_a_required_part_is_rejected(missing):
    data = {key: value for key, value in LESSON_JSON.items() if key != missing}
    with pytest.raises(SchemaError, match=missing):
        parse_lesson(data, "t", "beginner")


def test_lesson_lists_are_lenient():
    data = dict(LESSON_JSON, key_points="first point\nsecond point", prerequisites=None)
    lesson = parse_lesson(data, "t", "expert")
    assert lesson.key_points == ("first point", "second point")
    assert lesson.prerequisites == ()
    assert lesson.level == "beginner"  # unknown levels fall back safely


def test_lesson_round_trips_through_a_dict():
    lesson = parse_lesson(LESSON_JSON, "photosynthesis", "beginner")
    assert Lesson.from_dict(lesson.to_dict()) == lesson


@pytest.mark.parametrize(
    "reply, expected",
    [("B", 1), ("b)", 1), ("Option C", 2), ("The answer is D.", 3), ("(A)", 0), ("Water", 1), ("E", None), ("maybe", None)],
)
def test_parse_option_letter(reply, expected):
    assert parse_option_letter(reply, 4, ("Chlorophyll", "Water", "Salt", "Air")) == expected


def test_concepts_are_matched_to_lesson_concepts():
    assert normalize_concept("  Neural Networks. ") == "neural networks"
    assert match_concept("Neural Network", ("neural networks", "backpropagation")) == "neural networks"
    assert match_concept("gradient descent", ("neural networks",)) == "gradient descent"


def test_parse_quiz_keeps_good_questions_and_explains_dropped_ones():
    bad_mc = dict(MC, options=["Same", "same", "Other"])
    data = {"multiple_choice": [MC, bad_mc], "short_answer": [SA]}
    quiz, notes = parse_quiz(data, "quiz1", "photosynthesis", "beginner", ("photosynthesis", "chlorophyll"), 2, 1)
    assert [q.kind for q in quiz.questions] == ["multiple_choice", "short_answer"]
    assert quiz.questions[0].correct_index == 0
    assert any("two options are the same" in note for note in notes)
    assert any("asked for 2, got 1" in note for note in notes)


def test_option_letter_prefixes_are_removed():
    data = {"multiple_choice": [dict(MC, options=["A) Chlorophyll", "B) Water", "C) Salt"])]}
    quiz, _ = parse_quiz(data, "q", "t", "beginner", (), 1, 0)
    assert quiz.questions[0].options == ("Chlorophyll", "Water", "Salt")


def test_quiz_with_no_usable_questions_is_rejected():
    data = {"multiple_choice": [dict(MC, answer="Z")]}
    with pytest.raises(SchemaError, match="no usable questions"):
        parse_quiz(data, "q", "t", "beginner", (), 1, 0)


def test_public_view_contains_no_answers():
    quiz, _ = parse_quiz({"multiple_choice": [MC], "short_answer": [SA]}, "q", "t", "beginner", (), 1, 1)
    views = public_view(quiz)
    allowed = {"question_id", "kind", "concept", "prompt", "options"}
    assert {f.name for f in fields(QuestionView)} == allowed
    shown = repr(views)
    assert "Chlorophyll is the green pigment" not in shown  # explanation
    assert "Sugar and oxygen" not in shown  # model answer
    assert "oxygen'" not in shown  # rubric key points
    assert views[0].options == ("Chlorophyll", "Water", "Salt", "Air")


def test_quiz_round_trips_through_a_dict():
    quiz, _ = parse_quiz({"multiple_choice": [MC], "short_answer": [SA]}, "q", "t", "beginner", (), 1, 1)
    again = Quiz.from_dict(quiz.to_dict())
    assert again == quiz
    assert isinstance(again.questions[0], MultipleChoiceQuestion)
    assert isinstance(again.questions[1], ShortAnswerQuestion)


def test_assessment_result_scores_only_graded_questions():
    def result(score, graded=True):
        return QuestionResult("q", "short_answer", "c", "p", "a", score, graded, "f", "x", (PointResult("p", True),))

    assessment = AssessmentResult("quiz", (result(1.0), result(0.5), result(0.0, graded=False)))
    assert assessment.score == 0.75
    assert assessment.correct_count == 2
    assert AssessmentResult.from_dict(assessment.to_dict()) == assessment