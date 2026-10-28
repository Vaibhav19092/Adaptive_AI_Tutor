"""Phase 5: quiz generation, option shuffling, and the answer-key self-check."""

import json

from adaptive_tutor.assessment.quiz_generator import (
    QUIZ_PROMPT,
    generate_quiz,
    shuffle_options,
    verify_answer_keys,
)
from adaptive_tutor.assessment.schemas import MultipleChoiceQuestion, Quiz, ShortAnswerQuestion

QUIZ_REPLY = json.dumps({
    "multiple_choice": [
        {"concept": "chlorophyll", "question": "What absorbs light?", "options": ["Chlorophyll", "Water", "Salt", "Air"],
         "answer": "A", "explanation": "It is the green pigment."},
    ],
    "short_answer": [
        {"concept": "photosynthesis", "question": "What is made?", "key_points": ["sugar", "oxygen"], "model_answer": "Sugar."},
    ],
})


class Chain:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs = []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def mc(correct_index=0):
    return MultipleChoiceQuestion("q1", "c", "Pick one", ("w", "x", "y", "z"), correct_index, "because")


def test_shuffle_moves_the_correct_index_with_the_options():
    for seed in ("a", "b", "c", "d", "e"):
        shuffled = shuffle_options(mc(), seed)
        assert shuffled.correct_option == "w"
        assert sorted(shuffled.options) == ["w", "x", "y", "z"]
    assert shuffle_options(mc(), "same") == shuffle_options(mc(), "same")  # repeatable


def test_generate_quiz_fills_the_prompt_and_keeps_the_answer_correct():
    chain = Chain(QUIZ_REPLY)
    quiz, notes, tries = generate_quiz(chain, "quiz1", "photosynthesis", "beginner", "Material text",
                                       ["photosynthesis", "chlorophyll"], 1, 1, retries=1)
    assert tries == 1 and notes == []
    assert quiz.questions[0].correct_option == "Chlorophyll"
    QUIZ_PROMPT.invoke(chain.inputs[0])  # every prompt variable is filled in
    assert chain.inputs[0]["concepts"] == "photosynthesis, chlorophyll"


def test_answer_key_check_drops_questions_the_check_disagrees_with():
    first = mc(correct_index=0)
    second = MultipleChoiceQuestion("q2", "c", "Pick again", ("a", "b", "c"), 1, "because")
    third = MultipleChoiceQuestion("q3", "c", "And again", ("a", "b", "c"), 2, "because")
    short = ShortAnswerQuestion("q4", "c", "Explain", ("p",), "answer")
    quiz = Quiz("quiz", "t", "beginner", (first, second, third, short))

    checked, notes = verify_answer_keys(Chain("A", "C", "I'm not sure"), quiz, "material")

    assert [q.question_id for q in checked.questions] == ["q1", "q3", "q4"]
    assert any("q2" in note and "dropped" in note for note in notes)
    assert any("q3" in note and "unclear" in note for note in notes)