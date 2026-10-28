"""Phase 8: understanding the learner's request."""

import pytest

from adaptive_tutor.learner.intent import clean_topic, detect_intent, parse_intent


@pytest.mark.parametrize(
    "request_text, intent",
    [
        ("Teach me photosynthesis", "lesson"),
        ("Explain embeddings", "lesson"),
        ("I want to learn about RAG", "lesson"),
        ("Quiz me on photosynthesis", "quiz"),
        ("Test me", "quiz"),
        ("What should I study next?", "next"),
        ("Any recommendations?", "next"),
        ("photosynthesis", None),
    ],
)
def test_detect_intent(request_text, intent):
    assert detect_intent(request_text) == intent


@pytest.mark.parametrize("reply, intent", [("QUIZ", "quiz"), ("Lesson.", "lesson"), ("next", "next"), ("hmm", None)])
def test_parse_intent(reply, intent):
    assert parse_intent(reply) == intent


@pytest.mark.parametrize(
    "request_text, topic",
    [
        ("Please teach me about photosynthesis!", "photosynthesis"),
        ("Quiz me on the water cycle", "the water cycle"),
        ("Can you explain vector databases?", "vector databases"),
        ("What is an embedding?", "an embedding"),
        ("Test me", ""),
        ("Quiz me", ""),
        ("neural networks", "neural networks"),
    ],
)
def test_clean_topic(request_text, topic):
    assert clean_topic(request_text) == topic