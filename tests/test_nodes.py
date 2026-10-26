from langgraph.graph import END

from adaptive_tutor.graph.nodes import (
    make_validate_question_node,
    route_after_generation,
    route_after_validation,
)


def test_validate_question_strips_whitespace():
    validate = make_validate_question_node(max_length=100)
    result = validate({"question": "   What is RAG?   "})
    assert result == {"question": "What is RAG?", "steps": ["validate_question"]}


def test_validate_question_rejects_empty_question():
    validate = make_validate_question_node(max_length=100)
    result = validate({"question": "   "})
    assert result["error"] == "Please type a question first."


def test_validate_question_rejects_long_question():
    validate = make_validate_question_node(max_length=10)
    result = validate({"question": "x" * 11})
    assert "under 10 characters" in result["error"]


def test_route_after_validation():
    assert route_after_validation({"question": "ok"}) == "generate_answer"
    assert route_after_validation({"error": "bad"}) == "handle_error"


def test_route_after_generation():
    assert route_after_generation({"answer": "hi"}) == END
    assert route_after_generation({"error": "bad"}) == "handle_error"