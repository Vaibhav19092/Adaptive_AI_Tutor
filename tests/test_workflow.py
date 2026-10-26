"""Run the real LangGraph workflow with a fake model, so Ollama isn't needed."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from adaptive_tutor.graph.workflow import ask_tutor, build_tutor_graph


def test_successful_question_runs_both_nodes(settings):
    fake_llm = FakeListChatModel(responses=["An embedding is a list of numbers."])
    graph = build_tutor_graph(fake_llm, settings)

    result = ask_tutor(graph, "  What is an embedding?  ")

    assert result["answer"] == "An embedding is a list of numbers."
    assert result["question"] == "What is an embedding?"
    assert not result.get("error")
    assert result["steps"] == ["validate_question", "generate_answer"]


def test_empty_question_never_calls_the_model(settings):
    def fail_if_called(_):
        raise AssertionError("The model should not be called")

    graph = build_tutor_graph(RunnableLambda(fail_if_called), settings)
    result = ask_tutor(graph, "")

    assert result["error"] == "Please type a question first."
    assert result["steps"] == ["validate_question", "handle_error"]


def test_model_failure_goes_to_error_node(settings):
    def ollama_is_down(_):
        raise ConnectionError("refused")

    graph = build_tutor_graph(RunnableLambda(ollama_is_down), settings)
    result = ask_tutor(graph, "What is LangGraph?")

    assert "Cannot reach Ollama" in result["error"]
    assert result["steps"] == ["validate_question", "generate_answer", "handle_error"]


def test_empty_model_answer_is_treated_as_error(settings):
    graph = build_tutor_graph(FakeListChatModel(responses=["   "]), settings)
    result = ask_tutor(graph, "What is LangGraph?")

    assert "empty answer" in result["error"]