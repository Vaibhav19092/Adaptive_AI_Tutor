"""Run the real LangGraph workflow with fake models and retrievers, so Ollama isn't needed."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from adaptive_tutor.graph.workflow import ask_tutor, build_tutor_graph
from adaptive_tutor.retrieval.retriever import RetrievedChunk


def fail_if_called(_):
    raise AssertionError("The model should not be called")


class FakeRetriever:
    """Pretends to search documents; returns whatever chunks we give it."""

    def __init__(self, chunks: list[RetrievedChunk], count: int = 1) -> None:
        self._chunks = chunks
        self._count = count
        self.questions: list[str] = []

    def document_count(self) -> int:
        return self._count

    def search(self, question: str) -> list[RetrievedChunk]:
        self.questions.append(question)
        return list(self._chunks)


def make_chunk(score: float, page: int | None = 2) -> RetrievedChunk:
    return RetrievedChunk(
        text="Mitochondria produce energy for the cell.",
        source="biology.pdf",
        page=page,
        chunk_index=0,
        document_id="doc1",
        score=score,
    )


# ---------- Phase 1 (general knowledge) ----------

def test_successful_question_runs_both_nodes(settings):
    fake_llm = FakeListChatModel(responses=["An embedding is a list of numbers."])
    graph = build_tutor_graph(fake_llm, settings)

    result = ask_tutor(graph, "  What is an embedding?  ")

    assert result["answer"] == "An embedding is a list of numbers."
    assert result["question"] == "What is an embedding?"
    assert not result.get("error")
    assert result["steps"] == ["validate_question", "generate_answer"]


def test_empty_question_never_calls_the_model(settings):
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


# ---------- Phase 2 (my documents) ----------

def test_document_question_is_answered_from_retrieved_passages(settings):
    prompts: list[str] = []

    def fake_llm(prompt_value):
        prompts.append(prompt_value.to_string())
        return "Mitochondria make energy [1]."

    retriever = FakeRetriever([make_chunk(score=0.8)])
    graph = build_tutor_graph(RunnableLambda(fake_llm), settings, retriever)

    result = ask_tutor(graph, "What do mitochondria do?", use_documents=True)

    assert result["steps"] == ["validate_question", "retrieve_documents", "generate_grounded_answer"]
    assert result["answer"] == "Mitochondria make energy [1]."
    assert [chunk.citation for chunk in result["sources"]] == ["biology.pdf, page 2"]
    # The model really received the numbered passage and the question.
    assert "[1] (biology.pdf, page 2)" in prompts[0]
    assert "Mitochondria produce energy for the cell." in prompts[0]
    assert "What do mitochondria do?" in prompts[0]


def test_low_scores_mean_insufficient_evidence_and_no_model_call(settings):
    retriever = FakeRetriever([make_chunk(score=0.2)])
    graph = build_tutor_graph(RunnableLambda(fail_if_called), settings, retriever)

    result = ask_tutor(graph, "Who won the 1998 World Cup?", use_documents=True)

    assert result["steps"] == ["validate_question", "retrieve_documents", "report_insufficient_evidence"]
    assert result["insufficient_evidence"] is True
    assert "couldn't find" in result["answer"]
    assert result["best_score"] == 0.2
    assert result["sources"] == []


def test_only_passages_above_the_threshold_are_used(settings):
    retriever = FakeRetriever([make_chunk(score=0.9), make_chunk(score=0.3, page=7)])
    graph = build_tutor_graph(FakeListChatModel(responses=["Answer [1]."]), settings, retriever)

    result = ask_tutor(graph, "What do mitochondria do?", use_documents=True)

    assert [chunk.page for chunk in result["sources"]] == [2]


def test_empty_library_gives_a_helpful_message(settings):
    retriever = FakeRetriever([], count=0)
    graph = build_tutor_graph(RunnableLambda(fail_if_called), settings, retriever)

    result = ask_tutor(graph, "What is a cell?", use_documents=True)

    assert "haven't added any documents" in result["error"]
    assert result["steps"] == ["validate_question", "retrieve_documents", "handle_error"]


def test_document_mode_without_a_retriever(settings):
    graph = build_tutor_graph(RunnableLambda(fail_if_called), settings)
    result = ask_tutor(graph, "What is a cell?", use_documents=True)

    assert "not set up" in result["error"]


def test_retrieval_failure_goes_to_error_node(settings):
    class BrokenRetriever(FakeRetriever):
        def search(self, question):
            raise ConnectionError("refused")

    graph = build_tutor_graph(RunnableLambda(fail_if_called), settings, BrokenRetriever([]))
    result = ask_tutor(graph, "What is a cell?", use_documents=True)

    assert "Cannot reach Ollama" in result["error"]
    assert result["steps"] == ["validate_question", "retrieve_documents", "handle_error"]


def test_general_mode_does_not_search_documents(settings):
    retriever = FakeRetriever([make_chunk(score=0.9)])
    graph = build_tutor_graph(FakeListChatModel(responses=["General answer."]), settings, retriever)

    result = ask_tutor(graph, "What is a cell?", use_documents=False)

    assert result["steps"] == ["validate_question", "generate_answer"]
    assert retriever.questions == []
    assert not result.get("sources")