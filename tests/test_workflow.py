"""Run the real LangGraph workflow with fake models and retrievers, so Ollama isn't needed."""

from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from adaptive_tutor.graph.workflow import ask_tutor, build_tutor_graph
from adaptive_tutor.research.search_provider import SearchQuotaError, WebResult
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


# ---------- Phase 3 (web research) ----------

LONG_TEXT = "Mitochondria produce energy for the cell through cellular respiration. " * 7


class FakeSearchProvider:
    """Pretends to be a web search service; returns the results we give it."""

    name = "FakeSearch"
    destination = "search.example"

    def __init__(self, results: list[WebResult] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int) -> list[WebResult]:
        self.queries.append((query, max_results))
        if self._error:
            raise self._error
        return list(self._results)


class FakePageFetcher:
    """Pretends to read web pages."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.urls: list[str] = []

    def fetch_text(self, url: str) -> str:
        self.urls.append(url)
        return self._text


def make_web_result(
    url: str = "https://en.wikipedia.org/wiki/Mitochondrion",
    content: str = LONG_TEXT,
    title: str = "Mitochondrion",
    content_source: str = "article intro",
) -> WebResult:
    return WebResult(title=title, url=url, content=content, provider="FakeSearch", rank=1, content_source=content_source)


def scripted_llm(*replies: str):
    """A fake model that gives each reply in turn and records every prompt it receives."""
    prompts: list[str] = []
    remaining = list(replies)

    def fake(prompt_value):
        prompts.append(prompt_value.to_string())
        if not remaining:
            raise AssertionError("The model was called more times than expected")
        return remaining.pop(0)

    return RunnableLambda(fake), prompts


def test_web_question_searches_filters_and_cites_web_sources(settings):
    llm, prompts = scripted_llm('"mitochondria energy"', "Mitochondria make energy [W1].")
    provider = FakeSearchProvider([make_web_result()])
    retriever = FakeRetriever([make_chunk(score=0.9)])
    graph = build_tutor_graph(llm, settings, retriever, provider)

    result = ask_tutor(graph, "What do mitochondria do?", use_web_research=True)

    assert result["steps"] == ["validate_question", "generate_search_query", "search_web", "generate_web_answer"]
    assert result["search_query"] == "mitochondria energy"  # quotes from the model removed
    assert provider.queries == [("mitochondria energy", settings.search_max_results)]
    assert result["answer"] == "Mitochondria make energy [W1]."
    assert [source.title for source in result["web_sources"]] == ["Mitochondrion"]
    assert result["web_sources"][0].quality_score == 0.9
    # The answer prompt contains the web result, numbered [W1], and the question.
    assert "[W1] Mitochondrion (en.wikipedia.org)" in prompts[1]
    assert "What do mitochondria do?" in prompts[1]
    # Document evidence is kept completely separate: the documents were not searched.
    assert retriever.questions == []
    assert not result.get("sources")


def test_only_the_query_is_sent_never_the_full_prompt(settings):
    llm, _ = scripted_llm("cell energy", "Answer [W1].")
    provider = FakeSearchProvider([make_web_result()])
    graph = build_tutor_graph(llm, settings, search_provider=provider)

    ask_tutor(graph, "My teacher Mrs Smith said cells make energy. How?", use_web_research=True)

    assert provider.queries[0][0] == "cell energy"


def test_low_quality_results_mean_no_answer_model_call(settings):
    llm, prompts = scripted_llm("mitochondria energy")  # only ONE reply: the answer model must not run
    junk = make_web_result(url="http://shop.example/shoes", content="Buy shoes!", title="Shoes", content_source="snippet")
    graph = build_tutor_graph(llm, settings, search_provider=FakeSearchProvider([junk]))

    result = ask_tutor(graph, "What do mitochondria do?", use_web_research=True)

    assert result["steps"] == ["validate_question", "generate_search_query", "search_web", "report_no_web_results"]
    assert result["insufficient_evidence"] is True
    assert "didn't find any results good enough" in result["answer"]
    assert result["web_sources"] == []
    assert result["web_results_found"] == 1
    assert "below RESEARCH_MIN_QUALITY" in result["rejected_web_sources"][0].quality_notes[-1]
    assert len(prompts) == 1


def test_blocked_sites_are_filtered_out(settings):
    settings = replace(settings, research_blocked_domains=("wikipedia.org",))
    llm, _ = scripted_llm("mitochondria energy")
    graph = build_tutor_graph(llm, settings, search_provider=FakeSearchProvider([make_web_result()]))

    result = ask_tutor(graph, "What do mitochondria do?", use_web_research=True)

    assert result["steps"][-1] == "report_no_web_results"
    assert "RESEARCH_BLOCKED_DOMAINS" in result["rejected_web_sources"][0].quality_notes[0]


def test_search_service_failure_goes_to_error_node(settings):
    llm, _ = scripted_llm("mitochondria energy")
    error = SearchQuotaError("FakeSearch says the free usage limit has been reached.")
    graph = build_tutor_graph(llm, settings, search_provider=FakeSearchProvider(error=error))

    result = ask_tutor(graph, "What do mitochondria do?", use_web_research=True)

    assert result["error"] == "FakeSearch says the free usage limit has been reached."
    assert result["steps"] == ["validate_question", "generate_search_query", "search_web", "handle_error"]


def test_unexpected_provider_bug_does_not_crash(settings):
    llm, _ = scripted_llm("q")
    graph = build_tutor_graph(llm, settings, search_provider=FakeSearchProvider(error=KeyError("oops")))

    result = ask_tutor(graph, "What is a cell?", use_web_research=True)

    assert "unexpected" in result["error"]
    assert result["steps"][-1] == "handle_error"


def test_web_mode_without_a_provider_explains_how_to_turn_it_on(settings):
    llm, _ = scripted_llm("cell")
    graph = build_tutor_graph(llm, settings)

    result = ask_tutor(graph, "What is a cell?", use_web_research=True)

    assert "RESEARCH_ENABLED=true" in result["error"]
    assert result["steps"] == ["validate_question", "generate_search_query", "search_web", "handle_error"]


def test_query_generation_failure_stops_before_searching(settings):
    def ollama_is_down(_):
        raise ConnectionError("refused")

    provider = FakeSearchProvider([make_web_result()])
    graph = build_tutor_graph(RunnableLambda(ollama_is_down), settings, search_provider=provider)

    result = ask_tutor(graph, "What is a cell?", use_web_research=True)

    assert "Cannot reach Ollama" in result["error"]
    assert result["steps"] == ["validate_question", "generate_search_query", "handle_error"]
    assert provider.queries == []  # nothing was sent to the internet


def test_blank_query_from_the_model_falls_back_to_the_question(settings):
    llm, _ = scripted_llm("   ", "Answer [W1].")
    provider = FakeSearchProvider([make_web_result()])
    graph = build_tutor_graph(llm, settings, search_provider=provider)

    ask_tutor(graph, "What do mitochondria do?", use_web_research=True)

    assert provider.queries[0][0] == "What do mitochondria do?"


def test_page_fetcher_only_replaces_short_snippets(settings):
    llm, _ = scripted_llm("mitochondria energy", "Answer [W1] [W2].")
    snippet = make_web_result(url="https://biology.example/m", content="Mitochondria energy.", content_source="snippet")
    intro = make_web_result()  # an article intro: should not be fetched again
    fetcher = FakePageFetcher(LONG_TEXT + " Full page text.")
    graph = build_tutor_graph(llm, settings, search_provider=FakeSearchProvider([snippet, intro]), page_fetcher=fetcher)

    result = ask_tutor(graph, "What do mitochondria do?", use_web_research=True)

    assert fetcher.urls == ["https://biology.example/m"]
    by_url = {source.url: source for source in result["web_sources"]}
    assert by_url["https://biology.example/m"].content_source == "web page"
    assert by_url["https://en.wikipedia.org/wiki/Mitochondrion"].content_source == "article intro"


def test_documents_and_web_cannot_be_used_together(settings):
    graph = build_tutor_graph(RunnableLambda(fail_if_called), settings)
    with pytest.raises(ValueError, match="not both"):
        ask_tutor(graph, "What is a cell?", use_documents=True, use_web_research=True)


def test_document_mode_never_calls_the_search_provider(settings):
    provider = FakeSearchProvider([make_web_result()])
    graph = build_tutor_graph(
        FakeListChatModel(responses=["Answer [1]."]), settings, FakeRetriever([make_chunk(score=0.9)]), provider
    )

    result = ask_tutor(graph, "What do mitochondria do?", use_documents=True)

    assert provider.queries == []
    assert not result.get("web_sources")