"""Phase 4: Self-RAG checks and Corrective RAG loops, run through the real LangGraph graph.

A FakeTaskModel recognises which job it is asked to do (grade, check evidence,
check the answer, ...) from the system prompt, and replies from a script.
No Ollama or internet is needed.
"""

from collections import defaultdict
from dataclasses import replace

from langchain_core.runnables import RunnableLambda

from adaptive_tutor.graph.checking_nodes import (
    EVIDENCE_SYSTEM_PROMPT,
    GRADE_SYSTEM_PROMPT,
    GROUNDING_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    UNRELIABLE_ANSWER_MESSAGE,
)
from adaptive_tutor.graph.nodes import (
    CONTRADICTION_INSTRUCTION,
    GROUNDED_SYSTEM_PROMPT,
    SEARCH_QUERY_SYSTEM_PROMPT,
    WEB_GROUNDED_SYSTEM_PROMPT,
)
from adaptive_tutor.graph.workflow import ask_tutor, build_tutor_graph
from adaptive_tutor.research.search_provider import WebResult
from adaptive_tutor.retrieval.retriever import RetrievedChunk

QUESTION = "What do mitochondria do?"
LONG_TEXT = "Mitochondria produce energy for the cell through cellular respiration. " * 7


class FakeTaskModel:
    """Replies according to the job in the prompt. The last reply for a job repeats."""

    TASKS = {
        "grade": GRADE_SYSTEM_PROMPT,
        "evidence": EVIDENCE_SYSTEM_PROMPT,
        "grounding": GROUNDING_SYSTEM_PROMPT,
        "rewrite": REWRITE_SYSTEM_PROMPT,
        "search_query": SEARCH_QUERY_SYSTEM_PROMPT,
        "web_answer": WEB_GROUNDED_SYSTEM_PROMPT,
        "answer": GROUNDED_SYSTEM_PROMPT,
    }

    def __init__(self, **replies) -> None:
        self.replies = {task: list(values) for task, values in replies.items()}
        self.prompts: dict[str, list[str]] = defaultdict(list)

    def respond(self, prompt_value) -> str:
        text = prompt_value.to_string()
        task = next((name for name, marker in self.TASKS.items() if marker in text), None)
        assert task is not None, f"Unknown prompt: {text[:80]}"
        self.prompts[task].append(text)
        queue = self.replies.get(task)
        assert queue, f"The test gave no reply for the '{task}' job"
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def runnable(self) -> RunnableLambda:
        return RunnableLambda(self.respond)

    def calls(self, task: str) -> int:
        return len(self.prompts[task])


class FakeRetriever:
    """Returns a different list of passages for each search, and records each search."""

    def __init__(self, *results_per_search: list[RetrievedChunk]) -> None:
        self._results = list(results_per_search)
        self.searches: list[tuple[str, int | None]] = []

    def document_count(self) -> int:
        return 1

    def search(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.searches.append((question, top_k))
        index = min(len(self.searches) - 1, len(self._results) - 1)
        return list(self._results[index])


class FakeSearchProvider:
    name = "FakeSearch"
    destination = "search.example"

    def __init__(self, results: list[WebResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[WebResult]:
        self.queries.append(query)
        return list(self._results)


def chunk(index: int, text: str = "Mitochondria produce energy for the cell.", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(text=text, source="biology.pdf", page=index, chunk_index=index, document_id="doc1", score=score)


def web_result(title: str = "Mitochondrion", url: str = "https://en.wikipedia.org/wiki/Mitochondrion") -> WebResult:
    return WebResult(title=title, url=url, content=LONG_TEXT, provider="FakeSearch", rank=1, content_source="article intro")


def ask_checked(graph, **kwargs):
    return ask_tutor(graph, QUESTION, use_documents=True, check_evidence=True, **kwargs)


# ---------- The happy path ----------

def test_checked_document_answer_runs_every_check(settings):
    model = FakeTaskModel(grade=["yes", "no"], evidence=["SUFFICIENT"], answer=["They make energy [1]."], grounding=["yes"])
    relevant, off_topic = chunk(1), chunk(2, text="Plants have chloroplasts.")
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([relevant, off_topic]))

    result = ask_checked(graph)

    assert result["steps"] == [
        "validate_question", "retrieve_documents", "grade_documents",
        "check_document_evidence", "generate_grounded_answer", "check_answer",
    ]
    assert result["answer"] == "They make energy [1]."
    assert result["grounded"] is True
    assert result["sources"] == [relevant]  # only the passage judged relevant
    assert result["rejected_sources"] == [off_topic]
    assert result["evidence_verdict"] == "sufficient"
    # The answer writer never saw the off-topic passage.
    assert "chloroplasts" not in model.prompts["answer"][0]
    assert model.calls("grade") == 2
    assert len(result["corrections"]) == 4  # search, relevance, evidence, answer check


def test_basic_mode_is_unchanged_when_checking_is_off(settings):
    model = FakeTaskModel(answer=["They make energy [1]."])
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1)]))

    result = ask_tutor(graph, QUESTION, use_documents=True)

    assert result["steps"] == ["validate_question", "retrieve_documents", "generate_grounded_answer"]
    assert model.calls("grade") == 0
    assert not result.get("corrections")


def test_general_knowledge_is_never_checked(settings):
    graph = build_tutor_graph(RunnableLambda(lambda _: "General answer."), settings)

    # Even if checking is requested, a general-knowledge answer has no sources to check against.
    result = ask_tutor(graph, QUESTION, check_evidence=True)

    assert result["steps"] == ["validate_question", "generate_answer"]
    assert result["answer"] == "General answer."


# ---------- Corrective RAG: search again ----------

def test_irrelevant_passages_trigger_a_rewritten_wider_search(settings):
    model = FakeTaskModel(
        grade=["no", "yes"], rewrite=['"mitochondria ATP production"'],
        evidence=["SUFFICIENT"], answer=["ATP is made [1]."], grounding=["yes"],
    )
    first, second = chunk(1, text="Cells divide."), chunk(2, text="Mitochondria make ATP.", score=0.45)
    retriever = FakeRetriever([first], [second])
    graph = build_tutor_graph(model.runnable(), settings, retriever)

    result = ask_checked(graph)

    assert result["steps"] == [
        "validate_question", "retrieve_documents", "grade_documents", "rewrite_query",
        "retrieve_documents", "grade_documents", "check_document_evidence",
        "generate_grounded_answer", "check_answer",
    ]
    # Second search: rewritten query, twice as many passages.
    assert retriever.searches == [(QUESTION, None), ("mitochondria ATP production", 8)]
    # Score 0.45 is below 0.5 but passes the wider search's 0.4 threshold, then the relevance check.
    assert result["sources"] == [second]
    assert result["retrieval_attempts"] == 2
    assert result["grounded"] is True


def test_retries_stop_at_the_limit_then_the_tutor_says_so(settings):
    model = FakeTaskModel(grade=["no"], rewrite=["cell energy"])
    retriever = FakeRetriever([chunk(1)], [chunk(2)], [chunk(3)])
    graph = build_tutor_graph(model.runnable(), settings, retriever)  # MAX_RETRIEVAL_RETRIES = 1

    result = ask_checked(graph)

    assert len(retriever.searches) == 2  # first search + one retry, never more
    assert result["steps"][-1] == "report_insufficient_evidence"
    assert "couldn't find" in result["answer"]
    assert model.calls("answer") == 0  # no answer is written without evidence


def test_zero_retries_means_no_rewrite(settings):
    model = FakeTaskModel(grade=["no"])
    graph = build_tutor_graph(model.runnable(), replace(settings, max_retrieval_retries=0), FakeRetriever([chunk(1)]))

    result = ask_checked(graph)

    assert "rewrite_query" not in result["steps"]
    assert model.calls("rewrite") == 0


def test_passages_already_graded_are_not_graded_again(settings):
    model = FakeTaskModel(grade=["yes"], evidence=["INSUFFICIENT"], rewrite=["cell energy"])
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1)], [chunk(1)]))

    result = ask_checked(graph)

    assert model.calls("grade") == 1  # the same passage came back on the retry
    assert model.calls("evidence") == 2
    assert result["steps"][-1] == "report_insufficient_evidence"
    # Related passages exist but aren't enough: a different, honest message.
    assert "related" in result["answer"]
    assert result["sources"] == [chunk(1)]


def test_unclear_grades_count_as_not_relevant(settings):
    model = FakeTaskModel(grade=["Maybe, hard to say."], rewrite=["x"])
    graph = build_tutor_graph(model.runnable(), replace(settings, max_retrieval_retries=0), FakeRetriever([chunk(1)]))

    result = ask_checked(graph)

    assert result["sources"] == []
    assert "unclear" in result["corrections"][1]


# ---------- Corrective RAG: web fallback ----------

def test_web_fallback_when_the_learner_allows_it(settings):
    model = FakeTaskModel(
        grade=["no", "yes"], search_query=["mitochondria energy"], evidence=["SUFFICIENT"],
        web_answer=["They make energy [W1]."], grounding=["yes"],
    )
    provider = FakeSearchProvider([web_result()])
    graph = build_tutor_graph(
        model.runnable(), replace(settings, max_retrieval_retries=0), FakeRetriever([chunk(1)]), provider
    )

    result = ask_checked(graph, allow_web_fallback=True)

    assert result["steps"] == [
        "validate_question", "retrieve_documents", "grade_documents", "start_web_fallback",
        "generate_search_query", "search_web", "grade_web_results", "check_web_evidence",
        "generate_web_answer", "check_answer",
    ]
    assert result["used_web_fallback"] is True
    assert provider.queries == ["mitochondria energy"]
    assert result["answer"] == "They make energy [W1]."
    assert result["grounded"] is True
    assert result["sources"] == []  # document and web evidence stay separate


def test_no_web_fallback_without_permission(settings):
    model = FakeTaskModel(grade=["no"])
    provider = FakeSearchProvider([web_result()])
    graph = build_tutor_graph(
        model.runnable(), replace(settings, max_retrieval_retries=0), FakeRetriever([chunk(1)]), provider
    )

    result = ask_checked(graph, allow_web_fallback=False)

    assert provider.queries == []  # nothing was sent to the internet
    assert result["steps"][-1] == "report_insufficient_evidence"


def test_web_fallback_with_no_good_results_says_so(settings):
    model = FakeTaskModel(grade=["no"], search_query=["mitochondria energy"])
    provider = FakeSearchProvider([web_result()])
    graph = build_tutor_graph(
        model.runnable(), replace(settings, max_retrieval_retries=0), FakeRetriever([chunk(1)]), provider
    )

    result = ask_checked(graph, allow_web_fallback=True)

    assert result["steps"][-1] == "report_no_web_results"
    assert "Your documents didn't cover this" in result["answer"]


# ---------- Self-RAG: checking the answer ----------

def test_unsupported_answer_is_rewritten_once_then_accepted(settings):
    model = FakeTaskModel(
        grade=["yes"], evidence=["SUFFICIENT"],
        answer=["They also control the weather [1].", "They make energy [1]."], grounding=["no", "yes"],
    )
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1)]))

    result = ask_checked(graph)

    assert result["steps"].count("generate_grounded_answer") == 2
    assert result["answer"] == "They make energy [1]."
    assert result["grounded"] is True
    # The second attempt was told why the first one was rejected.
    assert "previous answer was rejected" in model.prompts["answer"][1]
    assert "previous answer was rejected" not in model.prompts["answer"][0]


def test_answer_that_stays_unsupported_is_never_shown(settings):
    model = FakeTaskModel(grade=["yes"], evidence=["SUFFICIENT"], answer=["Weather control [1]."], grounding=["no"])
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1)]))

    result = ask_checked(graph)

    assert result["steps"][-1] == "report_unreliable_answer"
    assert result["answer"] == UNRELIABLE_ANSWER_MESSAGE
    assert "Weather" not in result["answer"]
    assert result["unreliable_answer"] is True
    assert model.calls("answer") == 2  # first try + MAX_ANSWER_RETRIES (1)


def test_wrong_citation_is_caught_without_asking_the_checker(settings):
    model = FakeTaskModel(
        grade=["yes"], evidence=["SUFFICIENT"], answer=["Energy [3].", "Energy [1]."], grounding=["yes"]
    )
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1)]))

    result = ask_checked(graph)

    assert model.calls("grounding") == 1  # only the second answer needed the model check
    assert any("cites [3]" in note for note in result["corrections"])
    assert result["answer"] == "Energy [1]."


def test_contradictory_evidence_gets_a_both_sides_answer(settings):
    model = FakeTaskModel(
        grade=["yes"], evidence=["CONTRADICTORY"], answer=["Source 1 says X [1]; source 2 says Y [2]."], grounding=["yes"]
    )
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1), chunk(2)]))

    result = ask_checked(graph)

    assert result["evidence_verdict"] == "contradictory"
    assert CONTRADICTION_INSTRUCTION in model.prompts["answer"][0]
    assert result["grounded"] is True


# ---------- Web research mode with checking ----------

def test_checked_web_research_grades_results(settings):
    model = FakeTaskModel(
        search_query=["mitochondria energy"], grade=["yes", "no"], evidence=["SUFFICIENT"],
        web_answer=["Energy [W1]."], grounding=["yes"],
    )
    good, off_topic = web_result(), web_result("Mitosis", "https://en.wikipedia.org/wiki/Mitosis")
    graph = build_tutor_graph(model.runnable(), settings, search_provider=FakeSearchProvider([good, off_topic]))

    result = ask_tutor(graph, QUESTION, use_web_research=True, check_evidence=True)

    assert result["steps"] == [
        "validate_question", "generate_search_query", "search_web", "grade_web_results",
        "check_web_evidence", "generate_web_answer", "check_answer",
    ]
    assert [r.title for r in result["web_sources"]] == ["Mitochondrion"]
    assert "relevance check" in result["rejected_web_sources"][-1].quality_notes[-1]


# ---------- Safety: errors and loop limits ----------

def test_checker_failure_goes_to_the_error_node(settings):
    model = FakeTaskModel(grade=[ConnectionError("refused")])
    graph = build_tutor_graph(model.runnable(), settings, FakeRetriever([chunk(1)]))

    result = ask_checked(graph)

    assert "Cannot reach Ollama" in result["error"]
    assert result["steps"][-1] == "handle_error"


def test_worst_case_still_finishes(settings):
    """Largest retry limits, every check failing: the graph must still stop on its own."""
    model = FakeTaskModel(
        grade=["no", "no", "no", "yes"], rewrite=["q2"], search_query=["q3"], evidence=["SUFFICIENT"],
        web_answer=["Made up [W1]."], grounding=["no"],
    )
    settings = replace(settings, max_retrieval_retries=2, max_answer_retries=2)
    retriever = FakeRetriever([chunk(1)], [chunk(2)], [chunk(3)])
    graph = build_tutor_graph(model.runnable(), settings, retriever, FakeSearchProvider([web_result()]))

    result = ask_checked(graph, allow_web_fallback=True)

    assert len(retriever.searches) == 3  # first search + 2 retries
    assert model.calls("web_answer") == 3  # first answer + 2 retries
    assert result["steps"][-1] == "report_unreliable_answer"
    assert len(result["steps"]) <= 25