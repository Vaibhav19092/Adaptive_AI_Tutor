"""Build the long-lived objects the app needs, all in one place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.state import CompiledStateGraph

from adaptive_tutor.assessment.answer_evaluator import RUBRIC_PROMPT, ShortAnswerGrader
from adaptive_tutor.config import Settings
from adaptive_tutor.graph.learning_workflow import build_learning_graph
from adaptive_tutor.graph.workflow import build_tutor_graph
from adaptive_tutor.llm.provider import create_chat_model
from adaptive_tutor.memory.checkpointing import create_checkpointer
from adaptive_tutor.memory.repository import LearnerRepository
from adaptive_tutor.research.content_extraction import PageFetcher
from adaptive_tutor.research.providers import create_http_client, create_search_provider
from adaptive_tutor.research.search_provider import SearchProvider
from adaptive_tutor.retrieval.retriever import DocumentRetriever
from adaptive_tutor.retrieval.vector_store import DocumentStore, create_embeddings, create_vector_store


@dataclass(frozen=True)
class TutorServices:
    """Everything the app reuses between clicks."""

    graph: CompiledStateGraph  # Phases 1-4: question answering
    store: DocumentStore
    search_provider: SearchProvider | None = None  # None when web research is off
    # Phases 5-9
    learning_graph: CompiledStateGraph | None = None  # lessons, quizzes, grading, planning
    repository: LearnerRepository | None = None  # saved progress (SQLite)
    checkpointer: Any = None  # saved workflow state (SQLite), for paused quizzes
    short_answer_grader: ShortAnswerGrader | None = None  # reused by the grading-consistency check


def build_services(settings: Settings) -> TutorServices:
    llm = create_chat_model(settings)
    json_llm = create_chat_model(settings, json_mode=True)
    embeddings = create_embeddings(settings)
    store = DocumentStore(create_vector_store(settings, embeddings))
    retriever = DocumentRetriever(store, top_k=settings.retrieval_top_k)

    # Web research objects are only created when RESEARCH_ENABLED=true,
    # so with the default settings nothing in the app can contact the internet.
    search_provider = None
    page_fetcher = None
    if settings.research_enabled:
        http_client = create_http_client(settings)
        search_provider = create_search_provider(settings, http_client)
        if settings.research_fetch_pages:
            page_fetcher = PageFetcher(http_client)

    graph = build_tutor_graph(llm, settings, retriever, search_provider, page_fetcher)

    repository = LearnerRepository(settings.database_path)
    checkpointer = create_checkpointer(settings.checkpoint_path)
    learning_graph = build_learning_graph(
        llm, json_llm, settings, repository, retriever, search_provider, checkpointer
    )
    grader = ShortAnswerGrader(RUBRIC_PROMPT | json_llm | StrOutputParser(), settings.structured_output_retries)
    return TutorServices(
        graph=graph,
        store=store,
        search_provider=search_provider,
        learning_graph=learning_graph,
        repository=repository,
        checkpointer=checkpointer,
        short_answer_grader=grader,
    )