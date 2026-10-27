"""Build the long-lived objects the app needs, all in one place."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.workflow import build_tutor_graph
from adaptive_tutor.llm.provider import create_chat_model
from adaptive_tutor.research.content_extraction import PageFetcher
from adaptive_tutor.research.providers import create_http_client, create_search_provider
from adaptive_tutor.research.search_provider import SearchProvider
from adaptive_tutor.retrieval.retriever import DocumentRetriever
from adaptive_tutor.retrieval.vector_store import DocumentStore, create_embeddings, create_vector_store


@dataclass(frozen=True)
class TutorServices:
    """Everything the app reuses between clicks."""

    graph: CompiledStateGraph
    store: DocumentStore
    search_provider: SearchProvider | None = None  # None when web research is off


def build_services(settings: Settings) -> TutorServices:
    llm = create_chat_model(settings)
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
    return TutorServices(graph=graph, store=store, search_provider=search_provider)