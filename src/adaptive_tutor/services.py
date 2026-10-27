"""Build the long-lived objects the app needs, all in one place."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from adaptive_tutor.config import Settings
from adaptive_tutor.graph.workflow import build_tutor_graph
from adaptive_tutor.llm.provider import create_chat_model
from adaptive_tutor.retrieval.retriever import DocumentRetriever
from adaptive_tutor.retrieval.vector_store import DocumentStore, create_embeddings, create_vector_store


@dataclass(frozen=True)
class TutorServices:
    """Everything the app reuses between clicks."""

    graph: CompiledStateGraph
    store: DocumentStore


def build_services(settings: Settings) -> TutorServices:
    llm = create_chat_model(settings)
    embeddings = create_embeddings(settings)
    store = DocumentStore(create_vector_store(settings, embeddings))
    retriever = DocumentRetriever(store, top_k=settings.retrieval_top_k)
    graph = build_tutor_graph(llm, settings, retriever)
    return TutorServices(graph=graph, store=store)