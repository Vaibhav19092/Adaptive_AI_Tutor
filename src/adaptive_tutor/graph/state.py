"""The shared data that flows through the LangGraph workflow."""

import operator
from typing import Annotated, TypedDict

from adaptive_tutor.research.search_provider import WebResult
from adaptive_tutor.retrieval.retriever import RetrievedChunk


class TutorState(TypedDict, total=False):
    """One 'clipboard' passed from node to node.

    total=False means a key may be missing until some node fills it in.
    """

    # Input from the learner
    question: str
    use_documents: bool
    use_web_research: bool  # Phase 3

    # Results
    answer: str
    error: str

    # Phase 2: evidence from the learner's own documents
    sources: list[RetrievedChunk]  # passages that passed the relevance threshold
    best_score: float  # score of the closest passage, even if it was too low
    insufficient_evidence: bool  # True when the evidence doesn't cover the question

    # Phase 3: evidence from the internet, kept in SEPARATE keys from document evidence
    search_query: str  # the exact text sent to the search service
    web_sources: list[WebResult]  # results that passed the quality checks
    rejected_web_sources: list[WebResult]  # results filtered out, with the reasons
    web_results_found: int  # how many results the service returned before filtering

    # The reducer `operator.add` tells LangGraph to APPEND each node's steps
    # instead of replacing them, giving a trace of which nodes ran.
    steps: Annotated[list[str], operator.add]