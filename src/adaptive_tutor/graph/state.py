"""The shared data that flows through the LangGraph workflow."""

import operator
from typing import Annotated, TypedDict

from adaptive_tutor.retrieval.retriever import RetrievedChunk


class TutorState(TypedDict, total=False):
    """One 'clipboard' passed from node to node.

    total=False means a key may be missing until some node fills it in.
    """

    # Input from the learner
    question: str
    use_documents: bool

    # Results
    answer: str
    error: str

    # Phase 2: document retrieval
    sources: list[RetrievedChunk]  # passages that passed the relevance threshold
    best_score: float  # score of the closest passage, even if it was too low
    insufficient_evidence: bool  # True when the documents don't cover the question

    # The reducer `operator.add` tells LangGraph to APPEND each node's steps
    # instead of replacing them, giving a trace of which nodes ran.
    steps: Annotated[list[str], operator.add]