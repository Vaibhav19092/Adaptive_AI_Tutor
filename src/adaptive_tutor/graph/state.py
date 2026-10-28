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
    check_evidence: bool  # Phase 4: run the Self-RAG checks for this question
    allow_web_fallback: bool  # Phase 4: may search the web if documents fall short

    # Results
    answer: str
    error: str

    # Phase 2: evidence from the learner's own documents
    sources: list[RetrievedChunk]  # passages the answer is built from
    best_score: float  # score of the closest passage, even if it was too low
    insufficient_evidence: bool  # True when the evidence doesn't cover the question

    # Phase 3: evidence from the internet, kept in SEPARATE keys from document evidence
    search_query: str  # the exact text sent to the search service
    web_sources: list[WebResult]  # results that passed the quality checks
    rejected_web_sources: list[WebResult]  # results filtered out, with the reasons
    web_results_found: int  # how many results the service returned before filtering

    # Phase 4: Corrective RAG (searching again)
    retrieval_query: str  # the rewritten query used for the next document search
    retrieval_attempts: int  # how many document searches have run (bounded)
    approved_sources: list[RetrievedChunk]  # passages the checker judged relevant so far
    rejected_sources: list[RetrievedChunk]  # passages the checker judged not relevant
    used_web_fallback: bool  # True when documents fell short and the web was searched

    # Phase 4: Self-RAG (checking the evidence and the answer)
    evidence_verdict: str  # "sufficient", "insufficient", or "contradictory"
    answer_source: str  # "documents" or "web": which evidence the answer used
    answer_attempts: int  # how many answers have been written (bounded)
    grounded: bool  # True when the answer passed the grounding check
    grounding_problem: str  # why the last answer failed the check ("" if it passed)
    unreliable_answer: bool  # True when no answer passed the check

    # The reducer `operator.add` tells LangGraph to APPEND each node's steps
    # instead of replacing them, giving a trace of which nodes ran.
    steps: Annotated[list[str], operator.add]
    # Same idea: a plain-language log of every check and correction.
    corrections: Annotated[list[str], operator.add]