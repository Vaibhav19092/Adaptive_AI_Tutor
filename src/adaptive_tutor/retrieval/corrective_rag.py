"""Corrective RAG helpers: how to search again when the first search was poor.

Attempt 1 is the normal Phase 2 search. If the checker finds the passages
irrelevant or not enough, the tutor rewrites the query and searches again with
an alternative strategy: a WIDER search (more passages, slightly lower
similarity threshold). Widening is safe here because every passage still has
to pass the model's relevance check before it can be used.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_tutor.config import Settings

WIDER_SEARCH_FACTOR = 2  # retries look at twice as many passages...
RETRY_SCORE_DROP = 0.1  # ...and accept a similarity score 0.1 lower
MAX_TOP_K = 20


@dataclass(frozen=True)
class RetrievalPlan:
    """What one search attempt will do."""

    attempt: int  # 1 = first search, 2 = first retry, ...
    query: str
    top_k: int
    min_score: float
    strategy: str  # a short description shown to the learner

    def describe(self, found: int, kept: int) -> str:
        return (
            f"Search attempt {self.attempt} ({self.strategy}): looked at the {self.top_k} closest "
            f"passages, {kept} of {found} scored at least {self.min_score:.2f}"
        )


def plan_retrieval(attempts_so_far: int, question: str, rewritten_query: str | None, settings: Settings) -> RetrievalPlan:
    """Choose the query and strategy for the next search attempt."""
    if attempts_so_far == 0:
        return RetrievalPlan(
            attempt=1,
            query=question,
            top_k=settings.retrieval_top_k,
            min_score=settings.min_relevance_score,
            strategy="standard search",
        )
    return RetrievalPlan(
        attempt=attempts_so_far + 1,
        query=rewritten_query or question,
        top_k=min(MAX_TOP_K, settings.retrieval_top_k * WIDER_SEARCH_FACTOR),
        min_score=round(max(0.0, settings.min_relevance_score - RETRY_SCORE_DROP), 2),
        strategy="rewritten query, wider search",
    )