"""Phase 5: evidence for lessons and quizzes, stored as plain dictionaries.

The learning workflow is saved to disk by a LangGraph checkpointer (Phase 7), so
everything in its state must be simple data: strings, numbers, lists, dicts.
These helpers turn document passages (RetrievedChunk) and web results (WebResult)
into such dicts, and number them for the model exactly like Phases 2-3 do:
documents as [1], [2] and web results as [W1], [W2]. The two kinds are never mixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adaptive_tutor.research.search_provider import WEB_CONTEXT_MAX_CHARS
from adaptive_tutor.retrieval.self_rag import chunk_key

if TYPE_CHECKING:
    from adaptive_tutor.research.search_provider import WebResult
    from adaptive_tutor.retrieval.retriever import RetrievedChunk

Evidence = dict[str, Any]


def chunk_to_evidence(chunk: RetrievedChunk) -> Evidence:
    return {
        "kind": "document",
        "key": chunk_key(chunk),
        "citation": chunk.citation,
        "url": "",
        "text": chunk.text,
        "score": round(chunk.score, 3),
    }


def web_to_evidence(result: WebResult) -> Evidence:
    text = result.content
    if len(text) > WEB_CONTEXT_MAX_CHARS:
        text = text[:WEB_CONTEXT_MAX_CHARS].rsplit(" ", 1)[0] + " ..."
    return {
        "kind": "web",
        "key": result.url,
        "citation": result.citation,
        "url": result.url,
        "text": text,
        "score": round(result.quality_score, 3),
    }


def evidence_label(item: Evidence, number: int) -> str:
    """[1] for a document passage, [W1] for a web result."""
    return f"[W{number}]" if item.get("kind") == "web" else f"[{number}]"


def format_evidence(items: list[Evidence]) -> str:
    """Number the evidence for the model, the same way Phases 2-3 number sources."""
    if not items:
        return "(no sources)"
    return "\n\n".join(
        f"{evidence_label(item, number)} ({item['citation']})\n{item['text']}"
        for number, item in enumerate(items, start=1)
    )


def is_web_evidence(items: list[Evidence]) -> bool:
    return bool(items) and items[0].get("kind") == "web"