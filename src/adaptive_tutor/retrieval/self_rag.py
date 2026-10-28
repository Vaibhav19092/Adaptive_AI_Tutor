"""Self-RAG helpers: read the checker model's replies and check citations.

Self-RAG means the system checks its own work: Is this passage relevant?
Is the evidence enough? Is the answer supported by the sources?
These helpers are plain Python (no model calls), so they are easy to test.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # only for type hints; keeps this module free of LangChain
    from adaptive_tutor.retrieval.retriever import RetrievedChunk

EvidenceVerdict = Literal["sufficient", "insufficient", "contradictory"]

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
# INSUFFICIENT must be tried before SUFFICIENT, so the pattern lists it first.
_VERDICT = re.compile(r"\b(insufficient|contradictory|sufficient)\b", re.IGNORECASE)
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")


def parse_yes_no(reply: str) -> bool | None:
    """'Yes.' -> True, 'No, it is not' -> False, anything unclear -> None.

    The first 'yes' or 'no' in the reply wins, because small models sometimes
    add an explanation after their answer.
    """
    match = _YES_NO.search(reply or "")
    if not match:
        return None
    return match.group(1).lower() == "yes"


def parse_evidence_verdict(reply: str) -> EvidenceVerdict | None:
    """Find SUFFICIENT, INSUFFICIENT, or CONTRADICTORY in the reply (first one wins)."""
    match = _VERDICT.search(reply or "")
    return match.group(1).lower() if match else None  # type: ignore[return-value]


def find_citations(answer: str, web: bool = False) -> list[int]:
    """All citation numbers in an answer, in order.

    Documents: '[1]', '[2][3]', '[1, 2]'.  Web: '[W1]', '[W2, W3]'.
    Brackets that aren't citations, like '[note]', are ignored.
    """
    numbers: list[int] = []
    for inside in _BRACKETS.findall(answer or ""):
        for part in inside.split(","):
            token = part.strip()
            if web:
                if len(token) > 1 and token[0] in "Ww" and token[1:].isdigit():
                    numbers.append(int(token[1:]))
            elif token.isdigit():
                numbers.append(int(token))
    return numbers


def citation_problem(answer: str, source_count: int, web: bool = False) -> str | None:
    """A quick rule-based check before asking the model. Returns the problem, or None."""
    numbers = find_citations(answer, web=web)
    if not numbers:
        return "the answer has no citations"
    label = "W" if web else ""
    invalid = sorted({n for n in numbers if not 1 <= n <= source_count})
    if invalid:
        cited = ", ".join(f"[{label}{n}]" for n in invalid)
        return f"the answer cites {cited}, but there are only {source_count} source(s)"
    return None


def chunk_key(chunk: RetrievedChunk) -> str:
    """A stable name for a passage, so the same passage is never graded twice."""
    return f"{chunk.document_id}-{chunk.chunk_index}"