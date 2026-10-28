"""Self-RAG helpers: reading the checker's replies and checking citations. Pure Python."""

from adaptive_tutor.retrieval.self_rag import (
    chunk_key,
    citation_problem,
    find_citations,
    parse_evidence_verdict,
    parse_yes_no,
)


class FakeChunk:
    document_id = "doc1"
    chunk_index = 3


# ---------- parse_yes_no ----------

def test_plain_yes_and_no():
    assert parse_yes_no("yes") is True
    assert parse_yes_no("No") is False
    assert parse_yes_no("YES.") is True


def test_first_yes_or_no_wins_even_with_explanations():
    assert parse_yes_no("No, the passage is about plants.") is False
    assert parse_yes_no("Yes - it explains how ATP is made. No other details.") is True


def test_unclear_replies_give_none():
    assert parse_yes_no("") is None
    assert parse_yes_no("I'm not sure.") is None
    assert parse_yes_no("Not relevant") is None  # "not" is not "no"
    assert parse_yes_no("None of it") is None  # "none" is not "no"


# ---------- parse_evidence_verdict ----------

def test_evidence_verdicts():
    assert parse_evidence_verdict("SUFFICIENT") == "sufficient"
    assert parse_evidence_verdict("Insufficient.") == "insufficient"
    assert parse_evidence_verdict("The sources are CONTRADICTORY") == "contradictory"


def test_insufficient_is_not_mistaken_for_sufficient():
    assert parse_evidence_verdict("INSUFFICIENT - the dates are missing") == "insufficient"


def test_unclear_verdict_gives_none():
    assert parse_evidence_verdict("Maybe?") is None
    assert parse_evidence_verdict("") is None


# ---------- citations ----------

def test_find_document_citations():
    answer = "Cells make energy [1]. They have DNA [2][3]. Both matter [1, 2]. See [note]."
    assert find_citations(answer) == [1, 2, 3, 1, 2]


def test_find_web_citations():
    answer = "Plants use light [W1]. Also water [W2, W3]. Not a web cite [1]."
    assert find_citations(answer, web=True) == [1, 2, 3]
    assert find_citations("Energy [1].", web=True) == []


def test_answer_without_citations_is_a_problem():
    assert citation_problem("Mitochondria make energy.", 2) == "the answer has no citations"


def test_citation_to_a_missing_source_is_a_problem():
    problem = citation_problem("Energy [1]. Heat [5].", 2)
    assert problem == "the answer cites [5], but there are only 2 source(s)"
    assert "[W4]" in citation_problem("Energy [W4].", 3, web=True)


def test_valid_citations_are_fine():
    assert citation_problem("Energy [1]. Heat [2].", 2) is None
    assert citation_problem("Energy [W1].", 1, web=True) is None


def test_chunk_key():
    assert chunk_key(FakeChunk()) == "doc1-3"