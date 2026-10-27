"""Source-quality rules. Pure Python, so these run without any network or model."""

from adaptive_tutor.research.search_provider import WebResult, format_web_context
from adaptive_tutor.research.source_verification import (
    apply_hard_rules,
    normalize_url,
    score_result,
    score_results,
    search_terms,
)

LONG_TEXT = "Mitochondria produce energy for the cell through cellular respiration. " * 7


def make_result(url: str = "https://en.wikipedia.org/wiki/Mitochondrion", content: str = LONG_TEXT, title: str = "Mitochondrion") -> WebResult:
    return WebResult(title=title, url=url, content=content, provider="Test", rank=1)


# ---------- Hard rules ----------

def test_non_web_links_are_rejected():
    kept, rejected = apply_hard_rules([make_result(url="ftp://files.example.com/a"), make_result(url="javascript:alert(1)")])
    assert kept == []
    assert all("http or https" in r.quality_notes[0] for r in rejected)


def test_blocked_domains_include_their_subdomains():
    results = [make_result(url="https://www.spam.org/page"), make_result(url="https://blog.spam.org/x"),
               make_result(url="https://notspam.org/y")]
    kept, rejected = apply_hard_rules(results, blocked_domains=("spam.org",))
    assert [r.url for r in kept] == ["https://notspam.org/y"]
    assert len(rejected) == 2
    assert "RESEARCH_BLOCKED_DOMAINS" in rejected[0].quality_notes[0]


def test_duplicates_are_rejected_even_with_small_url_differences():
    results = [make_result(url="https://www.example.org/page/"), make_result(url="http://example.org/page#intro")]
    kept, rejected = apply_hard_rules(results)
    assert len(kept) == 1
    assert "duplicate" in rejected[0].quality_notes[0]


def test_private_and_local_addresses_are_rejected():
    results = [make_result(url="http://localhost:8080/admin"), make_result(url="http://192.168.1.10/"),
               make_result(url="http://127.0.0.1/x"), make_result(url="http://[::1]/y")]
    kept, rejected = apply_hard_rules(results)
    assert kept == []
    assert all("private or local" in r.quality_notes[0] for r in rejected)


# ---------- Scoring ----------

def test_trusted_long_on_topic_result_scores_high():
    scored = score_result(make_result(), search_terms("mitochondria energy"))
    assert scored.quality_score == 0.9
    assert "+ educational, government, or reference site" in scored.quality_notes
    assert "+ enough text to answer from" in scored.quality_notes
    assert "+ mentions most of the search words" in scored.quality_notes


def test_short_off_topic_result_is_rejected_with_reasons():
    result = make_result(url="http://random-blog.example/post", content="Buy cheap shoes today!", title="Shoes")
    accepted, rejected = score_results([result], "mitochondria energy", min_quality=0.4)
    assert accepted == []
    notes = rejected[0].quality_notes
    assert "− not encrypted (http instead of https)" in notes
    assert "− very little text to learn from" in notes
    assert "− doesn't mention any of the search words" in notes
    assert "below RESEARCH_MIN_QUALITY" in notes[-1]
    assert rejected[0].quality_score == 0.0


def test_ordinary_snippet_is_accepted():
    snippet = "Mitochondria are organelles that release energy from food. " * 3  # about 180 characters
    accepted, _ = score_results([make_result(url="https://biology-notes.example/m", content=snippet)], "mitochondria energy", 0.4)
    assert accepted[0].quality_score == 0.5
    assert "− only a short snippet" in accepted[0].quality_notes


def test_lookalike_domain_is_not_trusted():
    scored = score_result(make_result(url="https://fakewikipedia.org/x"), set())
    assert "+ educational, government, or reference site" not in scored.quality_notes


# ---------- Helpers ----------

def test_search_terms_drop_filler_words():
    assert search_terms("What is the difference between RAG and fine-tuning?") == {"rag", "fine", "tuning"}


def test_normalize_url():
    assert normalize_url("https://WWW.Example.org/Page/?a=1#top") == "example.org/Page?a=1"


def test_web_context_uses_w_numbers_and_trims_long_text():
    results = [make_result(), make_result(url="https://example.org/b", content="word " * 1000, title="Long")]
    context = format_web_context(results)
    assert context.startswith("[W1] Mitochondrion (en.wikipedia.org)\n")
    assert "[W2] Long (example.org)" in context
    assert len(context) < 3500
    assert format_web_context([]) == "(no web results)"