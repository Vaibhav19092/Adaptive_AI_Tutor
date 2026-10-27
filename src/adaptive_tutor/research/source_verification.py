"""Basic, explainable source-quality checks for web results.

These are simple rules of thumb, NOT fact-checking. A high score means
"looks like a reasonable source to learn from", not "this is true".
Phase 4 adds model-based relevance and grounding checks.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import replace
from urllib.parse import urlsplit

from adaptive_tutor.research.search_provider import WebResult

# Website types that are usually careful about accuracy.
TRUSTED_SUFFIXES = (".edu", ".gov", ".ac.uk", ".gov.uk", ".ac.in", ".gov.in", ".edu.au")
TRUSTED_SITES = (
    "wikipedia.org", "britannica.com", "arxiv.org", "python.org",
    "developer.mozilla.org", "w3.org", "khanacademy.org",
)

SHORT_TEXT_CHARS = 80  # less than this is barely a sentence
BRIEF_TEXT_CHARS = 200  # a typical search snippet
RICH_TEXT_CHARS = 400  # enough to actually learn something from

STOPWORDS = {
    "the", "and", "for", "are", "was", "what", "how", "why", "who", "when", "where",
    "which", "does", "with", "from", "that", "this", "into", "about", "your", "you",
    "can", "explain", "simple", "words", "between", "difference",
}


def search_terms(text: str) -> set[str]:
    """The meaningful words of a query: lowercase, 3+ letters, no filler words."""
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) >= 3 and word not in STOPWORDS}


def normalize_url(url: str) -> str:
    """Make two links to the same page look identical (for spotting duplicates)."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"  # http vs https and #fragment are ignored


def _matches(host: str, domain: str) -> bool:
    """'en.wikipedia.org' matches 'wikipedia.org'; 'notwikipedia.org' does not."""
    return host == domain or host.endswith("." + domain)


def _is_private_address(host: str) -> bool:
    """True for localhost and private network IPs (like 192.168.x.x), which we never use."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False  # a normal name like example.com
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_unspecified


def _hard_rule_problem(result: WebResult, blocked_domains: tuple[str, ...], seen: set[str]) -> str | None:
    """Rules that reject a result outright. Returns the reason, or None if it passes."""
    parts = urlsplit(result.url)
    if parts.scheme not in ("http", "https"):
        return "Rejected: not a normal web link (http or https)"
    host = (parts.hostname or "").lower().removeprefix("www.")
    if not host:
        return "Rejected: the link has no website name"
    if _is_private_address(host):
        return "Rejected: points to a private or local network address"
    if any(_matches(host, domain) for domain in blocked_domains):
        return "Rejected: this site is listed in RESEARCH_BLOCKED_DOMAINS"
    if normalize_url(result.url) in seen:
        return "Rejected: duplicate of another result"
    return None


def apply_hard_rules(
    results: list[WebResult],
    blocked_domains: tuple[str, ...] = (),
) -> tuple[list[WebResult], list[WebResult]]:
    """Split results into (kept, rejected). Runs BEFORE any page is downloaded,
    so blocked or private sites are never contacted."""
    kept: list[WebResult] = []
    rejected: list[WebResult] = []
    seen: set[str] = set()
    for result in results:
        problem = _hard_rule_problem(result, blocked_domains, seen)
        if problem:
            rejected.append(replace(result, quality_score=0.0, quality_notes=(problem,)))
        else:
            kept.append(result)
            seen.add(normalize_url(result.url))
    return kept, rejected


def score_result(result: WebResult, terms: set[str]) -> WebResult:
    """Give a 0-1 quality score, starting at 0.5, with a note for every change."""
    score = 0.5
    notes: list[str] = []

    host = result.domain
    if host.endswith(TRUSTED_SUFFIXES) or any(_matches(host, site) for site in TRUSTED_SITES):
        score += 0.2
        notes.append("+ educational, government, or reference site")

    if result.url.lower().startswith("http://"):
        score -= 0.1
        notes.append("− not encrypted (http instead of https)")

    length = len(result.content.strip())
    if length < SHORT_TEXT_CHARS:
        score -= 0.3
        notes.append("− very little text to learn from")
    elif length < BRIEF_TEXT_CHARS:
        score -= 0.1
        notes.append("− only a short snippet")
    elif length >= RICH_TEXT_CHARS:
        score += 0.1
        notes.append("+ enough text to answer from")

    if terms:
        words = set(re.findall(r"[a-z0-9]+", f"{result.title} {result.content}".lower()))
        overlap = len(terms & words) / len(terms)
        if overlap == 0:
            score -= 0.3
            notes.append("− doesn't mention any of the search words")
        elif overlap >= 0.5:
            score += 0.1
            notes.append("+ mentions most of the search words")

    score = round(max(0.0, min(1.0, score)), 2)
    return replace(result, quality_score=score, quality_notes=tuple(notes))


def score_results(
    results: list[WebResult],
    query: str,
    min_quality: float,
) -> tuple[list[WebResult], list[WebResult]]:
    """Score every result and split them into (accepted, rejected)."""
    terms = search_terms(query)
    accepted: list[WebResult] = []
    rejected: list[WebResult] = []
    for result in results:
        scored = score_result(result, terms)
        if scored.quality_score >= min_quality:
            accepted.append(scored)
        else:
            reason = f"Rejected: quality {scored.quality_score:.2f} is below RESEARCH_MIN_QUALITY ({min_quality:.2f})"
            rejected.append(replace(scored, quality_notes=scored.quality_notes + (reason,)))
    return accepted, rejected