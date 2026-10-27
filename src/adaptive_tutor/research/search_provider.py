"""The replaceable search-provider interface: what every web search service must offer.

This file has no network code at all. Any class with a `name`, a `destination`,
and a `search()` method can be plugged into the tutor, including fakes in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

# Longest piece of one web result given to the model (keeps prompts inside OLLAMA_NUM_CTX).
WEB_CONTEXT_MAX_CHARS = 1500


class SearchError(Exception):
    """Base class for web search problems. The message is shown to the learner."""


class SearchUnavailableError(SearchError):
    """The service can't be reached, timed out, or is having problems."""


class SearchConfigError(SearchError):
    """The service refused the request because of a setup problem (key, settings)."""


class SearchQuotaError(SearchError):
    """The free usage limit or rate limit has been reached."""


@dataclass(frozen=True)
class WebResult:
    """One search result from the internet, kept separate from document passages."""

    title: str
    url: str
    content: str
    provider: str  # display name of the service, e.g. "Wikipedia"
    rank: int  # 1 = the provider's top result
    content_source: str = "snippet"  # "snippet", "article intro", or "web page"
    quality_score: float = 0.0  # filled in by source_verification.py
    quality_notes: tuple[str, ...] = ()  # the reasons behind the score

    @property
    def domain(self) -> str:
        """The website name without 'www.', e.g. 'en.wikipedia.org'."""
        host = (urlsplit(self.url).hostname or "").lower()
        return host.removeprefix("www.")

    @property
    def citation(self) -> str:
        """A human-readable reference, like 'Mitochondrion (en.wikipedia.org)'."""
        return f"{self.title} ({self.domain})" if self.domain else self.title


class SearchProvider(Protocol):
    """What the graph needs from any web search service."""

    name: str  # shown to the learner, e.g. "Wikipedia"
    destination: str  # where queries are sent, e.g. "en.wikipedia.org"

    def search(self, query: str, max_results: int) -> list[WebResult]: ...


def format_web_context(results: list[WebResult]) -> str:
    """Number web results [W1], [W2], ... so they can't be confused with document passages [1]."""
    if not results:
        return "(no web results)"
    blocks = []
    for number, result in enumerate(results, start=1):
        text = result.content
        if len(text) > WEB_CONTEXT_MAX_CHARS:
            text = text[:WEB_CONTEXT_MAX_CHARS].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[W{number}] {result.citation}\n{text}")
    return "\n\n".join(blocks)