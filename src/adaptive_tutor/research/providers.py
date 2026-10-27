"""Concrete search providers. Each one turns a service's JSON reply into WebResult objects.

All three are free to use without a payment method (checked October 2026):
- Wikipedia: no account or key. Needs contact details in the User-Agent.
- SearXNG: software you run yourself (for example with Docker).
- Tavily: free account and API key (monthly free credits).
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from adaptive_tutor import __version__
from adaptive_tutor.config import Settings
from adaptive_tutor.research.search_provider import (
    SearchConfigError,
    SearchProvider,
    SearchQuotaError,
    SearchUnavailableError,
    WebResult,
)

logger = logging.getLogger(__name__)

NETWORK_HINT = (
    "If you are on a company network, it may block this service. "
    "Don't try to get around the block: ask your IT team, or keep web research off."
)

# 429 = too many requests. Tavily also uses 432 (plan limit) and 433 (pay-as-you-go limit).
QUOTA_STATUS_CODES = {429, 432, 433}


def build_user_agent(contact: str) -> str:
    """Say honestly who is calling. Wikimedia blocks apps that hide or pretend to be a browser."""
    contact_part = f"; {contact}" if contact else ""
    return f"AdaptiveAITutor/{__version__} (personal learning project{contact_part}) httpx/{httpx.__version__}"


def create_http_client(settings: Settings) -> httpx.Client:
    """One shared HTTP client with a time limit and an honest User-Agent.

    httpx uses the HTTP_PROXY / HTTPS_PROXY variables if your company sets them,
    which is the approved way through a company proxy.
    """
    return httpx.Client(
        timeout=settings.search_timeout_seconds,
        headers={"User-Agent": build_user_agent(settings.research_contact)},
        follow_redirects=False,  # never get silently sent to a different site
    )


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    provider: str,
    forbidden_hint: str = "",
    **kwargs: Any,
) -> Any:
    """Send one request and turn every kind of failure into a clear SearchError."""
    try:
        response = client.request(method, url, **kwargs)
    except httpx.TimeoutException as exc:
        raise SearchUnavailableError(
            f"{provider} did not answer within the time limit. "
            "Try again, or raise SEARCH_TIMEOUT_SECONDS in .env."
        ) from exc
    except httpx.HTTPError as exc:
        raise SearchUnavailableError(
            f"Cannot reach {provider}. Check your internet connection. {NETWORK_HINT}"
        ) from exc

    status = response.status_code
    if status == 401:
        raise SearchConfigError(f"{provider} rejected the API key. Check the key in .env.")
    if status == 403:
        raise SearchConfigError(f"{provider} refused the request (HTTP 403). {forbidden_hint or NETWORK_HINT}")
    if status in QUOTA_STATUS_CODES:
        raise SearchQuotaError(
            f"{provider} says the free usage or rate limit has been reached (HTTP {status}). "
            "Wait a while and try again. This app never switches to a paid plan."
        )
    if status >= 400:
        raise SearchUnavailableError(f"{provider} returned an error (HTTP {status}). Try again later.")
    if status >= 300:
        raise SearchUnavailableError(
            f"{provider} tried to redirect the request elsewhere (HTTP {status}). "
            f"Check its address in .env. {NETWORK_HINT}"
        )

    try:
        return response.json()
    except ValueError as exc:
        # Company proxies often answer with an HTML "blocked" page instead of JSON.
        raise SearchUnavailableError(f"{provider} sent a reply that isn't JSON. {NETWORK_HINT}") from exc


def _text(value: Any) -> str:
    """Turn a possibly-missing JSON value into a clean string."""
    return str(value or "").strip()


class WikipediaSearchProvider:
    """Searches Wikipedia articles through the free MediaWiki API (no key needed)."""

    name = "Wikipedia"

    def __init__(self, client: httpx.Client, language: str = "en") -> None:
        self._client = client
        self.destination = f"{language}.wikipedia.org"
        self._endpoint = f"https://{self.destination}/w/api.php"

    def search(self, query: str, max_results: int) -> list[WebResult]:
        # One request does two jobs: "generator=search" finds matching articles, and
        # "prop=extracts|info" returns each article's plain-text intro and its link.
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(max_results),
            "gsrnamespace": "0",  # normal articles only, not talk or user pages
            "prop": "extracts|info",
            "exintro": "1",  # only the introduction section
            "explaintext": "1",  # plain text, not HTML
            "exlimit": "max",
            "inprop": "url",
        }
        data = request_json(
            self._client,
            "GET",
            self._endpoint,
            self.name,
            forbidden_hint="Wikipedia may block apps without contact details. Check RESEARCH_CONTACT in .env.",
            params=params,
        )
        if not isinstance(data, dict):
            raise SearchUnavailableError("Wikipedia sent an unexpected reply.")
        if data.get("error"):
            error = data["error"]
            code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
            raise SearchUnavailableError(f"Wikipedia could not run this search (error code: {code}).")

        # No matches means there is no "query" key at all.
        pages = [page for page in data.get("query", {}).get("pages", []) if isinstance(page, dict)]
        pages.sort(key=lambda page: page.get("index", 0))  # "index" is the search rank

        results: list[WebResult] = []
        for page in pages[:max_results]:
            title = _text(page.get("title"))
            if not title:
                continue
            url = _text(page.get("fullurl")) or (
                f"https://{self.destination}/wiki/{quote(title.replace(' ', '_'))}"
            )
            results.append(
                WebResult(
                    title=title,
                    url=url,
                    content=_text(page.get("extract")),
                    provider=self.name,
                    rank=len(results) + 1,
                    content_source="article intro",
                )
            )
        return results


class SearxngSearchProvider:
    """Searches through a SearXNG server you run yourself (it must have JSON output enabled)."""

    name = "SearXNG"

    def __init__(self, client: httpx.Client, base_url: str) -> None:
        self._client = client
        self._endpoint = f"{base_url.rstrip('/')}/search"
        self.destination = urlsplit(base_url).netloc or base_url

    def search(self, query: str, max_results: int) -> list[WebResult]:
        data = request_json(
            self._client,
            "GET",
            self._endpoint,
            self.name,
            forbidden_hint=(
                "SearXNG only answers JSON requests when 'json' is listed under "
                "search: formats: in its settings.yml. Add it and restart SearXNG."
            ),
            params={"q": query, "format": "json"},
        )
        if not isinstance(data, dict):
            raise SearchUnavailableError("SearXNG sent an unexpected reply.")

        results: list[WebResult] = []
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            title, url = _text(item.get("title")), _text(item.get("url"))
            if not title or not url:
                continue
            results.append(
                WebResult(
                    title=title,
                    url=url,
                    content=_text(item.get("content")),
                    provider=self.name,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= max_results:
                break
        return results


class TavilySearchProvider:
    """Searches with Tavily's API (free plan: monthly credits, no payment method)."""

    name = "Tavily"
    destination = "api.tavily.com"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, client: httpx.Client, api_key: str) -> None:
        self._client = client
        self._api_key = api_key  # never logged or shown

    def search(self, query: str, max_results: int) -> list[WebResult]:
        body = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",  # the cheapest search type
            "include_answer": False,  # we want sources, not Tavily's own AI answer
            "include_raw_content": False,
            "include_images": False,
        }
        data = request_json(
            self._client,
            "POST",
            self.endpoint,
            self.name,
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        if not isinstance(data, dict):
            raise SearchUnavailableError("Tavily sent an unexpected reply.")

        results: list[WebResult] = []
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            title, url = _text(item.get("title")), _text(item.get("url"))
            if not title or not url:
                continue
            results.append(
                WebResult(
                    title=title,
                    url=url,
                    content=_text(item.get("content")),
                    provider=self.name,
                    rank=len(results) + 1,
                )
            )
        return results[:max_results]


def create_search_provider(settings: Settings, client: httpx.Client | None = None) -> SearchProvider | None:
    """Build the provider chosen in .env, or return None when research is switched off."""
    if not settings.research_enabled:
        return None

    client = client or create_http_client(settings)
    logger.info("Web research is on (provider: %s)", settings.search_provider)

    if settings.search_provider == "wikipedia":
        return WikipediaSearchProvider(client, settings.wikipedia_language)
    if settings.search_provider == "searxng":
        return SearxngSearchProvider(client, settings.searxng_base_url)
    if settings.search_provider == "tavily":
        return TavilySearchProvider(client, settings.tavily_api_key)
    raise ValueError(f"Unknown search provider: {settings.search_provider}")