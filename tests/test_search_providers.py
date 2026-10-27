"""Search providers, tested with httpx's MockTransport: no real network calls are made."""

import json
from dataclasses import replace

import httpx
import pytest

from adaptive_tutor.research.providers import (
    SearxngSearchProvider,
    TavilySearchProvider,
    WikipediaSearchProvider,
    build_user_agent,
    create_search_provider,
)
from adaptive_tutor.research.search_provider import (
    SearchConfigError,
    SearchQuotaError,
    SearchUnavailableError,
)


def make_client(handler) -> httpx.Client:
    """An httpx client whose 'internet' is the handler function below each test."""
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": build_user_agent("learner@example.com")},
    )


def reply_with(status: int, **kwargs):
    return lambda request: httpx.Response(status, **kwargs)


WIKIPEDIA_REPLY = {
    "batchcomplete": True,
    "query": {
        "pages": [
            {"pageid": 2, "title": "Cellular respiration", "index": 2,
             "extract": "Cellular respiration releases energy.",
             "fullurl": "https://en.wikipedia.org/wiki/Cellular_respiration"},
            {"pageid": 1, "title": "Mitochondrion", "index": 1,
             "extract": "A mitochondrion is an organelle.",
             "fullurl": "https://en.wikipedia.org/wiki/Mitochondrion"},
        ]
    },
}


# ---------- Wikipedia ----------

def test_wikipedia_results_come_back_in_search_order():
    requests: list[httpx.Request] = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=WIKIPEDIA_REPLY)

    results = WikipediaSearchProvider(make_client(handler)).search("mitochondria energy", max_results=5)

    assert [r.title for r in results] == ["Mitochondrion", "Cellular respiration"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].url == "https://en.wikipedia.org/wiki/Mitochondrion"
    assert results[0].content == "A mitochondrion is an organelle."
    assert results[0].provider == "Wikipedia"
    assert results[0].content_source == "article intro"

    sent = requests[0]
    assert sent.url.host == "en.wikipedia.org"
    assert sent.url.path == "/w/api.php"
    assert sent.url.params["gsrsearch"] == "mitochondria energy"
    assert sent.url.params["gsrlimit"] == "5"
    assert "learner@example.com" in sent.headers["User-Agent"]


def test_wikipedia_with_no_matches_returns_an_empty_list():
    provider = WikipediaSearchProvider(make_client(reply_with(200, json={"batchcomplete": True})))
    assert provider.search("zzqxw", 5) == []


def test_wikipedia_language_changes_the_address():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(200, json={"batchcomplete": True})

    provider = WikipediaSearchProvider(make_client(handler), language="simple")
    provider.search("cell", 3)
    assert provider.destination == "simple.wikipedia.org"
    assert hosts == ["simple.wikipedia.org"]


def test_wikipedia_api_error_is_reported():
    provider = WikipediaSearchProvider(make_client(reply_with(200, json={"error": {"code": "maxlag"}})))
    with pytest.raises(SearchUnavailableError, match="maxlag"):
        provider.search("cell", 3)


def test_wikipedia_403_mentions_the_contact_setting():
    provider = WikipediaSearchProvider(make_client(reply_with(403, text="blocked")))
    with pytest.raises(SearchConfigError, match="RESEARCH_CONTACT"):
        provider.search("cell", 3)


# ---------- SearXNG ----------

def test_searxng_results_are_parsed_and_limited():
    requests: list[httpx.Request] = []
    items = [{"title": f"Result {i}", "url": f"https://site{i}.org/page", "content": "Some text"} for i in range(8)]
    items.insert(0, {"title": "", "url": "https://no-title.org"})  # incomplete items are skipped

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"query": "q", "results": items})

    provider = SearxngSearchProvider(make_client(handler), "http://localhost:8080")
    results = provider.search("vector databases", max_results=3)

    assert [r.title for r in results] == ["Result 0", "Result 1", "Result 2"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].content_source == "snippet"
    assert provider.destination == "localhost:8080"
    assert requests[0].url.path == "/search"
    assert requests[0].url.params["format"] == "json"
    assert requests[0].url.params["q"] == "vector databases"


def test_searxng_403_explains_how_to_enable_json():
    provider = SearxngSearchProvider(make_client(reply_with(403, text="Forbidden")), "http://localhost:8080")
    with pytest.raises(SearchConfigError, match="settings.yml"):
        provider.search("q", 3)


# ---------- Tavily ----------

def test_tavily_sends_the_key_as_a_bearer_token():
    requests: list[httpx.Request] = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"query": "q", "results": [
            {"title": "Photosynthesis", "url": "https://example.edu/photo", "content": "Plants use light.", "score": 0.9},
        ]})

    results = TavilySearchProvider(make_client(handler), api_key="tvly-test-key").search("photosynthesis", 4)

    assert [r.title for r in results] == ["Photosynthesis"]
    sent = requests[0]
    assert sent.method == "POST"
    assert sent.url == "https://api.tavily.com/search"
    assert sent.headers["Authorization"] == "Bearer tvly-test-key"
    body = json.loads(sent.content)
    assert body["query"] == "photosynthesis"
    assert body["max_results"] == 4
    assert body["search_depth"] == "basic"
    assert body["include_answer"] is False


def test_tavily_rejected_key_is_a_config_error():
    provider = TavilySearchProvider(make_client(reply_with(401, json={"detail": {"error": "bad key"}})), "wrong")
    with pytest.raises(SearchConfigError, match="API key"):
        provider.search("q", 3)


def test_the_api_key_is_not_in_the_provider_repr():
    provider = TavilySearchProvider(make_client(reply_with(200, json={})), "tvly-secret")
    assert "tvly-secret" not in repr(provider)


# ---------- Errors shared by every provider ----------

@pytest.mark.parametrize("status", [429, 432, 433])
def test_quota_and_rate_limits_never_switch_to_paid(status):
    provider = TavilySearchProvider(make_client(reply_with(status, json={})), "key")
    with pytest.raises(SearchQuotaError, match="never switches to a paid plan"):
        provider.search("q", 3)


def test_server_error_means_unavailable():
    provider = WikipediaSearchProvider(make_client(reply_with(503, text="down")))
    with pytest.raises(SearchUnavailableError, match="HTTP 503"):
        provider.search("q", 3)


def test_connection_failure_mentions_company_networks():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SearchUnavailableError, match="company network"):
        WikipediaSearchProvider(make_client(handler)).search("q", 3)


def test_timeout_is_explained():
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(SearchUnavailableError, match="SEARCH_TIMEOUT_SECONDS"):
        WikipediaSearchProvider(make_client(handler)).search("q", 3)


def test_html_instead_of_json_is_explained():
    provider = SearxngSearchProvider(make_client(reply_with(200, text="<html>Blocked by proxy</html>")), "http://localhost:8080")
    with pytest.raises(SearchUnavailableError, match="isn't JSON"):
        provider.search("q", 3)


def test_redirects_are_not_followed():
    provider = SearxngSearchProvider(
        make_client(reply_with(302, headers={"Location": "https://elsewhere.example/"})), "http://localhost:8080"
    )
    with pytest.raises(SearchUnavailableError, match="redirect"):
        provider.search("q", 3)


# ---------- Choosing a provider from settings ----------

def test_no_provider_when_research_is_off(settings):
    assert create_search_provider(settings) is None


def test_the_configured_provider_is_created(settings):
    client = make_client(reply_with(200, json={}))
    on = replace(settings, research_enabled=True)

    assert isinstance(create_search_provider(replace(on, search_provider="wikipedia"), client), WikipediaSearchProvider)
    assert isinstance(create_search_provider(replace(on, search_provider="searxng"), client), SearxngSearchProvider)
    assert isinstance(
        create_search_provider(replace(on, search_provider="tavily", tavily_api_key="k"), client), TavilySearchProvider
    )


def test_user_agent_is_honest():
    agent = build_user_agent("learner@example.com")
    assert agent.startswith("AdaptiveAITutor/")
    assert "learner@example.com" in agent
    assert "Mozilla" not in agent  # we never pretend to be a web browser