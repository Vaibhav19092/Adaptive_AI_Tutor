import pytest

from adaptive_tutor.config import PROJECT_ROOT, load_settings

ENV_VARS = [
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TEMPERATURE",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS",
    "OLLAMA_NUM_CTX",
    "MAX_QUESTION_LENGTH",
    "LOG_LEVEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
    "MIN_RELEVANCE_SCORE",
    "MAX_UPLOAD_MB",
    "RESEARCH_ENABLED",
    "SEARCH_PROVIDER",
    "SEARCH_MAX_RESULTS",
    "SEARCH_TIMEOUT_SECONDS",
    "RESEARCH_CONTACT",
    "WIKIPEDIA_LANGUAGE",
    "SEARXNG_BASE_URL",
    "TAVILY_API_KEY",
    "RESEARCH_FETCH_PAGES",
    "RESEARCH_MIN_QUALITY",
    "RESEARCH_BLOCKED_DOMAINS",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Make sure real environment variables don't affect these tests."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------- Phase 1 ----------

def test_defaults_used_when_environment_is_empty():
    settings = load_settings(use_dotenv=False)
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.temperature == 0.3


def test_values_read_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = load_settings(use_dotenv=False)
    assert settings.ollama_model == "qwen2.5:7b"
    assert settings.ollama_base_url == "http://localhost:11434"  # trailing slash removed
    assert settings.log_level == "DEBUG"


def test_temperature_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "5")
    with pytest.raises(ValueError, match="between"):
        load_settings(use_dotenv=False)


def test_non_numeric_timeout_raises(monkeypatch):
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "abc")
    with pytest.raises(ValueError, match="must be a number"):
        load_settings(use_dotenv=False)


def test_empty_model_name_raises(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "   ")
    with pytest.raises(ValueError, match="must not be empty"):
        load_settings(use_dotenv=False)


# ---------- Phase 2 ----------

def test_phase2_defaults():
    settings = load_settings(use_dotenv=False)
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.context_window == 4096
    assert settings.chroma_persist_dir == PROJECT_ROOT / "data" / "chroma"


def test_relative_store_path_is_inside_project(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "my_store")
    settings = load_settings(use_dotenv=False)
    assert settings.chroma_persist_dir == PROJECT_ROOT / "my_store"


def test_empty_prefixes_are_allowed(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "")
    settings = load_settings(use_dotenv=False)
    assert settings.embedding_query_prefix == ""
    assert settings.embedding_document_prefix == ""


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    with pytest.raises(ValueError, match="smaller than CHUNK_SIZE"):
        load_settings(use_dotenv=False)


def test_min_relevance_score_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "1.5")
    with pytest.raises(ValueError, match="MIN_RELEVANCE_SCORE"):
        load_settings(use_dotenv=False)


def test_top_k_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "0")
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        load_settings(use_dotenv=False)


# ---------- Phase 3 ----------

def test_research_is_off_by_default():
    settings = load_settings(use_dotenv=False)
    assert settings.research_enabled is False
    assert settings.search_provider == "wikipedia"
    assert settings.search_max_results == 5
    assert settings.research_fetch_pages is False
    assert settings.research_blocked_domains == ()
    assert settings.tavily_api_key == ""


def test_research_settings_are_read(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENABLED", "yes")
    monkeypatch.setenv("RESEARCH_CONTACT", "learner@example.com")
    monkeypatch.setenv("WIKIPEDIA_LANGUAGE", "Simple")
    monkeypatch.setenv("RESEARCH_FETCH_PAGES", "on")
    settings = load_settings(use_dotenv=False)
    assert settings.research_enabled is True
    assert settings.research_contact == "learner@example.com"
    assert settings.wikipedia_language == "simple"
    assert settings.research_fetch_pages is True


def test_bad_boolean_raises(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENABLED", "maybe")
    with pytest.raises(ValueError, match="true or false"):
        load_settings(use_dotenv=False)


def test_wikipedia_needs_a_contact_when_research_is_on(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENABLED", "true")
    with pytest.raises(ValueError, match="RESEARCH_CONTACT"):
        load_settings(use_dotenv=False)


def test_tavily_needs_a_key_when_research_is_on(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENABLED", "true")
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        load_settings(use_dotenv=False)


def test_missing_key_is_fine_while_research_is_off(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    assert load_settings(use_dotenv=False).search_provider == "tavily"


def test_api_key_is_hidden_when_settings_are_printed(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENABLED", "true")
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-very-secret")
    settings = load_settings(use_dotenv=False)
    assert settings.tavily_api_key == "tvly-very-secret"
    assert "tvly-very-secret" not in repr(settings)


def test_unknown_search_provider_raises(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "google")
    with pytest.raises(ValueError, match="SEARCH_PROVIDER"):
        load_settings(use_dotenv=False)


def test_blocked_domains_are_cleaned(monkeypatch):
    monkeypatch.setenv("RESEARCH_BLOCKED_DOMAINS", " Example.com, www.Spam.org ,,example.com")
    assert load_settings(use_dotenv=False).research_blocked_domains == ("example.com", "spam.org")


def test_blocked_domains_must_be_site_names(monkeypatch):
    monkeypatch.setenv("RESEARCH_BLOCKED_DOMAINS", "https://example.com/page")
    with pytest.raises(ValueError, match="site names only"):
        load_settings(use_dotenv=False)


def test_invalid_wikipedia_language_raises(monkeypatch):
    monkeypatch.setenv("WIKIPEDIA_LANGUAGE", "en.evil.example/")
    with pytest.raises(ValueError, match="WIKIPEDIA_LANGUAGE"):
        load_settings(use_dotenv=False)


def test_search_max_results_range(monkeypatch):
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "50")
    with pytest.raises(ValueError, match="SEARCH_MAX_RESULTS"):
        load_settings(use_dotenv=False)


def test_searxng_url_must_be_a_web_address(monkeypatch):
    monkeypatch.setenv("SEARXNG_BASE_URL", "localhost:8080")
    with pytest.raises(ValueError, match="SEARXNG_BASE_URL"):
        load_settings(use_dotenv=False)


def test_contact_must_be_one_plain_line(monkeypatch):
    monkeypatch.setenv("RESEARCH_CONTACT", "me@example.com\nX-Injected: 1")
    with pytest.raises(ValueError, match="RESEARCH_CONTACT"):
        load_settings(use_dotenv=False)