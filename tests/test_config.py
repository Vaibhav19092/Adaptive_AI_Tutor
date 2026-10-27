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