import pytest

from adaptive_tutor.config import load_settings

ENV_VARS = [
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TEMPERATURE",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS",
    "MAX_QUESTION_LENGTH",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Make sure real environment variables don't affect these tests."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_used_when_environment_is_empty():
    settings = load_settings(use_dotenv=False)
    assert settings.ollama_model == "qwen2.5:7b"
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