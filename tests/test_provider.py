import httpx
import pytest
from ollama import ResponseError

from adaptive_tutor.llm import provider
from adaptive_tutor.llm.provider import (
    ModelNotFoundError,
    OllamaUnavailableError,
    check_embedding_model_available,
    check_model_available,
    describe_llm_error,
    normalize_model_name,
)


def test_normalize_model_name_adds_latest_tag():
    assert normalize_model_name("llama3.2") == "llama3.2:latest"
    assert normalize_model_name("llama3.2:3b") == "llama3.2:3b"


def test_list_installed_models_when_ollama_is_down(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(provider.httpx, "get", fake_get)
    with pytest.raises(OllamaUnavailableError, match="Cannot reach Ollama"):
        provider.list_installed_models("http://localhost:11434")


def test_list_installed_models_parses_response(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(provider.httpx, "get", fake_get)
    assert provider.list_installed_models("http://localhost:11434") == ["llama3.2:3b", "qwen2.5:7b"]


def test_check_model_available_passes_when_installed(monkeypatch, settings):
    monkeypatch.setattr(provider, "list_installed_models", lambda base_url: ["test-model"])
    check_model_available(settings)  # should not raise


def test_check_model_available_raises_when_missing(monkeypatch, settings):
    monkeypatch.setattr(provider, "list_installed_models", lambda base_url: ["other:latest"])
    with pytest.raises(ModelNotFoundError, match="ollama pull test-model:latest"):
        check_model_available(settings)


def test_check_embedding_model_names_the_embedding_model(monkeypatch, settings):
    # Only the chat model is installed, so the embedding check must fail.
    monkeypatch.setattr(provider, "list_installed_models", lambda base_url: ["test-model:latest"])
    with pytest.raises(ModelNotFoundError, match="ollama pull nomic-embed-text"):
        check_embedding_model_available(settings)


def test_describe_missing_model_error(settings):
    message = describe_llm_error(ResponseError("model 'test-model' not found", 404), settings)
    assert "ollama pull" in message


def test_describe_error_can_name_the_embedding_model(settings):
    error = ResponseError("model not found", 404)
    message = describe_llm_error(error, settings, model_name="nomic-embed-text")
    assert "ollama pull nomic-embed-text" in message


def test_describe_connection_error(settings):
    message = describe_llm_error(ConnectionError("refused"), settings)
    assert "Cannot reach Ollama" in message


def test_describe_timeout_error(settings):
    message = describe_llm_error(httpx.ReadTimeout("timed out"), settings)
    assert "took longer" in message