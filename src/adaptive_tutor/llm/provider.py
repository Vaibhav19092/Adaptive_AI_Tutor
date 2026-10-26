"""Create the Ollama chat model and turn connection problems into clear messages."""

from __future__ import annotations

import logging

import httpx
from langchain_ollama import ChatOllama
from ollama import ResponseError

from adaptive_tutor.config import Settings

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 5.0


class LLMError(Exception):
    """Base class for problems talking to the local model."""


class OllamaUnavailableError(LLMError):
    """Ollama is not running or cannot be reached."""


class ModelNotFoundError(LLMError):
    """The configured model has not been downloaded with `ollama pull`."""


def normalize_model_name(name: str) -> str:
    """Ollama treats 'llama3.2' and 'llama3.2:latest' as the same model."""
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def list_installed_models(base_url: str) -> list[str]:
    """Ask Ollama which models are installed (same list as `ollama list`)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {base_url}. Is it running? "
            "Start the Ollama app, or run `ollama serve` in a terminal."
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaUnavailableError(
            f"Ollama at {base_url} sent an unexpected response."
        ) from exc

    return [model.get("name", "") for model in data.get("models", [])]


def check_model_available(settings: Settings) -> None:
    """Raise a helpful error if Ollama is down or the model is missing."""
    installed = list_installed_models(settings.ollama_base_url)
    wanted = normalize_model_name(settings.ollama_model)
    installed_normalized = {normalize_model_name(name) for name in installed if name}

    if wanted not in installed_normalized:
        installed_text = ", ".join(sorted(installed_normalized)) or "none"
        raise ModelNotFoundError(
            f"Model '{settings.ollama_model}' is not installed. "
            f"Run: ollama pull {settings.ollama_model} "
            f"(installed models: {installed_text})"
        )


def create_chat_model(settings: Settings) -> ChatOllama:
    """Build the LangChain chat model. This does not contact Ollama yet."""
    logger.info("Creating ChatOllama client for model '%s'", settings.ollama_model)
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        client_kwargs={"timeout": settings.request_timeout_seconds},
    )


def describe_llm_error(exc: Exception, settings: Settings) -> str:
    """Translate a technical exception into a message a learner can act on."""
    if isinstance(exc, LLMError):
        return str(exc)

    if isinstance(exc, ResponseError):
        if exc.status_code == 404 or "not found" in str(exc.error).lower():
            return (
                f"Model '{settings.ollama_model}' is not installed. "
                f"Run: ollama pull {settings.ollama_model}"
            )
        return f"Ollama returned an error: {exc.error}"

    if isinstance(exc, httpx.TimeoutException):
        return (
            f"The model took longer than {settings.request_timeout_seconds:.0f} seconds. "
            "Try a smaller model or increase OLLAMA_REQUEST_TIMEOUT_SECONDS in .env."
        )

    if isinstance(exc, (httpx.ConnectError, ConnectionError)):
        return (
            f"Cannot reach Ollama at {settings.ollama_base_url}. Is it running? "
            "Start the Ollama app, or run `ollama serve` in a terminal."
        )

    return (
        f"Unexpected error from the local model ({type(exc).__name__}). "
        "Check the terminal logs for details."
    )