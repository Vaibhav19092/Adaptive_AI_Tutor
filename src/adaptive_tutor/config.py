"""Application settings loaded from environment variables (.env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class Settings:
    """All configurable values in one place. frozen=True means read-only."""

    ollama_base_url: str
    ollama_model: str
    temperature: float
    request_timeout_seconds: float
    max_question_length: int
    log_level: str


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number, got {raw_value!r}") from exc


def load_settings(use_dotenv: bool = True) -> Settings:
    """Read settings from the environment, validating each value.

    use_dotenv=False lets tests ignore any real .env file on your machine.
    """
    if use_dotenv:
        load_dotenv()

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()
    if not model:
        raise ValueError("OLLAMA_MODEL must not be empty")

    temperature = _read_float("OLLAMA_TEMPERATURE", 0.3)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("OLLAMA_TEMPERATURE must be between 0.0 and 2.0")

    timeout = _read_float("OLLAMA_REQUEST_TIMEOUT_SECONDS", 180.0)
    if timeout <= 0:
        raise ValueError("OLLAMA_REQUEST_TIMEOUT_SECONDS must be greater than 0")

    max_length = _read_int("MAX_QUESTION_LENGTH", 2000)
    if max_length <= 0:
        raise ValueError("MAX_QUESTION_LENGTH must be greater than 0")

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")

    return Settings(
        ollama_base_url=base_url,
        ollama_model=model,
        temperature=temperature,
        request_timeout_seconds=timeout,
        max_question_length=max_length,
        log_level=log_level,
    )