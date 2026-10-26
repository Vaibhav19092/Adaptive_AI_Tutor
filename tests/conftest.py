"""Shared test fixtures."""

import pytest

from adaptive_tutor.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model:latest",
        temperature=0.0,
        request_timeout_seconds=10.0,
        max_question_length=100,
        log_level="INFO",
    )