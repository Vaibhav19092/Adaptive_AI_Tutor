"""Application settings loaded from environment variables (.env file)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
SEARCH_PROVIDERS = ("wikipedia", "searxng", "tavily")

_TRUE_WORDS = {"1", "true", "yes", "on"}
_FALSE_WORDS = {"0", "false", "no", "off"}

# Language codes like "en", "de", "simple" (Simple English Wikipedia), "zh-yue".
# The strict pattern stops anything else being slipped into the Wikipedia address.
_WIKIPEDIA_LANGUAGE_PATTERN = re.compile(r"^[a-z][a-z-]{1,15}$")

# This file is src/adaptive_tutor/config.py, so parents[2] is the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """All configurable values in one place. frozen=True means read-only."""

    # Phase 1: the chat model
    ollama_base_url: str
    ollama_model: str
    temperature: float
    request_timeout_seconds: float
    max_question_length: int
    log_level: str
    context_window: int = 4096

    # Phase 2: documents, embeddings, and retrieval
    embedding_model: str = "nomic-embed-text"
    embedding_query_prefix: str = "search_query:"
    embedding_document_prefix: str = "search_document:"
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection: str = "learning_documents"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    min_relevance_score: float = 0.5
    max_upload_mb: float = 20.0

    # Phase 3: optional external research (OFF unless you turn it on)
    research_enabled: bool = False
    search_provider: str = "wikipedia"
    search_max_results: int = 5
    search_timeout_seconds: float = 15.0
    research_contact: str = ""
    wikipedia_language: str = "en"
    searxng_base_url: str = "http://localhost:8080"
    # repr=False keeps the key out of printed settings, error pages, and logs.
    tavily_api_key: str = field(default="", repr=False)
    research_fetch_pages: bool = False
    research_min_quality: float = 0.4
    research_blocked_domains: tuple[str, ...] = ()


def _read_text(name: str, default: str, allow_empty: bool = False) -> str:
    value = os.getenv(name, default).strip()
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return value


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


def _read_bool(name: str, default: bool) -> bool:
    """Accept true/false, yes/no, on/off, or 1/0. An empty value means the default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    value = raw_value.strip().lower()
    if value in _TRUE_WORDS:
        return True
    if value in _FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be true or false, got {raw_value!r}")


def _read_domain_list(name: str) -> tuple[str, ...]:
    """Read a comma-separated list like 'example.com, spam.org'."""
    domains: list[str] = []
    for part in os.getenv(name, "").split(","):
        domain = part.strip().lower()
        if not domain:
            continue
        if "/" in domain or ":" in domain:
            raise ValueError(f"{name} must list site names only, like example.com (got {part.strip()!r})")
        domains.append(domain.removeprefix("www."))
    return tuple(dict.fromkeys(domains))  # removes repeats, keeps the order


def _resolve_path(raw_path: str) -> Path:
    """Relative paths are placed inside the project folder, wherever you run from."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_settings(use_dotenv: bool = True) -> Settings:
    """Read settings from the environment, validating each value.

    use_dotenv=False lets tests ignore any real .env file on your machine.
    """
    if use_dotenv:
        load_dotenv()

    # ----- Phase 1 -----
    model = _read_text("OLLAMA_MODEL", "llama3.2:3b")

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

    context_window = _read_int("OLLAMA_NUM_CTX", 4096)
    if context_window < 512:
        raise ValueError("OLLAMA_NUM_CTX must be at least 512")

    # ----- Phase 2 -----
    embedding_model = _read_text("EMBEDDING_MODEL", "nomic-embed-text")
    query_prefix = _read_text("EMBEDDING_QUERY_PREFIX", "search_query:", allow_empty=True)
    document_prefix = _read_text("EMBEDDING_DOCUMENT_PREFIX", "search_document:", allow_empty=True)

    persist_dir = _resolve_path(_read_text("CHROMA_PERSIST_DIR", "data/chroma"))
    collection = _read_text("CHROMA_COLLECTION", "learning_documents")

    chunk_size = _read_int("CHUNK_SIZE", 800)
    if chunk_size < 100:
        raise ValueError("CHUNK_SIZE must be at least 100 characters")

    chunk_overlap = _read_int("CHUNK_OVERLAP", 150)
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("CHUNK_OVERLAP must be 0 or more and smaller than CHUNK_SIZE")

    top_k = _read_int("RETRIEVAL_TOP_K", 4)
    if not 1 <= top_k <= 20:
        raise ValueError("RETRIEVAL_TOP_K must be between 1 and 20")

    min_score = _read_float("MIN_RELEVANCE_SCORE", 0.5)
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("MIN_RELEVANCE_SCORE must be between 0.0 and 1.0")

    max_upload_mb = _read_float("MAX_UPLOAD_MB", 20.0)
    if max_upload_mb <= 0:
        raise ValueError("MAX_UPLOAD_MB must be greater than 0")

    # ----- Phase 3 -----
    research_enabled = _read_bool("RESEARCH_ENABLED", False)

    search_provider = _read_text("SEARCH_PROVIDER", "wikipedia").lower()
    if search_provider not in SEARCH_PROVIDERS:
        raise ValueError(f"SEARCH_PROVIDER must be one of: {', '.join(SEARCH_PROVIDERS)}")

    search_max_results = _read_int("SEARCH_MAX_RESULTS", 5)
    if not 1 <= search_max_results <= 10:
        raise ValueError("SEARCH_MAX_RESULTS must be between 1 and 10")

    search_timeout = _read_float("SEARCH_TIMEOUT_SECONDS", 15.0)
    if not 0 < search_timeout <= 120:
        raise ValueError("SEARCH_TIMEOUT_SECONDS must be more than 0 and at most 120")

    contact = _read_text("RESEARCH_CONTACT", "", allow_empty=True)
    if contact and not (contact.isascii() and contact.isprintable()):
        raise ValueError("RESEARCH_CONTACT must be plain text on one line, like an email address")

    wikipedia_language = _read_text("WIKIPEDIA_LANGUAGE", "en").lower()
    if not _WIKIPEDIA_LANGUAGE_PATTERN.match(wikipedia_language):
        raise ValueError("WIKIPEDIA_LANGUAGE must be a language code like en, de, or simple")

    searxng_url = _read_text("SEARXNG_BASE_URL", "http://localhost:8080").rstrip("/")
    if not searxng_url.startswith(("http://", "https://")):
        raise ValueError("SEARXNG_BASE_URL must start with http:// or https://")

    tavily_key = _read_text("TAVILY_API_KEY", "", allow_empty=True)
    fetch_pages = _read_bool("RESEARCH_FETCH_PAGES", False)

    min_quality = _read_float("RESEARCH_MIN_QUALITY", 0.4)
    if not 0.0 <= min_quality <= 1.0:
        raise ValueError("RESEARCH_MIN_QUALITY must be between 0.0 and 1.0")

    blocked_domains = _read_domain_list("RESEARCH_BLOCKED_DOMAINS")

    # Some settings are only required once research is switched on.
    if research_enabled and search_provider == "tavily" and not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily. "
            "Get a free key at https://app.tavily.com, or choose another provider"
        )
    if research_enabled and search_provider == "wikipedia" and not contact:
        raise ValueError(
            "RESEARCH_CONTACT is required for Wikipedia (an email address or web page). "
            "Wikimedia asks every app to say who runs it"
        )

    return Settings(
        ollama_base_url=base_url,
        ollama_model=model,
        temperature=temperature,
        request_timeout_seconds=timeout,
        max_question_length=max_length,
        log_level=log_level,
        context_window=context_window,
        embedding_model=embedding_model,
        embedding_query_prefix=query_prefix,
        embedding_document_prefix=document_prefix,
        chroma_persist_dir=persist_dir,
        chroma_collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_top_k=top_k,
        min_relevance_score=min_score,
        max_upload_mb=max_upload_mb,
        research_enabled=research_enabled,
        search_provider=search_provider,
        search_max_results=search_max_results,
        search_timeout_seconds=search_timeout,
        research_contact=contact,
        wikipedia_language=wikipedia_language,
        searxng_base_url=searxng_url,
        tavily_api_key=tavily_key,
        research_fetch_pages=fetch_pages,
        research_min_quality=min_quality,
        research_blocked_domains=blocked_domains,
    )