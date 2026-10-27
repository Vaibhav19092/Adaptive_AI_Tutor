"""Clean up the model's reply into a short web search query.

Small local models often add extras like 'Search query: "..."' or
'Here is a query:'. These helpers strip that away.
"""

from __future__ import annotations

import re

MAX_QUERY_CHARS = 100

_LABEL = re.compile(r"^(search query|query|search|keywords)\s*:\s*", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^(\d+[.)]|[-*•])\s+")
_EDGE_CHARACTERS = " \"'`*“”‘’"


def shorten(text: str, max_chars: int = MAX_QUERY_CHARS) -> str:
    """Cut text to max_chars without splitting a word in half."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


def clean_search_query(raw: str, fallback: str) -> str:
    """Return the first useful line of the model's reply, or the fallback (the question)."""
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue  # skip blank lines and intros like "Here is a search query:"
        line = _LIST_MARKER.sub("", line.strip(_EDGE_CHARACTERS))
        line = _LABEL.sub("", line)
        line = re.sub(r"\s+", " ", line.strip(_EDGE_CHARACTERS)).replace("`", "")
        if line:
            return shorten(line)
    return shorten(re.sub(r"\s+", " ", fallback).strip())