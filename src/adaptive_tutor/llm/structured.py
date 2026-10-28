"""Phase 5: get structured data (JSON) out of a small local model, safely.

Lessons, quizzes, and grading results must be read by the app as data, not text.
The model is asked for JSON, and every reply goes through three steps:

1. extract_json: find the JSON in the reply (small models sometimes wrap it in a
   Markdown code fence, or add a sentence before it).
2. a parse function: check the data has every field we need, with the right types.
   It raises ValueError (or SchemaError) with a plain-language reason.
3. invoke_structured: if step 1 or 2 fails, ask again (bounded by
   STRUCTURED_OUTPUT_RETRIES) and tell the model what was wrong. If it still
   fails, raise StructuredOutputError with a message the learner can act on.

This module has no LangChain imports, so it can be tested anywhere.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# A Markdown code fence is three backticks. It is built with "`" * 3 so this file
# never contains three backticks in a row (they break copy-and-paste from a chat).
FENCE = "`" * 3
_FENCE = re.compile(FENCE + r"(?:json)?\s*(.*?)" + FENCE, re.DOTALL | re.IGNORECASE)

# Added to the prompt on a retry, so the model knows what to fix.
RETRY_FEEDBACK = (
    "\n\nYour previous reply could not be used because {problem}. "
    "Reply again with ONLY one valid JSON object in exactly the format described."
)


class StructuredOutputError(Exception):
    """The model never returned usable JSON. The message is shown to the learner."""


class Invokable(Protocol):
    """Anything with .invoke(), like a LangChain chain or a test fake."""

    def invoke(self, inputs: dict[str, Any]) -> Any: ...


def extract_json(text: str) -> Any:
    """Return the first JSON object or list found in the model's reply.

    '{"a": 1}'                       -> {'a': 1}
    'Here you go: {"a": 1} Thanks!'  -> {'a': 1}
    the same object inside a json code fence -> {'a': 1}
    Raises ValueError when there is no valid JSON.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("the reply was empty")

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for the first { or [ and read one JSON value from there.
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character in "{[":
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("the reply did not contain valid JSON")


def invoke_structured(
    chain: Invokable,
    inputs: dict[str, Any],
    parse: Callable[[Any], T],
    retries: int,
    what: str,
) -> tuple[T, int]:
    """Ask for JSON, check it with `parse`, and retry up to `retries` times.

    The prompt must contain a {format_feedback} variable: it is empty on the
    first try and explains the problem on a retry.
    Returns (parsed value, number of tries used).
    Model connection errors are NOT caught here; the graph nodes turn them into
    friendly messages with describe_llm_error.
    """
    problem = ""
    for attempt in range(1, retries + 2):
        payload = dict(inputs)
        payload["format_feedback"] = RETRY_FEEDBACK.format(problem=problem) if problem else ""
        reply = chain.invoke(payload)
        reply_text = reply if isinstance(reply, str) else str(getattr(reply, "content", reply))
        try:
            data = extract_json(reply_text)
            return parse(data), attempt
        except ValueError as exc:  # SchemaError and JSONDecodeError are ValueErrors too
            problem = str(exc) or "the reply was not in the expected format"
            logger.warning("Unusable %s from the model (try %d): %s", what, attempt, problem)

    raise StructuredOutputError(
        f"The local model didn't return a usable {what} after {retries + 1} tries "
        f"(last problem: {problem}). Try again, try a narrower topic, or use a larger model. "
        "You can also raise STRUCTURED_OUTPUT_RETRIES in .env."
    )