"""Phase 8: understand what the learner is asking for, before the workflow acts.

Three kinds of request:
  lesson - "Teach me photosynthesis", "Explain embeddings"
  quiz   - "Quiz me on photosynthesis", "Test me"
  next   - "What should I study next?"

Clear wording is recognised with simple rules (free and instant). Only when the
rules can't tell does the workflow ask the model for a one-word answer
(LESSON, QUIZ, or NEXT), read here by parse_intent. No LangChain imports.
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["lesson", "quiz", "next"]

MAX_TOPIC_CHARS = 80

_NEXT = re.compile(
    r"\b(what (should|do|can) i (study|learn|do|revise|review) next|what next|study next|learn next|"
    r"recommend\w*|suggest\w*|my progress|where am i|what('s| is) next)\b",
    re.IGNORECASE,
)
_QUIZ = re.compile(r"\b(quiz|test me|test my|assess|exam me|practi[cs]e|questions on|question me|check my understanding)\b", re.IGNORECASE)
_LESSON = re.compile(r"\b(teach|explain|lesson|learn|study|understand|introduce|introduction to|what is|what are|how does|how do)\b", re.IGNORECASE)
_ONE_WORD = re.compile(r"\b(lesson|quiz|next)\b", re.IGNORECASE)

# Request phrases removed to leave the topic: "Please quiz me on the water cycle" -> "the water cycle"
_TOPIC_PREFIXES = re.compile(
    r"^(please\s+)?(can you\s+|could you\s+)?("
    r"teach me( about)?|explain( to me)?|give me a lesson (on|about)|a lesson (on|about)|lesson (on|about)|"
    r"i want to (learn|study|understand)( about)?|help me (learn|understand)( about)?|introduce( me to)?|"
    r"quiz me( on| about)?|test me( on| about)?|test my (knowledge|understanding) (of|on)|"
    r"give me a quiz (on|about)|a quiz (on|about)|quiz (on|about)|practi[cs]e|ask me questions (on|about)|"
    r"what is|what are|how does|how do"
    r")\s*",
    re.IGNORECASE,
)


def detect_intent(request: str) -> Intent | None:
    """Rule-based guess. None means the rules can't tell (ask the model)."""
    text = request or ""
    if _NEXT.search(text):
        return "next"
    if _QUIZ.search(text):
        return "quiz"
    if _LESSON.search(text):
        return "lesson"
    return None


def parse_intent(reply: str) -> Intent | None:
    """Read the model's one-word reply: 'Quiz.' -> 'quiz'. Unclear -> None."""
    match = _ONE_WORD.search(reply or "")
    return match.group(1).lower() if match else None  # type: ignore[return-value]


def clean_topic(request: str) -> str:
    """'Please teach me about photosynthesis!' -> 'photosynthesis'. May return ''."""
    text = re.sub(r"\s+", " ", request or "").strip()
    previous = None
    while previous != text:  # remove several prefixes, e.g. "please can you teach me"
        previous = text
        text = _TOPIC_PREFIXES.sub("", text).strip()
    text = text.strip(" ?!.,;:\"'`")
    if text.lower() in {"me", "it", "this", "something", "anything"}:
        return ""
    if len(text) > MAX_TOPIC_CHARS:
        text = text[:MAX_TOPIC_CHARS].rsplit(" ", 1)[0]
    return text