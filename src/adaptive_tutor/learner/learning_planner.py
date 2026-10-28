"""Phase 6: decide what the learner should do next, with a reason for every suggestion.

Rules, in priority order:
1. prerequisite - struggling with a concept whose prerequisite is weak or unseen:
                  study the prerequisite first.
2. remediate    - mastery under 0.4: a simpler lesson on it, then practice.
3. revise       - mastered, but its review date has arrived: a short quiz.
4. practice     - mastery 0.4 to 0.8: another quiz.
5. start        - seen in a lesson but never assessed: take a first quiz.
6. advance      - every concept in a topic is mastered: try the next level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

from adaptive_tutor.assessment.schemas import LEVELS, normalize_topic
from adaptive_tutor.learner.mastery import ConceptProgress, concept_status

Action = Literal["prerequisite", "remediate", "revise", "practice", "start", "advance"]

PRIORITY: dict[str, int] = {"prerequisite": 0, "remediate": 1, "revise": 2, "practice": 3, "start": 4, "advance": 5}

# Which activity each recommendation leads to in the app.
ACTIVITY_FOR_ACTION: dict[str, str] = {
    "prerequisite": "lesson",
    "remediate": "lesson",
    "revise": "quiz",
    "practice": "quiz",
    "start": "quiz",
    "advance": "lesson",
}


@dataclass(frozen=True)
class Recommendation:
    action: Action
    concept: str  # for "advance", this is the topic
    topic: str
    reason: str
    activity: str  # "lesson" or "quiz"
    level: str = "beginner"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recommendation:
        return cls(**data)


def next_level(level: str) -> str | None:
    """beginner -> intermediate -> advanced -> None."""
    index = LEVELS.index(level) if level in LEVELS else 0
    return LEVELS[index + 1] if index + 1 < len(LEVELS) else None


def _make(action: Action, concept: str, topic: str, reason: str, level: str = "beginner") -> Recommendation:
    return Recommendation(action, concept, topic, reason, ACTIVITY_FOR_ACTION[action], level)


def recommend(
    progress: list[ConceptProgress],
    prerequisites: dict[str, list[str]],
    today: date,
    topic_levels: dict[str, str] | None = None,
    limit: int = 5,
) -> list[Recommendation]:
    """Build the list of next activities, most important first."""
    by_concept = {item.concept: item for item in progress}
    topic_levels = topic_levels or {}
    # One suggestion per concept: if two rules match, the higher-priority one wins.
    best: dict[str, tuple[tuple[int, float, str], Recommendation]] = {}

    def add(recommendation: Recommendation, mastery: float) -> None:
        sort_key = (PRIORITY[recommendation.action], mastery, recommendation.concept)
        name = f"topic:{recommendation.concept}" if recommendation.action == "advance" else recommendation.concept
        current = best.get(name)
        if current is None or sort_key < current[0]:
            best[name] = (sort_key, recommendation)

    for item in progress:
        status = concept_status(item, today)
        if status == "remediate":
            for prerequisite in prerequisites.get(item.concept, []):
                known = by_concept.get(prerequisite)
                if known is None or concept_status(known, today) in ("new", "remediate"):
                    add(
                        _make(
                            "prerequisite",
                            prerequisite,
                            known.topic if known else item.topic,
                            f"You're finding “{item.concept}” hard, and it builds on “{prerequisite}”, "
                            "which isn't solid yet. Study that first.",
                        ),
                        known.mastery if known else 0.0,
                    )
            add(
                _make(
                    "remediate",
                    item.concept,
                    item.topic,
                    f"Mastery is {item.mastery:.2f} (under 0.40). A simpler lesson and some practice will help.",
                ),
                item.mastery,
            )
        elif status == "revise":
            days = (today - item.last_reviewed).days if item.last_reviewed else 0
            add(
                _make("revise", item.concept, item.topic, f"Mastered, last reviewed {days} day(s) ago. A quick review keeps it fresh."),
                item.mastery,
            )
        elif status == "practice":
            add(
                _make("practice", item.concept, item.topic, f"Mastery is {item.mastery:.2f}. Another quiz will make it solid (0.80+)."),
                item.mastery,
            )
        elif status == "new":
            add(_make("start", item.concept, item.topic, "You studied this in a lesson but haven't been quizzed on it yet."), 0.0)

    # Advance: every concept of a topic is mastered (and none is due for review).
    topics: dict[str, list[ConceptProgress]] = {}
    for item in progress:
        topics.setdefault(item.topic, []).append(item)
    for topic, items in topics.items():
        if all(concept_status(item, today) == "mastered" for item in items):
            current = topic_levels.get(topic, "beginner")
            upcoming = next_level(current)
            if upcoming:
                reason = f"You've mastered every concept in “{topic}” at {current} level. Try {upcoming} level."
                add(_make("advance", topic, topic, reason, level=upcoming), 1.0)
            else:
                reason = f"You've mastered “{topic}” at advanced level. Pick a new topic, or revise it later."
                add(_make("advance", topic, topic, reason, level=current), 1.0)

    ordered = sorted(best.values(), key=lambda pair: pair[0])
    return [recommendation for _, recommendation in ordered[:limit]]


def focus_concepts(progress: list[ConceptProgress], topic: str, today: date, limit: int = 3) -> list[str]:
    """Concepts in this topic the learner finds hard (or should revise), weakest first."""
    wanted = {"remediate", "practice", "revise"}
    items = [item for item in progress if item.topic == normalize_topic(topic) and concept_status(item, today) in wanted]
    items.sort(key=lambda item: (item.mastery, item.concept))
    return [item.concept for item in items[:limit]]


def activity_request(recommendation: Recommendation) -> tuple[str, str, str]:
    """The (request, action, level) that starts a recommended activity in the app.

    A remediation lesson is on the whole topic: the workflow then finds the weak
    concepts in the profile and focuses the lesson on them.
    """
    if recommendation.action == "prerequisite":
        return f"Teach me {recommendation.concept}", "lesson", "beginner"
    if recommendation.action in ("remediate", "advance"):
        return f"Teach me {recommendation.topic}", "lesson", recommendation.level
    return f"Quiz me on {recommendation.topic}", "quiz", recommendation.level