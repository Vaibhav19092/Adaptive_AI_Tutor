"""Phase 6: simple, explainable mastery estimates for each concept.

The rule (an "exponential moving average"):
    new mastery = old mastery + 0.4 x (latest score - old mastery)
So mastery moves 40% of the way toward your latest score. Four good answers in a
row take a new concept from 0.00 to about 0.87; one wrong answer pulls it down again.

Review dates use simple spaced repetition: after each well-answered question the
gap before the next review grows (1, 3, 7, 14, then 30 days). A poor answer
resets the gap to 1 day.

These numbers are reasonable rules of thumb for a learning project. They are NOT
a scientifically validated measure of knowledge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from typing import Any, Literal

LEARNING_RATE = 0.4
REMEDIATE_BELOW = 0.4  # mastery under this -> re-teach the concept
MASTERED_AT = 0.8  # mastery at or above this -> mastered (review it later)
WELL_ANSWERED = 0.8  # a question score at or above this grows the review gap
REVIEW_INTERVALS_DAYS = (1, 3, 7, 14, 30)

ConceptStatus = Literal["new", "remediate", "practice", "mastered", "revise"]


@dataclass(frozen=True)
class ConceptProgress:
    """Everything the tutor remembers about one concept for one learner."""

    concept: str
    topic: str
    mastery: float = 0.0
    attempts: int = 0  # graded questions answered on this concept
    correct: int = 0  # how many of those counted as correct
    streak: int = 0  # well-answered questions in a row
    last_reviewed: date | None = None
    next_review: date | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["last_reviewed"] = self.last_reviewed.isoformat() if self.last_reviewed else None
        data["next_review"] = self.next_review.isoformat() if self.next_review else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptProgress:
        values = dict(data)
        for key in ("last_reviewed", "next_review"):
            values[key] = date.fromisoformat(values[key]) if values.get(key) else None
        return cls(**values)


def review_interval_days(streak: int) -> int:
    """streak 0 or 1 -> 1 day, 2 -> 3 days, 3 -> 7, 4 -> 14, 5 or more -> 30."""
    index = max(0, min(streak, len(REVIEW_INTERVALS_DAYS)) - 1)
    return REVIEW_INTERVALS_DAYS[index]


def update_progress(progress: ConceptProgress, score: float, today: date) -> ConceptProgress:
    """Apply one graded question (score 0.0 to 1.0) to a concept."""
    score = max(0.0, min(1.0, score))
    mastery = round(progress.mastery + LEARNING_RATE * (score - progress.mastery), 3)
    streak = progress.streak + 1 if score >= WELL_ANSWERED else 0
    return replace(
        progress,
        mastery=mastery,
        attempts=progress.attempts + 1,
        correct=progress.correct + (1 if score >= 0.5 else 0),
        streak=streak,
        last_reviewed=today,
        next_review=today + timedelta(days=review_interval_days(streak)),
    )


def concept_status(progress: ConceptProgress, today: date) -> ConceptStatus:
    """Turn the numbers into one plain-language status."""
    if progress.attempts == 0:
        return "new"  # seen in a lesson, never assessed
    if progress.mastery < REMEDIATE_BELOW:
        return "remediate"
    if progress.mastery < MASTERED_AT:
        return "practice"
    if progress.next_review is not None and today >= progress.next_review:
        return "revise"
    return "mastered"


def describe_change(before: ConceptProgress, after: ConceptProgress) -> str:
    """'neural networks: mastery 0.40 -> 0.64 (next review in 3 days)'."""
    days = (after.next_review - after.last_reviewed).days if after.next_review and after.last_reviewed else None
    review = f" (next review in {days} day{'s' if days != 1 else ''})" if days is not None else ""
    return f"{after.concept}: mastery {before.mastery:.2f} → {after.mastery:.2f}{review}"