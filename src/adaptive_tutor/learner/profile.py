"""Phase 6-7: who the learner is. Each learner's records are kept apart by learner_id.

There are no passwords: this is a local learning app, so anyone using this
computer can open any profile. Profiles keep progress apart; they don't protect it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

MAX_NAME_CHARS = 40
_NAME_PATTERN = re.compile(r"^[\w .'-]+$")  # letters (any language), digits, space . ' - _


@dataclass(frozen=True)
class LearnerProfile:
    learner_id: str  # stable id, never shown in the UI, used in every database row
    name: str  # display name chosen by the learner
    created_at: str  # ISO date and time


def clean_learner_name(raw: str) -> str:
    """Tidy up and check a display name. Raises ValueError with a plain message."""
    name = re.sub(r"\s+", " ", raw or "").strip()
    if not name:
        raise ValueError("Please type a name for the new learner.")
    if len(name) > MAX_NAME_CHARS:
        raise ValueError(f"Please keep the name under {MAX_NAME_CHARS} characters.")
    if not _NAME_PATTERN.match(name):
        raise ValueError("Names can use letters, numbers, spaces, and . ' - _ only.")
    return name


def make_learner_id(name: str) -> str:
    """'Moksh Saini' -> 'moksh-saini-3f9a1c2b'. The random part keeps ids unique forever."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:20] or "learner"
    return f"{slug}-{uuid.uuid4().hex[:8]}"