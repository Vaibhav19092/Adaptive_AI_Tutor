"""Phase 7: saved learning progress in a local SQLite database.

SQLite is a whole database in one file (data/tutor.sqlite), built into Python,
so no server and no new package are needed.

Separation of learners: every table has a learner_id column, and every method
takes a learner_id and filters by it. Deleting a learner removes all their rows
(ON DELETE CASCADE). The tests check that two learners never see each other's data.

Privacy: lesson and quiz text is stored because the learner wants to review it;
metric events store only numbers and labels, never questions or answers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from adaptive_tutor.learner.mastery import ConceptProgress
from adaptive_tutor.learner.profile import LearnerProfile, clean_learner_name, make_learner_id

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS concepts (
    learner_id    TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept       TEXT NOT NULL,
    topic         TEXT NOT NULL,
    mastery       REAL NOT NULL DEFAULT 0,
    attempts      INTEGER NOT NULL DEFAULT 0,
    correct       INTEGER NOT NULL DEFAULT 0,
    streak        INTEGER NOT NULL DEFAULT 0,
    last_reviewed TEXT,
    next_review   TEXT,
    PRIMARY KEY (learner_id, concept)
);
CREATE TABLE IF NOT EXISTS prerequisites (
    learner_id   TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept      TEXT NOT NULL,
    prerequisite TEXT NOT NULL,
    PRIMARY KEY (learner_id, concept, prerequisite)
);
CREATE TABLE IF NOT EXISTS misconceptions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    concept    TEXT NOT NULL,
    note       TEXT NOT NULL,
    quiz_id    TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id     TEXT PRIMARY KEY,
    learner_id    TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    topic         TEXT NOT NULL,
    level         TEXT NOT NULL,
    evidence_kind TEXT NOT NULL,
    lesson_json   TEXT NOT NULL,
    sources_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
    quiz_id     TEXT PRIMARY KEY,
    learner_id  TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    thread_id   TEXT NOT NULL,
    lesson_id   TEXT,
    topic       TEXT NOT NULL,
    level       TEXT NOT NULL,
    status      TEXT NOT NULL,
    quiz_json   TEXT NOT NULL,
    result_json TEXT,
    score       REAL,
    created_at  TEXT NOT NULL,
    graded_at   TEXT
);
CREATE TABLE IF NOT EXISTS question_results (
    learner_id  TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    quiz_id     TEXT NOT NULL,
    question_id TEXT NOT NULL,
    concept     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    score       REAL NOT NULL,
    correct     INTEGER NOT NULL,
    graded      INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (learner_id, quiz_id, question_id)
);
CREATE TABLE IF NOT EXISTS activities (
    thread_id  TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id  TEXT REFERENCES learners(learner_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    data_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_activities_learner ON activities(learner_id, status);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class LearnerRepository:
    """All reads and writes of saved progress. Tests use a database in a temporary folder."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        logger.info("Progress database ready at %s", self.path.name)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One short connection per operation: safe with Streamlit's threads.

        The `with` block commits on success, rolls back on any error, and always closes.
        """
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- Learners ----------

    def create_learner(self, name: str) -> LearnerProfile:
        name = clean_learner_name(name)
        profile = LearnerProfile(make_learner_id(name), name, _now())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO learners (learner_id, name, created_at) VALUES (?, ?, ?)",
                    (profile.learner_id, profile.name, profile.created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A learner called “{name}” already exists. Choose another name.") from exc
        return profile

    def list_learners(self) -> list[LearnerProfile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT learner_id, name, created_at FROM learners ORDER BY name COLLATE NOCASE").fetchall()
        return [LearnerProfile(row["learner_id"], row["name"], row["created_at"]) for row in rows]

    def get_learner(self, learner_id: str) -> LearnerProfile | None:
        with self._connect() as conn:
            row = conn.execute("SELECT learner_id, name, created_at FROM learners WHERE learner_id = ?", (learner_id,)).fetchone()
        return LearnerProfile(row["learner_id"], row["name"], row["created_at"]) if row else None

    def thread_ids(self, learner_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT thread_id FROM activities WHERE learner_id = ?", (learner_id,)).fetchall()
        return [row["thread_id"] for row in rows]

    def reset_progress(self, learner_id: str) -> list[str]:
        """Delete everything learned and tested, but keep the profile. Returns the old thread ids."""
        threads = self.thread_ids(learner_id)
        with self._connect() as conn:
            for table in ("concepts", "prerequisites", "misconceptions", "lessons", "assessments",
                          "question_results", "activities", "events"):
                conn.execute(f"DELETE FROM {table} WHERE learner_id = ?", (learner_id,))  # noqa: S608 - fixed names
        logger.info("Progress reset for one learner (%d activities removed)", len(threads))
        return threads

    def delete_learner(self, learner_id: str) -> list[str]:
        """Delete the learner and every row that belongs to them. Returns their thread ids."""
        threads = self.thread_ids(learner_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM learners WHERE learner_id = ?", (learner_id,))
        logger.info("Learner deleted (%d activities removed)", len(threads))
        return threads

    # ---------- Concepts and mastery ----------

    def get_progress(self, learner_id: str) -> list[ConceptProgress]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT concept, topic, mastery, attempts, correct, streak, last_reviewed, next_review "
                "FROM concepts WHERE learner_id = ? ORDER BY topic, concept",
                (learner_id,),
            ).fetchall()
        return [
            ConceptProgress(
                concept=row["concept"],
                topic=row["topic"],
                mastery=row["mastery"],
                attempts=row["attempts"],
                correct=row["correct"],
                streak=row["streak"],
                last_reviewed=date.fromisoformat(row["last_reviewed"]) if row["last_reviewed"] else None,
                next_review=date.fromisoformat(row["next_review"]) if row["next_review"] else None,
            )
            for row in rows
        ]

    def add_concepts(self, learner_id: str, topic: str, concepts: list[str]) -> None:
        """Register concepts seen in a lesson (status "new"). Existing progress is kept."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO concepts (learner_id, concept, topic) VALUES (?, ?, ?)",
                [(learner_id, concept, topic) for concept in concepts],
            )

    def add_prerequisites(self, learner_id: str, concepts: list[str], prerequisites: list[str]) -> None:
        rows = [(learner_id, c, p) for c in concepts for p in prerequisites if c != p]
        with self._connect() as conn:
            conn.executemany("INSERT OR IGNORE INTO prerequisites VALUES (?, ?, ?)", rows)

    def get_prerequisites(self, learner_id: str) -> dict[str, list[str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT concept, prerequisite FROM prerequisites WHERE learner_id = ? ORDER BY concept, prerequisite",
                (learner_id,),
            ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row["concept"], []).append(row["prerequisite"])
        return result

    def get_misconceptions(self, learner_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT concept, note, quiz_id, created_at FROM misconceptions WHERE learner_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (learner_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def topic_levels(self, learner_id: str) -> dict[str, str]:
        """The highest level of each topic the learner has been quizzed on."""
        order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT topic, level FROM assessments WHERE learner_id = ? AND status = 'graded'", (learner_id,)
            ).fetchall()
        levels: dict[str, str] = {}
        for row in rows:
            if order.get(row["level"], 0) >= order.get(levels.get(row["topic"], "beginner"), 0):
                levels[row["topic"]] = row["level"]
        return levels

    # ---------- Lessons and quizzes ----------

    def save_activity_material(
        self,
        learner_id: str,
        thread_id: str,
        quiz: dict[str, Any],
        lesson: dict[str, Any] | None,
        lesson_id: str | None,
        sources: list[dict[str, Any]],
        evidence_kind: str,
    ) -> None:
        """Save a new lesson (if any) and its quiz in one transaction. The quiz starts 'open'."""
        now = _now()
        with self._connect() as conn:
            if lesson is not None and lesson_id:
                conn.execute(
                    "INSERT OR REPLACE INTO lessons VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (lesson_id, learner_id, quiz["topic"], quiz["level"], evidence_kind,
                     json.dumps(lesson), json.dumps(sources), now),
                )
            conn.execute(
                "INSERT OR REPLACE INTO assessments "
                "(quiz_id, learner_id, thread_id, lesson_id, topic, level, status, quiz_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)",
                (quiz["quiz_id"], learner_id, thread_id, lesson_id, quiz["topic"], quiz["level"], json.dumps(quiz), now),
            )

    def list_lessons(self, learner_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT lesson_id, topic, level, evidence_kind, lesson_json, sources_json, created_at "
                "FROM lessons WHERE learner_id = ? ORDER BY created_at DESC LIMIT ?",
                (learner_id, limit),
            ).fetchall()
        return [
            {
                "lesson_id": row["lesson_id"],
                "topic": row["topic"],
                "level": row["level"],
                "evidence_kind": row["evidence_kind"],
                "lesson": json.loads(row["lesson_json"]),
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_assessment(
        self,
        learner_id: str,
        quiz_id: str,
        result: dict[str, Any],
        progress: list[ConceptProgress],
        misconceptions: list[dict[str, str]],
    ) -> None:
        """Save a graded quiz, its question results, the new mastery values, and any
        possible misconceptions, all in ONE transaction (all saved, or none)."""
        now = _now()
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE assessments SET status = 'graded', result_json = ?, score = ?, graded_at = ? "
                "WHERE quiz_id = ? AND learner_id = ?",
                (json.dumps(result), result.get("score"), now, quiz_id, learner_id),
            ).rowcount
            if updated != 1:
                raise ValueError("This quiz doesn't belong to the current learner.")
            conn.executemany(
                "INSERT OR REPLACE INTO question_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (learner_id, quiz_id, item["question_id"], item["concept"], item["kind"], item["score"],
                     int(item["correct"]), int(item["graded"]), now)
                    for item in result["results"]
                ],
            )
            conn.executemany(
                "INSERT INTO concepts (learner_id, concept, topic, mastery, attempts, correct, streak, last_reviewed, next_review) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (learner_id, concept) DO UPDATE SET mastery = excluded.mastery, "
                "attempts = excluded.attempts, correct = excluded.correct, streak = excluded.streak, "
                "last_reviewed = excluded.last_reviewed, next_review = excluded.next_review",
                [
                    (learner_id, p.concept, p.topic, p.mastery, p.attempts, p.correct, p.streak,
                     p.last_reviewed.isoformat() if p.last_reviewed else None,
                     p.next_review.isoformat() if p.next_review else None)
                    for p in progress
                ],
            )
            conn.executemany(
                "INSERT INTO misconceptions (learner_id, concept, note, quiz_id, created_at) VALUES (?, ?, ?, ?, ?)",
                [(learner_id, m["concept"], m["note"], quiz_id, now) for m in misconceptions],
            )

    def set_quiz_status(self, learner_id: str, quiz_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE assessments SET status = ? WHERE quiz_id = ? AND learner_id = ?", (status, quiz_id, learner_id))

    def get_quiz(self, learner_id: str, quiz_id: str) -> dict[str, Any] | None:
        """The full quiz (with answers). Only used after grading, e.g. for the consistency check."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT quiz_json FROM assessments WHERE quiz_id = ? AND learner_id = ?", (quiz_id, learner_id)
            ).fetchone()
        return json.loads(row["quiz_json"]) if row else None

    def list_assessments(self, learner_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Quiz history, newest first. Answers are only included for graded quizzes."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT quiz_id, topic, level, status, score, result_json, created_at, graded_at "
                "FROM assessments WHERE learner_id = ? ORDER BY created_at DESC LIMIT ?",
                (learner_id, limit),
            ).fetchall()
        history = []
        for row in rows:
            item = {key: row[key] for key in ("quiz_id", "topic", "level", "status", "score", "created_at", "graded_at")}
            item["result"] = json.loads(row["result_json"]) if row["result_json"] else None
            history.append(item)
        return history

    def question_results(self, learner_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT quiz_id, question_id, concept, kind, score, correct, graded, created_at "
                "FROM question_results WHERE learner_id = ? ORDER BY created_at",
                (learner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------- Activities (one LangGraph thread each) ----------

    def start_activity(self, learner_id: str, thread_id: str) -> None:
        now = _now()
        with self._connect() as conn:
            # Only one open activity per learner: an unfinished one is abandoned.
            conn.execute(
                "UPDATE activities SET status = 'abandoned', updated_at = ? WHERE learner_id = ? AND status = 'open'",
                (now, learner_id),
            )
            conn.execute("INSERT INTO activities VALUES (?, ?, 'open', ?, ?)", (thread_id, learner_id, now, now))

    def open_activity(self, learner_id: str) -> str | None:
        """The thread id of the learner's unfinished activity, so it can be resumed after a restart."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT thread_id FROM activities WHERE learner_id = ? AND status = 'open' "
                "ORDER BY created_at DESC LIMIT 1",
                (learner_id,),
            ).fetchone()
        return row["thread_id"] if row else None

    def close_activity(self, learner_id: str, thread_id: str, status: str = "done") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE activities SET status = ?, updated_at = ? WHERE thread_id = ? AND learner_id = ?",
                (status, _now(), thread_id, learner_id),
            )
            if status == "discarded":
                conn.execute(
                    "UPDATE assessments SET status = 'discarded' WHERE thread_id = ? AND learner_id = ? AND status = 'open'",
                    (thread_id, learner_id),
                )

    # ---------- Metric events (Phase 9) ----------

    def record_event(self, learner_id: str | None, kind: str, duration_ms: int, data: dict[str, Any]) -> None:
        """Store one timing/quality record. `data` must hold only numbers and labels, never text the learner wrote."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (learner_id, kind, duration_ms, data_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (learner_id, kind, int(duration_ms), json.dumps(data), _now()),
            )

    def list_events(self, learner_id: str | None = None, kind: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        query = "SELECT learner_id, kind, duration_ms, data_json, created_at FROM events WHERE 1 = 1"
        params: list[Any] = []
        if learner_id is not None:
            query += " AND learner_id = ?"
            params.append(learner_id)
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"learner_id": row["learner_id"], "kind": row["kind"], "duration_ms": row["duration_ms"],
             "data": json.loads(row["data_json"]), "created_at": row["created_at"]}
            for row in rows
        ]