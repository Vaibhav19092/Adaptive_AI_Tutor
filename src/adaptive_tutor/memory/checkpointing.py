"""Phase 7: LangGraph checkpointing, so an unfinished activity survives a restart.

A checkpointer saves the workflow's state after every step, under a thread id.
Our learning workflow pauses while the learner answers a quiz (an "interrupt").
Because the paused state is saved in data/checkpoints.sqlite, you can close the
app, start it again tomorrow, and submit the same quiz.

Thread ids are stable and readable: "<learner_id>:<random part>". Each activity
(one lesson + quiz) gets its own thread, and the repository remembers which
thread belongs to which learner, so learners never share state.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The longest learning run is about 25 steps (see learning_workflow.py); this is the safety net.
LEARNING_RECURSION_LIMIT = 60


def create_checkpointer(path: Path) -> Any:
    """A SqliteSaver writing to `path`. Needs the langgraph-checkpoint-sqlite package."""
    from langgraph.checkpoint.sqlite import SqliteSaver  # imported here: only the app needs it

    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit may use the connection from different threads.
    # SqliteSaver guards the connection with its own lock.
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()  # creates its tables the first time
    logger.info("Checkpoints saved in %s", path.name)
    return saver


def make_thread_id(learner_id: str) -> str:
    """A new, unique thread id for one activity of one learner."""
    return f"{learner_id}:{uuid.uuid4().hex[:12]}"


def learner_of_thread(thread_id: str) -> str:
    """'moksh-3f9a1c2b:9e1d...' -> 'moksh-3f9a1c2b'."""
    return thread_id.rsplit(":", 1)[0]


def thread_config(thread_id: str) -> dict[str, Any]:
    """The config LangGraph needs to save and load one thread."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": LEARNING_RECURSION_LIMIT}


def delete_threads(checkpointer: Any, thread_ids: list[str]) -> int:
    """Remove saved checkpoints (used when a learner is reset or deleted). Returns how many."""
    deleted = 0
    for thread_id in thread_ids:
        try:
            checkpointer.delete_thread(thread_id)
            deleted += 1
        except Exception:  # noqa: BLE001 - an old thread may already be gone
            logger.warning("Could not delete one saved checkpoint thread")
    return deleted