"""Phase 7: thread ids, and a REAL restart test with the SQLite checkpointer."""

import importlib.util
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from adaptive_tutor.memory.checkpointing import (
    LEARNING_RECURSION_LIMIT,
    create_checkpointer,
    delete_threads,
    learner_of_thread,
    make_thread_id,
    thread_config,
)

HAS_SQLITE_SAVER = importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None


def test_thread_ids_are_unique_and_name_their_learner():
    first, second = make_thread_id("ada-1234abcd"), make_thread_id("ada-1234abcd")
    assert first != second
    assert learner_of_thread(first) == "ada-1234abcd"
    config = thread_config(first)
    assert config["configurable"]["thread_id"] == first
    assert config["recursion_limit"] == LEARNING_RECURSION_LIMIT


def test_delete_threads_counts_what_was_removed():
    class Saver:
        def __init__(self):
            self.deleted = []

        def delete_thread(self, thread_id):
            if thread_id == "broken":
                raise RuntimeError("gone")
            self.deleted.append(thread_id)

    saver = Saver()
    assert delete_threads(saver, ["a", "broken", "b"]) == 2
    assert saver.deleted == ["a", "b"]
    assert delete_threads(InMemorySaver(), ["never-existed"]) == 1


@pytest.mark.skipif(not HAS_SQLITE_SAVER, reason="needs langgraph-checkpoint-sqlite")
def test_paused_quiz_survives_closing_and_reopening_the_checkpoint_file(tmp_path, settings, repository):
    from test_learning_workflow import FakeRetriever, build, chunk, correct_answers, happy_model

    from adaptive_tutor.graph.learning_workflow import pending_activity, start_activity, submit_answers

    quiz_settings = replace(settings, quiz_multiple_choice=1, quiz_short_answer=1)
    learner = repository.create_learner("Ada").learner_id
    path = tmp_path / "checkpoints.sqlite"

    saver = create_checkpointer(path)
    graph = build(happy_model(), quiz_settings, repository, FakeRetriever([chunk(1)]), saver=saver)
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    assert outcome.waiting
    saver.conn.close()  # the app stops

    reopened = create_checkpointer(path)  # the app starts again
    graph = build(happy_model(), quiz_settings, repository, FakeRetriever([chunk(1)]), saver=reopened)
    resumed = pending_activity(graph, repository, learner)
    assert resumed is not None and resumed.thread_id == outcome.thread_id

    done = submit_answers(graph, repository, learner, resumed.thread_id, correct_answers(resumed))
    assert done.state["saved"] is True
    assert delete_threads(reopened, [outcome.thread_id]) == 1
    reopened.conn.close()