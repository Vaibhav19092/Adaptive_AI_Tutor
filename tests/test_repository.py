"""Phase 7: the SQLite progress database, and keeping learners apart."""

from datetime import timedelta

import pytest

from adaptive_tutor.learner.mastery import ConceptProgress
from adaptive_tutor.memory.repository import LearnerRepository

QUIZ = {"quiz_id": "quiz1", "topic": "biology", "level": "beginner", "questions": []}
RESULT = {
    "quiz_id": "quiz1", "score": 0.5, "correct_count": 1, "question_count": 2,
    "results": [
        {"question_id": "q1", "concept": "cells", "kind": "multiple_choice", "score": 1.0, "correct": True, "graded": True},
        {"question_id": "q2", "concept": "cells", "kind": "short_answer", "score": 0.0, "correct": False, "graded": True},
    ],
}


def save_quiz(repository, learner_id, quiz=QUIZ, thread="t1"):
    repository.start_activity(learner_id, thread)
    repository.save_activity_material(learner_id, thread, quiz, {"title": "L"}, "lesson1", [{"citation": "bio.pdf"}], "documents")


def test_learners_are_created_listed_and_unique(repository):
    ada = repository.create_learner("  Ada   Lovelace ")
    assert ada.name == "Ada Lovelace" and ada.learner_id.startswith("ada-lovelace-")
    with pytest.raises(ValueError, match="already exists"):
        repository.create_learner("ada lovelace")
    with pytest.raises(ValueError):
        repository.create_learner("<script>")
    assert [p.name for p in repository.list_learners()] == ["Ada Lovelace"]


def test_progress_survives_reopening_the_database(tmp_path, today):
    path = tmp_path / "tutor.sqlite"
    first = LearnerRepository(path)
    learner = first.create_learner("Ada")
    save_quiz(first, learner.learner_id)
    progress = ConceptProgress("cells", "biology", 0.4, 1, 1, 1, today, today + timedelta(days=1))
    first.record_assessment(learner.learner_id, "quiz1", RESULT, [progress], [{"concept": "cells", "note": "mixes up X and Y"}])

    reopened = LearnerRepository(path)  # like restarting the app
    assert reopened.get_progress(learner.learner_id) == [progress]
    history = reopened.list_assessments(learner.learner_id)
    assert history[0]["status"] == "graded" and history[0]["score"] == 0.5
    assert reopened.get_misconceptions(learner.learner_id)[0]["note"] == "mixes up X and Y"
    assert len(reopened.question_results(learner.learner_id)) == 2
    assert reopened.list_lessons(learner.learner_id)[0]["sources"] == [{"citation": "bio.pdf"}]
    assert reopened.topic_levels(learner.learner_id) == {"biology": "beginner"}


def test_learners_never_see_each_others_records(repository, today):
    ada = repository.create_learner("Ada").learner_id
    bob = repository.create_learner("Bob").learner_id
    save_quiz(repository, ada, thread="ada-thread")
    repository.add_concepts(ada, "biology", ["cells"])
    repository.add_prerequisites(ada, ["cells"], ["atoms"])
    repository.record_event(ada, "learning_run", 10, {"score": 1})

    assert repository.get_progress(bob) == []
    assert repository.get_prerequisites(bob) == {}
    assert repository.list_assessments(bob) == []
    assert repository.list_lessons(bob) == []
    assert repository.open_activity(bob) is None
    assert repository.list_events(learner_id=bob) == []
    # Bob can't grade Ada's quiz.
    with pytest.raises(ValueError, match="doesn't belong"):
        repository.record_assessment(bob, "quiz1", RESULT, [], [])


def test_only_one_open_activity_per_learner(repository):
    ada = repository.create_learner("Ada").learner_id
    repository.start_activity(ada, "first")
    repository.start_activity(ada, "second")
    assert repository.open_activity(ada) == "second"
    repository.close_activity(ada, "second", "discarded")
    assert repository.open_activity(ada) is None
    assert set(repository.thread_ids(ada)) == {"first", "second"}


def test_discarding_marks_the_open_quiz(repository):
    ada = repository.create_learner("Ada").learner_id
    save_quiz(repository, ada)
    repository.close_activity(ada, "t1", "discarded")
    assert repository.list_assessments(ada)[0]["status"] == "discarded"


def test_reset_keeps_the_profile_and_delete_removes_everything(repository):
    ada = repository.create_learner("Ada").learner_id
    save_quiz(repository, ada)
    repository.add_concepts(ada, "biology", ["cells"])

    assert repository.reset_progress(ada) == ["t1"]
    assert repository.get_learner(ada) is not None
    assert repository.get_progress(ada) == [] and repository.list_assessments(ada) == []

    save_quiz(repository, ada, thread="t2")
    assert repository.delete_learner(ada) == ["t2"]
    assert repository.get_learner(ada) is None
    assert repository.list_lessons(ada) == []


def test_events_store_numbers_only(repository):
    repository.record_event(None, "question", 1234, {"grounded": True})
    events = repository.list_events(kind="question")
    assert events[0]["duration_ms"] == 1234 and events[0]["data"] == {"grounded": True}