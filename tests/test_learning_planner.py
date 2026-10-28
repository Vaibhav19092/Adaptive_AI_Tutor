"""Phase 6: recommendations for remediation, revision, practice, and advancement."""

from datetime import timedelta

from adaptive_tutor.learner.learning_planner import focus_concepts, next_level, recommend
from adaptive_tutor.learner.mastery import ConceptProgress


def concept(name, mastery, attempts=2, topic="biology", **extra):
    return ConceptProgress(name, topic, mastery=mastery, attempts=attempts, **extra)


def test_struggling_concept_gets_remediation(today):
    top = recommend([concept("photosynthesis", 0.2)], {}, today)[0]
    assert (top.action, top.concept, top.activity) == ("remediate", "photosynthesis", "lesson")


def test_weak_prerequisite_comes_first(today):
    progress = [concept("respiration", 0.1), concept("cells", 0.0, attempts=0)]
    suggestions = recommend(progress, {"respiration": ["cells"]}, today)
    assert [(s.action, s.concept) for s in suggestions[:2]] == [("prerequisite", "cells"), ("remediate", "respiration")]


def test_unseen_prerequisite_is_also_suggested(today):
    suggestions = recommend([concept("respiration", 0.1)], {"respiration": ["mitochondria"]}, today)
    assert suggestions[0].action == "prerequisite" and suggestions[0].concept == "mitochondria"


def test_order_is_remediate_then_revise_then_practice_then_start(today):
    progress = [
        concept("new one", 0.0, attempts=0),
        concept("halfway", 0.6),
        concept("due", 0.9, last_reviewed=today - timedelta(days=7), next_review=today),
        concept("weak", 0.1),
    ]
    assert [s.action for s in recommend(progress, {}, today)] == ["remediate", "revise", "practice", "start"]


def test_mastered_topic_suggests_the_next_level(today):
    progress = [concept("cells", 0.9, next_review=today + timedelta(days=5))]
    top = recommend(progress, {}, today, topic_levels={"biology": "beginner"})[0]
    assert (top.action, top.concept, top.level) == ("advance", "biology", "intermediate")
    assert next_level("advanced") is None


def test_one_suggestion_per_concept_and_a_limit(today):
    progress = [concept(f"c{i}", 0.1) for i in range(8)]
    suggestions = recommend(progress, {}, today, limit=5)
    assert len(suggestions) == 5
    assert len({s.concept for s in suggestions}) == 5


def test_focus_concepts_are_the_weakest_in_the_topic(today):
    progress = [concept("a", 0.7), concept("b", 0.2), concept("c", 0.95, next_review=today + timedelta(days=9)),
                concept("x", 0.1, topic="chemistry")]
    assert focus_concepts(progress, "Biology", today) == ["b", "a"]