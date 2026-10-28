"""Phase 6: the mastery rule and spaced review dates."""

from datetime import timedelta

import pytest

from adaptive_tutor.learner.mastery import ConceptProgress, concept_status, describe_change, review_interval_days, update_progress


def test_mastery_moves_40_percent_toward_the_latest_score(today):
    progress = ConceptProgress("photosynthesis", "biology")
    after_one = update_progress(progress, 1.0, today)
    assert after_one.mastery == 0.4
    after_two = update_progress(after_one, 1.0, today)
    assert after_two.mastery == 0.64
    after_wrong = update_progress(after_two, 0.0, today)
    assert after_wrong.mastery == pytest.approx(0.384)
    assert (after_wrong.attempts, after_wrong.correct, after_wrong.streak) == (3, 2, 0)


@pytest.mark.parametrize("streak, days", [(0, 1), (1, 1), (2, 3), (3, 7), (4, 14), (5, 30), (9, 30)])
def test_review_gaps_grow_with_a_streak(streak, days):
    assert review_interval_days(streak) == days


def test_review_date_resets_after_a_poor_answer(today):
    progress = ConceptProgress("c", "t", mastery=0.9, attempts=4, correct=4, streak=3)
    good = update_progress(progress, 1.0, today)
    assert good.next_review == today + timedelta(days=14)
    poor = update_progress(good, 0.2, today)
    assert poor.next_review == today + timedelta(days=1) and poor.streak == 0


def test_statuses(today):
    assert concept_status(ConceptProgress("c", "t"), today) == "new"
    assert concept_status(ConceptProgress("c", "t", mastery=0.3, attempts=1), today) == "remediate"
    assert concept_status(ConceptProgress("c", "t", mastery=0.6, attempts=2), today) == "practice"
    later = ConceptProgress("c", "t", mastery=0.9, attempts=4, next_review=today + timedelta(days=3))
    assert concept_status(later, today) == "mastered"
    assert concept_status(later, today + timedelta(days=3)) == "revise"


def test_scores_are_clamped_and_changes_are_described(today):
    before = ConceptProgress("c", "t")
    after = update_progress(before, 5.0, today)
    assert after.mastery == 0.4
    assert describe_change(before, after) == "c: mastery 0.00 → 0.40 (next review in 1 day)"
    assert ConceptProgress.from_dict(after.to_dict()) == after