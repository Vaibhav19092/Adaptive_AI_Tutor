"""Corrective RAG: the plan for each search attempt. Pure Python."""

from dataclasses import replace

from adaptive_tutor.retrieval.corrective_rag import plan_retrieval


def test_first_attempt_is_the_normal_search(settings):
    plan = plan_retrieval(0, "What do mitochondria do?", None, settings)
    assert plan.attempt == 1
    assert plan.query == "What do mitochondria do?"
    assert plan.top_k == settings.retrieval_top_k == 4
    assert plan.min_score == settings.min_relevance_score == 0.5
    assert plan.strategy == "standard search"


def test_retry_uses_the_rewritten_query_and_searches_wider(settings):
    plan = plan_retrieval(1, "What do mitochondria do?", "mitochondria ATP energy", settings)
    assert plan.attempt == 2
    assert plan.query == "mitochondria ATP energy"
    assert plan.top_k == 8  # twice as many passages
    assert plan.min_score == 0.4  # 0.1 lower threshold


def test_retry_without_a_rewrite_falls_back_to_the_question(settings):
    assert plan_retrieval(1, "What is RAG?", None, settings).query == "What is RAG?"


def test_wider_search_has_limits(settings):
    plan = plan_retrieval(1, "q", "q2", replace(settings, retrieval_top_k=15, min_relevance_score=0.05))
    assert plan.top_k == 20  # never more than 20
    assert plan.min_score == 0.0  # never below 0


def test_plan_description_for_the_learner(settings):
    text = plan_retrieval(0, "q", None, settings).describe(found=4, kept=2)
    assert text == "Search attempt 1 (standard search): looked at the 4 closest passages, 2 of 4 scored at least 0.50"