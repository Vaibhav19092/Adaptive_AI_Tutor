"""Phases 5-8: the learning workflow, run through the real LangGraph graph.

A FakeTaskModel recognises each job (lesson, quiz, grading, ...) from the first line
of its system prompt and replies from a script. An in-memory checkpointer lets the
graph pause for quiz answers exactly as it does in the app. No Ollama needed.
"""

import json
import re
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from adaptive_tutor.assessment.answer_evaluator import RUBRIC_SYSTEM_PROMPT
from adaptive_tutor.assessment.quiz_generator import KEY_CHECK_SYSTEM_PROMPT, QUIZ_SYSTEM_PROMPT
from adaptive_tutor.graph.checking_nodes import (
    EVIDENCE_SYSTEM_PROMPT,
    GRADE_SYSTEM_PROMPT,
    GROUNDING_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)
from adaptive_tutor.graph.learning_nodes import (
    INTENT_SYSTEM_PROMPT,
    NO_LEARNER_MESSAGE,
    NO_MATERIAL_MESSAGE,
    NO_RECOMMENDATIONS_MESSAGE,
    UNRELIABLE_LESSON_MESSAGE,
)
from adaptive_tutor.graph.learning_workflow import (
    build_learning_graph,
    discard_activity,
    pending_activity,
    start_activity,
    submit_answers,
)
from adaptive_tutor.graph.nodes import NO_DOCUMENTS_MESSAGE, SEARCH_QUERY_SYSTEM_PROMPT
from adaptive_tutor.learner.mastery import ConceptProgress
from adaptive_tutor.memory.checkpointing import LEARNING_RECURSION_LIMIT
from adaptive_tutor.research.search_provider import WebResult
from adaptive_tutor.retrieval.retriever import RetrievedChunk
from conftest import TODAY

LESSON_REPLY = json.dumps({
    "title": "Photosynthesis",
    "summary": "How plants make their food.",
    "explanation": "Plants use sunlight to turn water and carbon dioxide into sugar [1].",
    "analogy": "A leaf is like a tiny solar-powered kitchen.",
    "example": "A sunflower turning toward the sun.",
    "key_points": ["Light energy is needed [1]", "Sugar is produced [1]"],
    "concepts": ["photosynthesis", "chlorophyll"],
    "prerequisites": ["cells"],
})
WEB_LESSON_REPLY = LESSON_REPLY.replace("[1]", "[W1]")
GENERAL_LESSON_REPLY = LESSON_REPLY.replace(" [1]", "")

QUIZ_REPLY = json.dumps({
    "multiple_choice": [{
        "concept": "chlorophyll", "question": "Which pigment absorbs light?",
        "options": ["Chlorophyll", "Haemoglobin", "Keratin", "Melanin"], "answer": "A",
        "explanation": "Chlorophyll is the green pigment in leaves.",
    }],
    "short_answer": [{
        "concept": "photosynthesis", "question": "What does photosynthesis produce?",
        "key_points": ["sugar", "oxygen"], "model_answer": "Sugar and oxygen.",
    }],
})

RUBRIC_ALL = json.dumps({"points": [{"point": 1, "covered": True}, {"point": 2, "covered": True}],
                         "misconception": "", "feedback": "Well done."})
RUBRIC_HALF = json.dumps({"points": [{"point": 1, "covered": True}, {"point": 2, "covered": False}],
                          "misconception": "Thinks oxygen is absorbed", "feedback": "Nearly."})


def answer_key_letter(prompt: str) -> str:
    """Plays a model that answers the key check correctly: the letter next to 'Chlorophyll'."""
    match = re.search(r"([A-E])\) Chlorophyll", prompt)
    return match.group(1) if match else "?"


class FakeTaskModel:
    """Replies by job. A reply can be text, an exception to raise, or a function of the prompt."""

    TASKS = {
        "intent": INTENT_SYSTEM_PROMPT,
        "grade": GRADE_SYSTEM_PROMPT,
        "evidence": EVIDENCE_SYSTEM_PROMPT,
        "grounding": GROUNDING_SYSTEM_PROMPT,
        "rewrite": REWRITE_SYSTEM_PROMPT,
        "search_query": SEARCH_QUERY_SYSTEM_PROMPT,
        "lesson": "You are a patient, friendly tutor who writes short structured lessons.",
        "quiz": QUIZ_SYSTEM_PROMPT,
        "key_check": KEY_CHECK_SYSTEM_PROMPT,
        "rubric": RUBRIC_SYSTEM_PROMPT,
    }

    def __init__(self, **replies) -> None:
        self.replies = {task: list(values) for task, values in replies.items()}
        self.prompts: dict[str, list[str]] = defaultdict(list)

    def respond(self, prompt_value) -> str:
        text = prompt_value.to_string()
        # The first line of each system prompt identifies the job (later lines contain JSON braces).
        task = next((name for name, marker in self.TASKS.items() if marker.split("\n")[0] in text), None)
        assert task is not None, f"Unknown prompt: {text[:80]}"
        self.prompts[task].append(text)
        queue = self.replies.get(task)
        assert queue, f"The test gave no reply for the '{task}' job"
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply(text) if callable(reply) else reply

    def runnable(self) -> RunnableLambda:
        return RunnableLambda(self.respond)

    def calls(self, task: str) -> int:
        return len(self.prompts[task])


def happy_model(**overrides) -> FakeTaskModel:
    replies = dict(
        grade=["yes"], evidence=["SUFFICIENT"], lesson=[LESSON_REPLY], grounding=["yes"],
        quiz=[QUIZ_REPLY], key_check=[answer_key_letter], rubric=[RUBRIC_ALL], intent=["LESSON"],
        rewrite=["photosynthesis light energy"], search_query=["photosynthesis plants"],
    )
    replies.update(overrides)
    return FakeTaskModel(**replies)


class FakeRetriever:
    def __init__(self, *results_per_search):
        self._results = list(results_per_search) or [[]]
        self.searches = []

    def document_count(self) -> int:
        return 1

    def search(self, question, top_k=None):
        self.searches.append((question, top_k))
        return list(self._results[min(len(self.searches) - 1, len(self._results) - 1)])


class EmptyRetriever(FakeRetriever):
    def document_count(self) -> int:
        return 0


class FakeSearchProvider:
    name = "FakeSearch"
    destination = "search.example"

    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, query, max_results):
        self.queries.append(query)
        return list(self._results)


def chunk(index, text="Plants use sunlight to make sugar in their leaves.", score=0.8):
    return RetrievedChunk(text, "biology.pdf", index, index, "doc1", score)


def web_result():
    text = "Photosynthesis is the process plants use to make sugar from sunlight, water and carbon dioxide. " * 4
    return WebResult("Photosynthesis", "https://en.wikipedia.org/wiki/Photosynthesis", text, "FakeSearch", 1,
                     content_source="article intro")


@pytest.fixture
def quiz_settings(settings):
    return replace(settings, quiz_multiple_choice=1, quiz_short_answer=1)


@pytest.fixture
def learner(repository):
    return repository.create_learner("Ada").learner_id


def build(model, settings, repository, retriever=None, search_provider=None, saver=None):
    llm = model.runnable()
    return build_learning_graph(llm, llm, settings, repository, retriever, search_provider,
                                saver or InMemorySaver(), clock=lambda: TODAY)


def correct_answers(outcome):
    """The answers a learner who knows the material would give, using only the PUBLIC view."""
    answers = {}
    for view in outcome.questions:
        answers[view.question_id] = view.options.index("Chlorophyll") if view.options else "Sugar and oxygen are made."
    return answers


# ---------- The full cycle ----------

def test_lesson_quiz_pause_then_grade_update_plan_and_save(quiz_settings, repository, learner):
    model = happy_model(grade=["yes", "no"])
    retriever = FakeRetriever([chunk(1), chunk(2, text="Unrelated text about rocks.")])
    graph = build(model, quiz_settings, repository, retriever)

    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")

    assert outcome.waiting
    assert outcome.state["steps"] == [
        "load_profile", "understand_request", "retrieve_material", "check_material",
        "generate_lesson", "check_lesson", "create_quiz",
    ]
    assert outcome.state["lesson"]["title"] == "Photosynthesis"
    assert [item["citation"] for item in outcome.state["evidence"]] == ["biology.pdf, page 1"]
    assert len(outcome.state["rejected_evidence"]) == 1
    # Before submission only the public view is shown: no answers anywhere in it.
    shown = repr(outcome.questions)
    assert "green pigment" not in shown and "Sugar and oxygen" not in shown
    assert repository.list_assessments(learner)[0]["status"] == "open"
    assert repository.open_activity(learner) == outcome.thread_id

    done = submit_answers(graph, repository, learner, outcome.thread_id, correct_answers(outcome))

    assert not done.waiting
    assert done.state["steps"][-5:] == ["wait_for_answers", "grade_answers", "update_profile", "recommend_next", "save_progress"]
    assert done.state["assessment"]["score"] == 1.0
    progress = {p.concept: p for p in repository.get_progress(learner)}
    assert progress["chlorophyll"].mastery == 0.4 and progress["photosynthesis"].mastery == 0.4
    assert repository.get_prerequisites(learner) == {"chlorophyll": ["cells"], "photosynthesis": ["cells"]}
    assert repository.list_assessments(learner)[0]["status"] == "graded"
    assert repository.open_activity(learner) is None
    assert done.state["recommendations"][0]["action"] == "practice"
    assert any("mastery 0.00 → 0.40" in note for note in done.state["mastery_notes"])
    kinds = {event["kind"] for event in repository.list_events(learner_id=learner)}
    assert kinds == {"learning_run", "grading"}


def test_wrong_answers_are_explained_and_lower_mastery(quiz_settings, repository, learner):
    model = happy_model(rubric=[RUBRIC_HALF])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    mc = next(view for view in outcome.questions if view.options)
    wrong = mc.options.index("Keratin")

    done = submit_answers(graph, repository, learner, outcome.thread_id, {mc.question_id: wrong, "q2": "Sugar"})

    results = {item["concept"]: item for item in done.state["assessment"]["results"]}
    assert "You chose “Keratin”" in results["chlorophyll"]["feedback"]
    assert results["photosynthesis"]["score"] == 0.5
    assert repository.get_progress(learner)[0].mastery == 0.0  # chlorophyll stays at 0.0
    notes = {m["note"] for m in repository.get_misconceptions(learner)}
    assert "Thinks oxygen is absorbed" in notes


def test_paused_quiz_survives_a_restart(quiz_settings, repository, learner):
    saver = InMemorySaver()  # in the app this is the SQLite checkpointer
    first_app = build(happy_model(), quiz_settings, repository, FakeRetriever([chunk(1)]), saver=saver)
    outcome = start_activity(first_app, repository, learner, "Teach me photosynthesis")

    restarted_app = build(happy_model(), quiz_settings, repository, FakeRetriever([chunk(1)]), saver=saver)
    resumed = pending_activity(restarted_app, repository, learner)

    assert resumed is not None and resumed.waiting
    assert resumed.thread_id == outcome.thread_id
    assert [q.question_id for q in resumed.questions] == [q.question_id for q in outcome.questions]
    done = submit_answers(restarted_app, repository, learner, resumed.thread_id, correct_answers(resumed))
    assert done.state["saved"] is True


def test_two_learners_have_separate_activities(quiz_settings, repository, learner):
    bob = repository.create_learner("Bob").learner_id
    graph = build(happy_model(), quiz_settings, repository, FakeRetriever([chunk(1)]))
    ada_outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")

    assert pending_activity(graph, repository, bob) is None
    with pytest.raises(ValueError, match="different learner"):
        submit_answers(graph, repository, bob, ada_outcome.thread_id, {})
    assert repository.get_progress(bob) == []


def test_discarded_quiz_changes_nothing(quiz_settings, repository, learner):
    graph = build(happy_model(), quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    discard_activity(repository, learner, outcome.thread_id)
    assert pending_activity(graph, repository, learner) is None
    assert repository.list_assessments(learner)[0]["status"] == "discarded"
    assert all(p.attempts == 0 for p in repository.get_progress(learner))


# ---------- Corrective RAG and honest refusals ----------

def test_weak_documents_are_retried_then_refused(quiz_settings, repository, learner):
    model = happy_model(grade=["no"])
    retriever = FakeRetriever([chunk(1)], [chunk(2)])
    graph = build(model, quiz_settings, repository, retriever)

    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")

    assert not outcome.waiting
    assert outcome.state["steps"] == [
        "load_profile", "understand_request", "retrieve_material", "check_material",
        "rewrite_material_query", "retrieve_material", "check_material", "report_no_material",
    ]
    assert outcome.state["message"] == NO_MATERIAL_MESSAGE and outcome.state["refused"]
    assert retriever.searches[1] == ("photosynthesis light energy", 8)  # rewritten query, wider search
    assert model.calls("lesson") == 0
    assert repository.open_activity(learner) is None


def test_web_research_when_documents_fall_short_and_allowed(quiz_settings, repository, learner):
    model = happy_model(grade=["no", "yes"], lesson=[WEB_LESSON_REPLY])
    provider = FakeSearchProvider([web_result()])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]), provider)

    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis", allow_web=True)

    assert outcome.waiting
    assert "research_web" in outcome.state["steps"]
    assert outcome.state["evidence_kind"] == "web"
    assert outcome.state["search_query"] == "photosynthesis plants"
    assert all(item["kind"] == "web" for item in outcome.state["evidence"])  # never mixed with documents
    assert "[W1]" in model.prompts["lesson"][0]


def test_web_is_not_used_without_permission(quiz_settings, repository, learner):
    provider = FakeSearchProvider([web_result()])
    graph = build(happy_model(grade=["no"]), quiz_settings, repository, FakeRetriever([chunk(1)]), provider)
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis", allow_web=False)
    assert provider.queries == []
    assert outcome.state["refused"]


def test_partly_covered_topic_is_taught_from_what_the_documents_say(quiz_settings, repository, learner):
    model = happy_model(evidence=["INSUFFICIENT"])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    assert "use_partial_material" in outcome.state["steps"]
    assert outcome.state["partial_evidence"] is True
    assert model.calls("grade") == 1  # the same passage is never graded twice
    assert outcome.waiting


def test_unsupported_lesson_is_rewritten_then_hidden(quiz_settings, repository, learner):
    model = happy_model(grounding=["no"])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    assert outcome.state["steps"][-5:] == [
        "generate_lesson", "check_lesson", "generate_lesson", "check_lesson", "report_unreliable_lesson",
    ]
    assert outcome.state["message"] == UNRELIABLE_LESSON_MESSAGE
    assert outcome.state["lesson"] == {}  # the unsupported lesson can't be shown
    assert "previous lesson was rejected" in model.prompts["lesson"][1]
    assert model.calls("quiz") == 0


def test_lesson_without_citations_fails_the_rule_check_without_a_model_call(quiz_settings, repository, learner):
    model = happy_model(lesson=[GENERAL_LESSON_REPLY])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    assert model.calls("grounding") == 0
    assert outcome.state["refused"]


# ---------- Other paths ----------

def test_general_knowledge_lesson_in_fast_mode(quiz_settings, repository, learner):
    model = happy_model(lesson=[GENERAL_LESSON_REPLY])
    retriever = FakeRetriever([chunk(1)])
    graph = build(model, quiz_settings, repository, retriever)
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis",
                             source_mode="general", check_evidence=False)
    assert outcome.state["steps"] == ["load_profile", "understand_request", "generate_lesson", "create_quiz"]
    assert outcome.state["evidence_kind"] == "general"
    assert retriever.searches == [] and model.calls("key_check") == 0
    assert "No sources are available" in model.prompts["lesson"][0]


def test_quiz_only_targets_weak_concepts(quiz_settings, repository, learner):
    weak = ConceptProgress("chlorophyll", "photosynthesis", mastery=0.2, attempts=2, last_reviewed=TODAY - timedelta(days=1),
                           next_review=TODAY)
    repository.start_activity(learner, "seed")
    repository.save_activity_material(learner, "seed", {"quiz_id": "old", "topic": "photosynthesis", "level": "beginner",
                                                        "questions": []}, None, None, [], "general")
    repository.record_assessment(learner, "old", {"score": 0.0, "results": []}, [weak], [])
    model = happy_model()
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))

    outcome = start_activity(graph, repository, learner, "Quiz me on photosynthesis")

    assert "generate_lesson" not in outcome.state["steps"]
    assert outcome.state["focus_concepts"] == ["chlorophyll"]
    assert "finds hard: chlorophyll" in model.prompts["quiz"][0]
    assert any("extra focus on chlorophyll" in note for note in outcome.state["decisions"])


def test_model_decides_when_the_wording_is_unclear(quiz_settings, repository, learner):
    model = happy_model(intent=["QUIZ"])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "photosynthesis")
    assert model.calls("intent") == 1
    assert outcome.state["chosen_action"] == "quiz"


def test_what_next_uses_the_planner_without_model_calls(quiz_settings, repository, learner):
    model = happy_model()
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    empty = start_activity(graph, repository, learner, "What should I study next?")
    assert empty.state["message"] == NO_RECOMMENDATIONS_MESSAGE

    repository.add_concepts(learner, "photosynthesis", ["chlorophyll"])
    outcome = start_activity(graph, repository, learner, "What should I study next?")
    assert outcome.state["steps"] == ["load_profile", "understand_request", "recommend_next"]
    assert outcome.state["recommendations"][0]["action"] == "start"
    assert sum(len(prompts) for prompts in model.prompts.values()) == 0
    assert not outcome.waiting


def test_grading_failure_keeps_the_quiz_open(quiz_settings, repository, learner):
    model = happy_model(rubric=[ConnectionError("refused"), RUBRIC_ALL])
    graph = build(model, quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")

    failed = submit_answers(graph, repository, learner, outcome.thread_id, correct_answers(outcome))
    assert failed.waiting and "still open" in failed.state["error"]
    assert repository.open_activity(learner) == outcome.thread_id

    done = submit_answers(graph, repository, learner, outcome.thread_id, correct_answers(outcome))
    assert not done.waiting and done.state["saved"]


def test_ungraded_short_answer_does_not_change_mastery(quiz_settings, repository, learner):
    graph = build(happy_model(rubric=["garbage"]), quiz_settings, repository, FakeRetriever([chunk(1)]))
    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis")
    done = submit_answers(graph, repository, learner, outcome.thread_id, correct_answers(outcome))
    progress = {p.concept: p for p in repository.get_progress(learner)}
    assert progress["chlorophyll"].attempts == 1
    assert progress["photosynthesis"].attempts == 0  # not graded -> not counted
    assert done.state["assessment"]["score"] == 1.0


def test_friendly_errors(quiz_settings, repository, learner):
    graph = build(happy_model(), quiz_settings, repository, EmptyRetriever())
    assert start_activity(graph, repository, "nobody", "Teach me x").state["error"] == NO_LEARNER_MESSAGE
    assert start_activity(graph, repository, learner, "Teach me x").state["error"] == NO_DOCUMENTS_MESSAGE
    assert "type what you want" in start_activity(graph, repository, learner, "  ").state["error"]
    assert "topic" in start_activity(graph, repository, learner, "Quiz me").state["error"]
    with pytest.raises(ValueError):
        start_activity(graph, repository, learner, "x", action="dance")


def test_worst_case_run_stays_bounded(quiz_settings, repository, learner):
    settings = replace(quiz_settings, max_retrieval_retries=2, max_answer_retries=2)
    model = happy_model(grade=["no", "no", "no", "yes"], lesson=[WEB_LESSON_REPLY], grounding=["no"])
    retriever = FakeRetriever([chunk(1)], [chunk(2)], [chunk(3)])
    graph = build(model, settings, repository, retriever, FakeSearchProvider([web_result()]))

    outcome = start_activity(graph, repository, learner, "Teach me photosynthesis", allow_web=True)

    steps = outcome.state["steps"]
    assert steps.count("retrieve_material") == 3 and steps.count("generate_lesson") == 3
    assert steps[-1] == "report_unreliable_lesson"
    assert len(steps) < LEARNING_RECURSION_LIMIT