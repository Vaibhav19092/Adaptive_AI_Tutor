"""Phase 5: the shapes of lessons, quizzes, and results, and how to check them.

The model writes lessons and quizzes as JSON. Before the app uses that JSON,
these functions check it field by field, like a teacher checking a form:
is every box filled in, is the answer letter one of the options, ...?

Design choices:
- Plain frozen dataclasses with explicit checks (no Pydantic), so every error
  message is written in plain language and every rule is easy to read and test.
- Lenient where it's safe (extra items are trimmed, "B)" is read as "B"),
  strict where it matters (a question without a valid answer is dropped).
- Correct answers live only in the full Quiz. The UI shows QuestionView objects,
  which have no answer fields at all, so answers can't leak before submission.

No LangChain imports: this module can be tested anywhere.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LEVELS = ("beginner", "intermediate", "advanced")
OPTION_LETTERS = "ABCDE"
MIN_OPTIONS, MAX_OPTIONS = 3, 5
MAX_CONCEPT_CHARS = 60
MAX_KEY_POINTS = 6
MAX_LESSON_CONCEPTS = 5
MAX_PREREQUISITES = 3
PASS_SCORE = 0.5  # a question scoring at least this counts as "correct"

QuestionKind = Literal["multiple_choice", "short_answer"]

_OPTION_PREFIX = re.compile(r"^\(?[A-Ea-e][).:]\s+")
_ANSWER_LABEL = re.compile(r"^(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is)?\s*[:\-]?\s*", re.IGNORECASE)
_LETTER_AT_START = re.compile(r"^\W*\(?([A-Ea-e])\)?(?:[.):,\s]|$)")
_LETTER_AFTER_LABEL = re.compile(r"\b(?:option|answer)\W*(?:is\W*)?\(?([A-E])\)?\b", re.IGNORECASE)


class SchemaError(ValueError):
    """The model's JSON is missing something or has the wrong shape."""


# ---------- Small field readers ----------

def _text(data: dict[str, Any], key: str, what: str, required: bool = True) -> str:
    value = data.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise SchemaError(f"the {what} has no '{key}'")
        return ""
    if isinstance(value, (list, tuple)):  # e.g. an explanation written as a list of sentences
        value = " ".join(str(item) for item in value)
    return re.sub(r"[ \t]+", " ", str(value)).strip()


def _text_list(data: dict[str, Any], key: str, what: str, minimum: int, maximum: int) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        value = []
    if isinstance(value, str):  # a single string with one item per line
        value = [line for line in value.splitlines()]
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"'{key}' in the {what} must be a list")
    items = []
    for item in value:
        text = str(item).strip().lstrip("-*• ").strip() if item is not None else ""
        if text and text.lower() not in (existing.lower() for existing in items):
            items.append(text)
    if len(items) < minimum:
        raise SchemaError(f"the {what} needs at least {minimum} item(s) in '{key}'")
    return tuple(items[:maximum])


def _require_dict(data: Any, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaError(f"the {what} must be a JSON object")
    return data


# ---------- Concepts ----------

def normalize_concept(raw: Any) -> str:
    """'  Neural Networks. ' -> 'neural networks'. Concept labels are compared in this form."""
    text = re.sub(r"\s+", " ", str(raw or "")).strip().strip(".,;:!?\"'`").strip().lower()
    if not text:
        raise SchemaError("a concept label is empty")
    if len(text) > MAX_CONCEPT_CHARS:
        text = text[:MAX_CONCEPT_CHARS].rsplit(" ", 1)[0]
    return text


def normalize_topic(raw: Any) -> str:
    """'  Photosynthesis  basics ' -> 'photosynthesis basics'. Topics are stored in this form."""
    text = re.sub(r"\s+", " ", str(raw or "")).strip().strip(".,;:!?\"'`").strip().lower()
    return text[:80]


def _concept_words(concept: str) -> set[str]:
    """Words of a concept, with a simple plural 's' removed: 'neural networks' -> {'neural', 'network'}."""
    words = set()
    for word in re.findall(r"[a-z0-9]+", concept.lower()):
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(word)
    return words


def match_concept(label: str, allowed: tuple[str, ...] | list[str]) -> str:
    """Map a question's concept label onto one of the lesson's concepts, if they overlap.

    'Neural Network' with allowed ('neural networks', 'backpropagation') -> 'neural networks'.
    With no overlap at all the question keeps its own (normalized) label.
    """
    concept = normalize_concept(label)
    if not allowed or concept in allowed:
        return concept
    words = _concept_words(concept)
    best, best_overlap = concept, 0
    for candidate in allowed:
        overlap = len(words & _concept_words(candidate))
        if overlap > best_overlap:
            best, best_overlap = candidate, overlap
    return best


# ---------- Lessons ----------

@dataclass(frozen=True)
class Lesson:
    """A short structured lesson, written for one level."""

    topic: str
    level: str
    title: str
    summary: str
    explanation: str
    analogy: str
    example: str
    key_points: tuple[str, ...]
    concepts: tuple[str, ...]  # normalized labels, used for quiz questions and mastery
    prerequisites: tuple[str, ...] = ()  # concepts a learner should know first

    def fact_text(self) -> str:
        """The factual parts that must be supported by the sources.

        The analogy and example are the tutor's own teaching aids, so they are not
        checked against the sources (they are labelled as such in the app).
        """
        points = "\n".join(f"- {point}" for point in self.key_points)
        return f"{self.explanation}\n\nKey points:\n{points}"

    def as_material(self) -> str:
        """The whole lesson as text, used as the material a quiz is written from."""
        return f"{self.title}\n\n{self.summary}\n\n{self.fact_text()}\n\nExample: {self.example}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("key_points", "concepts", "prerequisites"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        values = dict(data)
        for key in ("key_points", "concepts", "prerequisites"):
            values[key] = tuple(values.get(key, ()))
        return cls(**values)


def parse_lesson(data: Any, topic: str, level: str) -> Lesson:
    """Check the model's lesson JSON and build a Lesson. Raises SchemaError."""
    data = _require_dict(data, "lesson")
    concepts_raw = _text_list(data, "concepts", "lesson", 1, MAX_LESSON_CONCEPTS)
    concepts = tuple(dict.fromkeys(normalize_concept(item) for item in concepts_raw))
    prerequisites = tuple(
        item
        for item in dict.fromkeys(normalize_concept(p) for p in _text_list(data, "prerequisites", "lesson", 0, MAX_PREREQUISITES))
        if item not in concepts
    )
    return Lesson(
        topic=topic,
        level=level if level in LEVELS else "beginner",
        title=_text(data, "title", "lesson"),
        summary=_text(data, "summary", "lesson"),
        explanation=_text(data, "explanation", "lesson"),
        analogy=_text(data, "analogy", "lesson"),
        example=_text(data, "example", "lesson"),
        key_points=_text_list(data, "key_points", "lesson", 2, MAX_KEY_POINTS),
        concepts=concepts,
        prerequisites=prerequisites,
    )


# ---------- Quiz questions ----------

@dataclass(frozen=True)
class MultipleChoiceQuestion:
    question_id: str
    concept: str
    prompt: str
    options: tuple[str, ...]
    correct_index: int  # SECRET until the learner submits
    explanation: str  # SECRET until the learner submits
    kind: QuestionKind = field(default="multiple_choice", init=False)

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]


@dataclass(frozen=True)
class ShortAnswerQuestion:
    question_id: str
    concept: str
    prompt: str
    key_points: tuple[str, ...]  # SECRET: the rubric used for grading
    model_answer: str  # SECRET until the learner submits
    kind: QuestionKind = field(default="short_answer", init=False)


Question = MultipleChoiceQuestion | ShortAnswerQuestion


@dataclass(frozen=True)
class QuestionView:
    """What the learner may see BEFORE submitting: no answers, no rubric, no explanation."""

    question_id: str
    kind: QuestionKind
    concept: str
    prompt: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quiz:
    quiz_id: str
    topic: str
    level: str
    questions: tuple[Question, ...]

    def concepts(self) -> list[str]:
        return list(dict.fromkeys(question.concept for question in self.questions))

    def to_dict(self) -> dict[str, Any]:
        questions = []
        for question in self.questions:
            data = asdict(question)
            data["kind"] = question.kind
            for key in ("options", "key_points"):
                if key in data:
                    data[key] = list(data[key])
            questions.append(data)
        return {"quiz_id": self.quiz_id, "topic": self.topic, "level": self.level, "questions": questions}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Quiz:
        questions: list[Question] = []
        for item in data["questions"]:
            values = {key: value for key, value in item.items() if key != "kind"}
            if item["kind"] == "multiple_choice":
                values["options"] = tuple(values["options"])
                questions.append(MultipleChoiceQuestion(**values))
            else:
                values["key_points"] = tuple(values["key_points"])
                questions.append(ShortAnswerQuestion(**values))
        return cls(quiz_id=data["quiz_id"], topic=data["topic"], level=data["level"], questions=tuple(questions))


def public_view(quiz: Quiz) -> list[QuestionView]:
    """The quiz as the learner sees it before submitting. Answers are left out on purpose."""
    views = []
    for question in quiz.questions:
        options = question.options if isinstance(question, MultipleChoiceQuestion) else ()
        views.append(QuestionView(question.question_id, question.kind, question.concept, question.prompt, options))
    return views


def parse_option_letter(reply: Any, option_count: int, options: tuple[str, ...] = ()) -> int | None:
    """Read an answer like 'B', 'b)', 'Option C', 'The answer is D' -> a 0-based index.

    Also accepts the exact text of an option. Returns None when unclear.
    """
    if reply is None:
        return None
    text = _ANSWER_LABEL.sub("", str(reply).strip())
    for index, option in enumerate(options):
        if text.lower().strip(" .") == option.lower().strip(" ."):
            return index
    match = _LETTER_AT_START.match(text) or _LETTER_AFTER_LABEL.search(str(reply))
    if not match:
        return None
    index = OPTION_LETTERS.index(match.group(1).upper())
    return index if index < option_count else None


def _clean_option(option: Any) -> str:
    return _OPTION_PREFIX.sub("", str(option or "").strip()).strip()


def _parse_multiple_choice(item: Any, question_id: str, allowed: tuple[str, ...]) -> MultipleChoiceQuestion:
    item = _require_dict(item, "question")
    prompt = _text(item, "question", "multiple-choice question")
    options_raw = item.get("options")
    if not isinstance(options_raw, (list, tuple)):
        raise SchemaError("'options' must be a list")
    options = tuple(_clean_option(option) for option in options_raw)
    if any(not option for option in options):
        raise SchemaError("an option is empty")
    if len({option.lower() for option in options}) != len(options):
        raise SchemaError("two options are the same")
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise SchemaError(f"it has {len(options)} options (needs {MIN_OPTIONS} to {MAX_OPTIONS})")
    correct_index = parse_option_letter(item.get("answer"), len(options), options)
    if correct_index is None:
        raise SchemaError(f"its answer {item.get('answer')!r} is not one of the option letters")
    return MultipleChoiceQuestion(
        question_id=question_id,
        concept=match_concept(_text(item, "concept", "question"), allowed),
        prompt=prompt,
        options=options,
        correct_index=correct_index,
        explanation=_text(item, "explanation", "question", required=False) or "No explanation was given.",
    )


def _parse_short_answer(item: Any, question_id: str, allowed: tuple[str, ...]) -> ShortAnswerQuestion:
    item = _require_dict(item, "question")
    return ShortAnswerQuestion(
        question_id=question_id,
        concept=match_concept(_text(item, "concept", "question"), allowed),
        prompt=_text(item, "question", "short-answer question"),
        key_points=_text_list(item, "key_points", "short-answer question", 1, 4),
        model_answer=_text(item, "model_answer", "short-answer question"),
    )


def parse_quiz(
    data: Any,
    quiz_id: str,
    topic: str,
    level: str,
    allowed_concepts: tuple[str, ...],
    multiple_choice: int,
    short_answer: int,
) -> tuple[Quiz, list[str]]:
    """Check the model's quiz JSON. Bad questions are dropped (with a reason), not fatal.

    Returns (quiz, notes about dropped questions). Raises SchemaError only when no
    usable question is left, so invoke_structured can ask the model again.
    """
    data = _require_dict(data, "quiz")
    notes: list[str] = []
    questions: list[Question] = []

    sections = (
        ("multiple_choice", multiple_choice, _parse_multiple_choice, "Multiple-choice"),
        ("short_answer", short_answer, _parse_short_answer, "Short-answer"),
    )
    for key, wanted, parser, label in sections:
        if wanted == 0:
            continue
        items = data.get(key) or []
        if not isinstance(items, list):
            items = []
        kept = 0
        for position, item in enumerate(items, start=1):
            if kept == wanted:
                break
            try:
                questions.append(parser(item, f"q{len(questions) + 1}", allowed_concepts))
                kept += 1
            except SchemaError as exc:
                notes.append(f"{label} question {position} was dropped: {exc}")
        if kept < wanted:
            notes.append(f"{label}: asked for {wanted}, got {kept} usable question(s)")

    if not questions:
        reason = "; ".join(notes) or "it had no questions"
        raise SchemaError(f"the quiz had no usable questions ({reason})")
    return Quiz(quiz_id=quiz_id, topic=topic, level=level, questions=tuple(questions)), notes


# ---------- Grading results ----------

@dataclass(frozen=True)
class PointResult:
    """One rubric key point: did the learner's answer cover it, and why?"""

    point: str
    covered: bool
    reason: str = ""


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    kind: QuestionKind
    concept: str
    prompt: str
    learner_answer: str
    score: float  # 0.0 to 1.0
    graded: bool  # False = could not be graded reliably; left out of mastery
    feedback: str  # explains what was right or wrong
    correct_answer: str  # shown only after submission
    point_results: tuple[PointResult, ...] = ()
    misconception: str = ""  # a POSSIBLE misunderstanding, worded cautiously

    @property
    def correct(self) -> bool:
        return self.graded and self.score >= PASS_SCORE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["point_results"] = [asdict(point) for point in self.point_results]
        data["correct"] = self.correct
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionResult:
        values = {key: value for key, value in data.items() if key != "correct"}
        values["point_results"] = tuple(PointResult(**point) for point in values.get("point_results", ()))
        return cls(**values)


@dataclass(frozen=True)
class AssessmentResult:
    quiz_id: str
    results: tuple[QuestionResult, ...]

    @property
    def graded_results(self) -> list[QuestionResult]:
        return [result for result in self.results if result.graded]

    @property
    def score(self) -> float | None:
        """Average score of the graded questions, or None if nothing could be graded."""
        graded = self.graded_results
        return round(sum(result.score for result in graded) / len(graded), 3) if graded else None

    @property
    def correct_count(self) -> int:
        return sum(1 for result in self.results if result.correct)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quiz_id": self.quiz_id,
            "score": self.score,
            "correct_count": self.correct_count,
            "question_count": len(self.results),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentResult:
        return cls(data["quiz_id"], tuple(QuestionResult.from_dict(item) for item in data["results"]))