"""Phase 5: reading JSON from model replies, with a bounded retry."""

import pytest

from adaptive_tutor.llm.structured import FENCE, RETRY_FEEDBACK, StructuredOutputError, extract_json, invoke_structured


class ScriptedChain:
    """Returns the scripted replies in order and remembers every input."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs = []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        return self.replies.pop(0)


def need_name(data):
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError("it has no name")
    return data["name"]


@pytest.mark.parametrize(
    "reply",
    [
        '{"name": "Ada"}',
        'Sure! Here is the JSON: {"name": "Ada"} Hope it helps.',
        FENCE + 'json\n{"name": "Ada"}\n' + FENCE,
        FENCE + '\n{"name": "Ada"}\n' + FENCE,
    ],
)
def test_extract_json_finds_the_object(reply):
    assert extract_json(reply) == {"name": "Ada"}


@pytest.mark.parametrize("reply", ["", "no json here", "{broken: json"])
def test_extract_json_rejects_replies_without_json(reply):
    with pytest.raises(ValueError):
        extract_json(reply)


def test_first_good_reply_is_used():
    chain = ScriptedChain('{"name": "Ada"}')
    value, tries = invoke_structured(chain, {"x": 1}, need_name, retries=1, what="thing")
    assert (value, tries) == ("Ada", 1)
    assert chain.inputs[0]["format_feedback"] == ""


def test_retry_tells_the_model_what_was_wrong():
    chain = ScriptedChain('{"age": 3}', '{"name": "Ada"}')
    value, tries = invoke_structured(chain, {"x": 1}, need_name, retries=1, what="thing")
    assert (value, tries) == ("Ada", 2)
    assert chain.inputs[1]["format_feedback"] == RETRY_FEEDBACK.format(problem="it has no name")


def test_gives_up_after_the_retry_limit_with_a_friendly_message():
    chain = ScriptedChain("nope", "still nope", "never")
    with pytest.raises(StructuredOutputError, match="after 2 tries"):
        invoke_structured(chain, {}, need_name, retries=1, what="lesson")
    assert len(chain.inputs) == 2  # bounded: 1 try + 1 retry


def test_model_errors_are_not_swallowed():
    class Broken:
        def invoke(self, inputs):
            raise ConnectionError("Ollama is down")

    with pytest.raises(ConnectionError):
        invoke_structured(Broken(), {}, need_name, retries=2, what="lesson")


def test_json_mode_model_asks_ollama_for_json(settings):
    pytest.importorskip("langchain_ollama")
    from adaptive_tutor.llm.provider import create_chat_model

    model = create_chat_model(settings, json_mode=True)
    assert model.format == "json"
    assert model.temperature <= 0.2