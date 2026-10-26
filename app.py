"""Streamlit user interface for the Adaptive AI Tutor (Phase 1)."""

import logging
import sys
from pathlib import Path

import streamlit as st

# Make the "src" folder importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adaptive_tutor.config import Settings, load_settings  # noqa: E402
from adaptive_tutor.graph.workflow import ask_tutor, build_tutor_graph  # noqa: E402
from adaptive_tutor.llm.provider import (  # noqa: E402
    LLMError,
    check_model_available,
    create_chat_model,
)
from adaptive_tutor.logging_setup import setup_logging  # noqa: E402

st.set_page_config(page_title="Adaptive AI Tutor", page_icon="🎓")

try:
    settings = load_settings()
except ValueError as exc:
    st.error(f"Configuration problem: {exc}. Check your .env file.")
    st.stop()

setup_logging(settings.log_level)
logger = logging.getLogger("adaptive_tutor.app")


@st.cache_resource
def get_tutor_graph(_settings: Settings):
    """Build the graph once and reuse it. Restart the app after editing .env.

    The leading underscore tells Streamlit not to try hashing this argument.
    """
    llm = create_chat_model(_settings)
    return build_tutor_graph(llm, _settings)


# ---------- Sidebar: model status ----------
with st.sidebar:
    st.header("Local model")
    st.write(f"**Model:** `{settings.ollama_model}`")
    st.write(f"**Ollama URL:** `{settings.ollama_base_url}`")
    try:
        check_model_available(settings)
        st.success("Ollama is running and the model is installed.")
    except LLMError as exc:
        st.error(str(exc))
    if st.button("Re-check connection"):
        st.rerun()

# ---------- Main area ----------
st.title("🎓 Adaptive AI Tutor")
st.caption(
    "Phase 1: answers come from the local model's own knowledge, "
    "not from your documents yet, so they can be wrong."
)

with st.form("question_form"):
    question = st.text_area(
        "Ask a question",
        placeholder="e.g. What is an embedding, in simple words?",
        height=120,
    )
    submitted = st.form_submit_button("Ask the tutor")

if submitted:
    with st.spinner("Thinking... local models on a CPU can take a while."):
        try:
            result = ask_tutor(get_tutor_graph(settings), question)
        except Exception:  # noqa: BLE001 - safety net; nodes already handle known errors
            logger.exception("Unexpected workflow failure")
            st.error("Something unexpected went wrong. Check the terminal logs.")
            st.stop()

    if result.get("error"):
        st.error(result["error"])
    else:
        st.markdown(result["answer"])

    with st.expander("Show LangGraph workflow steps"):
        st.write(" → ".join(result.get("steps", [])))