"""Streamlit user interface for the Adaptive AI Tutor (Phases 1-9).

app.py stays thin: it loads settings, builds the services once, and draws three tabs.
The pages themselves are in src/adaptive_tutor/ui/.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Make the "src" folder importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adaptive_tutor.config import Settings, load_settings  # noqa: E402
from adaptive_tutor.logging_setup import setup_logging  # noqa: E402
from adaptive_tutor.services import TutorServices, build_services  # noqa: E402
from adaptive_tutor.ui.ask_page import render_ask_page  # noqa: E402
from adaptive_tutor.ui.learn_page import render_learn_page  # noqa: E402
from adaptive_tutor.ui.progress_page import render_progress_page  # noqa: E402
from adaptive_tutor.ui.sidebar import render_sidebar  # noqa: E402

st.set_page_config(page_title="Adaptive AI Tutor", page_icon="🎓", layout="wide")

try:
    settings = load_settings()
except ValueError as exc:
    st.error(f"Configuration problem: {exc}. Check your .env file.")
    st.stop()

setup_logging(settings.log_level)
logger = logging.getLogger("adaptive_tutor.app")


@st.cache_resource
def get_services(_settings: Settings) -> TutorServices:
    """Build the models, document library, databases, and graphs once. Restart after editing .env.

    The leading underscore tells Streamlit not to try hashing this argument.
    """
    return build_services(_settings)


try:
    services = get_services(settings)
except ImportError as exc:
    st.error(f"A package is missing ({exc.name}). Run: pip install -r requirements.txt, then restart the app.")
    st.stop()
except Exception:  # noqa: BLE001 - e.g. a data folder cannot be created
    logger.exception("Could not start the tutor services")
    st.error("Could not open the document library or the progress database. Check the terminal logs.")
    st.stop()

learner = render_sidebar(services, settings)

st.title("🎓 Adaptive AI Tutor")
ask_tab, learn_tab, progress_tab = st.tabs(["💬 Ask a question", "📘 Learn & practise", "📊 My progress"])
with ask_tab:
    render_ask_page(services, settings, learner)
with learn_tab:
    render_learn_page(services, learner)
with progress_tab:
    render_progress_page(services, settings, learner)