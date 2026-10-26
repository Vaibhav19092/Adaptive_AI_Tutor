# Adaptive AI Tutor (Phase 1)

A local, zero-cost AI tutor built with Ollama, LangChain, LangGraph, and Streamlit.
Phase 1 answers questions using a local model; later phases add RAG, quizzes, and memory.

## Requirements
- Python 3.10+
- Ollama (https://ollama.com) with a model pulled, e.g. `ollama pull qwen2.5:7b`
- ~8 GB RAM for 3B models, ~16 GB for 7B models

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

## Run
```bash
streamlit run app.py
```
Open http://localhost:8501.

## Test
```bash
pytest -v
``` 
Tests use a fake model, so Ollama doesn't need to be running.

## Configuration
All settings live in `.env` (see `.env.example`). Restart the app after changing them.

## Troubleshooting
| Message | Fix |
|---|---|
| Cannot reach Ollama | Start the Ollama app or run `ollama serve` |
| Model ... is not installed | Run `ollama pull <model>` or fix `OLLAMA_MODEL` |
| Model took longer than ... | Use a smaller model or raise `OLLAMA_REQUEST_TIMEOUT_SECONDS` |

## Privacy
Everything runs locally. Streamlit is restricted to localhost and usage stats are disabled
in `.streamlit/config.toml`. Questions are not written to logs (only their length).