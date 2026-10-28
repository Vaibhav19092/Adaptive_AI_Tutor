"""Phase 9: measuring local resource usage."""

import httpx
import pytest

from adaptive_tutor.evaluation.resources import ollama_loaded_models, process_memory_mb


def test_process_memory_is_a_positive_number_or_unknown():
    memory = process_memory_mb()
    assert memory is None or memory > 0


def test_ollama_loaded_models_reads_api_ps():
    if not hasattr(httpx, "MockTransport"):
        pytest.skip("needs the real httpx package")

    def handler(request):
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json={"models": [{"name": "llama3.2:3b", "size": 3 * 1024**3, "size_vram": 1024**3}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    models = ollama_loaded_models("http://localhost:11434/", client=client)
    assert models == [{"name": "llama3.2:3b", "size_mb": 3072, "gpu_mb": 1024, "on_gpu": "33%"}]