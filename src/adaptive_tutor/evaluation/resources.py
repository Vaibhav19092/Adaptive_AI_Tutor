"""Phase 9: local resource usage, measured without extra packages.

- process_memory_mb(): memory used by this Python process (the app), on
  Windows, Linux, and macOS. Returns None if it can't be measured.
- ollama_loaded_models(): which models Ollama currently holds in memory, and how
  much (from Ollama's own /api/ps endpoint on this computer).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MB = 1024 * 1024


def process_memory_mb() -> float | None:
    """Resident memory of this process in MB (what the operating system has given it)."""
    try:
        if sys.platform == "win32":
            return _windows_memory_mb()
        status = Path("/proc/self/status")
        if status.exists():  # Linux
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)  # the value is in kB
        import resource  # macOS (and other Unix): peak memory

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / MB if sys.platform == "darwin" else peak / 1024, 1)
    except Exception:  # noqa: BLE001 - measuring memory must never break the app
        logger.debug("Could not measure process memory", exc_info=True)
        return None


def _windows_memory_mb() -> float | None:
    """Ask Windows (psapi.GetProcessMemoryInfo) for this process's working set."""

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.WorkingSetSize / MB, 1)


def ollama_loaded_models(base_url: str, client: httpx.Client | None = None, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Models Ollama has in memory right now: name, total MB, and MB on the GPU.

    Raises httpx.HTTPError if Ollama can't be reached (the dashboard shows a message).
    """
    url = f"{base_url.rstrip('/')}/api/ps"
    response = client.get(url, timeout=timeout) if client else httpx.get(url, timeout=timeout)
    response.raise_for_status()
    models = []
    for model in response.json().get("models", []):
        size = int(model.get("size") or 0)
        vram = int(model.get("size_vram") or 0)
        models.append({
            "name": model.get("name", ""),
            "size_mb": round(size / MB),
            "gpu_mb": round(vram / MB),
            "on_gpu": f"{round(100 * vram / size)}%" if size else "0%",
        })
    return models