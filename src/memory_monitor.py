"""Peak RSS / VRAM / swap sampling for a running subprocess.

llama-bench does not report process memory, so we poll it out-of-band
via psutil while the subprocess runs.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

import psutil


class MemoryMonitor:
    """Polls RSS of a psutil.Process (by pid) on a background thread.

    Usage:
        proc = subprocess.Popen([...])
        with MemoryMonitor(proc.pid) as mon:
            proc.wait()
        peak_bytes = mon.peak_rss_bytes
    """

    def __init__(self, pid: int, poll_interval_sec: float = 0.05):
        self.pid = pid
        self.poll_interval_sec = poll_interval_sec
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    def _poll_loop(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    rss += child.memory_info().rss
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.poll_interval_sec)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2 * self.poll_interval_sec + 1)


class VramMonitor:
    """Peak device VRAM usage via a streaming `nvidia-smi -lms` subprocess.

    Process RSS does not include VRAM. Per-process VRAM is often unavailable
    (e.g. WSL), so this records device-level used memory: peak during the run
    minus the value sampled before the run started. No-op if nvidia-smi absent.
    """

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @property
    def peak_delta_bytes(self) -> int | None:
        return None if self._proc is None else max(0, self.peak_bytes - self.baseline_bytes)

    def _query_cmd(self, *extra: str) -> list[str]:
        return ["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=memory.used",
                "--format=csv,noheader,nounits", *extra]

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if line.isdigit():
                self.peak_bytes = max(self.peak_bytes, int(line) * 1024**2)

    def __enter__(self) -> "VramMonitor":
        if shutil.which("nvidia-smi"):
            out = subprocess.run(self._query_cmd(), capture_output=True, text=True).stdout.strip()
            self.baseline_bytes = int(out) * 1024**2 if out.isdigit() else 0
            self.peak_bytes = self.baseline_bytes
            self._proc = subprocess.Popen(
                self._query_cmd("-lms", str(self.interval_ms)),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
            if self._thread:
                self._thread.join(timeout=2)


def swap_used_bytes() -> int:
    return int(psutil.swap_memory().used)
