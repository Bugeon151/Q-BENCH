"""Peak RSS memory sampling for a running subprocess.

llama-bench does not report process memory, so we poll it out-of-band
via psutil while the subprocess runs.
"""

from __future__ import annotations

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
