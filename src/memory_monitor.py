"""Peak RSS / VRAM / swap sampling for a running subprocess.

llama-bench does not report process memory, so we poll it out-of-band
via psutil while the subprocess runs.
"""

from __future__ import annotations

import shutil
import statistics
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


TELEMETRY_FIELDS = ("temperature.gpu", "clocks.sm", "power.draw")


def parse_smi_line(line: str) -> tuple[int, list[float | None]] | None:
    """'1234, 65, 2100, 80.52' -> (memory.used MiB, [temp C, SM MHz, power W]); fields may be [N/A]."""
    parts = [p.strip() for p in line.split(",")]
    if not parts or not parts[0].isdigit():
        return None
    vals: list[float | None] = []
    for p in parts[1:1 + len(TELEMETRY_FIELDS)]:
        try:
            vals.append(float(p))
        except ValueError:
            vals.append(None)
    return int(parts[0]), vals + [None] * (len(TELEMETRY_FIELDS) - len(vals))


class VramMonitor:
    """Peak device VRAM usage and GPU telemetry via a streaming `nvidia-smi -lms` subprocess.

    Process RSS does not include VRAM. Per-process VRAM is often unavailable
    (e.g. WSL), so this records device-level used memory: peak during the run
    minus the value sampled before the run started. Temperature, SM clock and
    board power are sampled alongside (thermal throttling check on laptops).
    No-op if nvidia-smi absent.
    """

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.samples: list[list[float | None]] = []
        self._fields: tuple[str, ...] = TELEMETRY_FIELDS
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @property
    def peak_delta_bytes(self) -> int | None:
        return None if self._proc is None else max(0, self.peak_bytes - self.baseline_bytes)

    @property
    def telemetry(self) -> dict:
        """Temperature max, SM clock min/median, power median over the run ({} without samples)."""
        cols = [[s[i] for s in self.samples if s[i] is not None] for i in range(len(TELEMETRY_FIELDS))]
        temp, clk, pwr = cols
        out = {"telemetry_samples": len(self.samples)}
        if temp:
            out["gpu_temp_max_c"] = max(temp)
        if clk:
            out["gpu_sm_clock_min_mhz"] = min(clk)
            out["gpu_sm_clock_median_mhz"] = statistics.median(clk)
        if pwr:
            out["gpu_power_median_w"] = statistics.median(pwr)
        return out if self.samples else {}

    def _query_cmd(self, *extra: str) -> list[str]:
        return ["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=" + ",".join(("memory.used", *self._fields)),
                "--format=csv,noheader,nounits", *extra]

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            parsed = parse_smi_line(line)
            if parsed:
                self.peak_bytes = max(self.peak_bytes, parsed[0] * 1024**2)
                self.samples.append(parsed[1])

    def __enter__(self) -> "VramMonitor":
        if shutil.which("nvidia-smi"):
            parsed = parse_smi_line(subprocess.run(self._query_cmd(), capture_output=True, text=True).stdout.strip())
            if parsed is None:  # a telemetry field this driver rejects: keep the VRAM measurement alone
                self._fields = ()
                parsed = parse_smi_line(subprocess.run(self._query_cmd(), capture_output=True, text=True).stdout.strip())
            self.baseline_bytes = parsed[0] * 1024**2 if parsed else 0
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
