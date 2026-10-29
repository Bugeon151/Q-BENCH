"""nvidia-smi parsing and GPU telemetry summary (no GPU needed)."""
import os
import sys

import pytest

import memory_monitor
from memory_monitor import VramMonitor, parse_smi_line


def test_parse_smi_line():
    assert parse_smi_line("1234, 65, 2100, 80.52\n") == (1234, [65.0, 2100.0, 80.52])
    assert parse_smi_line("1234, 65, 2100, [N/A]") == (1234, [65.0, 2100.0, None])
    assert parse_smi_line("1234") == (1234, [None, None, None])
    assert parse_smi_line("Field \"x\" is not a valid field to query.") is None
    assert parse_smi_line("") is None


def test_telemetry_summary():
    m = VramMonitor()
    assert m.telemetry == {}
    m.samples = [[60.0, 2500.0, 90.0], [72.0, 2100.0, None], [70.0, 2300.0, 80.0]]
    assert m.telemetry == {"telemetry_samples": 3, "gpu_temp_max_c": 72.0, "gpu_sm_clock_min_mhz": 2100.0,
                           "gpu_sm_clock_median_mhz": 2300.0, "gpu_power_median_w": 85.0}


@pytest.mark.skipif(os.name == "nt", reason="fake nvidia-smi is a shell script")
@pytest.mark.parametrize("rejects_fields", [False, True])
def test_vram_monitor_with_fake_smi(tmp_path, monkeypatch, rejects_fields):
    fake = tmp_path / "nvidia-smi"
    reject = 'case "$*" in *temperature*) echo "Field \\"temperature.gpu\\" is not a valid field to query."; exit 2;; esac\n'
    fake.write_text("#!/bin/sh\n" + (reject if rejects_fields else "") +
                    'case "$*" in *-lms*) for v in 1500 3500 2500; do echo "$v, 70, 2400, 85"; done; sleep 5;;'
                    ' *) echo "1000, 50, 300, 10";; esac\n', newline="\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    with VramMonitor(100) as m:
        import time
        time.sleep(0.5)
    assert m.baseline_bytes == 1000 * 1024**2
    assert m.peak_delta_bytes == 2500 * 1024**2
    if rejects_fields:
        assert m._fields == ()
    else:
        assert m.telemetry["gpu_temp_max_c"] == 70.0
