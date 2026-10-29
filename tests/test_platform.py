"""Cross-platform helpers: llama.cpp binary resolution, Git Bash shell helpers, UTF-8 I/O."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common import ROOT_DIR, resolve_bin
from quality_eval import run_program

REL = "external/llama.cpp/build/bin/llama-bench"


def touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_resolve_bin_posix(tmp_path):
    assert resolve_bin(REL, tmp_path, windows=False) == tmp_path / REL


@pytest.mark.parametrize("layout", ["Release/llama-bench.exe", "llama-bench.exe"])
def test_resolve_bin_windows_layouts(tmp_path, layout):
    exe = touch(tmp_path / "external/llama.cpp/build/bin" / layout)
    assert resolve_bin(REL, tmp_path, windows=True) == exe


def test_resolve_bin_windows_prefers_single_config(tmp_path):
    ninja = touch(tmp_path / f"{REL}.exe")
    touch(tmp_path / "external/llama.cpp/build/bin/Release/llama-bench.exe")
    assert resolve_bin(REL, tmp_path, windows=True) == ninja


def test_resolve_bin_windows_missing(tmp_path):
    assert resolve_bin(REL, tmp_path, windows=True) == tmp_path / f"{REL}.exe"


def test_common_cli_bin_exit_code():
    r = subprocess.run([sys.executable, str(ROOT_DIR / "src/common.py"), "bin", "benchmark.llama_bench_bin"],
                       capture_output=True, text=True)
    path = Path(r.stdout.strip())
    assert path.name.startswith("llama-bench")
    assert r.returncode == (0 if path.is_file() else 1)


def run_sh(tmp_path: Path, venv_rel: str, body: str) -> subprocess.CompletedProcess:
    """Sources scripts/common.sh with ROOT_DIR=tmp_path and a fake venv python."""
    fake = touch(tmp_path / ".venv" / venv_rel)
    fake.write_text('#!/usr/bin/env bash\nprintf "a b\\r\\nQ4_K_M\\r\\n"; exit "${1:-0}"\n')
    fake.chmod(0o755)
    script = f'set -euo pipefail\nROOT_DIR="{tmp_path}"\nsource "{ROOT_DIR}/scripts/common.sh"\n{body}'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("venv_rel", ["bin/python", "Scripts/python.exe"])
def test_common_sh_venv_and_crlf(tmp_path, venv_rel):
    r = run_sh(tmp_path, venv_rel, 'while IFS= read -r l; do printf "[%s]" "$l"; done < <(py 0)\n'
                                    'printf "|%s" "$PYTHON" "$PYTHONUTF8"')
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"[a b][Q4_K_M]|{tmp_path}/.venv/{venv_rel}|1"


def test_common_sh_keeps_exit_status(tmp_path):
    r = run_sh(tmp_path, "bin/python", 'if ! OUT="$(py 3)"; then echo "failed:$OUT"; fi')
    assert r.stdout.strip().replace("\n", " ") == "failed:a b Q4_K_M"


def test_run_program_unicode():
    assert run_program("s = '한글 ✓'\nassert len(s) == 4\n", 10)


@pytest.mark.skipif(os.name != "nt", reason="Windows registry lookup")
def test_windows_cpu_name():
    from common import _cpu_string
    assert _cpu_string() and "Family" not in _cpu_string()
