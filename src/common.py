"""Shared helpers: config, model paths, hardware detection, memory budget, logging.

CLI (used by shell scripts):
    python src/common.py hwid                      -> prints hardware id
    python src/common.py models [NAME|all]         -> "<name> <hf_repo>" per line
    python src/common.py fits MODEL QUANT CTX      -> exit 0 if within memory budget
"""

from __future__ import annotations

import fnmatch
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import psutil
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
GIB = 1024**3


def load_config() -> dict:
    with open(ROOT_DIR / "config" / "experiment.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- models

def select_models(cfg: dict, name: str | None = None) -> list[dict]:
    """name=None -> default_model, 'all' -> every model, else exact name."""
    models = cfg["models"]
    target = name or cfg["default_model"]
    if target == "all":
        return models
    sel = [m for m in models if m["name"] == target]
    if not sel:
        raise SystemExit(f"Unknown model '{target}'. Choices: {[m['name'] for m in models]} or 'all'")
    return sel


def quant_levels(cfg: dict) -> list[str]:
    return ["F16", *cfg["quantization"]["levels"]]


def hf_dir(cfg: dict, model: str) -> Path:
    return ROOT_DIR / cfg["model_dirs"]["hf"] / model


def gguf_path(cfg: dict, model: str, quant: str) -> Path:
    if quant == "F16":
        return ROOT_DIR / cfg["model_dirs"]["base"] / f"{model}-F16.gguf"
    return ROOT_DIR / cfg["quantization"]["output_dir"] / f"{model}-{quant}.gguf"


# --------------------------------------------------------------------------- hardware

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit():
            gpus.append({"name": parts[0], "vram_bytes": int(parts[1]) * 1024**2, "driver": parts[2]})
    return gpus


def _os_string() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Linux":
        name = "Linux"
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    name = line.split("=", 1)[1].strip('"')
        except OSError:
            pass
        wsl = " (WSL2)" if "microsoft" in platform.release().lower() else ""
        return f"{name}{wsl} kernel {platform.release()}"
    return f"{system} {platform.release()}"


def _cpu_string() -> str:
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@lru_cache(maxsize=1)
def detect_hw() -> dict:
    """Runtime hardware description. backend: Metal | CUDA | CPU."""
    cpu = _cpu_string()
    mem = psutil.virtual_memory().total
    gpus = nvidia_gpus()
    if gpus:
        g = gpus[0]
        backend, gpu, vram, driver = "CUDA", g["name"], g["vram_bytes"], g["driver"]
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        backend, gpu, vram, driver = "Metal", cpu, 0, ""
    else:
        backend, gpu, vram, driver = "CPU", "", 0, ""
    info = {
        "os": _os_string(), "cpu": cpu, "gpu": gpu, "gpu_driver": driver,
        "memory_bytes": mem, "vram_bytes": vram, "backend": backend,
    }
    override = (load_config().get("hardware") or {}).get("id")
    if override:
        info["hw_id"] = override
    elif backend == "CUDA":
        info["hw_id"] = _slug(f"{gpu} {round(vram / GIB)}gb")
    else:
        info["hw_id"] = _slug(f"{cpu} {round(mem / GIB)}gb")
    return info


# --------------------------------------------------------------------------- memory budget

def kv_cache_bytes(cfg: dict, model: str, ctx: int) -> int:
    """f16 K+V cache estimate from the HF config.json (0 if unavailable)."""
    p = hf_dir(cfg, model) / "config.json"
    if not p.exists():
        return 0
    c = json.loads(p.read_text())
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    n_kv = c.get("num_key_value_heads", c["num_attention_heads"])
    return 2 * c["num_hidden_layers"] * n_kv * head_dim * ctx * 2


def memory_budget_bytes(cfg: dict, hw: dict, allow_host_offload: bool = False) -> int:
    b = cfg["memory_budget"]
    if hw["backend"] == "CUDA":
        budget = hw["vram_bytes"] * b["cuda_fraction"]
        if allow_host_offload:
            budget += hw["memory_bytes"] * b["cpu_fraction"]
        return int(budget)
    frac = b["metal_fraction"] if hw["backend"] == "Metal" else b["cpu_fraction"]
    return int(hw["memory_bytes"] * frac)


def check_budget(
    cfg: dict, model: str, quant: str, ctx: int, hw: dict | None = None,
    model_bytes: int | None = None, allow_host_offload: bool = False,
) -> tuple[bool, str]:
    """Returns (ok, reason). Estimate = file size + f16 KV cache + fixed overhead."""
    hw = hw or detect_hw()
    b = cfg["memory_budget"]
    key = f"{hw['backend']}/{model}/{quant}"
    if any(fnmatch.fnmatch(key, pat) for pat in b.get("exclude") or []):
        return False, f"{key} matches memory_budget.exclude"
    if model_bytes is None:
        p = gguf_path(cfg, model, quant)
        model_bytes = p.stat().st_size if p.exists() else 0
    need = model_bytes + kv_cache_bytes(cfg, model, ctx) + int(b["overhead_gb"] * GIB)
    budget = memory_budget_bytes(cfg, hw, allow_host_offload)
    msg = f"est. {need / 1e9:.2f} GB vs budget {budget / 1e9:.2f} GB ({hw['backend']})"
    return need <= budget or not b["skip_over_budget"], msg


# --------------------------------------------------------------------------- logging

def make_logger(cfg: dict, tag: str):
    logs_dir = ROOT_DIR / cfg["paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{tag}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"

    def log(msg: str) -> None:
        line = f"[{tag}] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")

    return log


def main(argv: list[str]) -> int:
    cfg = load_config()
    cmd = argv[0] if argv else ""
    if cmd == "hwid":
        print(detect_hw()["hw_id"])
    elif cmd == "models":
        for m in select_models(cfg, argv[1] if len(argv) > 1 else None):
            print(m["name"], m["hf_repo"])
    elif cmd == "fits" and len(argv) == 4:
        ok, msg = check_budget(cfg, argv[1], argv[2], int(argv[3]))
        print(msg)
        return 0 if ok else 1
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
