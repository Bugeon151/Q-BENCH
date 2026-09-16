"""Pipeline stage: Warm-up + Benchmark + Raw Data Collection.

Runs llama-bench across every prepared quantization level (F16 + configured
quant levels), for three test groups (pp / tg / pg), each with a discarded
warm-up phase followed by a measurement phase whose per-repetition samples
are flattened into data/raw/benchmark_raw.csv.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from memory_monitor import MemoryMonitor

ROOT_DIR = Path(__file__).resolve().parent.parent

HARDWARE_SOC = "Apple M5"
HARDWARE_MEMORY_GB = 24
HARDWARE_BACKEND = "Metal"

RAW_CSV_FIELDS = [
    "quant",
    "test_type",
    "run_index",
    "n_prompt",
    "n_gen",
    "n_depth",
    "latency_ms",
    "tokens_per_sec",
    "model_size_bytes",
    "peak_mem_bytes",
    "timestamp",
    "hardware_soc",
    "hardware_memory_gb",
    "hardware_backend",
]


def load_config() -> dict:
    with open(ROOT_DIR / "config" / "experiment.yaml") as f:
        return yaml.safe_load(f)


def model_paths(cfg: dict) -> dict[str, Path]:
    base_gguf = ROOT_DIR / cfg["base_model"]["base_gguf"]
    stem = base_gguf.name.removesuffix("-F16.gguf")
    quant_dir = ROOT_DIR / cfg["quantization"]["output_dir"]
    paths = {"F16": base_gguf}
    for level in cfg["quantization"]["levels"]:
        paths[level] = quant_dir / f"{stem}-{level}.gguf"
    return paths


# test_type -> extra llama-bench CLI args (beyond -m/-r/-o/--no-warmup)
TEST_GROUPS = {
    "pp": lambda ctx: ["-p", str(ctx), "-n", "0"],
    "tg": lambda ctx: ["-d", str(ctx), "-p", "0", "-n", "256"],
    "pg": lambda ctx: ["-pg", f"{ctx},256"],
}


def run_llama_bench(
    llama_bench_bin: str, model_path: Path, extra_args: list[str], repetitions: int, capture: bool
) -> tuple[str, int | None]:
    """Runs llama-bench, optionally capturing stdout (jsonl) and peak RSS."""
    cmd = [
        llama_bench_bin,
        "-m",
        str(model_path),
        "-r",
        str(repetitions),
        "--no-warmup",
        "-o",
        "jsonl",
        *extra_args,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_mem = None
    poll_interval = 0.05
    with MemoryMonitor(proc.pid, poll_interval_sec=poll_interval) as mon:
        stdout, stderr = proc.communicate()
    peak_mem = mon.peak_rss_bytes
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({' '.join(cmd)}):\n{stderr}")
    return stdout or "", peak_mem


def parse_jsonl_rows(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def already_done(raw_csv: Path, quant: str, test_type: str) -> bool:
    if not raw_csv.exists():
        return False
    with open(raw_csv) as f:
        for row in csv.DictReader(f):
            if row["quant"] == quant and row["test_type"] == test_type:
                return True
    return False


def append_rows(raw_csv: Path, rows: list[dict]) -> None:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not raw_csv.exists()
    with open(raw_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = load_config()
    llama_bench_bin = str(ROOT_DIR / cfg["benchmark"]["llama_bench_bin"])
    warmup_runs = cfg["benchmark"]["warmup_runs"]
    measurement_runs = cfg["benchmark"]["measurement_runs"]
    ctx = cfg["inference"]["context_length"]
    raw_csv = ROOT_DIR / cfg["paths"]["raw_csv"]

    logs_dir = ROOT_DIR / cfg["paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"benchmark_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"

    def log(msg: str) -> None:
        line = f"[benchmark] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    paths = model_paths(cfg)
    for quant, gguf_path in paths.items():
        if not gguf_path.exists():
            log(f"FAIL: missing model file for {quant}: {gguf_path}")
            sys.exit(1)

    for quant, gguf_path in paths.items():
        model_size_bytes = os.path.getsize(gguf_path)
        for test_type, args_fn in TEST_GROUPS.items():
            if already_done(raw_csv, quant, test_type):
                log(f"SKIP {quant}/{test_type}: already in {raw_csv}")
                continue

            extra_args = args_fn(ctx)
            log(f"{quant}/{test_type}: warm-up ({warmup_runs} runs, discarded)")
            run_llama_bench(llama_bench_bin, gguf_path, extra_args, warmup_runs, capture=False)

            log(f"{quant}/{test_type}: measurement ({measurement_runs} runs)")
            stdout, peak_mem_bytes = run_llama_bench(
                llama_bench_bin, gguf_path, extra_args, measurement_runs, capture=True
            )
            bench_rows = parse_jsonl_rows(stdout)

            rows = []
            timestamp = datetime.now(timezone.utc).isoformat()
            for bench_row in bench_rows:
                samples_ns = bench_row["samples_ns"]
                samples_ts = bench_row["samples_ts"]
                n_prompt = bench_row["n_prompt"]
                n_gen = bench_row["n_gen"]
                n_depth = bench_row["n_depth"]
                for i, (ns, ts) in enumerate(zip(samples_ns, samples_ts)):
                    rows.append(
                        {
                            "quant": quant,
                            "test_type": test_type,
                            "run_index": i,
                            "n_prompt": n_prompt,
                            "n_gen": n_gen,
                            "n_depth": n_depth,
                            "latency_ms": ns / 1e6,
                            "tokens_per_sec": ts,
                            "model_size_bytes": model_size_bytes,
                            "peak_mem_bytes": peak_mem_bytes,
                            "timestamp": timestamp,
                            "hardware_soc": HARDWARE_SOC,
                            "hardware_memory_gb": HARDWARE_MEMORY_GB,
                            "hardware_backend": HARDWARE_BACKEND,
                        }
                    )
            append_rows(raw_csv, rows)
            log(f"{quant}/{test_type}: wrote {len(rows)} raw rows")

    log("Benchmark complete.")


if __name__ == "__main__":
    main()
