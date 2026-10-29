"""Pipeline stage: Warm-up + Benchmark + Raw Data Collection.

Iterates model x quant x ctx x test group (pp / tg / pg). Each group runs a
discarded warm-up followed by a measurement phase whose per-repetition samples
are flattened into data/raw/<hw_id>/benchmark_raw.csv together with
llama-bench metadata, runtime hardware info, peak RSS/VRAM and swap usage.

  pp: -p ctx -n 0             prompt processing of ctx tokens (TTFT)
  tg: -d ctx -p 0 -n N        generation of N tokens at KV depth ctx
  pg: -pg ctx,N -p 0 -n 0     prompt + generation (-p 0 -n 0 suppress the
                              default pp512/tg128 tests llama-bench adds)

The v0.1 file data/raw/benchmark_raw.csv (legacy schema) is never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import (ROOT_DIR, check_budget, detect_hw, gguf_path, load_config, make_logger,
                    quant_levels, resolve_bin, select_models)
from memory_monitor import MemoryMonitor, VramMonitor, swap_used_bytes

TEST_GROUPS = {
    "pp": lambda ctx, n: ["-p", str(ctx), "-n", "0"],
    "tg": lambda ctx, n: ["-d", str(ctx), "-p", "0", "-n", str(n)],
    "pg": lambda ctx, n: ["-pg", f"{ctx},{n}", "-p", "0", "-n", "0"],
}

# llama-bench JSONL fields preserved verbatim in the raw CSV
BENCH_META_FIELDS = [
    "build_commit", "n_threads", "n_gpu_layers", "flash_attn", "n_batch", "n_ubatch",
    "type_k", "type_v", "gpu_info", "cpu_info", "backends",
]

RAW_CSV_FIELDS = [
    "hw_id", "model", "quant", "ctx", "test_type", "run_index",
    "n_prompt", "n_gen", "n_depth", "latency_ms", "tokens_per_sec",
    "model_size_bytes", "peak_mem_bytes", "peak_vram_bytes",
    "swap_used_before_bytes", "swap_used_after_bytes", "timestamp",
    "os", "hardware_cpu", "hardware_gpu", "hardware_memory_bytes", "hardware_vram_bytes",
    "hardware_backend", *BENCH_META_FIELDS,
]


def raw_csv_path(cfg: dict, hw_id: str) -> Path:
    return ROOT_DIR / cfg["paths"]["raw_dir"] / hw_id / "benchmark_raw.csv"


def run_llama_bench(cmd: list[str], capture: bool, poll_sec: float, vram_ms: int) -> tuple[str, int, int | None]:
    """Runs llama-bench; returns (stdout, peak RSS bytes, peak VRAM delta bytes|None)."""
    with VramMonitor(vram_ms) as vmon:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )
        with MemoryMonitor(proc.pid, poll_interval_sec=poll_sec) as mon:
            stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({' '.join(cmd)}):\n{stderr[-4000:]}")
    return stdout or "", mon.peak_rss_bytes, vmon.peak_delta_bytes


def parse_jsonl_rows(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def to_raw_rows(bench_rows: list[dict], base: dict) -> list[dict]:
    """Flattens llama-bench JSONL records into one raw row per repetition."""
    rows = []
    for b in bench_rows:
        meta = {k: b.get(k) for k in BENCH_META_FIELDS}
        for i, (ns, ts) in enumerate(zip(b["samples_ns"], b["samples_ts"])):
            rows.append({
                **base, **meta, "run_index": i,
                "n_prompt": b["n_prompt"], "n_gen": b["n_gen"], "n_depth": b["n_depth"],
                "latency_ms": ns / 1e6, "tokens_per_sec": ts,
            })
    return rows


def done_keys(raw_csv: Path) -> set[tuple[str, str, str, str]]:
    if not raw_csv.exists():
        return set()
    with open(raw_csv, encoding="utf-8") as f:
        return {(r["model"], r["quant"], r["ctx"], r["test_type"]) for r in csv.DictReader(f)}


def append_rows(raw_csv: Path, rows: list[dict]) -> None:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not raw_csv.exists()
    with open(raw_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="model name from config, or 'all' (default: default_model)")
    ap.add_argument("--contexts", choices=["default", "all"], default="default",
                    help="default: inference.context_length (v0.1); all: inference.contexts (v0.3)")
    args = ap.parse_args()

    cfg = load_config()
    bcfg = cfg["benchmark"]
    hw = detect_hw()
    log = make_logger(cfg, "benchmark")
    raw_csv = raw_csv_path(cfg, hw["hw_id"])
    contexts = cfg["inference"]["contexts"] if args.contexts == "all" else [cfg["inference"]["context_length"]]
    n_gen = cfg["inference"]["max_output_tokens"]
    bin_path = str(resolve_bin(bcfg["llama_bench_bin"]))
    common_args = ["-ngl", str(bcfg["n_gpu_layers"]), "-fa", str(bcfg["flash_attn"]), "--no-warmup", "-o", "jsonl"]
    log(f"hw_id={hw['hw_id']} backend={hw['backend']} os={hw['os']} -> {raw_csv}")

    done = done_keys(raw_csv)
    for model in select_models(cfg, args.model):
        name = model["name"]
        for quant in quant_levels(cfg):
            path = gguf_path(cfg, name, quant)
            if not path.exists():
                raise SystemExit(f"FAIL: missing model file {path}. Run scripts/quantize.sh first.")
            size = path.stat().st_size
            for ctx in contexts:
                ok, why = check_budget(cfg, name, quant, ctx + n_gen, hw, size)
                if not ok:
                    log(f"SKIP {name}/{quant}/ctx{ctx}: over memory budget ({why})")
                    continue
                for test_type, args_fn in TEST_GROUPS.items():
                    tag = f"{name}/{quant}/ctx{ctx}/{test_type}"
                    if (name, quant, str(ctx), test_type) in done:
                        log(f"SKIP {tag}: already in {raw_csv}")
                        continue
                    cmd = [bin_path, "-m", str(path), *common_args, *args_fn(ctx, n_gen)]
                    log(f"{tag}: warm-up ({bcfg['warmup_runs']} runs, discarded)")
                    run_llama_bench([*cmd, "-r", str(bcfg["warmup_runs"])], False,
                                    bcfg["memory_poll_interval_sec"], bcfg["vram_poll_interval_ms"])
                    log(f"{tag}: measurement ({bcfg['measurement_runs']} runs)")
                    swap_before = swap_used_bytes()
                    stdout, peak_rss, peak_vram = run_llama_bench(
                        [*cmd, "-r", str(bcfg["measurement_runs"])], True,
                        bcfg["memory_poll_interval_sec"], bcfg["vram_poll_interval_ms"])
                    swap_after = swap_used_bytes()
                    base = {
                        "hw_id": hw["hw_id"], "model": name, "quant": quant, "ctx": ctx, "test_type": test_type,
                        "model_size_bytes": size, "peak_mem_bytes": peak_rss, "peak_vram_bytes": peak_vram,
                        "swap_used_before_bytes": swap_before, "swap_used_after_bytes": swap_after,
                        "timestamp": datetime.now(timezone.utc).isoformat(), "os": hw["os"],
                        "hardware_cpu": hw["cpu"], "hardware_gpu": hw["gpu"],
                        "hardware_memory_bytes": hw["memory_bytes"], "hardware_vram_bytes": hw["vram_bytes"],
                        "hardware_backend": hw["backend"],
                    }
                    rows = to_raw_rows(parse_jsonl_rows(stdout), base)
                    append_rows(raw_csv, rows)
                    if swap_after > swap_before:
                        log(f"WARN {tag}: swap grew {(swap_after - swap_before) / 1e9:.2f} GB during measurement")
                    log(f"{tag}: wrote {len(rows)} raw rows")

    log("Benchmark complete.")


if __name__ == "__main__":
    main()
