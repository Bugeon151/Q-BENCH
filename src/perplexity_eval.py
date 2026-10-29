"""Pipeline stage (v0.2): perplexity on wikitext-2 raw test via llama-perplexity.

Writes one row per (model, quant) to data/quality/<hw_id>/perplexity.csv.
Completed rows are skipped on re-run.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import (ROOT_DIR, check_budget, detect_hw, gguf_path, load_config, make_logger,
                    quant_levels, resolve_bin, select_models)

PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
FIELDS = ["hw_id", "model", "quant", "ctx", "chunks", "ppl", "ppl_err", "n_gpu_layers",
          "build_commit", "elapsed_sec", "timestamp"]


def parse_ppl(output: str) -> tuple[float, float]:
    m = PPL_RE.search(output)
    if not m:
        raise ValueError("no 'Final estimate: PPL' line in llama-perplexity output")
    return float(m.group(1)), float(m.group(2))


def pinned_commit() -> str:
    p = ROOT_DIR / "config" / "llama_cpp_commit.txt"  # verified against external/llama.cpp by setup.sh
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def ensure_dataset(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest.parent.parent)
    if not dest.exists():
        raise SystemExit(f"FAIL: {dest} not found after extracting {url}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="model name from config, or 'all' (default: default_model)")
    ap.add_argument("--quants", nargs="+", help="quant levels in run order (default: all, F16 first)")
    args = ap.parse_args()

    cfg = load_config()
    pcfg = cfg["perplexity"]
    hw = detect_hw()
    log = make_logger(cfg, "perplexity")
    out_csv = ROOT_DIR / cfg["paths"]["quality_dir"] / hw["hw_id"] / "perplexity.csv"
    text = ensure_dataset(pcfg["dataset_url"], ROOT_DIR / pcfg["dataset_file"])

    done = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            done = {(r["model"], r["quant"]) for r in csv.DictReader(f)}

    for model in select_models(cfg, args.model):
        name = model["name"]
        for quant in args.quants or quant_levels(cfg):
            tag = f"{name}/{quant}"
            if (name, quant) in done:
                log(f"SKIP {tag}: already in {out_csv}")
                continue
            ok, why = check_budget(cfg, name, quant, pcfg["ctx"], hw, allow_host_offload=True)
            if not ok:
                log(f"SKIP {tag}: over memory budget ({why})")
                continue
            cmd = [str(resolve_bin(pcfg["llama_perplexity_bin"])), "-m", str(gguf_path(cfg, name, quant)),
                   "-f", str(text), "-c", str(pcfg["ctx"]), "-ngl", str(pcfg["n_gpu_layers"])]
            if pcfg.get("chunks"):
                cmd += ["--chunks", str(pcfg["chunks"])]
            log(f"{tag}: running llama-perplexity")
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            output = proc.stdout + proc.stderr
            (ROOT_DIR / cfg["paths"]["logs_dir"] / f"perplexity_{name}_{quant}.log").write_text(output, encoding="utf-8")
            if proc.returncode != 0:
                raise SystemExit(f"FAIL {tag}: llama-perplexity exit {proc.returncode}\n{output[-4000:]}")
            ppl, err = parse_ppl(output)
            row = {
                "hw_id": hw["hw_id"], "model": name, "quant": quant, "ctx": pcfg["ctx"],
                "chunks": pcfg.get("chunks") or "all", "ppl": ppl, "ppl_err": err,
                "n_gpu_layers": pcfg["n_gpu_layers"], "build_commit": pinned_commit(),
                "elapsed_sec": round(time.time() - t0, 1), "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            new = not out_csv.exists()
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            log(f"{tag}: PPL = {ppl:.4f} +/- {err:.5f}")

    log("Perplexity complete.")


if __name__ == "__main__":
    main()
