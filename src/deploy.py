"""Deploy a local model on this machine: pick, verify, accept, serve, and record an audit manifest.

Steps (no network needed when models/ and the Hugging Face dataset cache are already present):
  1. detect     hardware via common.detect_hw (backend, VRAM, RAM)
  2. select     (model, quant) with recommend.py from the measured tables, or --model/--quant
  3. verify     SHA256 of the GGUF and of the llama-server binary
  4. accept     run the first --items items per task of the fixed quality sample
                (prompts/quality.json) on this machine and compare item by item with the
                results recorded for the same (model, quant) in data/quality (src/gate.py);
                a significant drop (FAIL) blocks the deployment
  5. serve      start llama-server on 127.0.0.1 (OpenAI-compatible) and send one smoke request;
                with --serve it keeps running, otherwise it is stopped after the check
  6. manifest   deploy/<UTC time>_<model>_<quant>.json with everything above

    python src/deploy.py [--vram-gb N] [--min-tps T] [--model M --quant Q] [--items 20] [--serve]
Exit code: 0 deployed (or verified), 1 acceptance FAIL, 2 nothing fits / missing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gate
import quality_eval as qe
from common import GIB, ROOT_DIR, detect_hw, gguf_path, load_config, make_logger, resolve_bin
from recommend import candidates, load_model_cfgs, recommend

TASKS = ("mmlu", "gsm8k", "humaneval")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def select(cfg: dict, hw: dict, vram_gib: float, min_tps: float) -> pd.DataFrame:
    """Recommendation table for this machine from the committed result tables."""
    tables = ROOT_DIR / cfg["paths"]["tables_dir"]
    summary = pd.read_csv(ROOT_DIR / cfg["paths"]["summary_table"])
    quality = pd.read_csv(tables / "quality.csv")
    gates = pd.read_csv(tables / "gate.csv")
    measured_here = hw["hw_id"] if hw["hw_id"] in set(summary.hw_id) else None
    c = candidates(summary, quality, gates, load_model_cfgs(), cfg["inference"]["context_length"],
                   cfg["inference"]["max_output_tokens"], cfg["memory_budget"]["overhead_gb"], measured_here)
    return recommend(c, vram_gib, min_tps=min_tps)


def reference_items(cfg: dict, model: str, quant: str) -> pd.DataFrame:
    """Recorded per-item results for (model, quant) from any measured hardware (first hw_id)."""
    items = gate.load_items(ROOT_DIR / cfg["paths"]["quality_dir"])
    d = items[(items.model == model) & (items.quant == quant)]
    if d.empty:
        return d
    return d[d.hw_id == sorted(set(d.hw_id))[0]]


def acceptance(cfg: dict, sample: dict, model: str, quant: str, n_items: int, log) -> dict:
    """Runs the first n_items of each task against the running server; gates vs recorded results."""
    ref = reference_items(cfg, model, quant)
    q = cfg["quality"]
    out = {"items_per_task": n_items, "reference_hw_id": sorted(set(ref.hw_id))[0] if len(ref) else None,
           "tasks": {}}
    verdicts = []
    for task in TASKS:
        rows = qe.load_items(sample, task)[:n_items]
        got = {}
        for row in rows:
            text = qe.chat(cfg, qe.PROMPTS[task](row), q["datasets"][task]["max_tokens"])
            got[row["id"]] = bool(qe.score(task, row, text, q["exec_timeout_sec"])[1])
        r = ref[ref.task == task].set_index("id").correct
        res = gate.compare_task(r, pd.Series(got), gate.MARGIN, gate.ALPHA / len(TASKS)) if len(r) else \
            {"n": len(got), "cand_acc": sum(got.values()) / max(len(got), 1), "verdict": "NO_REFERENCE"}
        log(f"accept {task}: {res.get('cand_acc', 0):.3f} on {res['n']} items "
            f"(recorded {res.get('ref_acc', float('nan')):.3f}) -> {res['verdict']}")
        out["tasks"][task] = res
        verdicts.append(res["verdict"])
    out["verdict"] = "FAIL" if "FAIL" in verdicts else "OK"  # small samples: only clear regressions block
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vram-gb", type=float, help="GPU memory budget, GiB (default: detected VRAM x 0.95)")
    ap.add_argument("--min-tps", type=float, default=0.0)
    ap.add_argument("--model")
    ap.add_argument("--quant")
    ap.add_argument("--items", type=int, default=20, help="acceptance items per task (0 = skip)")
    ap.add_argument("--serve", action="store_true", help="keep llama-server running after the checks")
    args = ap.parse_args()

    cfg = load_config()
    log = make_logger(cfg, "deploy")
    hw = detect_hw()
    vram = args.vram_gb if args.vram_gb is not None else hw["vram_bytes"] / GIB * cfg["memory_budget"]["cuda_fraction"]
    manifest: dict = {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "hardware": {k: hw[k] for k in ("hw_id", "os", "cpu", "gpu", "gpu_driver", "backend",
                                                      "memory_bytes", "vram_bytes")}}

    # 2. select
    if args.model and args.quant:
        model, quant = args.model, args.quant
        manifest["selection"] = {"mode": "manual"}
    else:
        rec = select(cfg, hw, vram, args.min_tps)
        if rec.empty:
            log(f"no measured configuration fits {vram:.2f} GiB with >= {args.min_tps} tok/s")
            return 2
        model, quant = rec.model.iloc[0], rec.quant.iloc[0]
        manifest["selection"] = {"mode": "recommend", "vram_gib": round(vram, 3), "min_tps": args.min_tps,
                                 "top": rec.head(3).replace({np.nan: None}).to_dict(orient="records")}
    log(f"selected {model} {quant} (budget {vram:.2f} GiB)")

    # 3. verify files
    path, server_bin = gguf_path(cfg, model, quant), resolve_bin(cfg["quality"]["llama_server_bin"])
    for p in (path, server_bin):
        if not p.is_file():
            log(f"missing {p}")
            return 2
    prebuilt = server_bin.parent / "PREBUILT.txt"
    commit_file = ROOT_DIR / "config" / "llama_cpp_commit.txt"
    manifest["model"] = {"name": model, "quant": quant, "gguf": path.relative_to(ROOT_DIR).as_posix(),
                         "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["runtime"] = {"llama_server": server_bin.relative_to(ROOT_DIR).as_posix(),
                           "llama_server_sha256": sha256(server_bin),
                           "llama_cpp_commit": commit_file.read_text(encoding="utf-8").strip(),
                           "prebuilt": prebuilt.read_text(encoding="utf-8").splitlines() if prebuilt.exists() else None}
    log(f"gguf sha256 {manifest['model']['sha256'][:16]}...")

    # 4-5. serve, accept, smoke
    sample = qe.load_sample(cfg, log)
    manifest["sample"] = {"seed": sample.get("seed"),
                          "datasets": {t: {k: v for k, v in d.items() if k != "ids"} for t, d in sample["datasets"].items()}}
    log_path = ROOT_DIR / cfg["paths"]["logs_dir"] / f"deploy_server_{model}_{quant}.log"
    t0 = time.time()
    server = qe.start_server(cfg, path, log_path)
    status = 0
    try:
        manifest["server"] = {"url": f"http://127.0.0.1:{cfg['quality']['port']}/v1", "load_sec": round(time.time() - t0, 1),
                              "args": {"n_parallel": cfg["quality"]["n_parallel"],
                                       "ctx_per_slot": cfg["quality"]["ctx_per_slot"],
                                       "n_gpu_layers": cfg["quality"]["n_gpu_layers"]}}
        if args.items > 0:
            manifest["acceptance"] = acceptance(cfg, sample, model, quant, args.items, log)
            if manifest["acceptance"]["verdict"] == "FAIL":
                status = 1
        t1 = time.time()
        reply = qe.chat(cfg, "Reply with the single word: ready", 8)
        manifest["smoke"] = {"reply": reply.strip()[:40], "sec": round(time.time() - t1, 2)}
        log(f"smoke reply {reply.strip()[:40]!r}")
        manifest["status"] = "FAILED_ACCEPTANCE" if status else ("SERVING" if args.serve else "VERIFIED")
    finally:
        out = ROOT_DIR / "deploy" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{model}_{quant}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8", newline="\n")
        log(f"manifest {out.relative_to(ROOT_DIR).as_posix()} status={manifest.get('status', 'ERROR')}")
        if not (args.serve and status == 0):
            server.terminate()
            server.wait(timeout=30)
    if args.serve and status == 0:
        log(f"serving at {manifest['server']['url']} (Ctrl+C to stop)")
        server.wait()
    return status


if __name__ == "__main__":
    sys.exit(main())
