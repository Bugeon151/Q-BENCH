"""Diagnose a collapsed low-bit quantization and re-quantize it with an importance matrix.

For each target (model, quant) of config imatrix.targets:
  1. kernel check  llama-perplexity on the CPU only (-ngl 0, first `check_chunks` chunks) against
                   the running estimate after the same chunks in the GPU run's log: the same
                   value means the loss is in the quantized weights, not in a GPU kernel
  2. imatrix       llama-imatrix on wikitext-2 *train* from the F16 model (the test split used for
                   perplexity stays unseen) -> models/imatrix/<model>.gguf
  3. requantize    llama-quantize --imatrix -> models/quantized/<model>-<quant>-imat.gguf
The new files are ordinary quant levels named `<quant>-imat`: evaluate them with
perplexity_eval.py / quality_eval.py --quants Q2_K-imat ...

    python src/imatrix_fix.py            -> data/quality/<hw_id>/kernel_check.csv
"""

from __future__ import annotations

import csv
import re
import subprocess
import time
from pathlib import Path

from common import ROOT_DIR, detect_hw, gguf_path, load_config, make_logger, resolve_bin
from perplexity_eval import parse_ppl

RUNNING_RE = re.compile(r"\[(\d+)\]([0-9.]+(?:e[+-]?\d+)?)")


def running_ppl(log_text: str, chunk: int) -> float | None:
    """Running PPL estimate after `chunk` chunks from llama-perplexity output ('[20]22505.8397,')."""
    for n, v in RUNNING_RE.findall(log_text):
        if int(n) == chunk:
            return float(v)
    return None


def run(cmd: list[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    log_path.write_text(out, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"FAIL: {Path(cmd[0]).name} exit {proc.returncode}; see {log_path}\n{out[-2000:]}")
    return out


def main() -> None:
    cfg = load_config()
    ic, pcfg = cfg["imatrix"], cfg["perplexity"]
    log = make_logger(cfg, "imatrix")
    hw = detect_hw()
    logs = ROOT_DIR / cfg["paths"]["logs_dir"]
    test_text = ROOT_DIR / pcfg["dataset_file"]
    train_text = ROOT_DIR / ic["train_file"]
    if not train_text.is_file():
        raise SystemExit(f"FAIL: {train_text} missing (extracted with the perplexity dataset)")
    ppl_bin, imat_bin = resolve_bin(pcfg["llama_perplexity_bin"]), resolve_bin(ic["llama_imatrix_bin"])
    quant_bin = resolve_bin(cfg["benchmark"]["llama_quantize_bin"])
    out_csv = ROOT_DIR / cfg["paths"]["quality_dir"] / hw["hw_id"] / "kernel_check.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            done = {(r["model"], r["quant"]) for r in csv.DictReader(f)}
    n = ic["check_chunks"]

    for model, quant in ic["targets"]:
        tag = f"{model}/{quant}"
        src = gguf_path(cfg, model, quant)
        # 1. kernel check
        gpu_log = logs / f"perplexity_{model}_{quant}.log"
        if (model, quant) not in done and gpu_log.exists() and src.is_file():
            gpu = running_ppl(gpu_log.read_text(encoding="utf-8", errors="replace"), n)
            t0 = time.time()
            cpu, _ = parse_ppl(run([str(ppl_bin), "-m", str(src), "-f", str(test_text), "-c", str(pcfg["ctx"]),
                                    "--chunks", str(n), "-ngl", "0"], logs / f"kernel_check_{model}_{quant}.log"))
            row = {"hw_id": hw["hw_id"], "model": model, "quant": quant, "chunks": n, "ppl_gpu": gpu, "ppl_cpu": cpu,
                   "cpu_over_gpu": cpu / gpu if gpu else None, "elapsed_sec": round(time.time() - t0, 1)}
            new = not out_csv.exists()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            log(f"{tag}: PPL after {n} chunks GPU {gpu} vs CPU {cpu:.4f}")
        # 2. imatrix from F16 on the train split
        imat = ROOT_DIR / ic["dir"] / f"{model}.gguf"
        if not imat.exists():
            imat.parent.mkdir(parents=True, exist_ok=True)
            log(f"{model}: llama-imatrix on {ic['train_file']} ({ic['chunks']} chunks)")
            run([str(imat_bin), "-m", str(gguf_path(cfg, model, "F16")), "-f", str(train_text), "-o", str(imat),
                 "--chunks", str(ic["chunks"]), "-c", str(pcfg["ctx"]), "-ngl", str(pcfg["n_gpu_layers"]), "--no-ppl"],
                logs / f"imatrix_{model}.log")
        # 3. requantize
        dst = gguf_path(cfg, model, f"{quant}-imat")
        if not dst.exists():
            log(f"{tag}: llama-quantize --imatrix -> {dst.name}")
            part = dst.with_suffix(".part")
            run([str(quant_bin), "--imatrix", str(imat), str(gguf_path(cfg, model, "F16")), str(part), quant],
                logs / f"quantize_{model}_{quant}-imat.log")
            part.replace(dst)
    log("imatrix complete")


if __name__ == "__main__":
    main()
