"""Capacity: how many concurrent users one machine serves within a latency SLA.

Starts llama-server for (model, quant) with loadtest.slots slots (KV cache slots x ctx_per_slot),
then for each concurrency level c runs a closed loop of c users: each user sends a chat request
(GSM8K questions from the fixed quality sample, natural output length up to max_tokens), reads
the streamed reply and sends the next, until max(min_requests, c x requests_per_user) replies are
in. Per request (streamed; final chunk carries usage):
  ttft_s   time to first content token, queueing included
  tpot_s   (latency - ttft) / (output tokens - 1)
  tokens   usage.completion_tokens
Per level: p50 / p95 / p99 of ttft and tpot, aggregate output tokens/s.
SLA: MLPerf Inference v5.0 Llama-2-70B limits (MLCommons 2025): server TTFT 2 s / TPOT 200 ms,
interactive TTFT 450 ms / TPOT 40 ms, applied at loadtest.sla_percentile.
capacity_<sla> = largest level that meets the SLA (levels above the first server-SLA miss are
not run: latency only grows with load).

    python src/loadtest.py [--model M --quant Q] [--force]     (default: loadtest.targets)
    python src/loadtest.py --summarize-only
Per-request rows: data/capacity/<hw_id>/<model>_<quant>.csv; summary: results/tables/capacity.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import quality_eval as qe
from common import ROOT_DIR, check_budget, detect_hw, gguf_path, load_config, make_logger

PCTS = (50, 95, 99)


def stream_chat(url: str, prompt: str, max_tokens: int, timeout: float, opener=qe._NO_PROXY) -> dict:
    """One streamed chat completion; timings measured on the client."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0, ttft, chunks, tokens = time.perf_counter(), None, 0, None
    with opener.open(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            if j.get("usage"):
                tokens = j["usage"].get("completion_tokens")
    latency = time.perf_counter() - t0
    tokens = tokens if tokens is not None else chunks
    ttft = latency if ttft is None else ttft
    return {"ttft_s": ttft, "latency_s": latency, "tokens": tokens,
            "tpot_s": (latency - ttft) / (tokens - 1) if tokens > 1 else math.nan}


def run_level(send, prompts: list[str], level: int, n_requests: int) -> tuple[list[dict], float]:
    """Closed loop: `level` workers, each sends its next request when the previous reply ends."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(level) as ex:
        rows = list(ex.map(send, [prompts[i % len(prompts)] for i in range(n_requests)]))
    return rows, time.perf_counter() - t0


def meets(row: pd.Series, sla: dict, pct: int) -> bool:
    return bool(row[f"ttft_p{pct}_s"] <= sla["ttft_s"] and row[f"tpot_p{pct}_s"] <= sla["tpot_s"])


def summarize(raw: pd.DataFrame, lt: dict) -> pd.DataFrame:
    """raw: per-request rows (hw_id, model, quant, level, level_wall_s, ttft_s, tpot_s, tokens)."""
    out = []
    for key, d in raw.groupby(["hw_id", "model", "quant", "level"]):
        row = dict(zip(["hw_id", "model", "quant", "level"], key), n=len(d),
                   out_tps_total=d.tokens.sum() / d.level_wall_s.iloc[0], tokens_mean=d.tokens.mean())
        for p in PCTS:
            row[f"ttft_p{p}_s"] = float(np.percentile(d.ttft_s, p))
            row[f"tpot_p{p}_s"] = float(np.nanpercentile(d.tpot_s, p)) if d.tpot_s.notna().any() else math.nan
        row["user_tps_p50"] = 1 / row["tpot_p50_s"] if row["tpot_p50_s"] > 0 else math.nan
        out.append(row)
    s = pd.DataFrame(out)
    if s.empty:
        return s
    pct = lt["sla_percentile"]
    for name, sla in lt["sla"].items():
        s[f"meets_{name}"] = s.apply(meets, axis=1, sla=sla, pct=pct)
    return s.sort_values(["hw_id", "model", "quant", "level"]).reset_index(drop=True)


def capacity(summary: pd.DataFrame, slas) -> pd.DataFrame:
    """Largest level meeting each SLA per (hw, model, quant); 0 if none."""
    rows = []
    for key, d in summary.groupby(["hw_id", "model", "quant"]):
        r = dict(zip(["hw_id", "model", "quant"], key))
        for name in slas:
            ok = d[d[f"meets_{name}"]]
            r[f"capacity_{name}"] = int(ok.level.max()) if len(ok) else 0
            r[f"out_tps_total_at_{name}"] = float(ok.loc[ok.level.idxmax(), "out_tps_total"]) if len(ok) else math.nan
        r["max_out_tps_total"] = float(d.out_tps_total.max())
        rows.append(r)
    return pd.DataFrame(rows)


def raw_path(cfg: dict, hw_id: str, model: str, quant: str) -> Path:
    return ROOT_DIR / cfg["paths"]["capacity_dir"] / hw_id / f"{model}_{quant}.csv"


def measure(cfg: dict, hw: dict, model: str, quant: str, log, force: bool = False) -> pd.DataFrame:
    """Runs (or reuses) the sweep; returns the per-level summary for (model, quant) on this machine."""
    lt = cfg["loadtest"]
    out = raw_path(cfg, hw["hw_id"], model, quant)
    if out.exists() and not force:
        log(f"{model}/{quant}: reuse {out.relative_to(ROOT_DIR).as_posix()}")
        return summarize(pd.read_csv(out), lt)
    path = gguf_path(cfg, model, quant)
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    ok, why = check_budget(cfg, model, quant, lt["slots"] * lt["ctx_per_slot"], hw, allow_host_offload=True)
    if not ok:
        raise SystemExit(f"{model}/{quant}: over memory budget ({why})")
    sample = qe.load_sample(cfg, log)
    prompts = [qe.PROMPTS["gsm8k"](row) for row in qe.load_items(sample, "gsm8k")]
    url = f"http://127.0.0.1:{cfg['quality']['port']}/v1/chat/completions"
    timeout = cfg["quality"]["request_timeout_sec"]

    def send(p: str) -> dict:
        return stream_chat(url, p, lt["max_tokens"], timeout)

    log(f"{model}/{quant}: starting llama-server, {lt['slots']} slots x {lt['ctx_per_slot']} ctx ({why})")
    server = qe.start_server(cfg, path, ROOT_DIR / cfg["paths"]["logs_dir"] / f"loadtest_{model}_{quant}.log",
                             n_parallel=lt["slots"], ctx_per_slot=lt["ctx_per_slot"])
    rows, offset = [], 0
    try:
        run_level(send, prompts[-lt["warmup_requests"]:], 1, lt["warmup_requests"])
        for level in lt["levels"]:
            n = max(lt["min_requests"], level * lt["requests_per_user"])
            res, wall = run_level(send, prompts[offset:] + prompts[:offset], level, n)
            offset = (offset + n) % len(prompts)
            rows += [{"hw_id": hw["hw_id"], "model": model, "quant": quant, "level": level,
                      "level_wall_s": wall, **r} for r in res]
            s = summarize(pd.DataFrame(rows), lt).iloc[-1]
            log(f"{model}/{quant} c={level}: TTFT p50 {s.ttft_p50_s:.2f}s p95 {s.ttft_p95_s:.2f}s, "
                f"TPOT p50 {s.tpot_p50_s * 1000:.0f}ms p95 {s.tpot_p95_s * 1000:.0f}ms, "
                f"{s.out_tps_total:.1f} tok/s total, server SLA {'OK' if s.meets_server else 'MISS'}")
            if not s.meets_server:
                break
    finally:
        server.terminate()
        server.wait(timeout=30)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    return summarize(pd.DataFrame(rows), lt)


def write_tables(cfg: dict) -> pd.DataFrame:
    files = sorted((ROOT_DIR / cfg["paths"]["capacity_dir"]).glob("*/*.csv"))
    if not files:
        return pd.DataFrame()
    lt = cfg["loadtest"]
    s = summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True), lt)
    tables = ROOT_DIR / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    s.to_csv(tables / "capacity_levels.csv", index=False)
    cap = capacity(s, lt["sla"])
    cap.to_csv(tables / "capacity.csv", index=False)
    return cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model")
    ap.add_argument("--quant")
    ap.add_argument("--force", action="store_true", help="re-measure even if results exist")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    log = make_logger(cfg, "loadtest")
    if not args.summarize_only:
        hw = detect_hw()
        targets = [(args.model, args.quant)] if args.model and args.quant else cfg["loadtest"]["targets"]
        for model, quant in targets:
            measure(cfg, hw, model, quant, log, args.force)
    cap = write_tables(cfg)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(cap.to_string(index=False) if not cap.empty else "no capacity results")


if __name__ == "__main__":
    main()
