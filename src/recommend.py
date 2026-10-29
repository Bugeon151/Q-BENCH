"""Pick a model and quantization for a GPU memory budget from measured results.

Reads results/tables/{summary,quality,gate}.csv (run src/analyze.py first).
For each (model, quant) with complete quality results:
  need_gib  GPU memory needed (GiB) at --ctx: the measured peak VRAM of a full-GPU run when one
            exists (any CUDA hw_id), else GGUF size + f16 KV cache + overhead from
            config/model_configs/<model>.json (marked "estimated")
  fits      "gpu" if need_gib <= --vram-gb, "offload" if it fits in VRAM + --ram-gb, else "no"
  tps       generation tokens/s measured on --hw (blank if not measured there)
  tps_pred  with --bandwidth-gbs (unmeasured hardware): predicted range from src/perfmodel.py,
            efficiency of the measured devices x the given peak bandwidth (full-GPU configs only)
  quality   mean accuracy over the quality tasks; gate = regression verdict vs the model's reference
            (F16; Q8_0 when F16 was not evaluated)
  flips     share of answers whose correctness differs from the reference (src/consistency.py), if measured
  capacity_server / capacity_interactive  concurrent users within the latency SLAs on --hw
            (src/loadtest.py), if measured
Candidates must fit, must not FAIL the gate, and must reach --min-tps (measured speed, else the
low end of the predicted range). Ranking: quality, then speed.

    python src/recommend.py --vram-gb 8 [--ram-gb 32] [--min-tps 30] [--ctx 4096] [--hw HW_ID]
    python src/recommend.py --vram-gb 24 --bandwidth-gbs 1008 --min-tps 60   # unmeasured GPU
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import GIB, ROOT_DIR, load_config
from perfmodel import kv_bytes_per_token, predict_tps  # noqa: F401 (re-exported)

TASKS = ["mmlu", "gsm8k", "humaneval"]


def load_model_cfgs(root: Path = ROOT_DIR) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "config" / "model_configs").glob("*.json"))}


def candidates(summary: pd.DataFrame, quality: pd.DataFrame, gates: pd.DataFrame, model_cfgs: dict[str, dict],
               ctx: int, n_gen: int, overhead_gb: float, hw: str | None,
               bandwidth_gbs: float | None = None, efficiencies: list[float] | None = None) -> pd.DataFrame:
    tasks = [t for t in TASKS if t in quality]
    q = quality.dropna(subset=tasks).assign(quality=lambda d: d[tasks].mean(axis=1))
    q = q.drop_duplicates(["model", "quant"])[["model", "quant", "quality", *tasks]]
    s = summary[summary.ctx == ctx]
    gpu = s[(~s.partial_offload.astype(bool)) & s.peak_vram_bytes.notna()]
    measured = gpu.groupby(["model", "quant"]).peak_vram_bytes.max() / GIB
    size = summary.groupby(["model", "quant"]).model_size_bytes.first()
    refs = set(zip(gates.model, gates.reference)) if "reference" in gates and len(gates) else set()
    refs |= {(m, "F16") for m in q.model if not any(rm == m for rm, _ in refs)}
    rows = []
    for r in q.itertuples(index=False):
        key = (r.model, r.quant)
        if key in measured.index:
            need, basis = float(measured[key]), "measured"
        elif key in size.index and r.model in model_cfgs:
            kv = kv_bytes_per_token(model_cfgs[r.model]) * (ctx + n_gen)
            need, basis = (size[key] + kv) / GIB + overhead_gb, "estimated"
        else:
            continue
        tps = s[(s.hw_id == hw) & (s.model == r.model) & (s.quant == r.quant)].generation_throughput_tps
        offl = s[(s.hw_id == hw) & (s.model == r.model) & (s.quant == r.quant)].partial_offload
        g = gates[(gates.model == r.model) & (gates.candidate == r.quant) & (gates.task == "overall")].verdict
        row = {"model": r.model, "quant": r.quant, "need_gib": need, "need_basis": basis,
               "quality": r.quality, **{t: getattr(r, t) for t in tasks},
               "gate": "REFERENCE" if (r.model, r.quant) in refs else (g.iloc[0] if len(g) else ""),
               "tps": float(tps.iloc[0]) if len(tps) else float("nan"),
               "tps_offloaded": bool(offl.iloc[0]) if len(offl) else False}
        if bandwidth_gbs and efficiencies and key in size.index and r.model in model_cfgs:
            b_tok = size[key] + kv_bytes_per_token(model_cfgs[r.model]) * ctx
            row["tps_pred_low"], row["tps_pred_high"] = predict_tps(bandwidth_gbs, b_tok, efficiencies)
        rows.append(row)
    return pd.DataFrame(rows)


def recommend(c: pd.DataFrame, vram_gb: float, ram_gb: float = 0.0, min_tps: float = 0.0,
              allow_offload: bool = False) -> pd.DataFrame:
    if c.empty:
        return c
    c = c.assign(fits=["gpu" if n <= vram_gb else "offload" if n <= vram_gb + ram_gb else "no" for n in c.need_gib])
    ok = c.fits.eq("gpu") | (allow_offload & c.fits.eq("offload"))
    ok &= c.gate.ne("FAIL")
    speed = c.tps.fillna(c["tps_pred_low"]) if "tps_pred_low" in c else c.tps
    ok &= speed.isna() | (speed >= min_tps)
    return c[ok].assign(_speed=speed[ok]).sort_values(["quality", "_speed"], ascending=[False, False]) \
        .drop(columns="_speed").reset_index(drop=True)


def annotate(rec: pd.DataFrame, tables: Path, hw: str | None) -> pd.DataFrame:
    """Adds answer flips vs F16 (flips.csv) and concurrent users within the server SLA on hw
    (capacity.csv) when those tables exist."""
    if rec.empty:
        return rec
    f = tables / "flips.csv"
    if f.exists():
        fl = pd.read_csv(f)
        fl = fl[(fl.kind == "vs_ref") & (fl.task == "all")].drop_duplicates(["model", "quant"])
        rec = rec.merge(fl[["model", "quant", "flips"]], on=["model", "quant"], how="left")
        refs = set(zip(fl.model, fl.reference)) if "reference" in fl else {(m, "F16") for m in rec.model}
        rec.loc[[(m, q) in refs for m, q in zip(rec.model, rec.quant)], "flips"] = 0.0
    c = tables / "capacity.csv"
    if c.exists() and hw:
        cap = pd.read_csv(c)
        cap = cap[cap.hw_id == hw][["model", "quant", "capacity_server", "capacity_interactive"]]
        rec = rec.merge(cap, on=["model", "quant"], how="left")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vram-gb", type=float, required=True,
                    help="GPU memory for the model, GiB (a GPU sold as 8 GB has 8 GiB)")
    ap.add_argument("--ram-gb", type=float, default=0.0, help="host RAM usable for CPU offload, GiB")
    ap.add_argument("--min-tps", type=float, default=0.0, help="minimum generation tokens/s (if measured)")
    ap.add_argument("--ctx", type=int, help="context length (default: inference.context_length)")
    ap.add_argument("--hw", help="hw_id whose measured speeds to report (default: first CUDA hw)")
    ap.add_argument("--bandwidth-gbs", type=float,
                    help="peak memory bandwidth of unmeasured target hardware: predict speed instead")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    tables = ROOT_DIR / cfg["paths"]["tables_dir"]
    summary = pd.read_csv(ROOT_DIR / cfg["paths"]["summary_table"])
    quality = pd.read_csv(tables / "quality.csv")
    gates = pd.read_csv(tables / "gate.csv") if (tables / "gate.csv").exists() else pd.DataFrame(
        columns=["model", "candidate", "task", "verdict"])
    ctx = args.ctx or cfg["inference"]["context_length"]
    if args.bandwidth_gbs:  # unmeasured target: measured speeds of other devices do not apply
        hw = args.hw
        pm = tables / "perf_model.csv"
        effs = sorted(set(pd.read_csv(pm).efficiency.dropna().round(4))) if pm.exists() else []
        if not effs:
            raise SystemExit("run src/perfmodel.py first (results/tables/perf_model.csv)")
    else:
        hw = args.hw or next((h for h in summary.hw_id.unique() if h.startswith("nvidia")), None)
        effs = None
    c = candidates(summary, quality, gates, load_model_cfgs(), ctx, cfg["inference"]["max_output_tokens"],
                   cfg["memory_budget"]["overhead_gb"], hw, args.bandwidth_gbs, effs)
    rec = annotate(recommend(c, args.vram_gb, args.ram_gb, args.min_tps, allow_offload=args.ram_gb > 0), tables, hw)
    src = (f"speeds predicted for {args.bandwidth_gbs} GB/s from efficiencies {effs}" if args.bandwidth_gbs
           else f"speeds measured on {hw}")
    print(f"budget: {args.vram_gb} GiB VRAM + {args.ram_gb} GiB RAM, ctx {ctx}, min {args.min_tps} tok/s; {src}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(rec.head(args.top).to_string(index=False) if not rec.empty else "no configuration fits")


if __name__ == "__main__":
    main()
