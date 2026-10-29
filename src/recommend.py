"""Pick a model and quantization for a GPU memory budget from measured results.

Reads results/tables/{summary,quality,gate}.csv (run src/analyze.py first).
For each (model, quant) with complete quality results:
  need_gib  GPU memory needed (GiB) at --ctx: the measured peak VRAM of a full-GPU run when one
            exists (any CUDA hw_id), else GGUF size + f16 KV cache + overhead from
            config/model_configs/<model>.json (marked "estimated")
  fits      "gpu" if need_gib <= --vram-gb, "offload" if it fits in VRAM + --ram-gb, else "no"
  tps       generation tokens/s measured on --hw (blank if not measured there)
  quality   mean accuracy over the quality tasks; gate = regression verdict vs F16
Candidates must fit, must not FAIL the gate, and must reach --min-tps when speed is known.
Ranking: quality, then speed. Speeds are only reported for hardware that was measured.

    python src/recommend.py --vram-gb 8 [--ram-gb 32] [--min-tps 30] [--ctx 4096] [--hw HW_ID]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import GIB, ROOT_DIR, load_config

TASKS = ["mmlu", "gsm8k", "humaneval"]


def kv_bytes_per_token(model_cfg: dict) -> int:
    """f16 K+V bytes per token from an HF config.json."""
    head_dim = model_cfg.get("head_dim") or model_cfg["hidden_size"] // model_cfg["num_attention_heads"]
    n_kv = model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"])
    return 2 * model_cfg["num_hidden_layers"] * n_kv * head_dim * 2


def load_model_cfgs(root: Path = ROOT_DIR) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "config" / "model_configs").glob("*.json"))}


def candidates(summary: pd.DataFrame, quality: pd.DataFrame, gates: pd.DataFrame, model_cfgs: dict[str, dict],
               ctx: int, n_gen: int, overhead_gb: float, hw: str | None) -> pd.DataFrame:
    tasks = [t for t in TASKS if t in quality]
    q = quality.dropna(subset=tasks).assign(quality=lambda d: d[tasks].mean(axis=1))
    q = q.drop_duplicates(["model", "quant"])[["model", "quant", "quality", *tasks]]
    s = summary[summary.ctx == ctx]
    gpu = s[(~s.partial_offload.astype(bool)) & s.peak_vram_bytes.notna()]
    measured = gpu.groupby(["model", "quant"]).peak_vram_bytes.max() / GIB
    size = summary.groupby(["model", "quant"]).model_size_bytes.first()
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
        rows.append({"model": r.model, "quant": r.quant, "need_gib": need, "need_basis": basis,
                     "quality": r.quality, **{t: getattr(r, t) for t in tasks},
                     "gate": "REFERENCE" if r.quant == "F16" else (g.iloc[0] if len(g) else ""),
                     "tps": float(tps.iloc[0]) if len(tps) else float("nan"),
                     "tps_offloaded": bool(offl.iloc[0]) if len(offl) else False})
    return pd.DataFrame(rows)


def recommend(c: pd.DataFrame, vram_gb: float, ram_gb: float = 0.0, min_tps: float = 0.0,
              allow_offload: bool = False) -> pd.DataFrame:
    if c.empty:
        return c
    c = c.assign(fits=["gpu" if n <= vram_gb else "offload" if n <= vram_gb + ram_gb else "no" for n in c.need_gib])
    ok = c.fits.eq("gpu") | (allow_offload & c.fits.eq("offload"))
    ok &= c.gate.ne("FAIL")
    ok &= c.tps.isna() | (c.tps >= min_tps)
    return c[ok].sort_values(["quality", "tps"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vram-gb", type=float, required=True,
                    help="GPU memory for the model, GiB (a GPU sold as 8 GB has 8 GiB)")
    ap.add_argument("--ram-gb", type=float, default=0.0, help="host RAM usable for CPU offload, GiB")
    ap.add_argument("--min-tps", type=float, default=0.0, help="minimum generation tokens/s (if measured)")
    ap.add_argument("--ctx", type=int, help="context length (default: inference.context_length)")
    ap.add_argument("--hw", help="hw_id whose measured speeds to report (default: first CUDA hw)")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    tables = ROOT_DIR / cfg["paths"]["tables_dir"]
    summary = pd.read_csv(ROOT_DIR / cfg["paths"]["summary_table"])
    quality = pd.read_csv(tables / "quality.csv")
    gates = pd.read_csv(tables / "gate.csv") if (tables / "gate.csv").exists() else pd.DataFrame(
        columns=["model", "candidate", "task", "verdict"])
    ctx = args.ctx or cfg["inference"]["context_length"]
    hw = args.hw or next((h for h in summary.hw_id.unique() if h.startswith("nvidia")), None)
    c = candidates(summary, quality, gates, load_model_cfgs(), ctx, cfg["inference"]["max_output_tokens"],
                   cfg["memory_budget"]["overhead_gb"], hw)
    rec = recommend(c, args.vram_gb, args.ram_gb, args.min_tps, allow_offload=args.ram_gb > 0)
    print(f"budget: {args.vram_gb} GB VRAM + {args.ram_gb} GB RAM, ctx {ctx}, min {args.min_tps} tok/s; "
          f"speeds measured on {hw}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(rec.head(args.top).to_string(index=False) if not rec.empty else "no configuration fits")


if __name__ == "__main__":
    main()
