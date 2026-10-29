"""Decode-speed model: predict generation tokens/s from memory bandwidth instead of measuring.

Token generation reads every weight and the KV cache once per token, so it is
memory-bandwidth bound:
    bytes/token  B = GGUF size + f16 KV bytes/token x depth (ctx)
    1-parameter  tps = BW_eff / B                      (BW_eff = efficiency x peak bandwidth)
    2-parameter  1/tps = B / BW_eff + t0               (t0: fixed per-token overhead)
Fits use full-GPU tg rows of results/tables/summary.csv (partial offload excluded).
Leave-one-hardware-out: each device is predicted from the other devices' efficiency
(BW_eff / published peak) times its own published peak bandwidth
(config/hardware_specs.yaml; every published value is evaluated when sources disagree).

Outputs results/tables/perf_model.csv, perf_model_points.csv, figures/perf_model.png.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from common import ROOT_DIR, load_config  # noqa: E402

GB = 1e9


def kv_bytes_per_token(c: dict) -> int:
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c.get("num_key_value_heads", c["num_attention_heads"]) * head_dim * 2


def decode_points(summary: pd.DataFrame, model_cfgs: dict[str, dict]) -> pd.DataFrame:
    s = summary[summary.generation_throughput_tps.notna() & summary.model.isin(model_cfgs)]
    if "partial_offload" in s:
        s = s[~s.partial_offload.astype(bool)]
    kv = [kv_bytes_per_token(model_cfgs[m]) * ctx for m, ctx in zip(s.model, s.ctx)]
    return s.assign(bytes_per_token=s.model_size_bytes + kv)[
        ["hw_id", "model", "ctx", "quant", "bytes_per_token", "generation_throughput_tps"]].reset_index(drop=True)


def fit_1p(b: np.ndarray, tps: np.ndarray) -> float:
    """BW_eff (bytes/s) minimizing squared log error of tps = BW_eff / b."""
    return float(np.exp(np.mean(np.log(tps * b))))


def fit_2p(b: np.ndarray, tps: np.ndarray) -> tuple[float, float]:
    """(BW_eff bytes/s, t0 s) from least squares on 1/tps = b / BW_eff + t0."""
    (a, t0), *_ = np.linalg.lstsq(np.c_[b, np.ones_like(b)], 1 / tps, rcond=None)
    return float(1 / a), float(t0)


def ape(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / true


def evaluate(points: pd.DataFrame, specs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pts = [], []
    fits = {}
    for hw, d in points.groupby("hw_id"):
        b, tps = d.bytes_per_token.to_numpy(float), d.generation_throughput_tps.to_numpy(float)
        bw1 = fit_1p(b, tps)
        bw2, t0 = fit_2p(b, tps)
        fits[hw] = bw1
        p1, p2 = bw1 / b, 1 / (b / bw2 + t0)
        for spec in specs.get(hw, {}).get("bandwidth_gbs", [np.nan]):
            rows.append({"hw_id": hw, "n": len(d), "peak_bw_gbs": spec, "bw_eff_gbs": bw1 / GB,
                         "efficiency": bw1 / GB / spec, "mape_1p": ape(p1, tps).mean(), "max_ape_1p": ape(p1, tps).max(),
                         "bw_eff_2p_gbs": bw2 / GB, "t0_ms": t0 * 1e3,
                         "mape_2p": ape(p2, tps).mean(), "max_ape_2p": ape(p2, tps).max()})
        pts.append(d.assign(pred_1p=p1, pred_2p=p2))
    out = pd.DataFrame(rows)
    # leave-one-hardware-out: efficiency from the other devices x own published peak bandwidth
    loho = []
    for r in out.itertuples(index=False):
        others = out[(out.hw_id != r.hw_id) & out.efficiency.notna()]
        if others.empty or np.isnan(r.peak_bw_gbs):
            loho.append((np.nan, np.nan))
            continue
        d = points[points.hw_id == r.hw_id]
        b, tps = d.bytes_per_token.to_numpy(float), d.generation_throughput_tps.to_numpy(float)
        errs = [ape(eff * r.peak_bw_gbs * GB / b, tps).mean() for eff in others.efficiency]
        loho.append((min(errs), max(errs)))
    out["loho_mape_min"], out["loho_mape_max"] = zip(*loho) if loho else ([], [])
    return out, pd.concat(pts, ignore_index=True) if pts else pd.DataFrame()


def predict_tps(peak_bw_gbs: float, bytes_per_token: float, efficiencies: list[float]) -> tuple[float, float]:
    """Range of predicted tokens/s on unmeasured hardware from the measured efficiencies."""
    v = [e * peak_bw_gbs * GB / bytes_per_token for e in efficiencies]
    return min(v), max(v)


def figure(pts: pd.DataFrame, out) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (hw, model), d in pts.groupby(["hw_id", "model"]):
        ax.scatter(d.generation_throughput_tps, d.pred_2p, s=18, label=f"{hw[:24]} / {model.replace('-Instruct', '')}")
    lim = [0, max(pts.generation_throughput_tps.max(), pts.pred_2p.max()) * 1.05]
    ax.plot(lim, lim, color="grey", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Measured generation tokens/s", ylabel="Predicted (2-parameter model)",
           title="Decode speed from bytes per token")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    summary = pd.read_csv(ROOT_DIR / paths["summary_table"])
    model_cfgs = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted((ROOT_DIR / "config" / "model_configs").glob("*.json"))}
    specs = yaml.safe_load((ROOT_DIR / "config" / "hardware_specs.yaml").read_text(encoding="utf-8")) or {}
    fit, pts = evaluate(decode_points(summary, model_cfgs), specs)
    tables, figs = ROOT_DIR / paths["tables_dir"], ROOT_DIR / paths["figures_dir"]
    fit.to_csv(tables / "perf_model.csv", index=False)
    pts.to_csv(tables / "perf_model_points.csv", index=False)
    if not pts.empty:
        figure(pts, figs / "perf_model.png")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(fit.to_string(index=False))


if __name__ == "__main__":
    main()
