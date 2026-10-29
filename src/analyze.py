"""Pipeline stage: Data Processing + Statistical Analysis + Visualization + Report.

Inputs
  data/raw/benchmark_raw.csv            v0.1 legacy schema (hw_id "legacy_v0.1")
  data/raw/<hw_id>/benchmark_raw.csv    current schema, one dir per hardware
  data/raw/<hw_id>/gpu_telemetry.jsonl  GPU temperature / SM clock / power per measurement (CUDA)
  data/quality/<hw_id>/{perplexity,quality}.csv   v0.2 quality results
  data/quality/<hw_id>/<model>/<quant>/<task>.jsonl  per-item results (paired tests vs F16)
Outputs
  data/processed/benchmark_stats.csv    per (hw_id, model, ctx, quant, test_type)
  results/tables/summary.csv            per (hw_id, model, ctx, quant), CV warnings
  results/tables/quality.csv            PPL / accuracy (Wilson 95% CI) / retention / McNemar p vs F16
  results/tables/hardware_comparison.csv
  results/tables/gate.csv               regression gate per quant vs F16 (src/gate.py)
  results/figures/<hw_id>/<model>/      PRD figures 1-5, 7 and v0.3 context plots
  results/figures/quality/              PRD figure 6
  results/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import gate  # noqa: E402
from common import ROOT_DIR, load_config  # noqa: E402

QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q3_K_M-imat", "Q2_K", "Q2_K-imat"]  # -imat: src/imatrix_fix.py
LEGACY_HW_ID = "legacy_v0.1"
LEGACY_MODEL = "Qwen2.5-7B-Instruct"
KEYS = ["hw_id", "model", "ctx", "quant"]
GB = 1e9
TASKS = ["mmlu", "gsm8k", "humaneval"]


def ordered(quants) -> list[str]:
    return [q for q in QUANT_ORDER if q in set(quants)]


def ci95(series: pd.Series) -> float:
    n = len(series)
    if n < 2:
        return 0.0
    sem = series.std(ddof=1) / (n**0.5)
    return float(sem * stats.t.ppf(0.975, df=n - 1))

# --------------------------------------------------------------------------- loading


def load_raw(cfg: dict) -> pd.DataFrame:
    frames = []
    legacy = ROOT_DIR / cfg["paths"]["raw_csv"]
    if legacy.exists():
        df = pd.read_csv(legacy)
        df["hw_id"], df["model"], df["ctx"] = LEGACY_HW_ID, LEGACY_MODEL, cfg["inference"]["context_length"]
        frames.append(df)
    frames += [pd.read_csv(p) for p in sorted((ROOT_DIR / cfg["paths"]["raw_dir"]).glob("*/benchmark_raw.csv"))]
    if not frames:
        raise SystemExit("No raw data found. Run src/benchmark.py first.")
    raw = pd.concat(frames, ignore_index=True)
    for col in ("peak_vram_bytes", "swap_used_before_bytes", "swap_used_after_bytes"):
        if col not in raw:
            raw[col] = np.nan
    return raw


def load_telemetry(cfg: dict) -> pd.DataFrame:
    files = sorted((ROOT_DIR / cfg["paths"]["raw_dir"]).glob("*/gpu_telemetry.jsonl"))
    recs = [json.loads(line) for f in files for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(recs)


def load_items(cfg: dict) -> pd.DataFrame:
    """Per-item correctness: hw_id, model, quant, task, id, correct."""
    return gate.load_items(ROOT_DIR / cfg["paths"]["quality_dir"])


def wilson_ci(correct: int, n: int) -> tuple[float, float]:
    if not n:
        return np.nan, np.nan
    ci = stats.binomtest(int(correct), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def mcnemar_p(a: pd.Series, b: pd.Series) -> float:
    """Exact two-sided McNemar test on paired per-item correctness (aligned by id)."""
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    return gate.mcnemar_p(both.iloc[:, 0].to_numpy(bool), both.iloc[:, 1].to_numpy(bool))


def load_quality(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    qdir = ROOT_DIR / cfg["paths"]["quality_dir"]
    ppl = [pd.read_csv(p) for p in sorted(qdir.glob("*/perplexity.csv"))]
    acc = [pd.read_csv(p) for p in sorted(qdir.glob("*/quality.csv"))]
    return (pd.concat(ppl, ignore_index=True) if ppl else pd.DataFrame(),
            pd.concat(acc, ignore_index=True) if acc else pd.DataFrame())

# --------------------------------------------------------------------------- statistics


def compute_stats(raw: pd.DataFrame) -> pd.DataFrame:
    # Legacy pg rows include llama-bench default pp512/tg128 tests; keep only true pg rows.
    pg = raw[raw.test_type == "pg"]
    raw = pd.concat([raw[raw.test_type != "pg"], pg[(pg.n_prompt > 0) & (pg.n_gen > 0)]])
    raw = raw.assign(swap_growth_bytes=raw.swap_used_after_bytes - raw.swap_used_before_bytes)
    if "n_gpu_layers" not in raw:  # legacy v0.1 schema
        raw = raw.assign(n_gpu_layers=np.nan)
    by = [*KEYS, "test_type"]
    grouped = raw.groupby(by, sort=False)
    stats_df = grouped.agg(
        latency_ms_mean=("latency_ms", "mean"),
        latency_ms_median=("latency_ms", "median"),
        latency_ms_std=("latency_ms", "std"),
        tokens_per_sec_mean=("tokens_per_sec", "mean"),
        tokens_per_sec_median=("tokens_per_sec", "median"),
        tokens_per_sec_std=("tokens_per_sec", "std"),
        model_size_bytes=("model_size_bytes", "first"),
        peak_mem_bytes=("peak_mem_bytes", "max"),
        peak_vram_bytes=("peak_vram_bytes", "max"),
        swap_growth_bytes=("swap_growth_bytes", "max"),
        n_samples=("latency_ms", "count"),
        partial_offload=("n_gpu_layers", lambda s: s.astype(str).str.startswith("fit:").any()),
    ).reset_index()
    ci = pd.DataFrame({
        "latency_ms_ci95": grouped["latency_ms"].apply(ci95),
        "tokens_per_sec_ci95": grouped["tokens_per_sec"].apply(ci95),
    }).reset_index()
    stats_df = stats_df.merge(ci, on=by)
    stats_df["tokens_per_sec_cv"] = stats_df.tokens_per_sec_std / stats_df.tokens_per_sec_mean
    return sort_keys(stats_df)


def sort_keys(df: pd.DataFrame, by: tuple = ("hw_id", "model", "ctx")) -> pd.DataFrame:
    q = df.quant.map({k: i for i, k in enumerate(QUANT_ORDER)})
    return df.assign(_q=q).sort_values([*by, "_q"], kind="stable").drop(columns="_q").reset_index(drop=True)


def summarize(stats_df: pd.DataFrame, cv_threshold: float) -> pd.DataFrame:
    g = stats_df.groupby(KEYS)
    out = pd.DataFrame({
        "model_size_bytes": g.model_size_bytes.first(),
        "peak_mem_bytes": g.peak_mem_bytes.max(),
    })
    pick = {"pp": [("prompt_throughput_tps", "tokens_per_sec_mean"), ("prompt_throughput_ci95", "tokens_per_sec_ci95"),
                   ("ttft_ms", "latency_ms_mean"), ("ttft_ci95", "latency_ms_ci95")],
            "tg": [("generation_throughput_tps", "tokens_per_sec_mean"), ("generation_throughput_ci95", "tokens_per_sec_ci95")],
            "pg": [("total_latency_ms", "latency_ms_mean"), ("total_latency_ci95", "latency_ms_ci95")]}
    for tt, cols in pick.items():
        sub = stats_df[stats_df.test_type == tt].set_index(KEYS)
        for dst, src in cols:
            out[dst] = sub[src]
    order = ["prompt_throughput_tps", "generation_throughput_tps", "ttft_ms", "total_latency_ms"]
    out = out[["model_size_bytes", "peak_mem_bytes", *order,
               *[c for c in out.columns if c.endswith("_ci95")]]]
    out["peak_vram_bytes"] = g.peak_vram_bytes.max()
    out["swap_growth_bytes"] = g.swap_growth_bytes.max()
    out["partial_offload"] = g.partial_offload.any()  # GPU+CPU split via llama-bench -fitt
    cv = stats_df.pivot_table(index=KEYS, columns="test_type", values="tokens_per_sec_cv")
    for tt in ("pp", "tg", "pg"):
        out[f"{tt}_cv"] = cv[tt] if tt in cv else np.nan
    out["cv_warning"] = [
        ";".join(f"{tt}:{row[f'{tt}_cv'] * 100:.1f}%" for tt in ("pp", "tg", "pg")
                 if pd.notna(row[f"{tt}_cv"]) and row[f"{tt}_cv"] > cv_threshold)
        for _, row in out.iterrows()
    ]
    return sort_keys(out.reset_index())


def quality_table(ppl: pd.DataFrame, acc: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (quality_hw_id, model, quant): PPL, accuracy per task with Wilson 95% CI,
    retention vs the reference (F16, else Q8_0: gate.pick_reference) and exact McNemar p-value vs
    the reference on the same items."""
    parts = []
    if not acc.empty:
        acc = acc[acc.n == acc.expected_n].copy()  # complete tasks only
        ci = [wilson_ci(c, n) for c, n in zip(acc.correct, acc.n)]
        acc["ci_low"], acc["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
        idx = ["hw_id", "model", "quant"]
        parts.append(acc.pivot_table(index=idx, columns="task", values="accuracy"))
        for col in ("ci_low", "ci_high"):
            w = acc.pivot_table(index=idx, columns="task", values=col)
            w.columns = [f"{t}_{col}" for t in w.columns]
            parts.append(w)
        if items is not None and not items.empty:
            done = set(zip(acc.hw_id, acc.model, acc.quant, acc.task))
            keep = np.array([k in done for k in zip(items.hw_id, items.model, items.quant, items.task)], dtype=bool)
            items = items[keep]
            per = items.set_index("id")
            pv = {}
            refq = {k: gate.pick_reference(d.quant) for k, d in items.groupby(["hw_id", "model"])}
            for (hw, model, task), g in per.groupby(["hw_id", "model", "task"]):
                ref = g[g.quant == refq.get((hw, model))].correct
                if ref.empty:
                    continue
                for quant, gq in g.groupby("quant"):
                    pv.setdefault((hw, model, quant), {})[f"{task}_p_vs_ref"] = mcnemar_p(ref, gq.correct)
            if pv:
                parts.append(pd.DataFrame.from_dict(pv, orient="index").rename_axis(idx))
    if not ppl.empty:
        parts.append(ppl.set_index(["hw_id", "model", "quant"])[["ppl", "ppl_err"]])
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts, axis=1).reset_index().rename(columns={"hw_id": "quality_hw_id"})
    tasks = [t for t in TASKS if t in q]
    refq = {k: gate.pick_reference(d.quant) for k, d in q.groupby(["quality_hw_id", "model"])}
    q.insert(3, "reference", [refq[k] for k in zip(q.quality_hw_id, q.model)])
    ref = q[q.quant == q.reference].set_index(["quality_hw_id", "model"])
    key = list(zip(q.quality_hw_id, q.model))
    for t in tasks:
        q[f"{t}_retention"] = q[t].values / ref[t].reindex(key).values
    if "ppl" in q:
        q["ppl_ratio_vs_ref"] = q.ppl.values / ref["ppl"].reindex(key).values
    ret = [f"{t}_retention" for t in tasks]
    q["quality_retention"] = q[ret].mean(axis=1) if ret else np.nan
    q["retention_basis"] = np.where(q.quality_retention.notna(), "task_accuracy", "")
    if "ppl_ratio_vs_ref" in q:  # fallback when no complete task data: F16 PPL / PPL
        fb = q.quality_retention.isna() & q.ppl_ratio_vs_ref.notna()
        q.loc[fb, "quality_retention"] = 1 / q.loc[fb, "ppl_ratio_vs_ref"]
        q.loc[fb, "retention_basis"] = "perplexity"
    return sort_keys(q, ["quality_hw_id", "model"])


def hardware_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Same (model, ctx, quant) across hardware. legacy_v0.1 is the Apple M5 v0.1 run
    (7B, ctx 4096; same llama-bench test definitions), kept as the M5 reference."""
    s = summary
    if s.hw_id.nunique() < 2:
        return pd.DataFrame()
    metrics = ["generation_throughput_tps", "prompt_throughput_tps", "ttft_ms", "peak_mem_bytes", "peak_vram_bytes"]
    wide = s.pivot_table(index=["model", "ctx", "quant"], columns="hw_id", values=metrics)
    gen = wide["generation_throughput_tps"]
    wide = wide[gen.notna().sum(axis=1) >= 2]  # only configurations measured on 2+ hardware
    wide.columns = [f"{m}@{hw}" for m, hw in wide.columns]
    return sort_keys(wide.reset_index(), ("model", "ctx"))

# --------------------------------------------------------------------------- figures


def bar(df: pd.DataFrame, y: str, title: str, ylabel: str, out: Path, err: str | None = None, scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = df[err] / scale if err and err in df else None
    ax.bar(df.quant, df[y] / scale, yerr=yerr, capsize=4 if yerr is not None else 0)
    ax.set(title=title, xlabel="Quantization", ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def perf_figures(s: pd.DataFrame, out: Path) -> None:
    """PRD figures 1-5 for one (hw_id, model, ctx). Error bars = 95% CI (t)."""
    out.mkdir(parents=True, exist_ok=True)
    bar(s, "model_size_bytes", "Quantization vs Model Size", "GGUF size (GB)", out / "model_size.png", scale=GB)
    if s.peak_vram_bytes.notna().any():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(s))
        ax.bar(x - 0.2, s.peak_mem_bytes / GB, 0.4, label="Peak RSS (host)")
        ax.bar(x + 0.2, s.peak_vram_bytes / GB, 0.4, label="Peak VRAM (device, delta)")
        ax.set_xticks(x, s.quant)
        ax.set(title="Quantization vs Peak Memory", xlabel="Quantization", ylabel="GB")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "peak_memory.png", dpi=150)
        plt.close(fig)
    else:
        bar(s, "peak_mem_bytes", "Quantization vs Peak Memory", "Peak RSS (GB)", out / "peak_memory.png", scale=GB)
    bar(s, "prompt_throughput_tps", "Quantization vs Prompt Processing Throughput", "tokens/s",
        out / "prompt_throughput.png", "prompt_throughput_ci95")
    bar(s, "generation_throughput_tps", "Quantization vs Generation Throughput", "tokens/s",
        out / "generation_throughput.png", "generation_throughput_ci95")
    bar(s, "ttft_ms", "Quantization vs Time To First Token", "TTFT (ms)", out / "ttft.png", "ttft_ci95")


def ctx_figures(s: pd.DataFrame, out: Path) -> None:
    """v0.3: throughput and TTFT vs context length, one line per quant."""
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (y, err, title) in zip(axes, [("prompt_throughput_tps", "prompt_throughput_ci95", "Prompt processing"),
                                          ("generation_throughput_tps", "generation_throughput_ci95", "Generation (depth = ctx)")]):
        for q in ordered(s.quant):
            d = s[s.quant == q].sort_values("ctx")
            ax.errorbar(d.ctx, d[y], yerr=d[err], marker="o", capsize=3, label=q)
        ax.set(title=title, xlabel="Context length (tokens)", ylabel="tokens/s", xscale="log")
        ax.set_xticks(sorted(s.ctx.unique()), [str(c) for c in sorted(s.ctx.unique())])
        ax.minorticks_off()
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ctx_throughput.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in ordered(s.quant):
        d = s[s.quant == q].sort_values("ctx")
        ax.errorbar(d.ctx, d.ttft_ms / 1000, yerr=d.ttft_ci95 / 1000, marker="o", capsize=3, label=q)
    ax.set(title="TTFT vs Context Length", xlabel="Context length (tokens)", ylabel="TTFT (s)", xscale="log")
    ax.set_xticks(sorted(s.ctx.unique()), [str(c) for c in sorted(s.ctx.unique())])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ctx_ttft.png", dpi=150)
    plt.close(fig)


def quality_figure(q: pd.DataFrame, out: Path) -> None:
    """PRD figure 6: PPL (+/- err) and task accuracy per quant."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t in q]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if "ppl" in q:
        axes[0].bar(q.quant, q.ppl, yerr=q.ppl_err, capsize=4)
        if q.ppl.max() > 10 * q.ppl.min():  # a collapsed quant (e.g. PPL 2e4) would flatten the rest
            axes[0].set_yscale("log")
    axes[0].set(title="Perplexity (wikitext-2, lower is better)", xlabel="Quantization", ylabel="PPL")
    x = np.arange(len(q))
    for i, t in enumerate(tasks):
        axes[1].bar(x + (i - (len(tasks) - 1) / 2) * 0.25, q[t] * 100, 0.25, label=t)
    axes[1].set_xticks(x, q.quant)
    axes[1].set(title="Task accuracy", xlabel="Quantization", ylabel="Accuracy (%)")
    if tasks:
        axes[1].legend(fontsize=8)
    fig.suptitle(f"Quantization vs Model Quality - {q.model.iloc[0]}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where no other point has x>= and y>= with one strictly greater (maximize both)."""
    return np.array([not np.any((x >= xi) & (y >= yi) & ((x > xi) | (y > yi))) for xi, yi in zip(x, y)])


def pareto_figure(d: pd.DataFrame, out: Path, title: str) -> None:
    """PRD figure 7: generation throughput vs quality retention."""
    d = d.dropna(subset=["generation_throughput_tps", "quality_retention"])
    if d.empty:
        return
    front = pareto_mask(d.generation_throughput_tps.values, d.quality_retention.values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(d.generation_throughput_tps, d.quality_retention * 100, c=np.where(front, "C3", "C0"))
    f = d[front].sort_values("generation_throughput_tps")
    ax.plot(f.generation_throughput_tps, f.quality_retention * 100, "C3--", label="Pareto frontier")
    for _, r in d.iterrows():
        ax.annotate(r.quant, (r.generation_throughput_tps, r.quality_retention * 100), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set(title=title, xlabel="Generation throughput (tokens/s)", ylabel="Quality retention vs F16 (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def hardware_figure(s: pd.DataFrame, out: Path, title: str) -> None:
    hws = sorted(s.hw_id.unique())
    quants = ordered(s.quant)
    x = np.arange(len(quants))
    w = 0.8 / len(hws)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, hw in enumerate(hws):
        d = s[s.hw_id == hw].set_index("quant").reindex(quants)
        ax.bar(x + (i - (len(hws) - 1) / 2) * w, d.generation_throughput_tps, w,
               yerr=d.generation_throughput_ci95, capsize=3, label=hw)
    ax.set_xticks(x, quants)
    ax.set(title=title, xlabel="Quantization", ylabel="Generation throughput (tokens/s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

# --------------------------------------------------------------------------- main


def hardware_meta(raw: pd.DataFrame, cfg: dict) -> dict:
    meta = {}
    for hw, d in raw.groupby("hw_id"):
        if hw == LEGACY_HW_ID:
            h = cfg["hardware"]
            meta[hw] = {"soc": h["soc"], "memory_gb": h["memory_gb"], "backend": h["backend"], "os": h["os"],
                        "note": "v0.1 run; pg rows include llama-bench default tests (filtered)"}
            continue
        cols = ["os", "hardware_cpu", "hardware_gpu", "hardware_memory_bytes", "hardware_vram_bytes",
                "hardware_backend", "build_commit", "backends", "gpu_info", "cpu_info", "n_gpu_layers", "flash_attn"]
        meta[hw] = {c: sorted(map(str, d[c].dropna().unique())) for c in cols if c in d}
    return meta


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    raw = load_raw(cfg)
    stats_df = compute_stats(raw)
    summary = summarize(stats_df, cfg["analysis"]["cv_warn_threshold"])
    tele = load_telemetry(cfg)
    if not tele.empty:
        tg = tele[tele.test_type == "tg"].groupby(KEYS)[["gpu_temp_max_c", "gpu_sm_clock_median_mhz"]].last()
        summary = summary.merge(tg.add_prefix("tg_").reset_index(), on=KEYS, how="left")
    ppl, acc = load_quality(cfg)
    items = load_items(cfg)
    quality = quality_table(ppl, acc, items)
    gates = gate.gate_all(items, acc.groupby("task").expected_n.first().to_dict()) if not acc.empty else pd.DataFrame()
    hwcmp = hardware_comparison(summary)

    tables = ROOT_DIR / paths["tables_dir"]
    figs = ROOT_DIR / paths["figures_dir"]
    for p in (ROOT_DIR / paths["processed_stats"], ROOT_DIR / paths["summary_table"]):
        p.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(ROOT_DIR / paths["processed_stats"], index=False)
    summary.to_csv(ROOT_DIR / paths["summary_table"], index=False)
    if not quality.empty:
        quality.to_csv(tables / "quality.csv", index=False)
    if not hwcmp.empty:
        hwcmp.to_csv(tables / "hardware_comparison.csv", index=False)
    if not gates.empty:
        gates.to_csv(tables / "gate.csv", index=False)

    default_ctx = cfg["inference"]["context_length"]
    for (hw, model), s in summary.groupby(["hw_id", "model"]):
        out = figs / hw / model
        d = s[s.ctx == default_ctx]
        if not d.empty:
            perf_figures(d, out)
        if s.ctx.nunique() > 1:
            ctx_figures(s, out)
        if not quality.empty and not d.empty:
            qm = quality[quality.model == model].drop_duplicates("quant")  # first quality hw_id
            pareto_figure(d.merge(qm[["quant", "quality_retention"]], on="quant"), out / "pareto.png",
                          f"Quality-Performance Pareto - {model} ({hw}, ctx {default_ctx})")
    for (qhw, model), q in (quality.groupby(["quality_hw_id", "model"]) if not quality.empty else []):
        quality_figure(q, figs / "quality" / f"{qhw}_{model}.png")
    if not hwcmp.empty:
        for model, s in summary[summary.ctx == default_ctx].groupby("model"):
            if s.hw_id.nunique() > 1:
                (figs / "hardware").mkdir(parents=True, exist_ok=True)
                hardware_figure(s, figs / "hardware" / f"{model}_generation_throughput.png",
                                f"Generation throughput by hardware - {model} (ctx {default_ctx})")

    commit_file = ROOT_DIR / "config" / "llama_cpp_commit.txt"
    summary_json = {
        "experiment_config": cfg,
        "hardware": hardware_meta(raw, cfg),
        "llama_cpp_commit": commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else None,
        "per_quant_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "quality": quality.replace({np.nan: None}).to_dict(orient="records") if not quality.empty else [],
        "gate": {"margin": gate.MARGIN, "alpha": gate.ALPHA,
                 "overall": gates[gates.task == "overall"][["hw_id", "model", "candidate", "verdict"]]
                 .to_dict(orient="records") if not gates.empty else []},
    }
    with open(ROOT_DIR / paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, default=str)

    warns = summary[summary.cv_warning != ""]
    for _, r in warns.iterrows():
        print(f"WARN CV>{cfg['analysis']['cv_warn_threshold']:.0%}: {r.hw_id}/{r.model}/ctx{r.ctx}/{r.quant} {r.cv_warning}")
    print(f"Wrote {paths['processed_stats']}, {paths['summary_table']}, {paths['summary_json']}, figures in {paths['figures_dir']}")


if __name__ == "__main__":
    main()
