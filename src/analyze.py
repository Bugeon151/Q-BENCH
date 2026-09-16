"""Pipeline stage: Data Processing + Statistical Analysis + Visualization + Report.

Reads data/raw/benchmark_raw.csv, computes per (quant, test_type) statistics,
writes processed/summary tables, renders PRD figures 1-5, and writes
results/summary.json for reproducibility.

Quality-dependent figures (Model Quality, Pareto Frontier) are out of scope
until v0.2 adds perplexity/quality_eval data.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent

QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def load_config() -> dict:
    with open(ROOT_DIR / "config" / "experiment.yaml") as f:
        return yaml.safe_load(f)


def ordered(quants: list[str]) -> list[str]:
    return [q for q in QUANT_ORDER if q in quants]


def ci95(series: pd.Series) -> float:
    n = len(series)
    if n < 2:
        return 0.0
    sem = series.std(ddof=1) / (n**0.5)
    return float(sem * stats.t.ppf(0.975, df=n - 1))


def compute_stats(raw: pd.DataFrame) -> pd.DataFrame:
    pg = raw[raw.test_type == "pg"]
    raw = pd.concat([raw[raw.test_type != "pg"], pg[(pg.n_prompt > 0) & (pg.n_gen > 0)]])
    grouped = raw.groupby(["quant", "test_type"])
    stats_df = grouped.agg(
        latency_ms_mean=("latency_ms", "mean"),
        latency_ms_median=("latency_ms", "median"),
        latency_ms_std=("latency_ms", "std"),
        tokens_per_sec_mean=("tokens_per_sec", "mean"),
        tokens_per_sec_median=("tokens_per_sec", "median"),
        tokens_per_sec_std=("tokens_per_sec", "std"),
        model_size_bytes=("model_size_bytes", "first"),
        peak_mem_bytes=("peak_mem_bytes", "max"),
        n_samples=("latency_ms", "count"),
    ).reset_index()

    ci_latency = grouped["latency_ms"].apply(ci95).rename("latency_ms_ci95")
    ci_tps = grouped["tokens_per_sec"].apply(ci95).rename("tokens_per_sec_ci95")
    stats_df = stats_df.merge(ci_latency, on=["quant", "test_type"]).merge(
        ci_tps, on=["quant", "test_type"]
    )
    return stats_df


def per_quant_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    pp = stats_df[stats_df.test_type == "pp"].set_index("quant")
    tg = stats_df[stats_df.test_type == "tg"].set_index("quant")
    pg = stats_df[stats_df.test_type == "pg"].set_index("quant")

    quants = ordered(stats_df["quant"].unique().tolist())
    rows = []
    for q in quants:
        peak_mem = max(
            pp.loc[q, "peak_mem_bytes"] if q in pp.index else 0,
            tg.loc[q, "peak_mem_bytes"] if q in tg.index else 0,
            pg.loc[q, "peak_mem_bytes"] if q in pg.index else 0,
        )
        rows.append(
            {
                "quant": q,
                "model_size_bytes": pp.loc[q, "model_size_bytes"] if q in pp.index else None,
                "peak_mem_bytes": peak_mem,
                "prompt_throughput_tps": pp.loc[q, "tokens_per_sec_mean"] if q in pp.index else None,
                "generation_throughput_tps": tg.loc[q, "tokens_per_sec_mean"] if q in tg.index else None,
                "ttft_ms": pp.loc[q, "latency_ms_mean"] if q in pp.index else None,
                "total_latency_ms": pg.loc[q, "latency_ms_mean"] if q in pg.index else None,
            }
        )
    return pd.DataFrame(rows)


def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel("Quantization")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    raw_csv = ROOT_DIR / cfg["paths"]["raw_csv"]
    if not raw_csv.exists():
        raise SystemExit(f"Raw data not found: {raw_csv}. Run src/benchmark.py first.")

    raw = pd.read_csv(raw_csv)

    stats_df = compute_stats(raw)
    processed_path = ROOT_DIR / cfg["paths"]["processed_stats"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(processed_path, index=False)

    summary_df = per_quant_summary(stats_df)
    summary_table_path = ROOT_DIR / cfg["paths"]["summary_table"]
    summary_table_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_table_path, index=False)

    figures_dir = ROOT_DIR / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    make_bar_chart(
        summary_df, "quant", "model_size_bytes",
        "Quantization vs Model Size", "Size (bytes)",
        figures_dir / "model_size.png",
    )
    make_bar_chart(
        summary_df, "quant", "peak_mem_bytes",
        "Quantization vs Peak Memory", "Peak RSS (bytes)",
        figures_dir / "peak_memory.png",
    )
    make_bar_chart(
        summary_df, "quant", "prompt_throughput_tps",
        "Quantization vs Prompt Processing Throughput", "tokens/s",
        figures_dir / "prompt_throughput.png",
    )
    make_bar_chart(
        summary_df, "quant", "generation_throughput_tps",
        "Quantization vs Generation Throughput", "tokens/s",
        figures_dir / "generation_throughput.png",
    )
    make_bar_chart(
        summary_df, "quant", "ttft_ms",
        "Quantization vs Time To First Token", "TTFT (ms)",
        figures_dir / "ttft.png",
    )
    # PRD figures 6 (Model Quality) and 7 (Pareto Frontier) require
    # perplexity/quality_eval data introduced in v0.2 - intentionally omitted.

    llama_commit = subprocess.run(
        ["git", "-C", str(ROOT_DIR / "external/llama.cpp"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    summary_json = {
        "experiment_config": cfg,
        "hardware": {
            "soc": cfg["hardware"]["soc"],
            "memory_gb": cfg["hardware"]["memory_gb"],
            "backend": cfg["hardware"]["backend"],
        },
        "llama_cpp_commit": llama_commit,
        "per_quant_summary": summary_df.to_dict(orient="records"),
    }
    summary_json_path = ROOT_DIR / cfg["paths"]["summary_json"]
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    print(f"Wrote {processed_path}, {summary_table_path}, {summary_json_path}, and 5 figures to {figures_dir}")


if __name__ == "__main__":
    main()
