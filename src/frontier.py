"""Size vs bits: for a given amount of memory, a bigger model at fewer bits or a smaller one at more?

Dettmers & Zettlemoyer (ICML 2023) found 4-bit almost universally optimal for zero-shot accuracy
per total model bits. This checks the question on downstream tasks for the measured models:
  frontier.csv      per (model, quant): GGUF GiB, bits/weight (16 x size / F16 size), task accuracy,
                    mean of tasks, PPL ratio vs F16, flips vs F16, gate verdict, Pareto-optimal
                    (no other config is both smaller and more accurate)
  size_vs_bits.csv  matched-memory duels on the same items: each config of a larger model vs the
                    closest-size config of every smaller model (size ratio within 0.8-1.25);
                    per task and pooled: accuracy difference, paired bootstrap 95% CI, McNemar p
  ppl_screen.csv    Spearman correlation of ln(PPL / PPL_F16) with flips and with the drop in mean
                    accuracy: can a perplexity run (minutes) stand in for task evaluation (hours)?
Figures: frontier.png, ppl_screen.png.  Run after analyze.py and consistency.py.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import gate  # noqa: E402
from analyze import QUANT_ORDER, pareto_mask  # noqa: E402
from common import GIB, ROOT_DIR, load_config  # noqa: E402

TASKS = ["mmlu", "gsm8k", "humaneval"]
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4df"
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]  # ordinal blue, small -> large model
FAIL_RED = "#e34948"


def params_b(model: str) -> float:
    """'Qwen2.5-14B-Instruct' -> 14.0 (billions, from the name; used only for ordering)."""
    tok = next(t for t in model.split("-") if t.endswith("B") and t[:-1].replace(".", "").isdigit())
    return float(tok[:-1])


def frontier_table(quality: pd.DataFrame, summary: pd.DataFrame, flips: pd.DataFrame | None,
                   gates: pd.DataFrame | None) -> pd.DataFrame:
    tasks = [t for t in TASKS if t in quality]
    q = quality.dropna(subset=tasks).copy()
    size = summary.groupby(["model", "quant"]).model_size_bytes.first()
    q = q[[k in size.index for k in zip(q.model, q.quant)]]
    q["size_gib"] = [size[k] / GIB for k in zip(q.model, q.quant)]
    f16 = {m: size.get((m, "F16")) for m in q.model.unique()}
    q["bpw"] = [16 * size[(m, qq)] / f16[m] if f16[m] else np.nan for m, qq in zip(q.model, q.quant)]
    q["quality"] = q[tasks].mean(axis=1)
    if flips is not None and not flips.empty:
        f = flips[(flips.kind == "vs_F16") & (flips.task == "all")].set_index(["model", "quant"]).flips
        q["flips"] = [0.0 if qq == "F16" else f.get((m, qq), np.nan) for m, qq in zip(q.model, q.quant)]
    if gates is not None and not gates.empty:
        g = gates[gates.task == "overall"].set_index(["model", "candidate"]).verdict
        q["gate"] = ["REFERENCE" if qq == "F16" else g.get((m, qq), "") for m, qq in zip(q.model, q.quant)]
    q["params_b"] = q.model.map(params_b)
    q["pareto"] = False
    for hw, d in q.groupby("quality_hw_id"):
        q.loc[d.index, "pareto"] = pareto_mask(-d.size_gib.to_numpy(), d.quality.to_numpy())  # small and accurate
    order = {k: i for i, k in enumerate(QUANT_ORDER)}
    cols = ["quality_hw_id", "model", "params_b", "quant", "size_gib", "bpw", *tasks, "quality", "ppl",
            "ppl_ratio_vs_f16", *[c for c in ("flips", "gate") if c in q], "pareto"]
    return q.assign(_q=q.quant.map(order)).sort_values(["quality_hw_id", "params_b", "_q"])[cols] \
        .reset_index(drop=True)


def duel(items: pd.DataFrame, hw: str, big: tuple[str, str], small: tuple[str, str]) -> list[dict]:
    """big vs small (reference) on shared items, per task and pooled over tasks."""
    d = items[items.hw_id == hw]
    a = d[(d.model == big[0]) & (d.quant == big[1])]
    b = d[(d.model == small[0]) & (d.quant == small[1])]
    rows, pool_a, pool_b = [], [], []
    for task in [t for t in TASKS if t in set(a.task) & set(b.task)]:
        ra = a[a.task == task].set_index("id").correct
        rb = b[b.task == task].set_index("id").correct
        res = gate.compare_task(rb, ra, gate.MARGIN, gate.ALPHA / 3)
        rows.append({"task": task, **res})
        pool_a.append(ra.rename(lambda i, t=task: f"{t}:{i}"))
        pool_b.append(rb.rename(lambda i, t=task: f"{t}:{i}"))
    if rows:
        rows.append({"task": "all", **gate.compare_task(pd.concat(pool_b), pd.concat(pool_a), gate.MARGIN, gate.ALPHA)})
    return rows


def size_vs_bits(front: pd.DataFrame, items: pd.DataFrame, lo: float = 0.8, hi: float = 1.25) -> pd.DataFrame:
    out = []
    for hw, d in front.groupby("quality_hw_id"):
        for big in d.itertuples():
            for model in d[d.params_b < big.params_b].model.unique():
                s = d[d.model == model]
                r = s.size_gib / big.size_gib
                s = s[(r >= lo) & (r <= hi)]
                if s.empty:
                    continue
                small = s.loc[(s.size_gib - big.size_gib).abs().idxmin()]
                for row in duel(items, hw, (big.model, big.quant), (small.model, small.quant)):
                    out.append({"quality_hw_id": hw, "big": f"{big.model} {big.quant}", "big_gib": big.size_gib,
                                "big_bpw": big.bpw, "small": f"{small.model} {small.quant}",
                                "small_gib": small.size_gib, "small_bpw": small.bpw, **row})
    cols = ["quality_hw_id", "big", "big_gib", "big_bpw", "small", "small_gib", "small_bpw", "task", "n",
            "ref_acc", "cand_acc", "diff", "diff_ci_low", "diff_ci_high", "p_mcnemar"]
    t = pd.DataFrame(out)
    return t[cols].rename(columns={"ref_acc": "small_acc", "cand_acc": "big_acc", "diff": "big_minus_small"}) \
        if not t.empty else t


def ppl_screen(front: pd.DataFrame) -> pd.DataFrame:
    d = front[(front.quant != "F16") & front.ppl_ratio_vs_f16.notna()].copy()
    f16 = front[front.quant == "F16"].set_index(["quality_hw_id", "model"]).quality
    d["quality_drop"] = [f16.get(k, np.nan) - v for k, v in zip(zip(d.quality_hw_id, d.model), d.quality)]
    x = np.log(d.ppl_ratio_vs_f16)
    rows = []
    for name in [c for c in ("flips", "quality_drop") if c in d]:
        ok = x.notna() & d[name].notna()
        if ok.sum() >= 3:
            rho, p = stats.spearmanr(x[ok], d.loc[ok, name])
            rows.append({"metric": name, "n": int(ok.sum()), "spearman_rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def frontier_figure(front: pd.DataFrame, out) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style(ax)
    models = sorted(front.model.unique(), key=params_b)
    colors = dict(zip(models, RAMP[-len(models):] if len(models) <= len(RAMP) else RAMP))
    for m in models:
        d = front[front.model == m].sort_values("size_gib")
        ax.plot(d.size_gib, d.quality * 100, "-", color=colors[m], lw=2, label=m.replace("-Instruct", ""))
        ax.scatter(d.size_gib, d.quality * 100, s=30, color=colors[m], edgecolor=SURFACE, lw=1.5, zorder=3)
        fail = d[d.get("gate", pd.Series("", index=d.index)) == "FAIL"]
        ax.scatter(fail.size_gib, fail.quality * 100, s=90, facecolor="none", edgecolor=FAIL_RED, lw=1.5, zorder=4)
        for r in d.itertuples():
            ax.annotate(r.quant, (r.size_gib, r.quality * 100), xytext=(4, -10), textcoords="offset points",
                        fontsize=6.5, color=INK2)
    p = front[front.pareto].sort_values("size_gib")
    ax.step(p.size_gib, p.quality * 100, where="post", color=INK2, ls=":", lw=1.2, label="Pareto frontier")
    ax.scatter([], [], s=90, facecolor="none", edgecolor=FAIL_RED, lw=1.5, label="gate FAIL vs own F16")
    ax.set_xscale("log")
    ax.set_xlabel("GGUF size (GiB, log scale)", color=INK2)
    ax.set_ylabel("mean accuracy, MMLU/GSM8K/HumanEval (%)", color=INK2)
    ax.set_title("Accuracy per memory: model size vs bits per weight", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def screen_figure(front: pd.DataFrame, out) -> None:
    if "flips" not in front:
        return
    d = front[(front.quant != "F16") & front.ppl_ratio_vs_f16.notna() & front.flips.notna()]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    style(ax)
    models = sorted(d.model.unique(), key=params_b)
    colors = dict(zip(models, RAMP[-len(models):] if len(models) <= len(RAMP) else RAMP))
    for m in models:
        e = d[d.model == m]
        ax.scatter((e.ppl_ratio_vs_f16 - 1) * 100, e.flips * 100, s=36, color=colors[m], edgecolor=SURFACE,
                   lw=1.5, label=m.replace("-Instruct", ""), zorder=3)
        for r in e.itertuples():
            ax.annotate(r.quant, ((r.ppl_ratio_vs_f16 - 1) * 100, r.flips * 100), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("perplexity increase vs F16 (%, log scale)", color=INK2)
    ax.set_ylabel("answers flipped vs F16 (% of all items)", color=INK2)
    ax.set_title("A perplexity run predicts how many answers change", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    tables, figs = ROOT_DIR / cfg["paths"]["tables_dir"], ROOT_DIR / cfg["paths"]["figures_dir"]
    read = lambda name: pd.read_csv(tables / name) if (tables / name).exists() else None  # noqa: E731
    quality, summary = read("quality.csv"), read("summary.csv")
    if quality is None or summary is None or quality.empty:
        print("run analyze.py first")
        return
    front = frontier_table(quality, summary, read("flips.csv"), read("gate.csv"))
    items = gate.load_items(ROOT_DIR / cfg["paths"]["quality_dir"])
    duels, screen = size_vs_bits(front, items), ppl_screen(front)
    figs.mkdir(parents=True, exist_ok=True)
    front.to_csv(tables / "frontier.csv", index=False)
    duels.to_csv(tables / "size_vs_bits.csv", index=False)
    screen.to_csv(tables / "ppl_screen.csv", index=False)
    frontier_figure(front, figs / "frontier.png")
    screen_figure(front, figs / "ppl_screen.png")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(front.drop(columns=["quality_hw_id"]).to_string(index=False))
        if not duels.empty:
            print(duels[duels.task == "all"].drop(columns=["quality_hw_id", "task"]).to_string(index=False))
        print(screen.to_string(index=False))


if __name__ == "__main__":
    main()
