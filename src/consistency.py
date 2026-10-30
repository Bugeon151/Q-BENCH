"""Answer consistency: how many answers change when only the quantization changes.

Accuracy can stay flat while individual answers flip (Dutta et al. 2024, arXiv:2407.09141).
Per (hw, model, quant, task), against the reference on the same items (F16; Q8_0 for a model whose
F16 was not evaluated, see gate.pick_reference):
  flips         share of items whose correctness differs (correct->wrong + wrong->correct)
  c2w, w2c      item counts per direction
  pred_changed  share whose extracted answer differs (mmlu letter, gsm8k number; blank for humaneval)
  text_changed  share whose response text differs at all
Noise floor: the same (model, quant) re-run with identical settings (quality_eval.py
--out-root data/quality_rerun) shows the flips that occur without any change: llama-server's
continuous batching changes batch sizes from run to run, and outputs are not batch-invariant.
A vs_ref row uses the re-run of its own configuration when there is one (noise_basis "same"),
else the mean re-run flips of the model's quantized (non-reference) re-runs ("proxy:<quants>").
attributable = flips - noise floor is a conservative lower bound: both compared runs are noisy,
but a re-run already counts the noise of two runs of the quantized model, while the reference
side is often far less noisy. It is also loose: noise changes which items flip more than how many
(on this laptop a deterministic one-slot candidate run gave 4.6 % flips for 7B Q4_K_M vs 4.4 % with
4 slots, while the subtraction leaves 1.8 %), so prefer vs_ref_np1 where it exists.
Rows with task "all" pool the tasks.
One-slot runs (quality_eval.py --parallel 1, paths.quality_np1_dirs = [run, re-run]) remove
concurrent batching: kind "rerun_np1" (the two one-slot runs), "np1_vs_np4" (one-slot run vs the
4-slot run of the same configuration), "vs_ref_np1" (reference vs the one-slot run, noise floor
from "rerun_np1").

    python src/consistency.py   -> results/tables/flips.csv, results/figures/flips_<model>.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gate  # noqa: E402
from analyze import QUANT_ORDER, wilson_ci  # noqa: E402
from gate import pick_reference  # noqa: E402
from common import ROOT_DIR, load_config  # noqa: E402

KEY = ["hw_id", "model", "quant", "task"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
C_ACC, C_FLIP, C_NOISE = "#2a78d6", "#eb6834", "#52514e"


def complete(items: pd.DataFrame, full: pd.Series | None = None) -> pd.DataFrame:
    """Keep (hw, model, quant, task) runs that cover every item id of the task (full: ids per task,
    default: all ids seen for that task in items)."""
    if items.empty:
        return items
    full = items.groupby("task").id.nunique() if full is None else full
    n = items.groupby(KEY).id.transform("size")
    return items[n.to_numpy() == items.task.map(full).to_numpy()]


def pair(ref: pd.DataFrame, cand: pd.DataFrame) -> dict:
    """ref/cand: one task's rows (id, correct, pred, response) of two runs."""
    b = ref.set_index("id").add_suffix("_r").join(cand.set_index("id").add_suffix("_c"), how="inner")
    r, c = b.correct_r.to_numpy(bool), b.correct_c.to_numpy(bool)
    n, c2w, w2c = len(b), int((r & ~c).sum()), int((~r & c).sum())
    lo, hi = wilson_ci(c2w + w2c, n)
    has_pred = b.pred_r.notna() & b.pred_c.notna()
    return {"n": n, "flips": (c2w + w2c) / n, "flips_ci_low": lo, "flips_ci_high": hi, "c2w": c2w, "w2c": w2c,
            "acc_diff": float(c.mean() - r.mean()),
            "pred_changed": float((b.pred_r != b.pred_c)[has_pred].mean()) if has_pred.any() else np.nan,
            "text_changed": float((b.response_r != b.response_c).mean())}


def pooled(rows: pd.DataFrame) -> dict:
    n, c2w, w2c = int(rows.n.sum()), int(rows.c2w.sum()), int(rows.w2c.sum())
    lo, hi = wilson_ci(c2w + w2c, n)
    return {"n": n, "flips": (c2w + w2c) / n, "flips_ci_low": lo, "flips_ci_high": hi, "c2w": c2w, "w2c": w2c,
            "acc_diff": float((rows.acc_diff * rows.n).sum() / n), "pred_changed": np.nan,
            "text_changed": float((rows.text_changed * rows.n).sum() / n)}


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, kind: str, same_quant: bool) -> list[dict]:
    """same_quant=False: every quant of b vs the reference quant of a (same hw, model).
    same_quant=True: each (hw, model, quant) of b vs the same run key of a (re-run)."""
    out = []
    refs = {k: pick_reference(d.quant) for k, d in a.groupby(["hw_id", "model"])}
    for (hw, model, quant), cb in b.groupby(["hw_id", "model", "quant"]):
        ref_quant = quant if same_quant else refs.get((hw, model))
        if ref_quant is None or (not same_quant and quant == ref_quant):
            continue
        ca = a[(a.hw_id == hw) & (a.model == model) & (a.quant == ref_quant)]
        rows = []
        for task in sorted(set(ca.task) & set(cb.task)):
            rows.append({"hw_id": hw, "model": model, "quant": quant, "task": task, "kind": kind,
                         "reference": ref_quant, **pair(ca[ca.task == task], cb[cb.task == task])})
        if rows:
            out += rows + [{"hw_id": hw, "model": model, "quant": quant, "task": "all", "kind": kind,
                            "reference": ref_quant, **pooled(pd.DataFrame(rows))}]
    return out


def flip_table(items: pd.DataFrame, rerun: pd.DataFrame | None = None,
               np1: list[pd.DataFrame | None] | None = None) -> pd.DataFrame:
    full = items.groupby("task").id.nunique() if not items.empty else None
    items = complete(items)
    rows = compare_runs(items, items, "vs_ref", same_quant=False)
    if rerun is not None and not rerun.empty:
        rows += compare_runs(items, complete(rerun, full), "rerun", same_quant=True)
    one = [complete(d, full) if d is not None and not d.empty else pd.DataFrame() for d in (np1 or [])]
    if one and not one[0].empty:
        rows += compare_runs(items, one[0], "vs_ref_np1", same_quant=False)
        rows += compare_runs(items, one[0], "np1_vs_np4", same_quant=True)
        if len(one) > 1 and not one[1].empty:
            rows += compare_runs(one[0], one[1], "rerun_np1", same_quant=True)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["noise_floor"], t["noise_basis"] = np.nan, ""
    r1 = t[t.kind == "rerun_np1"].set_index(["hw_id", "model", "quant", "task"]).flips
    for i in t.index[t.kind == "vs_ref_np1"]:
        k = tuple(t.loc[i, ["hw_id", "model", "quant", "task"]])
        if k in r1.index:
            t.loc[i, ["noise_floor", "noise_basis"]] = r1[k], "same_np1"
    rr = t[t.kind == "rerun"]
    if not rr.empty:
        refs = t[t.kind == "vs_ref"].groupby(["hw_id", "model"]).reference.first()
        same = rr.set_index(["hw_id", "model", "quant", "task"]).flips
        proxy_rows = rr[[q != refs.get((h, m)) for h, m, q in zip(rr.hw_id, rr.model, rr.quant)]]
        proxy = proxy_rows.groupby(["hw_id", "model", "task"]).flips.mean()
        proxy_q = proxy_rows.groupby(["hw_id", "model"]).quant.agg(lambda q: "+".join(sorted(set(q))))
        for i in t.index[t.kind == "vs_ref"]:
            r = t.loc[i]
            if (r.hw_id, r.model, r.quant, r.task) in same.index:
                t.loc[i, ["noise_floor", "noise_basis"]] = same[(r.hw_id, r.model, r.quant, r.task)], "same"
            elif (r.hw_id, r.model, r.task) in proxy.index:
                t.loc[i, ["noise_floor", "noise_basis"]] = proxy[(r.hw_id, r.model, r.task)], \
                    f"proxy:{proxy_q[(r.hw_id, r.model)]}"
    t["attributable"] = np.where(t.kind.isin(["vs_ref", "vs_ref_np1"]), t.flips - t.noise_floor, np.nan)
    order = {q: i for i, q in enumerate(QUANT_ORDER)}
    return t.assign(_q=t.quant.map(order)).sort_values(["kind", "hw_id", "model", "_q", "task"]) \
        .drop(columns="_q").reset_index(drop=True)


def figure(t: pd.DataFrame, model: str, out) -> None:
    d = t[(t.kind == "vs_ref") & (t.model == model) & (t.task != "all")]
    tasks = [x for x in ("mmlu", "gsm8k", "humaneval") if x in set(d.task)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), sharey=True, facecolor=SURFACE)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        e = d[d.task == task]
        x = np.arange(len(e))
        ax.set_facecolor(SURFACE)
        ax.bar(x - 0.2, e.acc_diff.abs() * 100, 0.38, color=C_ACC, label="|accuracy change|")
        ax.bar(x + 0.2, e.flips * 100, 0.38, color=C_FLIP, label="answers flipped",
               yerr=[(e.flips - e.flips_ci_low) * 100, (e.flips_ci_high - e.flips) * 100],
               error_kw={"ecolor": INK2, "lw": 1})
        if e.noise_floor.notna().any():
            ax.axhline(e.noise_floor.iloc[0] * 100, color=C_NOISE, ls="--", lw=1.2, label="re-run noise floor")
        ax.set_xticks(x, e.quant, rotation=45, color=INK2, fontsize=8)
        ax.set_title(task, color=INK, fontsize=10)
        ax.grid(axis="y", color="#e5e4df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("% of items", color=INK2)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{model}: accuracy change vs answers flipped, each quant vs {e.reference.iloc[0]} (95% Wilson CI)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    items = gate.load_items(ROOT_DIR / paths["quality_dir"], extra=True)
    rerun_dir = ROOT_DIR / paths["quality_rerun_dir"]
    rerun = gate.load_items(rerun_dir, extra=True) if rerun_dir.exists() else None
    np1 = [gate.load_items(ROOT_DIR / d, extra=True) if (ROOT_DIR / d).exists() else None
           for d in paths.get("quality_np1_dirs", [])]
    t = flip_table(items, rerun, np1)
    if t.empty:
        print("no complete quality runs with a reference (F16 or Q8_0)")
        return
    tables, figs = ROOT_DIR / paths["tables_dir"], ROOT_DIR / paths["figures_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    t.to_csv(tables / "flips.csv", index=False)
    for model in sorted(set(t.model)):
        figure(t, model, figs / f"flips_{model}.png")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t[t.task == "all"][["kind", "model", "quant", "n", "acc_diff", "flips", "noise_floor",
                                  "attributable"]].to_string(index=False))


if __name__ == "__main__":
    main()
