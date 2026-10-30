"""Cause isolation: answer flips from quantization vs from concurrent batching (paper experiment).

Runs live in data/isolation/np<slots>_r<k>/<hw>/<model>/<quant>/<task>.jsonl, written by
quality_eval.py --parallel <slots> --out-root data/isolation/np<slots>_r<k>.
Per run pair (items pooled over tasks, same items, 95% Wilson CI), kinds:
  vs_ref       each quant vs the reference quant (F16, else Q8_0) inside ONE condition, run k vs run k
  rerun        identical re-runs of the same (model, quant, slots): pure nondeterminism
  np1_vs_np4   first one-slot run vs every multi-slot run of the same quant: what batching changes
One slot = no concurrent batching, so its vs_ref flips are the quantization effect alone, and
rerun flips at one slot test whether that condition is deterministic.

    python src/isolation.py  -> results/tables/isolation_runs.csv (pairs), results/tables/isolation.csv (summary)
"""

from __future__ import annotations

import itertools

import pandas as pd

import gate
from common import ROOT_DIR, load_config
from consistency import compare_runs, complete


def load_runs(root) -> dict[str, pd.DataFrame]:
    return {p.name: gate.load_items(p, extra=True) for p in sorted(root.iterdir()) if p.is_dir()} \
        if root.exists() else {}


def slots(name: str) -> int:
    return int(name.split("_")[0][2:])


def pairs_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    runs = {k: v for k, v in runs.items() if not v.empty}
    if not runs:
        return pd.DataFrame()
    full = pd.concat(runs.values()).groupby("task").id.nunique()
    runs = {k: complete(v, full) for k, v in runs.items()}
    names = sorted(runs, key=lambda n: (slots(n), n))
    rows = []

    def add(kind, a, b, same):
        rows.extend({**r, "slots": slots(b), "a": a, "b": b} for r in compare_runs(runs[a], runs[b], kind, same)
                    if r["task"] == "all")

    for n in names:
        add("vs_ref", n, n, False)
    for s in sorted({slots(n) for n in names}):
        for a, b in itertools.combinations([n for n in names if slots(n) == s], 2):
            add("rerun", a, b, True)
    one = [n for n in names if slots(n) == 1]
    for b in (n for n in names if slots(n) > 1):
        if one:
            add("np1_vs_np4", one[0], b, True)
    keep = ["kind", "hw_id", "model", "quant", "slots", "a", "b", "n", "flips", "flips_ci_low", "flips_ci_high",
            "c2w", "w2c", "acc_diff"]
    return pd.DataFrame(rows)[keep] if rows else pd.DataFrame()


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["hw_id", "model", "quant", "kind", "slots"]).flips
    s = g.agg(pairs="size", flips_mean="mean", flips_sd="std", flips_min="min", flips_max="max")
    return s.join(t.groupby(["hw_id", "model", "quant", "kind", "slots"]).n.first()).reset_index()


def main() -> None:
    cfg = load_config()
    t = pairs_table(load_runs(ROOT_DIR / cfg["paths"]["isolation_dir"]))
    if t.empty:
        print("no isolation runs yet")
        return
    out = ROOT_DIR / cfg["paths"]["tables_dir"]
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(t)
    t.to_csv(out / "isolation_runs.csv", index=False)
    s.to_csv(out / "isolation.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(s.drop(columns="hw_id").to_string(index=False))


if __name__ == "__main__":
    main()
