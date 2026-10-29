"""Regression gate: can a candidate (model, quant) replace a reference without losing task accuracy?

Per task, on the same items (data/quality/<hw_id>/<model>/<quant>/<task>.jsonl):
  diff      candidate accuracy - reference accuracy
  diff CI   95% paired bootstrap over items (fixed seed)
  p         exact two-sided McNemar test on discordant items
Task verdict:
  FAIL          significantly worse: diff < 0 and p < alpha / n_tasks (Bonferroni)
  PASS          non-inferior: lower CI bound > -margin
  INCONCLUSIVE  neither (more items needed to decide)
Overall: FAIL if any task FAILs, PASS if every task PASSes, else INCONCLUSIVE.

Planning (per task): p_discordant = share of items whose correctness differs; n_needed = items for
an 80% chance of PASS if the true difference is 0 at that discordance (Wald approximation for the
paired design, cf. Connor 1987, Nam 1997). Discordance, not accuracy, sets the cost of certifying.

CLI:
    python src/gate.py --model Qwen2.5-7B-Instruct --candidate Q3_K_M [--reference F16]
                       [--candidate-model Qwen2.5-14B-Instruct] [--hw HW_ID] [--margin 0.03] [--alpha 0.05]
Exit code: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (usable as a CI check).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT_DIR, load_config

MARGIN = 0.03      # largest accuracy drop (absolute) still accepted as "no regression"
ALPHA = 0.05
POWER = 0.80
REFERENCES = ("F16", "Q8_0")  # reference per model: F16, else Q8_0 when F16 was not evaluated (does not fit)


def pick_reference(quants) -> str | None:
    return next((r for r in REFERENCES if r in set(quants)), None)
N_BOOT = 10_000
SEED = 0


def load_items(quality_dir: Path, extra: bool = False) -> pd.DataFrame:
    """Per-item correctness: hw_id, model, quant, task, id, correct (+ pred, response if extra)."""
    rows = []
    for f in sorted(quality_dir.glob("*/*/*/*.jsonl")):
        hw, model, quant, task = f.parent.parent.parent.name, f.parent.parent.name, f.parent.name, f.stem
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                row = (hw, model, quant, task, r["id"], bool(r["correct"]))
                if extra:
                    row += (None if r.get("pred") is None else str(r["pred"]), r.get("response") or "")
                rows.append(row)
    cols = ["hw_id", "model", "quant", "task", "id", "correct"] + (["pred", "response"] if extra else [])
    return pd.DataFrame(rows, columns=cols)


def mcnemar_p(ref: np.ndarray, cand: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired booleans."""
    n01 = int((~ref & cand).sum())
    n10 = int((ref & ~cand).sum())
    return 1.0 if n01 + n10 == 0 else float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def paired_diff_ci(ref: np.ndarray, cand: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI of mean(cand) - mean(ref), resampling items."""
    d = cand.astype(float) - ref.astype(float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def items_needed(p_disc: float, margin: float = MARGIN, true_diff: float = 0.0,
                 alpha: float = ALPHA / 2, power: float = POWER) -> float:
    """Items for PASS (lower bound of the two-sided 1-2*alpha CI > -margin) with probability `power`.

    Per-item difference d in {-1, 0, 1}: Var(d) = p_disc - true_diff^2 (p_disc = P(d != 0))."""
    if margin + true_diff <= 0:
        return float("inf")
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return float(np.ceil(z * z * max(p_disc - true_diff ** 2, 0.0) / (margin + true_diff) ** 2))


def compare_task(ref: pd.Series, cand: pd.Series, margin: float, alpha: float) -> dict:
    """ref/cand: correctness indexed by item id. alpha is the per-task (already corrected) level."""
    both = pd.concat([ref.rename("ref"), cand.rename("cand")], axis=1, join="inner")
    r, c = both.ref.to_numpy(bool), both.cand.to_numpy(bool)
    diff = float(c.mean() - r.mean())
    lo, hi = paired_diff_ci(r, c)
    p = mcnemar_p(r, c)
    if diff < 0 and p < alpha:
        verdict = "FAIL"
    elif lo > -margin:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    p_disc = float((r != c).mean()) if len(both) else 0.0
    return {"n": len(both), "ref_acc": float(r.mean()), "cand_acc": float(c.mean()), "diff": diff,
            "diff_ci_low": lo, "diff_ci_high": hi, "p_mcnemar": p, "verdict": verdict,
            "p_discordant": p_disc, "n_needed": items_needed(p_disc, margin)}


def overall(verdicts: list[str]) -> str:
    if "FAIL" in verdicts:
        return "FAIL"
    return "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "INCONCLUSIVE"


def gate(items: pd.DataFrame, hw: str, model: str, candidate: str, reference: str = "F16",
         margin: float = MARGIN, alpha: float = ALPHA, candidate_model: str | None = None) -> pd.DataFrame:
    """One row per task plus an 'overall' row. Tasks need complete results for both sides.

    candidate_model compares across models (e.g. upgrading 7B F16 to 14B Q4_K_M) on the same items."""
    cand_model = candidate_model or model
    d = items[items.hw_id == hw]
    ref_d = d[(d.model == model) & (d.quant == reference)]
    cand_d = d[(d.model == cand_model) & (d.quant == candidate)]
    tasks = sorted(set(ref_d.task) & set(cand_d.task))
    rows = []
    for t in tasks:
        res = compare_task(ref_d[ref_d.task == t].set_index("id").correct,
                           cand_d[cand_d.task == t].set_index("id").correct, margin, alpha / len(tasks))
        rows.append({"task": t, **res})
    out = pd.DataFrame(rows)
    head = {"hw_id": hw, "model": model, "reference": reference, "candidate": candidate}
    if candidate_model:
        head["candidate_model"] = cand_model
    for i, (k, v) in enumerate(head.items()):
        out.insert(i, k, v)
    if not out.empty:
        out = pd.concat([out, pd.DataFrame([{**head, "task": "overall",
                                             "verdict": overall(out.verdict.tolist())}])], ignore_index=True)
    return out


def gate_all(items: pd.DataFrame, expected_n: dict[str, int], reference: str | None = None,
             margin: float = MARGIN, alpha: float = ALPHA) -> pd.DataFrame:
    """Gate every (hw, model, quant) against the reference (default: pick_reference per model),
    using only complete task files."""
    counts = items.groupby(["hw_id", "model", "quant", "task"]).size()
    complete = {k for k, n in counts.items() if n == expected_n.get(k[3], -1)}
    keep = np.array([k in complete for k in zip(items.hw_id, items.model, items.quant, items.task)], dtype=bool)
    items = items[keep]  # boolean array: an empty list would select columns instead of rows
    frames = []
    for (hw, model), d in items.groupby(["hw_id", "model"]):
        ref = reference or pick_reference(d.quant)
        if ref not in set(d.quant):
            continue
        for quant in sorted(set(d.quant) - {ref}):
            frames.append(gate(d, hw, model, quant, ref, margin, alpha))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", default="F16")
    ap.add_argument("--candidate-model", help="compare against another model (default: same --model)")
    ap.add_argument("--hw", help="quality hw_id (default: the only one present)")
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    items = load_items(ROOT_DIR / load_config()["paths"]["quality_dir"])
    hws = sorted(set(items.hw_id))
    hw = args.hw or (hws[0] if len(hws) == 1 else None)
    if hw is None:
        raise SystemExit(f"--hw required, one of {hws}")
    res = gate(items, hw, args.model, args.candidate, args.reference, args.margin, args.alpha, args.candidate_model)
    if res.empty:
        raise SystemExit("no shared complete tasks for reference and candidate")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(res.drop(columns=["hw_id", "model"]).to_string(index=False))
    return {"PASS": 0, "FAIL": 1}.get(res.verdict.iloc[-1], 2)


if __name__ == "__main__":
    sys.exit(main())
