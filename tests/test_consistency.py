"""Answer flips vs F16 and the re-run noise floor."""
import numpy as np
import pandas as pd
import pytest

import consistency as cs

COLS = ["hw_id", "model", "quant", "task", "id", "correct", "pred", "response"]


def run(quant, correct, task="mmlu", preds=None, hw="gpu", model="m"):
    preds = preds or ["A" if c else "B" for c in correct]
    return [(hw, model, quant, task, i, c, p, f"text {p}") for i, (c, p) in enumerate(zip(correct, preds))]


def test_flips_with_unchanged_accuracy():
    ref = [True] * 50 + [False] * 50
    cand = [False] * 10 + [True] * 40 + [True] * 10 + [False] * 40  # 10 c->w, 10 w->c
    t = cs.flip_table(pd.DataFrame(run("F16", ref) + run("Q4", cand), columns=COLS))
    r = t[(t.task == "mmlu") & (t.kind == "vs_F16")].iloc[0]
    assert r.acc_diff == 0 and r.flips == pytest.approx(0.2) and (r.c2w, r.w2c) == (10, 10)
    assert r.flips_ci_low < 0.2 < r.flips_ci_high and r.pred_changed == pytest.approx(0.2)
    assert set(t.task) == {"mmlu", "all"} and np.isnan(r.attributable)


def test_noise_floor_and_attributable():
    ref = [True] * 80 + [False] * 20
    q4 = [False] * 8 + [True] * 92
    q4_again = [False] * 6 + [True] * 94  # 2 items differ between identical runs
    items = pd.DataFrame(run("F16", ref) + run("Q4", q4), columns=COLS)
    rerun = pd.DataFrame(run("Q4", q4_again), columns=COLS)
    t = cs.flip_table(items, rerun).set_index(["kind", "task"])
    assert t.loc[("rerun", "mmlu"), "flips"] == pytest.approx(0.02)
    assert t.loc[("vs_F16", "mmlu"), "noise_floor"] == pytest.approx(0.02)
    assert t.loc[("vs_F16", "mmlu"), "attributable"] == pytest.approx(t.loc[("vs_F16", "mmlu"), "flips"] - 0.02)


def test_incomplete_runs_and_pooling():
    ref = [True, False] * 50
    rows = run("F16", ref) + run("Q4", ref[:60])  # Q4 incomplete -> ignored
    rows += run("F16", [True] * 40, task="gsm8k") + run("Q8", [True] * 100) + run("Q8", [False] * 40, task="gsm8k")
    t = cs.flip_table(pd.DataFrame(rows, columns=COLS))
    assert set(t.quant) == {"Q8"}
    pooled = t[t.task == "all"].iloc[0]
    assert pooled.n == 140 and pooled.flips == pytest.approx((50 + 40) / 140)
