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
    r = t[(t.task == "mmlu") & (t.kind == "vs_ref")].iloc[0]
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
    assert t.loc[("vs_ref", "mmlu"), "noise_floor"] == pytest.approx(0.02)
    assert t.loc[("vs_ref", "mmlu"), "attributable"] == pytest.approx(t.loc[("vs_ref", "mmlu"), "flips"] - 0.02)


def test_incomplete_runs_and_pooling():
    ref = [True, False] * 50
    rows = run("F16", ref) + run("Q4", ref[:60])  # Q4 incomplete -> ignored
    rows += run("F16", [True] * 40, task="gsm8k") + run("Q8", [True] * 100) + run("Q8", [False] * 40, task="gsm8k")
    t = cs.flip_table(pd.DataFrame(rows, columns=COLS))
    assert set(t.quant) == {"Q8"}
    pooled = t[t.task == "all"].iloc[0]
    assert pooled.n == 140 and pooled.flips == pytest.approx((50 + 40) / 140)


def test_q8_reference_when_f16_missing():
    ref = [True] * 80 + [False] * 20
    items = pd.DataFrame(run("Q8_0", ref, model="big") + run("Q4_K_M", [False] * 4 + [True] * 96, model="big"),
                         columns=COLS)
    t = cs.flip_table(items)
    r = t[(t.task == "mmlu")].iloc[0]
    assert (r.quant, r.reference, r.kind) == ("Q4_K_M", "Q8_0", "vs_ref") and r.flips == pytest.approx(0.24)


def test_noise_floor_same_config_or_quantized_proxy():
    """Q4 uses its own re-run; Q6 (no re-run) uses the quantized re-runs, never the F16 re-run."""
    ref = [True] * 80 + [False] * 20
    items = pd.DataFrame(run("F16", ref) + run("Q4_K_M", [False] * 10 + [True] * 90) +
                         run("Q6_K", [False] * 5 + [True] * 95), columns=COLS)
    rerun = pd.DataFrame(run("F16", [False] + [True] * 79 + [False] * 20) +       # 1 % noise
                         run("Q4_K_M", [False] * 6 + [True] * 94), columns=COLS)  # 4 % noise
    t = cs.flip_table(items, rerun)
    v = t[(t.kind == "vs_ref") & (t.task == "mmlu")].set_index("quant")
    assert v.loc["Q4_K_M", "noise_basis"] == "same" and v.loc["Q4_K_M", "noise_floor"] == pytest.approx(0.04)
    assert v.loc["Q6_K", "noise_basis"] == "proxy:Q4_K_M" and v.loc["Q6_K", "noise_floor"] == pytest.approx(0.04)
    assert v.loc["Q6_K", "attributable"] == pytest.approx(v.loc["Q6_K", "flips"] - 0.04)
