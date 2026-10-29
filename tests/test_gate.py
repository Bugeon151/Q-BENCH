"""Regression gate verdicts on synthetic paired results."""
import numpy as np
import pandas as pd
import pytest

import gate


def items(acc_by_quant: dict[str, list[bool]], task="mmlu", hw="gpu", model="m"):
    return pd.DataFrame([(hw, model, q, task, i, c) for q, cs in acc_by_quant.items() for i, c in enumerate(cs)],
                        columns=["hw_id", "model", "quant", "task", "id", "correct"])


def test_identical_passes():
    ref = [True] * 450 + [False] * 150
    res = gate.gate(items({"F16": ref, "Q4": ref}), "gpu", "m", "Q4")
    assert res.set_index("task").loc["mmlu", "verdict"] == "PASS"
    assert res.verdict.iloc[-1] == "PASS" and res.task.iloc[-1] == "overall"


def test_large_drop_fails():
    ref = [True] * 200
    cand = [True] * 160 + [False] * 40
    r = gate.gate(items({"F16": ref, "Q2": cand}), "gpu", "m", "Q2").set_index("task").loc["mmlu"]
    assert r.verdict == "FAIL" and r["diff"] == pytest.approx(-0.2) and r.p_mcnemar < 1e-10


def test_small_sample_inconclusive():
    ref = [True] * 30
    cand = [True] * 29 + [False]
    r = gate.gate(items({"F16": ref, "Q3": cand}), "gpu", "m", "Q3").set_index("task").loc["mmlu"]
    assert r.verdict == "INCONCLUSIVE"  # 1 of 30 worse: not significant, CI too wide for a 3%p margin


def test_bootstrap_ci_deterministic_and_covers():
    rng = np.random.default_rng(1)
    ref = rng.random(500) < 0.7
    cand = ref.copy()
    cand[:10] = False
    lo, hi = gate.paired_diff_ci(ref, cand)
    assert (lo, hi) == gate.paired_diff_ci(ref, cand)
    d = cand.mean() - ref.mean()
    assert lo <= d <= hi < 0.0 + 1e-12


def test_overall_and_bonferroni():
    assert gate.overall(["PASS", "PASS"]) == "PASS"
    assert gate.overall(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert gate.overall(["PASS", "FAIL"]) == "FAIL"
    # 6 discordant, all worse: exact McNemar p = 2 * 0.5**6 = 0.031 -> FAIL at 0.05, not at 0.05 / 3
    ref, cand = [True] * 100, [True] * 94 + [False] * 6
    one = gate.compare_task(pd.Series(ref), pd.Series(cand), 0.03, 0.05)
    three = gate.compare_task(pd.Series(ref), pd.Series(cand), 0.03, 0.05 / 3)
    assert one["verdict"] == "FAIL" and three["verdict"] != "FAIL"


def test_gate_all_skips_incomplete():
    ref = [True] * 10
    df = pd.concat([items({"F16": ref, "Q4": ref}), items({"Q5": ref[:5]})])
    out = gate.gate_all(df, {"mmlu": 10})
    assert set(out.candidate) == {"Q4"}


def test_gate_all_empty():
    empty = pd.DataFrame(columns=["hw_id", "model", "quant", "task", "id", "correct"])
    assert gate.gate_all(empty, {"mmlu": 10}).empty


def test_cross_model_upgrade():
    ref = [True] * 150 + [False] * 50          # 7B F16: 75%
    better = [True] * 190 + [False] * 10       # 14B Q4: 95%
    df = pd.concat([items({"F16": ref}, model="7b"), items({"Q4_K_M": better}, model="14b")])
    res = gate.gate(df, "gpu", "7b", "Q4_K_M", candidate_model="14b").set_index("task")
    assert res.loc["mmlu", "diff"] == pytest.approx(0.2) and res.loc["overall", "verdict"] == "PASS"
    assert res.loc["overall", "candidate_model"] == "14b"


def test_items_needed_formula_and_power():
    """n_needed gives ~80% PASS probability when the true difference is 0 (simulated paired data)."""
    p_disc, margin = 0.09, 0.03
    n = int(gate.items_needed(p_disc, margin))
    assert n == int(np.ceil((1.959964 + 0.841621) ** 2 * p_disc / margin ** 2))
    rng = np.random.default_rng(1)
    d = rng.choice([-1, 0, 1], p=[p_disc / 2, 1 - p_disc, p_disc / 2], size=(4000, n))
    lo = d.mean(1) - 1.959964 * d.std(1, ddof=1) / np.sqrt(n)
    assert abs((lo > -margin).mean() - 0.80) < 0.03
    assert gate.items_needed(0.1, 0.03, true_diff=-0.03) == float("inf")
    assert gate.items_needed(0.0, 0.03) == 0


def test_compare_task_reports_discordance():
    ref = pd.Series([True] * 90 + [False] * 10)
    cand = pd.Series([True] * 85 + [False] * 5 + [True] * 5 + [False] * 5)
    r = gate.compare_task(ref, cand, gate.MARGIN, gate.ALPHA)
    assert r["p_discordant"] == pytest.approx(0.10)
    assert r["n_needed"] == gate.items_needed(0.10)
