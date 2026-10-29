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
