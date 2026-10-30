"""Cause isolation: one-slot determinism, quantization flips per condition, batching flips."""
import pandas as pd
import pytest

import isolation as iso

COLS = ["hw_id", "model", "quant", "task", "id", "correct", "pred", "response"]


def run(quant, correct, preds=None):
    preds = preds or ["A" if c else "B" for c in correct]
    return [("gpu", "m", quant, "mmlu", i, c, p, f"t{p}") for i, (c, p) in enumerate(zip(correct, preds))]


def df(*parts):
    return pd.DataFrame([r for p in parts for r in p], columns=COLS)


REF = [True] * 80 + [False] * 20
Q4 = [False] * 8 + [True] * 72 + [False] * 20            # 8 flips vs REF
Q4_NOISY = [False] * 8 + [True] * 68 + [False] * 2 + [True] * 4 + [False] * 18  # differs from Q4 on 4 items


def test_one_slot_is_deterministic_and_isolates_quantization():
    runs = {"np1_r1": df(run("F16", REF), run("Q4", Q4)), "np1_r2": df(run("F16", REF), run("Q4", Q4)),
            "np4_r1": df(run("F16", REF), run("Q4", Q4_NOISY)), "np4_r2": df(run("F16", REF), run("Q4", Q4))}
    t = iso.pairs_table(runs)
    s = iso.summarize(t).set_index(["quant", "kind", "slots"])
    assert s.loc[("Q4", "rerun", 1), "flips_max"] == 0          # identical one-slot runs
    assert s.loc[("F16", "rerun", 1), "flips_max"] == 0
    assert s.loc[("Q4", "vs_ref", 1), "flips_mean"] == pytest.approx(0.08)
    assert s.loc[("Q4", "vs_ref", 4), "pairs"] == 2
    assert s.loc[("Q4", "vs_ref", 4), "flips_mean"] == pytest.approx((0.12 + 0.08) / 2)
    assert s.loc[("Q4", "rerun", 4), "flips_mean"] == pytest.approx(0.04)   # r1 vs r2 differ on 4 items
    assert s.loc[("Q4", "np1_vs_np4", 4), "pairs"] == 2
    assert s.loc[("Q4", "np1_vs_np4", 4), "flips_max"] == pytest.approx(0.04)


def test_incomplete_runs_dropped_and_empty():
    runs = {"np1_r1": df(run("F16", REF), run("Q4", Q4)), "np4_r1": df(run("F16", REF), run("Q4", Q4[:50]))}
    t = iso.pairs_table(runs)
    assert set(t[t.kind == "vs_ref"].slots) == {1}
    assert iso.pairs_table({}).empty


def test_slots_parse():
    assert iso.slots("np4_r3") == 4 and iso.slots("np1_r1") == 1
