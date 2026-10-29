"""Size vs bits: frontier, matched-memory duels, perplexity screen."""
import numpy as np
import pandas as pd
import pytest

import frontier as fr

HW = "gpu"


def tables():
    # model: {quant: (size bytes, ppl, accuracy)}
    spec = {"S-1.5B": {"F16": (3.0e9, 10.0, 0.50), "Q4_K_M": (1.0e9, 10.5, 0.48)},
            "L-7B": {"F16": (14.0e9, 8.0, 0.80), "Q4_K_M": (4.4e9, 8.1, 0.79), "Q2_K": (2.9e9, 11.0, 0.65)}}
    q, s, items = [], [], []
    for m, d in spec.items():
        for quant, (size, ppl, acc) in d.items():
            ppl_f16 = d["F16"][1]
            q.append({"quality_hw_id": HW, "model": m, "quant": quant, "mmlu": acc, "gsm8k": acc, "humaneval": acc,
                      "ppl": ppl, "ppl_ratio_vs_ref": ppl / ppl_f16})
            s.append({"hw_id": HW, "model": m, "quant": quant, "ctx": 4096, "model_size_bytes": size})
            for task in ("mmlu", "gsm8k", "humaneval"):
                n_ok = int(round(acc * 200))
                items += [(HW, m, quant, task, i, i < n_ok) for i in range(200)]
    flips = pd.DataFrame([{"kind": "vs_ref", "task": "all", "model": "S-1.5B", "quant": "Q4_K_M", "flips": 0.01},
                          {"kind": "vs_ref", "task": "all", "model": "L-7B", "quant": "Q4_K_M", "flips": 0.03},
                          {"kind": "vs_ref", "task": "all", "model": "L-7B", "quant": "Q2_K", "flips": 0.2}])
    gates = pd.DataFrame([{"model": "L-7B", "candidate": "Q2_K", "task": "overall", "verdict": "FAIL"}])
    return (pd.DataFrame(q), pd.DataFrame(s),
            pd.DataFrame(items, columns=["hw_id", "model", "quant", "task", "id", "correct"]), flips, gates)


def test_params_b():
    assert fr.params_b("Qwen2.5-14B-Instruct") == 14 and fr.params_b("Qwen2.5-1.5B-Instruct") == 1.5


def test_frontier_table():
    q, s, _, flips, gates = tables()
    f = fr.frontier_table(q, s, flips, gates).set_index(["model", "quant"])
    assert f.loc[("L-7B", "Q4_K_M"), "bpw"] == pytest.approx(16 * 4.4 / 14)
    assert f.loc[("L-7B", "Q2_K"), "gate"] == "FAIL" and f.loc[("L-7B", "F16"), "gate"] == "REFERENCE"
    # S F16 (3.0 GB, 0.50) is dominated by L Q2_K (2.9 GB, 0.65); S Q4 is the smallest -> Pareto
    assert not f.loc[("S-1.5B", "F16"), "pareto"] and f.loc[("L-7B", "Q2_K"), "pareto"]
    assert f.loc[("S-1.5B", "Q4_K_M"), "pareto"] and f.loc[("L-7B", "F16"), "flips"] == 0


def test_size_vs_bits_duel():
    q, s, items, flips, gates = tables()
    d = fr.size_vs_bits(fr.frontier_table(q, s, flips, gates), items)
    pooled = d[(d.task == "all") & (d.big == "L-7B Q2_K")].iloc[0]
    assert pooled.small == "S-1.5B F16" and pooled.n == 600
    assert pooled.big_minus_small == pytest.approx(0.15) and pooled.diff_ci_low > 0 and pooled.p_mcnemar < 1e-6
    assert set(d.big) == {"L-7B Q2_K"}  # L Q4 (4.4 GB) has no S config within 0.8-1.25x


def test_ppl_screen():
    q, s, _, flips, gates = tables()
    scr = fr.ppl_screen(fr.frontier_table(q, s, flips, gates)).set_index("metric")
    assert scr.loc["flips", "n"] == 3 and scr.loc["flips", "spearman_rho"] == pytest.approx(0.5)
    assert scr.loc["quality_drop", "spearman_rho"] == pytest.approx(1.0)


def test_figures(tmp_path):
    q, s, _, flips, gates = tables()
    f = fr.frontier_table(q, s, flips, gates)
    fr.frontier_figure(f, tmp_path / "a.png")
    fr.screen_figure(f, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "b.png").stat().st_size > 0
    fr.screen_figure(f.drop(columns="flips"), tmp_path / "c.png")
    assert not (tmp_path / "c.png").exists() and np.isfinite(f.quality).all()
