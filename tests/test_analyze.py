import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze
from common import load_config

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "summary_v0.1.csv"


def test_legacy_regression():
    """analyze on the v0.1 raw CSV reproduces the v0.1 summary.csv values."""
    cfg = load_config()
    raw = pd.read_csv(REPO / cfg["paths"]["raw_csv"]).assign(
        hw_id=analyze.LEGACY_HW_ID, model=analyze.LEGACY_MODEL, ctx=4096)
    for c in ("peak_vram_bytes", "swap_used_before_bytes", "swap_used_after_bytes"):
        raw[c] = np.nan
    summary = analyze.summarize(analyze.compute_stats(raw), 0.03).set_index("quant")
    old = pd.read_csv(FIXTURE).set_index("quant")
    assert list(summary.index) == list(old.index)
    np.testing.assert_allclose(summary[old.columns].to_numpy(float), old.to_numpy(float), rtol=0, atol=1e-9)
    assert summary.loc["F16", "cv_warning"] == "tg:8.0%"
    assert (summary.drop("F16").cv_warning == "").all()


def _raw(hw, ctxs, quants, rng):
    rows = []
    for ctx in ctxs:
        for qi, q in enumerate(quants):
            for tt in ("pp", "tg", "pg"):
                for i in range(10):
                    tps = (500 if tt == "pp" else 20 * (qi + 1)) * rng.normal(1, 0.01)
                    rows.append({"hw_id": hw, "model": "Qwen2.5-7B-Instruct", "quant": q, "ctx": ctx,
                                 "test_type": tt, "run_index": i, "n_prompt": 0 if tt == "tg" else ctx,
                                 "n_gen": 0 if tt == "pp" else 256, "n_depth": ctx if tt == "tg" else 0,
                                 "latency_ms": 1000 * ctx / tps, "tokens_per_sec": tps,
                                 "model_size_bytes": int(15e9 / (qi + 1)), "peak_mem_bytes": int(1e9),
                                 "peak_vram_bytes": int(8e9) if hw == "gpu" else None,
                                 "swap_used_before_bytes": 0, "swap_used_after_bytes": 0})
    return pd.DataFrame(rows)


def test_pipeline_synthetic(tmp_path, monkeypatch):
    cfg = load_config()
    rng = np.random.default_rng(0)
    for hw in ("mac", "gpu"):
        d = tmp_path / "data/raw" / hw
        d.mkdir(parents=True)
        _raw(hw, [512, 4096], ["F16", "Q4_K_M"], rng).to_csv(d / "benchmark_raw.csv", index=False)
    qd = tmp_path / "data/quality/gpu"
    qd.mkdir(parents=True)
    pd.DataFrame([{"hw_id": "gpu", "model": "Qwen2.5-7B-Instruct", "quant": q, "ppl": p, "ppl_err": 0.05}
                  for q, p in (("F16", 7.0), ("Q4_K_M", 7.2))]).to_csv(qd / "perplexity.csv", index=False)
    pd.DataFrame([{"hw_id": "gpu", "model": "Qwen2.5-7B-Instruct", "quant": q, "task": t, "n": 10,
                   "expected_n": 10, "correct": 0, "accuracy": a}
                  for q, a in (("F16", 0.8), ("Q4_K_M", 0.76)) for t in ("mmlu", "gsm8k", "humaneval")]
                 ).to_csv(qd / "quality.csv", index=False)
    shutil.copytree(REPO / "config", tmp_path / "config")
    monkeypatch.setattr(analyze, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(analyze, "load_config", lambda: cfg)
    analyze.main()

    summary = pd.read_csv(tmp_path / "results/tables/summary.csv")
    assert len(summary) == 8 and set(summary.ctx) == {512, 4096}
    quality = pd.read_csv(tmp_path / "results/tables/quality.csv").set_index("quant")
    assert quality.loc["Q4_K_M", "quality_retention"] == pytest.approx(0.95)
    assert quality.loc["Q4_K_M", "ppl_ratio_vs_f16"] == pytest.approx(7.2 / 7.0)
    assert (tmp_path / "results/tables/hardware_comparison.csv").exists()
    figs = tmp_path / "results/figures"
    for f in ("gpu/Qwen2.5-7B-Instruct/ttft.png", "gpu/Qwen2.5-7B-Instruct/ctx_throughput.png",
              "gpu/Qwen2.5-7B-Instruct/ctx_ttft.png", "gpu/Qwen2.5-7B-Instruct/pareto.png",
              "quality/gpu_Qwen2.5-7B-Instruct.png", "hardware/Qwen2.5-7B-Instruct_generation_throughput.png"):
        assert (figs / f).exists(), f


def test_pareto_mask():
    m = analyze.pareto_mask(np.array([1.0, 2.0, 3.0, 2.5]), np.array([1.0, 0.9, 0.5, 0.4]))
    assert m.tolist() == [True, True, True, False]
