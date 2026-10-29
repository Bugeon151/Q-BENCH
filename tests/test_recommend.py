"""Hardware-fit recommendation from measured tables."""
import pandas as pd
import pytest

from common import GIB
import recommend as rc
from recommend import candidates, kv_bytes_per_token, recommend

QWEN7B = {"num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}


def _inputs():
    hw = "nvidia-x-8gb"
    summary = pd.DataFrame([
        # model, quant, size GB, peak VRAM GB (None = not measured on GPU), tps, offloaded
        ("m7", "F16", 15.2, 6.5, 7.2, True),
        ("m7", "Q4_K_M", 4.7, 5.1, 69.3, False),
        ("m7", "Q2_K", 3.0, 3.6, 90.7, False),
        ("m14", "Q4_K_M", 9.0, None, None, False),
    ], columns=["model", "quant", "size", "vram", "tps", "partial_offload"])
    summary = summary.assign(hw_id=hw, ctx=4096, model_size_bytes=summary["size"] * GIB,
                             peak_vram_bytes=summary["vram"] * GIB, generation_throughput_tps=summary["tps"])
    quality = pd.DataFrame([("m7", "F16", .72, .94, .83), ("m7", "Q4_K_M", .72, .93, .80),
                            ("m7", "Q2_K", .67, .82, .60), ("m14", "Q4_K_M", .78, .95, .85)],
                           columns=["model", "quant", "mmlu", "gsm8k", "humaneval"])
    gates = pd.DataFrame([("m7", "Q4_K_M", "overall", "INCONCLUSIVE"), ("m7", "Q2_K", "overall", "FAIL")],
                         columns=["model", "candidate", "task", "verdict"])
    cfgs = {"m7": QWEN7B, "m14": {"num_hidden_layers": 48, "num_attention_heads": 40,
                                  "num_key_value_heads": 8, "hidden_size": 5120}}
    return summary, quality, gates, cfgs, 4096, 256, 0.75, None


def tables():
    summary, quality, gates, cfgs, ctx, n_gen, overhead, _ = _inputs()
    return candidates(summary, quality, gates, cfgs, ctx, n_gen, overhead, "nvidia-x-8gb")


def test_kv_bytes_per_token():
    assert kv_bytes_per_token(QWEN7B) == 2 * 28 * 4 * 128 * 2


def test_candidates_need_basis():
    c = tables().set_index(["model", "quant"])
    assert c.loc[("m7", "Q4_K_M"), "need_basis"] == "measured" and c.loc[("m7", "Q4_K_M"), "need_gib"] == 5.1
    assert c.loc[("m7", "F16"), "need_basis"] == "estimated"  # offloaded run: VRAM peak is not the need
    assert c.loc[("m14", "Q4_K_M"), "need_basis"] == "estimated" and c.loc[("m14", "Q4_K_M"), "need_gib"] > 9.0


def test_recommend_8gb_gpu_only():
    rec = recommend(tables(), vram_gb=8)
    assert list(zip(rec.model, rec.quant)) == [("m7", "Q4_K_M")]  # F16/14B do not fit, Q2_K fails the gate


def test_recommend_with_offload_and_min_tps():
    rec = recommend(tables(), vram_gb=8, ram_gb=24, allow_offload=True)
    assert (rec.model.iloc[0], rec.quant.iloc[0]) == ("m14", "Q4_K_M")  # best quality, speed unmeasured
    rec = recommend(tables(), vram_gb=8, ram_gb=24, min_tps=30, allow_offload=True)
    assert ("m7", "F16") not in set(zip(rec.model, rec.quant))  # 7.2 tok/s < 30


def test_predicted_speed_for_unmeasured_hardware():
    from recommend import candidates
    t = tables()
    hw_less = candidates(*_inputs(), bandwidth_gbs=1000, efficiencies=[0.6, 0.8])
    r = hw_less.set_index(["model", "quant"]).loc[("m7", "Q4_K_M")]
    b = 4.7 * GIB + kv_bytes_per_token(QWEN7B) * 4096
    assert r.tps_pred_low == pytest.approx(0.6e12 / b) and r.tps_pred_high == pytest.approx(0.8e12 / b)
    assert pd.isna(r.tps)  # no measured speed on the unmeasured target
    rec = recommend(hw_less, vram_gb=24, min_tps=0.6e12 / b + 1)  # filter uses the low end of the range
    assert ("m7", "Q4_K_M") not in set(zip(rec.model, rec.quant))
    assert len(t)


def test_annotate_flips_and_capacity(tmp_path):
    rec = pd.DataFrame({"model": ["m", "m"], "quant": ["F16", "Q4_K_M"], "quality": [0.8, 0.79]})
    pd.DataFrame([{"kind": "vs_F16", "task": "all", "model": "m", "quant": "Q4_K_M", "flips": 0.04},
                  {"kind": "rerun", "task": "all", "model": "m", "quant": "Q4_K_M", "flips": 0.01}]) \
        .to_csv(tmp_path / "flips.csv", index=False)
    pd.DataFrame([{"hw_id": "gpu", "model": "m", "quant": "Q4_K_M", "capacity_server": 8, "capacity_interactive": 2},
                  {"hw_id": "other", "model": "m", "quant": "F16", "capacity_server": 1, "capacity_interactive": 0}]) \
        .to_csv(tmp_path / "capacity.csv", index=False)
    out = rc.annotate(rec, tmp_path, "gpu").set_index("quant")
    assert out.loc["F16", "flips"] == 0 and out.loc["Q4_K_M", "flips"] == 0.04
    assert out.loc["Q4_K_M", "capacity_server"] == 8 and pd.isna(out.loc["F16", "capacity_server"])
    assert rc.annotate(rec, tmp_path / "missing", "gpu").columns.tolist() == rec.columns.tolist()
