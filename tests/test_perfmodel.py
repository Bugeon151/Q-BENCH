"""Decode-speed model fits and leave-one-hardware-out evaluation."""
import numpy as np
import pandas as pd
import pytest

import perfmodel

CFG7B = {"num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}


def test_fit_recovers_parameters():
    b = np.array([3e9, 4.7e9, 6.3e9, 8.1e9, 15.2e9])
    assert perfmodel.fit_1p(b, 300e9 / b) == pytest.approx(300e9)
    bw, t0 = perfmodel.fit_2p(b, 1 / (b / 370e9 + 0.0016))
    assert bw == pytest.approx(370e9) and t0 == pytest.approx(0.0016)


def test_decode_points_bytes_per_token():
    s = pd.DataFrame([{"hw_id": "h", "model": "m", "ctx": 4096, "quant": "Q4_K_M", "model_size_bytes": 4.7e9,
                       "generation_throughput_tps": 69.3, "partial_offload": False},
                      {"hw_id": "h", "model": "m", "ctx": 4096, "quant": "F16", "model_size_bytes": 15.2e9,
                       "generation_throughput_tps": 7.2, "partial_offload": True}])
    p = perfmodel.decode_points(s, {"m": CFG7B})
    assert len(p) == 1 and p.bytes_per_token.iloc[0] == 4.7e9 + 2 * 28 * 4 * 128 * 2 * 4096


def test_leave_one_hardware_out():
    # two devices with the same efficiency (0.75): cross-device prediction is exact
    rows = []
    for hw, peak in (("a", 150), ("b", 400)):
        for size in (3e9, 5e9, 8e9):
            rows.append({"hw_id": hw, "model": "m", "ctx": 0, "quant": str(size), "bytes_per_token": size,
                         "generation_throughput_tps": 0.75 * peak * 1e9 / size})
    fit, pts = perfmodel.evaluate(pd.DataFrame(rows), {"a": {"bandwidth_gbs": [150]}, "b": {"bandwidth_gbs": [400]}})
    assert fit.efficiency.tolist() == pytest.approx([0.75, 0.75])
    assert fit.loho_mape_max.tolist() == pytest.approx([0, 0], abs=1e-12)
    assert perfmodel.predict_tps(1000, 5e9, [0.7, 0.8]) == pytest.approx((140.0, 160.0))
