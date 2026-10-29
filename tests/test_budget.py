"""Memory-budget skip rules. GGUF sizes: 7B values measured in v0.1; 14B values are estimates."""
import json

import pytest

from common import GIB, check_budget, kv_cache_bytes, load_config

M5 = {"backend": "Metal", "memory_bytes": 24 * GIB, "vram_bytes": 0}
LAPTOP = {"backend": "CUDA", "memory_bytes": 32 * GIB, "vram_bytes": 8151 * 1024**2}  # nvidia-smi memory.total
QWEN = {  # num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_size (HF config.json)
    "Qwen2.5-7B-Instruct": (28, 28, 4, 3584),
    "Qwen2.5-14B-Instruct": (48, 40, 8, 5120),
}
SIZE = {
    ("Qwen2.5-7B-Instruct", "F16"): 15_237_853_760,
    ("Qwen2.5-7B-Instruct", "Q8_0"): 8_098_525_760,
    ("Qwen2.5-7B-Instruct", "Q4_K_M"): 4_683_074_112,
    ("Qwen2.5-14B-Instruct", "F16"): 29_540_000_000,
    ("Qwen2.5-14B-Instruct", "Q8_0"): 15_700_000_000,
    ("Qwen2.5-14B-Instruct", "Q4_K_M"): 8_990_000_000,
}


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["model_dirs"]["hf"] = str(tmp_path)
    for name, (l, h, kv, d) in QWEN.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(
            {"num_hidden_layers": l, "num_attention_heads": h, "num_key_value_heads": kv, "hidden_size": d}))
    return c


def fits(cfg, hw, model, quant, ctx=4096, **kw):
    return check_budget(cfg, model, quant, ctx, hw, SIZE[(model, quant)], **kw)[0]


def test_kv_cache(cfg):
    assert kv_cache_bytes(cfg, "Qwen2.5-7B-Instruct", 4096) == 2 * 28 * 4 * 128 * 4096 * 2


def test_m5_rules(cfg):
    assert fits(cfg, M5, "Qwen2.5-7B-Instruct", "F16", 8192 + 256)
    assert not fits(cfg, M5, "Qwen2.5-14B-Instruct", "F16")
    assert not fits(cfg, M5, "Qwen2.5-14B-Instruct", "Q8_0", 512)  # via memory_budget.exclude
    assert fits(cfg, M5, "Qwen2.5-14B-Instruct", "Q4_K_M", 8192 + 256)


def test_laptop_rules(cfg):
    assert not fits(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "F16")
    assert not fits(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "Q8_0", 512)
    assert fits(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "Q4_K_M", 8192 + 256)
    assert fits(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "F16", allow_host_offload=True)


def test_skip_disabled(cfg):
    cfg["memory_budget"]["skip_over_budget"] = False
    cfg["memory_budget"]["exclude"] = []
    assert fits(cfg, LAPTOP, "Qwen2.5-14B-Instruct", "F16")


def test_plan_combo_offload(cfg):
    from benchmark import gpu_args, plan_combo
    size = SIZE[("Qwen2.5-7B-Instruct", "F16")]
    assert plan_combo(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "F16", 4096, size)[0] == "offload"
    assert plan_combo(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "F16", 8192, size)[0] is None  # default ctx only
    assert plan_combo(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "Q4_K_M", 8192,
                      SIZE[("Qwen2.5-7B-Instruct", "Q4_K_M")])[0] == "gpu"
    assert plan_combo(cfg, M5, "Qwen2.5-14B-Instruct", "F16", 4096,
                      SIZE[("Qwen2.5-14B-Instruct", "F16")])[0] is None  # Metal: no partial offload
    cfg["benchmark"]["offload"]["enabled"] = False
    assert plan_combo(cfg, LAPTOP, "Qwen2.5-7B-Instruct", "F16", 4096, size)[0] is None
    assert gpu_args(cfg["benchmark"], True) == ["-fitt", "1024"]
    assert gpu_args(cfg["benchmark"], False) == ["-ngl", "99"]
