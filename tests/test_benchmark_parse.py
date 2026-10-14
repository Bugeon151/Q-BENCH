"""Parsing of llama-bench JSONL (fixture copied from tools/llama-bench/README.md @ c6824a9)."""
from pathlib import Path

from benchmark import BENCH_META_FIELDS, RAW_CSV_FIELDS, TEST_GROUPS, parse_jsonl_rows, to_raw_rows

FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench_readme.jsonl"


def test_parse_and_flatten():
    recs = parse_jsonl_rows(FIXTURE.read_text())
    assert len(recs) == 2
    rows = to_raw_rows(recs, {"hw_id": "hw", "model": "m", "quant": "Q4_K_M", "ctx": 512, "test_type": "pp"})
    assert len(rows) == 10  # 5 samples x 2 records
    r0 = rows[0]
    assert r0["latency_ms"] == 71551000 / 1e6 and r0["tokens_per_sec"] == 7155.74
    assert (r0["n_prompt"], r0["n_gen"], r0["n_depth"]) == (512, 0, 0)
    assert rows[5]["n_gen"] == 128 and rows[5]["run_index"] == 0
    meta = {k: r0[k] for k in BENCH_META_FIELDS}
    assert meta == {"build_commit": "8cf427ff", "n_threads": 8, "n_gpu_layers": -1, "flash_attn": -1,
                    "n_batch": 2048, "n_ubatch": 512, "type_k": "f16", "type_v": "f16",
                    "gpu_info": "NVIDIA GeForce RTX 4080", "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor",
                    "backends": "CUDA"}
    assert set(r0) <= set(RAW_CSV_FIELDS)


def test_test_groups_suppress_defaults():
    assert TEST_GROUPS["pg"](4096, 256) == ["-pg", "4096,256", "-p", "0", "-n", "0"]
    assert TEST_GROUPS["tg"](4096, 256) == ["-d", "4096", "-p", "0", "-n", "256"]
    assert TEST_GROUPS["pp"](512, 256) == ["-p", "512", "-n", "0"]
