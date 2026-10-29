"""Report tables: markdown rendering and section builders on small inputs."""
import math

import pandas as pd

import report


def test_md_blanks_nan():
    out = report.md(pd.DataFrame({"a": [1, None], "b": [math.nan, "x"]}))
    assert out.splitlines() == ["| a | b |", "|---|---|", "| 1.0 |  |", "|  | x |"]


def test_formatters_and_sort():
    assert report.pct(0.1234) == "12.3" and report.pp(-0.03) == "-3.0" and report.pct(float("nan")) == ""
    df = pd.DataFrame({"model": ["Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct"],
                       "quant": ["Q4_K_M", "F16", "F16"]})
    assert report.qsort(df).values.tolist() == [["Qwen2.5-1.5B-Instruct", "F16"], ["Qwen2.5-7B-Instruct", "F16"],
                                                ["Qwen2.5-7B-Instruct", "Q4_K_M"]]
    assert report.short("Qwen2.5-14B-Instruct") == "14B"


def test_capacity_section():
    cfg = {"loadtest": {"max_tokens": 256, "slots": 16, "sla_percentile": 95,
                        "sla": {"server": {"ttft_s": 2.0, "tpot_s": 0.2}}}}
    cap = pd.DataFrame([{"hw_id": "h", "model": "Qwen2.5-7B-Instruct", "quant": "Q4_K_M", "capacity_server": 8,
                         "capacity_interactive": 2, "out_tps_total_at_server": 300.0, "max_out_tps_total": 320.0}])
    text = "\n".join(report.capacity_section({"capacity": cap}, cfg))
    assert "| h | 7B | Q4_K_M | 8 | 2 | 300 | 320 |" in text and "TTFT <= 2000 ms" in text
    assert report.capacity_section({}, cfg) == []


def test_api_errors(tmp_path):
    d = tmp_path / "hw/m/Q3_K_M"
    d.mkdir(parents=True)
    (d / "gsm8k.jsonl").write_text('{"id": 1, "correct": true}\n{"id": 2, "correct": false, "api_error": "x"}\n')
    (d / "mmlu.jsonl").write_text('{"id": 1, "correct": true, "api_error": "y"}\n')
    assert report.api_errors(tmp_path) == {("m", "Q3_K_M"): 2}
