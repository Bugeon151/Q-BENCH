"""imatrix_fix: running-PPL parsing and the check -> imatrix -> requantize flow with fake binaries."""
import csv

import imatrix_fix as im

LOG = "[1]15152.6493,[2]19662.1331,[19]22399.7226,[20]22505.8397,[21]22293.9501,\nFinal estimate: PPL = 22581.78 +/- 206.39"


def test_running_ppl():
    assert im.running_ppl(LOG, 20) == 22505.8397 and im.running_ppl(LOG, 2) == 19662.1331
    assert im.running_ppl(LOG, 99) is None
    assert im.running_ppl("[3]1.2e+05,", 3) == 1.2e5


def test_flow(tmp_path, monkeypatch):
    for d in ("logs", "data/datasets/w", "models/base", "models/quantized"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "data/datasets/w/train.raw").write_text("text")
    (tmp_path / "data/datasets/w/test.raw").write_text("text")
    (tmp_path / "models/base/m-F16.gguf").write_bytes(b"f")
    (tmp_path / "models/quantized/m-Q2_K.gguf").write_bytes(b"q")
    (tmp_path / "logs/perplexity_m_Q2_K.log").write_text(LOG)
    cfg = {"imatrix": {"llama_imatrix_bin": "imat", "train_file": "data/datasets/w/train.raw", "chunks": 4,
                       "dir": "models/imatrix", "check_chunks": 20, "targets": [["m", "Q2_K"]]},
           "perplexity": {"llama_perplexity_bin": "ppl", "dataset_file": "data/datasets/w/test.raw", "ctx": 512,
                          "n_gpu_layers": "auto"},
           "benchmark": {"llama_quantize_bin": "quant"}, "paths": {"logs_dir": "logs", "quality_dir": "data/quality"}}
    calls = []

    def fake_run(cmd, log_path):
        calls.append(cmd)
        if cmd[0] == "imat":
            open(cmd[cmd.index("-o") + 1], "wb").close()
        if cmd[0] == "quant":
            open(cmd[-2], "wb").close()
        return "Final estimate: PPL = 22500.0000 +/- 200.0"

    monkeypatch.setattr(im, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(im, "load_config", lambda: cfg)
    monkeypatch.setattr(im, "make_logger", lambda c, t: lambda m: None)
    monkeypatch.setattr(im, "detect_hw", lambda: {"hw_id": "gpu"})
    monkeypatch.setattr(im, "resolve_bin", lambda rel: rel)
    monkeypatch.setattr(im, "gguf_path", lambda c, m, q: tmp_path / ("models/base" if q == "F16" else "models/quantized")
                        / f"{m}-{q}.gguf")
    monkeypatch.setattr(im, "run", fake_run)
    im.main()
    assert [c[0] for c in calls] == ["ppl", "imat", "quant"]
    assert calls[0][calls[0].index("-ngl") + 1] == "0" and calls[0][calls[0].index("--chunks") + 1] == "20"
    assert "train.raw" in calls[1][calls[1].index("-f") + 1] and calls[2][1] == "--imatrix" and calls[2][-1] == "Q2_K"
    assert (tmp_path / "models/quantized/m-Q2_K-imat.gguf").exists()
    rows = list(csv.DictReader(open(tmp_path / "data/quality/gpu/kernel_check.csv")))
    assert float(rows[0]["ppl_gpu"]) == 22505.8397 and abs(float(rows[0]["cpu_over_gpu"]) - 0.99974) < 1e-4
    calls.clear()
    im.main()  # everything done: resumes without running anything
    assert calls == []
