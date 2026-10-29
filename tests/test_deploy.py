"""Deployment flow with the server, model and evaluation mocked."""
import json
import sys

import pandas as pd

import deploy


def fake_rows(task, n=10):
    return [{"id": i} for i in range(n)]


def patch_eval(monkeypatch, correct: bool):
    monkeypatch.setattr(deploy.qe, "load_items", lambda sample, task: fake_rows(task))
    monkeypatch.setattr(deploy.qe, "PROMPTS", {t: (lambda row: "question") for t in deploy.TASKS})
    monkeypatch.setattr(deploy.qe, "chat", lambda cfg, prompt, max_tokens: "ready")
    monkeypatch.setattr(deploy.qe, "score", lambda task, row, text, t: (None, correct))
    ref = pd.DataFrame([("hw", "m", "Q4_K_M", t, i, True) for t in deploy.TASKS for i in range(10)],
                       columns=["hw_id", "model", "quant", "task", "id", "correct"])
    monkeypatch.setattr(deploy, "reference_items", lambda cfg, model, quant: ref)


def test_acceptance_ok_and_fail(monkeypatch):
    cfg = deploy.load_config()
    patch_eval(monkeypatch, True)
    ok = deploy.acceptance(cfg, {}, "m", "Q4_K_M", 10, lambda m: None)
    assert ok["verdict"] == "OK" and ok["tasks"]["mmlu"]["cand_acc"] == 1.0
    patch_eval(monkeypatch, False)  # 10 of 10 worse on every task: exact McNemar p = 0.002 < 0.05 / 3
    bad = deploy.acceptance(cfg, {}, "m", "Q4_K_M", 10, lambda m: None)
    assert bad["verdict"] == "FAIL" and bad["tasks"]["gsm8k"]["verdict"] == "FAIL"


def test_sha256(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert deploy.sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class FakeServer:
    def __init__(self):
        self.stopped = False

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0


def setup_main(tmp_path, monkeypatch, argv):
    gguf, server_bin = tmp_path / "models/m-Q4_K_M.gguf", tmp_path / "bin/llama-server"
    for p in (gguf, server_bin):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/llama_cpp_commit.txt").write_text("c6824a9\n")
    hw = {"hw_id": "gpu", "os": "os", "cpu": "cpu", "gpu": "gpu", "gpu_driver": "1", "backend": "CUDA",
          "memory_bytes": 1, "vram_bytes": 8 * 2**30}
    server = FakeServer()
    monkeypatch.setattr(deploy, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(deploy, "detect_hw", lambda: hw)
    monkeypatch.setattr(deploy, "make_logger", lambda cfg, tag: lambda m: None)
    monkeypatch.setattr(deploy, "gguf_path", lambda cfg, m, q: gguf)
    monkeypatch.setattr(deploy, "resolve_bin", lambda rel: server_bin)
    monkeypatch.setattr(deploy.qe, "load_sample", lambda cfg, log: {"seed": 42, "datasets": {"mmlu": {"n": 1, "ids": [0]}}})
    monkeypatch.setattr(deploy.qe, "start_server", lambda cfg, path, log_path: server)
    patch_eval(monkeypatch, True)
    monkeypatch.setattr(sys, "argv", ["deploy.py", "--model", "m", "--quant", "Q4_K_M", "--items", "10", *argv])
    assert deploy.main() == 0
    (out,) = (tmp_path / "deploy").glob("*.json")
    return json.loads(out.read_text()), gguf, server


def test_main_writes_manifest(tmp_path, monkeypatch):
    m, gguf, server = setup_main(tmp_path, monkeypatch, [])
    assert m["status"] == "VERIFIED" and m["model"]["sha256"] == deploy.sha256(gguf)
    assert m["acceptance"]["verdict"] == "OK" and m["smoke"]["reply"] == "ready" and server.stopped
    assert m["runtime"]["llama_cpp_commit"] == "c6824a9" and "ids" not in m["sample"]["datasets"]["mmlu"]


def test_main_capacity(tmp_path, monkeypatch):
    levels = pd.DataFrame([{"hw_id": "gpu", "model": "m", "quant": "Q4_K_M", "level": c, "n": 16,
                            "out_tps_total": 40.0 * c, "ttft_p95_s": 0.1 * c, "tpot_p95_s": 0.02 * c,
                            "meets_server": c <= 4, "meets_interactive": c <= 2} for c in (1, 2, 4, 8)])
    calls = []
    monkeypatch.setattr(deploy.loadtest, "measure", lambda cfg, hw, model, quant, log: calls.append(model) or levels)
    m, _, _ = setup_main(tmp_path, monkeypatch, ["--capacity"])
    assert calls == ["m"] and m["capacity"]["users"] == {"server": 4, "interactive": 2}
    assert m["capacity"]["levels"][0]["meets_server"] is True and m["status"] == "VERIFIED"
