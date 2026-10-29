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


def test_main_writes_manifest(tmp_path, monkeypatch):
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
    monkeypatch.setattr(sys, "argv", ["deploy.py", "--model", "m", "--quant", "Q4_K_M", "--items", "10"])
    assert deploy.main() == 0
    (out,) = (tmp_path / "deploy").glob("*.json")
    m = json.loads(out.read_text())
    assert m["status"] == "VERIFIED" and m["model"]["sha256"] == deploy.sha256(gguf)
    assert m["acceptance"]["verdict"] == "OK" and m["smoke"]["reply"] == "ready" and server.stopped
    assert m["runtime"]["llama_cpp_commit"] == "c6824a9" and "ids" not in m["sample"]["datasets"]["mmlu"]
