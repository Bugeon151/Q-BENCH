"""quality_eval.chat: normal reply, and the raw-completion fallback when llama-server cannot parse a reply."""
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import quality_eval as qe

PARSE_500 = {"error": {"code": 500, "message": "The model produced output that does not match the expected "
                       "peg-native format", "type": "server_error"}}


class Fake(BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append((self.path, body))
        if self.path == "/v1/chat/completions":
            if Fake.mode == "ok":
                return self.reply(200, {"choices": [{"message": {"content": "The answer is 55"}}]})
            return self.reply(500, PARSE_500 if Fake.mode in ("parse", "both") else {"error": {"message": "other"}})
        if self.path == "/apply-template":
            return self.reply(200, {"prompt": "<|im_start|>user\n" + body["messages"][0]["content"]})
        if self.path == "/completion":
            if Fake.mode == "both":
                return self.reply(500, PARSE_500)
            return self.reply(200, {"content": "(2) �orning\nThe answer is 55"})
        self.reply(404, {})

    def reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def cfg():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.seen = []
    yield {"quality": {"port": srv.server_port, "request_timeout_sec": 10}, "inference": {"temperature": 0, "seed": 42}}
    srv.shutdown()


def test_chat_ok(cfg):
    Fake.mode = "ok"
    errors = []
    assert qe.chat(cfg, "q", 16, errors) == "The answer is 55" and errors == []
    assert [p for p, _ in Fake.seen] == ["/v1/chat/completions"]


def test_chat_parse_error_falls_back_to_raw_completion(cfg):
    Fake.mode = "parse"
    errors = []
    text = qe.chat(cfg, "q", 16, errors)
    assert text.endswith("The answer is 55") and "peg-native" in errors[0]
    paths = [p for p, _ in Fake.seen]
    assert paths == ["/v1/chat/completions", "/apply-template", "/completion"]
    comp = Fake.seen[2][1]
    assert comp["prompt"] == "<|im_start|>user\nq" and comp["n_predict"] == 16 and comp["temperature"] == 0
    assert comp["seed"] == 42 and comp["cache_prompt"] is False
    assert qe.score("gsm8k", {"answer": "#### 55"}, text, 10)[1]


def test_chat_other_500_raises(cfg):
    Fake.mode = "other"
    with pytest.raises(urllib.error.HTTPError):
        qe.chat(cfg, "q", 16)


def test_chat_unrecoverable_parse_error_returns_empty(cfg):
    Fake.mode = "both"
    errors = []
    assert qe.chat(cfg, "q", 16, errors) == "" and len(errors) == 2
    assert [p for p, _ in Fake.seen] == ["/v1/chat/completions", "/apply-template", "/completion"]
    assert not qe.score("gsm8k", {"answer": "#### 55"}, "", 10)[1]


class Proc:
    def __init__(self, crash):
        self.crash, self.returncode = crash, None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if self.crash:
            self.returncode = 1
            return 1
        raise qe.subprocess.TimeoutExpired("llama-server", timeout)

    def terminate(self):
        self.returncode = 0


def main_with(monkeypatch, tmp_path, crashes):
    """Runs quality_eval.main with fakes; run_tasks raises ConnectionResetError while crashes remain."""
    procs, calls = [], []
    cfg = {"quality": {"n_parallel": 1, "ctx_per_slot": 64}, "paths": {"quality_dir": "q", "logs_dir": "logs"}}
    monkeypatch.setattr(qe, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(qe, "load_config", lambda: cfg)
    monkeypatch.setattr(qe, "make_logger", lambda c, t: lambda m: None)
    monkeypatch.setattr(qe, "load_sample", lambda c, log: {})
    monkeypatch.setattr(qe, "detect_hw", lambda: {"hw_id": "gpu"})
    monkeypatch.setattr(qe, "load_items", lambda sample, t: [{"id": 1}])
    monkeypatch.setattr(qe, "select_models", lambda c, m: [{"name": "m"}])
    monkeypatch.setattr(qe, "check_budget", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(qe, "gguf_path", lambda c, m, q: tmp_path / "m.gguf")
    monkeypatch.setattr(qe, "write_summary", lambda *a: "summary")

    def start(cfg_, path, log_path):
        procs.append(Proc(crash=len(procs) < crashes))
        calls.append(log_path.name)
        return procs[-1]

    def run(*a):
        if len(calls) <= crashes:
            raise ConnectionResetError(10054, "reset")

    monkeypatch.setattr(qe, "start_server", start)
    monkeypatch.setattr(qe, "run_tasks", run)
    monkeypatch.setattr(qe.sys, "argv", ["quality_eval.py", "--quants", "Q6_K", "--tasks", "mmlu"])
    return calls


def test_server_crash_is_restarted(monkeypatch, tmp_path):
    calls = main_with(monkeypatch, tmp_path, crashes=1)
    qe.main()
    assert calls == ["server_m_Q6_K.log", "server_m_Q6_K.restart1.log"]


def test_repeated_crash_gives_up(monkeypatch, tmp_path):
    main_with(monkeypatch, tmp_path, crashes=3)
    with pytest.raises(ConnectionResetError):
        qe.main()


def test_connection_error_with_live_server_is_not_retried(monkeypatch, tmp_path):
    """A connection error while llama-server keeps running is not a crash: no restart."""
    calls = main_with(monkeypatch, tmp_path, crashes=0)
    monkeypatch.setattr(qe, "run_tasks", lambda *a: (_ for _ in ()).throw(ConnectionResetError(10054, "reset")))
    with pytest.raises(ConnectionResetError):
        qe.main()
    assert calls == ["server_m_Q6_K.log"]
