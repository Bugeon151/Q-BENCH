"""Load test: streamed timing against a fake OpenAI-compatible server, SLA summary and capacity."""
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

import loadtest as lt

LT = {"sla_percentile": 95, "sla": {"server": {"ttft_s": 2.0, "tpot_s": 0.2},
                                    "interactive": {"ttft_s": 0.45, "tpot_s": 0.04}}}


class Fake(BaseHTTPRequestHandler):
    active = 0
    peak = 0
    lock = threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] and body["stream_options"]["include_usage"]
        with Fake.lock:
            Fake.active += 1
            Fake.peak = max(Fake.peak, Fake.active)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05)  # prefill
        for tok in ["a", "b", "c", "d", "e"]:
            self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": tok}}]})}\n\n'.encode())
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b'data: {"choices": [], "usage": {"completion_tokens": 5}}\n\ndata: [DONE]\n\n')
        with Fake.lock:
            Fake.active -= 1

    def log_message(self, *a):
        pass


@pytest.fixture
def url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
    srv.shutdown()


def test_stream_timing(url):
    r = lt.stream_chat(url, "hi", 8, 10, urllib.request.build_opener(urllib.request.ProxyHandler({})))
    assert r["tokens"] == 5
    assert 0.04 < r["ttft_s"] < r["latency_s"]
    assert r["tpot_s"] == pytest.approx((r["latency_s"] - r["ttft_s"]) / 4)


def test_closed_loop_concurrency(url):
    Fake.peak = 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows, wall = lt.run_level(lambda p: lt.stream_chat(url, p, 8, 10, opener), ["q"], 3, 9)
    assert len(rows) == 9 and Fake.peak == 3 and wall > 0


def raw(level, ttft, tpot, n=20, wall=10.0, tokens=100):
    return [{"hw_id": "h", "model": "m", "quant": "q", "level": level, "level_wall_s": wall,
             "ttft_s": ttft, "tpot_s": tpot, "tokens": tokens, "latency_s": ttft + tpot * 99} for _ in range(n)]


def test_summary_and_capacity():
    rows = raw(1, 0.2, 0.02) + raw(2, 0.3, 0.05) + raw(4, 1.0, 0.15) + raw(8, 3.0, 0.3)
    s = lt.summarize(pd.DataFrame(rows), LT)
    assert s.meets_interactive.tolist() == [True, False, False, False]
    assert s.meets_server.tolist() == [True, True, True, False]
    assert s.out_tps_total.iloc[0] == pytest.approx(20 * 100 / 10.0)
    assert s.user_tps_p50.iloc[0] == pytest.approx(50)
    cap = lt.capacity(s, LT["sla"]).iloc[0]
    assert (cap.capacity_server, cap.capacity_interactive) == (4, 1)
    assert cap.max_out_tps_total == pytest.approx(200)


def test_no_level_meets():
    s = lt.summarize(pd.DataFrame(raw(1, 5.0, 0.5)), LT)
    cap = lt.capacity(s, LT["sla"]).iloc[0]
    assert cap.capacity_server == 0 and pd.isna(cap.out_tps_total_at_server)
