"""Offline tests for agentchord_probe.gateway (no model calls: fake backend + a local stub upstream)."""
import json, os, threading, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from world_model.v2.agentchord_probe import gateway as gw

MSGS = [{"role": "system", "content": "plan"}, {"role": "user", "content": [{"type": "text", "text": "pour water"}]}]


def _start(backend, log, reuse=False):
    srv, lg = gw.serve(backend, log, 0, reuse=reuse)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, lg, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def _chat(url, messages=MSGS, **kw):
    body = dict(model="gpt-5", messages=messages, temperature=0.0, **kw)
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as f:
        return json.loads(f.read())


def test_fake_logs_and_replays(tmp_path):
    log = str(tmp_path / "llm_log.jsonl")
    srv, _, url = _start(gw.Fake(), log)
    a = _chat(url)["choices"][0]["message"]["content"]
    srv.shutdown()
    rows = [json.loads(l) for l in open(log)]
    assert len(rows) == 1 and rows[0]["reply"] == a and rows[0]["requested_model"] == "gpt-5" and rows[0]["backend"] == "fake"
    srv, _, url = _start(gw.Replay(), log)                  # replay: same request -> byte-identical reply, no model call
    assert _chat(url)["choices"][0]["message"]["content"] == a
    with pytest.raises(urllib.error.HTTPError) as e:      # unseen request -> 404, logged as a miss
        _chat(url, messages=[{"role": "user", "content": "other"}])
    assert e.value.code == 404
    srv.shutdown()
    rows = [json.loads(l) for l in open(log)]
    assert rows[1]["reused"] and rows[1]["backend"] == "fake" and rows[2]["error"] == "replay miss"


def test_key_ignores_model_name():
    assert gw.request_key(MSGS, 0.0) == gw.request_key(json.loads(json.dumps(MSGS)), 0.0)
    assert gw.request_key(MSGS, 0.0) != gw.request_key(MSGS, 0.5)


def test_flatten_text_and_images():
    p, imgs = gw.flatten([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}])
    assert p == "[USER]\na" and imgs == ["data:image/png;base64,AA=="]


def test_openai_upstream_caps_tokens_and_strips_think(tmp_path):
    seen = {}

    class Up(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"data": [{"id": "stub-qwen"}]})

        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send({"choices": [{"message": {"content": "<think>x</think>ANSWER"}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}})

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    be = gw.OpenAIUpstream(f"http://127.0.0.1:{up.server_address[1]}/v1", max_tokens=2048)
    srv, _, url = _start(be, str(tmp_path / "l.jsonl"))
    out = _chat(url, max_tokens=100000)
    srv.shutdown(); up.shutdown()
    assert out["choices"][0]["message"]["content"] == "ANSWER" and out["model"] == "stub-qwen"
    assert seen["max_tokens"] == 2048 and seen["model"] == "stub-qwen"


def test_stream_rejected(tmp_path):
    srv, _, url = _start(gw.Fake(), str(tmp_path / "l.jsonl"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _chat(url, stream=True)
    srv.shutdown()
    assert e.value.code == 400


def test_strip_images_keeps_text():
    m = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                                      {"type": "text", "text": "plan"}]}, {"role": "system", "content": "s"}]
    out = gw.strip_images(m)
    assert out[0]["content"] == [{"type": "text", "text": gw.DROP_NOTE}, {"type": "text", "text": "plan"}]
    assert out[1] == m[1] and m[0]["content"][0]["type"] == "image_url"   # input not mutated
