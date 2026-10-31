"""agentchord_probe/gateway.py -- one local OpenAI-compatible chat endpoint in front of several language-model backends.

AgentChord builds its agents with LangChain ChatOpenAI(model="gpt-5", base_url=$LLM_URL, api_key=$OPENAI_API_KEY). Point LLM_URL at
this gateway (http://127.0.0.1:<port>/v1) and pick the real backend here; their code is not edited. The model name they send is ignored
and every reply records the backend that actually answered.

Backends (--backend / $AC_BACKEND):
  openai  forward to an OpenAI-compatible server. --upstream URL, --upstream-model, --max-tokens (hard cap on output tokens).
          jenga's Qwen (OK'd by jenga 10-27, read-only): ssh -N -L 8781:127.0.0.1:8781 ilab3, model
          "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", text only, 16k context, --max-tokens <= 2048, --concurrency 1, pause on TOWER evals.
  codex   GPT via the Codex CLI on the user's ChatGPT login: `codex exec -m <model> --sandbox read-only --ephemeral` in an empty dir.
          Caveats (stored in every log row): Codex adds its own agent system prompt, temperature is not settable, plan rate limits apply.
  replay  answer only from earlier log rows with the same request key (no model call); a miss is an HTTP 404.
  fake    deterministic echo, for tests.

Every request is appended to --log (JSONL): key, backend, model, messages, reply, finish_reason, latency, usage, error.
--reuse returns a logged reply for a repeated key instead of calling the backend again (the row is still logged, with reused=true).
The request key = sha256 of the canonical (messages, temperature) only, so the same prompt maps to the same key across backends.

Binds 127.0.0.1 only. stdlib only.
Run:  python -m world_model.v2.agentchord_probe.gateway --backend codex --model gpt-5.5 --port 8790 --log RUN/llm_log.jsonl
"""
import argparse, base64, hashlib, json, os, re, subprocess, tempfile, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_THINK = re.compile(r"<think>.*?</think>", re.S)
CODEX_PREFACE = ("You are being used as a plain chat-completion backend for a robot task planner. Do not run commands, read files or "
                 "edit anything. Reply with exactly the assistant message that answers the conversation below, nothing else.\n\n")
CODEX_CAVEAT = "codex exec adds its own agent system prompt; temperature not settable; ChatGPT-plan rate limits"
DROP_NOTE = "[camera image omitted: this planner backend is text-only]"


def strip_images(messages):
    """Replace image parts by a text note (text-only upstream, e.g. jenga's Qwen3-Coder). Recorded as a caveat on every row."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{"type": "text", "text": DROP_NOTE} if p.get("type") == "image_url" else p for p in c]
        out.append({**m, "content": c})
    return out


def request_key(messages, temperature):
    canon = json.dumps({"messages": messages, "temperature": temperature}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _strip_think(text):
    text = _THINK.sub("", text or "")
    return text.rsplit("</think>", 1)[-1].strip()


def _parts(content):
    """OpenAI message content -> (text, [data-url images])."""
    if isinstance(content, str):
        return content, []
    texts, imgs = [], []
    for p in content or []:
        if p.get("type") == "text":
            texts.append(p.get("text", ""))
        elif p.get("type") == "image_url":
            u = p.get("image_url")
            imgs.append(u.get("url") if isinstance(u, dict) else u)
    return "\n".join(texts), imgs


def flatten(messages):
    """Chat messages -> one prompt string + image data URLs (for single-prompt backends)."""
    out, imgs = [], []
    for m in messages:
        t, im = _parts(m.get("content"))
        imgs += im
        out.append(f"[{m.get('role', 'user').upper()}]\n{t}")
    return "\n\n".join(out), imgs


class Log:
    def __init__(self, path):
        self.path, self.lock, self.by_key = path, threading.Lock(), {}
        if path and os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("reply") is not None and not r.get("error"):
                    self.by_key[r["key"]] = r

    def append(self, row):
        with self.lock:
            if row.get("reply") is not None and not row.get("error"):
                self.by_key.setdefault(row["key"], row)
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(row) + "\n")


class Backend:
    name, caveat = "base", ""

    def model(self):
        return self.name

    def __call__(self, body):  # -> (text, finish_reason, usage)
        raise NotImplementedError


class Fake(Backend):
    name = "fake"

    def __call__(self, body):
        prompt, _ = flatten(body["messages"])
        return "FAKE:" + hashlib.sha256(prompt.encode()).hexdigest()[:12], "stop", {}


class OpenAIUpstream(Backend):
    name = "openai"

    def __init__(self, url, model=None, max_tokens=8192, timeout_s=900.0, api_key=None, drop_images=False):
        self.url, self.max_tokens, self.timeout_s, self.drop_images = url.rstrip("/"), max_tokens, timeout_s, drop_images
        self.caveat = DROP_NOTE if drop_images else ""
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.served = model or self._get("/models")["data"][0]["id"]

    def model(self):
        return self.served

    def _get(self, path):
        with urllib.request.urlopen(urllib.request.Request(self.url + path, headers=self.headers), timeout=30) as f:
            return json.loads(f.read())

    def __call__(self, body):
        asked = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens)
        msgs = strip_images(body["messages"]) if self.drop_images else body["messages"]
        fwd = dict(model=self.served, messages=msgs, temperature=float(body.get("temperature") or 0.0),
                   max_tokens=min(asked, self.max_tokens), seed=0)  # max_tokens = hard cap (jenga's server: <= 2048)
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(fwd).encode(), headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as f:
            o = json.loads(f.read())
        c = o["choices"][0]
        return _strip_think(c["message"].get("content")), c.get("finish_reason"), o.get("usage", {})


class Codex(Backend):
    name, caveat = "codex", CODEX_CAVEAT

    def __init__(self, model, timeout_s=1800.0, reasoning=None, codex_bin="codex"):
        self.m, self.timeout_s, self.reasoning, self.bin = model, timeout_s, reasoning, codex_bin

    def model(self):
        return self.m

    def __call__(self, body):
        prompt, imgs = flatten(body["messages"])
        with tempfile.TemporaryDirectory(prefix="ac_codex_") as d:
            cmd = [self.bin, "exec", "-m", self.m, "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                   "--ignore-rules", "-C", d, "-o", os.path.join(d, "out.txt")]
            if self.reasoning:
                cmd += ["-c", f"model_reasoning_effort={self.reasoning}"]
            for k, u in enumerate(imgs):
                head, b64 = u.split(",", 1)
                ext = ".jpg" if "jpeg" in head else ".png"
                p = os.path.join(d, f"img{k}{ext}")
                open(p, "wb").write(base64.b64decode(b64))
                cmd += ["-i", p]
            cmd.append("-")  # prompt from stdin
            r = subprocess.run(cmd, input=CODEX_PREFACE + prompt, capture_output=True, text=True, timeout=self.timeout_s, cwd=d)
            out = os.path.join(d, "out.txt")
            text = open(out).read() if os.path.exists(out) else ""
            if r.returncode != 0 or not text.strip():
                raise RuntimeError(f"codex exit {r.returncode}: {(r.stderr or '')[-800:]}")
        return text.strip(), "stop", {}


class Replay(Backend):
    name = "replay"

    def __call__(self, body):
        raise KeyError("replay miss")


def make_handler(backend, log, reuse, sem):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send(200, {"object": "list", "data": [{"id": backend.model(), "object": "model"}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if body.get("stream"):
                return self._send(400, {"error": {"message": "stream not supported by gateway"}})
            temp = float(body.get("temperature") or 0.0)
            key = request_key(body.get("messages", []), temp)
            row = dict(ts=time.time(), key=key, backend=backend.name, model=backend.model(), requested_model=body.get("model"),
                       caveat=backend.caveat, messages=body.get("messages", []), temperature=temp, reused=False)
            t0 = time.time()
            hit = log.by_key.get(key) if (reuse or isinstance(backend, Replay)) else None
            try:
                if hit is not None:
                    text, fin, usage = hit["reply"], hit.get("finish_reason", "stop"), hit.get("usage", {})
                    row.update(reused=True, model=hit.get("model"), backend=hit.get("backend"), caveat=hit.get("caveat", ""))
                elif isinstance(backend, Replay):
                    row.update(error="replay miss", latency_s=0.0)
                    log.append(row)
                    return self._send(404, {"error": {"message": f"replay miss {key}"}})
                else:
                    with sem:
                        text, fin, usage = backend(body)
            except Exception as e:  # noqa: BLE001 -- report every backend failure to the client and the log
                row.update(error=f"{type(e).__name__}: {e}", latency_s=time.time() - t0)
                log.append(row)
                return self._send(502, {"error": {"message": row["error"]}})
            row.update(reply=text, finish_reason=fin, usage=usage, latency_s=time.time() - t0)
            log.append(row)
            self._send(200, {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
                             "model": row["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                                 "finish_reason": fin or "stop"}],
                             "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
    return H


def build_backend(a):
    if a.backend == "fake":
        return Fake()
    if a.backend == "replay":
        return Replay()
    if a.backend == "codex":
        return Codex(a.model or "gpt-5.5", reasoning=a.reasoning)
    if a.backend == "openai":
        return OpenAIUpstream(a.upstream, a.upstream_model, max_tokens=a.max_tokens, api_key=os.environ.get("AC_UPSTREAM_KEY"),
                              drop_images=a.drop_images)
    raise SystemExit(f"unknown backend {a.backend}")


def serve(backend, log_path, port, reuse=False, concurrency=1):
    log = Log(log_path)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(backend, log, reuse, threading.Semaphore(concurrency)))
    return srv, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=os.environ.get("AC_BACKEND", "fake"), choices=["openai", "codex", "replay", "fake"])
    ap.add_argument("--model", default=None, help="codex model (default gpt-5.5)")
    ap.add_argument("--reasoning", default=None, help="codex model_reasoning_effort override (default: codex's own)")
    ap.add_argument("--upstream", default=os.environ.get("AC_UPSTREAM_URL"), help="openai backend base URL (…/v1)")
    ap.add_argument("--upstream-model", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--drop-images", action="store_true", help="openai backend: replace image parts by a text note (text-only model)")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--log", required=True)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    srv, _ = serve(build_backend(a), a.log, a.port, a.reuse, a.concurrency)
    print(f"gateway backend={a.backend} on http://127.0.0.1:{a.port}/v1 log={a.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
