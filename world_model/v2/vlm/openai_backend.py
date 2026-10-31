"""vlm/openai_backend.py -- runner.run_one-compatible client for a model served by a LOCAL OpenAI-compatible server (vLLM).

Used for models that HF transformers cannot load (Cosmos 3: vLLM + the vllm-cosmos3 plugin). The server runs on the same node
as the daemon (127.0.0.1, launched by vlm/slurm/cosmos3_serve.sh); stdlib only, so it imports in any env.
URL from $VLM_OPENAI_URL (default http://127.0.0.1:8771/v1).  $VLM_ENABLE_THINKING=0|1 (unset = the model default) is sent as
chat_template_kwargs.enable_thinking (Cosmos 3 templates default it ON and open "<think>" inside the PROMPT, so the reply holds only "</think>").
A reply cut at max_tokens (finish_reason "length") is NOT an answer: texts[k] = "" (the raw text stays in raw_texts) -- callers must not
parse JSON fragments out of unfinished reasoning (seen 10-16: a mid-reasoning {...} with grams read as kg).
"""
import base64, json, mimetypes, os, re, time, urllib.request

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _answer(text, finish):
    if finish == "length":
        return ""
    text = _THINK.sub("", text)
    return text.rsplit("</think>", 1)[-1].strip()


class OpenAIChat:
    def __init__(self, model_id, base_url=None, timeout_s=600.0):
        self.model_id = model_id
        self.url = (base_url or os.environ.get("VLM_OPENAI_URL", "http://127.0.0.1:8771/v1")).rstrip("/")
        self.timeout_s = timeout_s
        self.served = self._get("/models")["data"][0]["id"]          # fails fast if the server is not up

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as f:
            return json.loads(f.read())

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as f:
            return json.loads(f.read())

    @staticmethod
    def _image_part(p):
        mt = mimetypes.guess_type(p)[0] or "image/png"
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}

    def run_one(self, r, max_new_tokens):
        content = [self._image_part(p) for p in r.get("images", [])] + [{"type": "text", "text": r["text"]}]
        msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [{"role": "user", "content": content}]
        temp = float(r.get("temperature", 0.0))
        n = int(r.get("n", 1)) if temp > 0 else 1
        body = dict(model=self.served, messages=msgs, max_tokens=int(max_new_tokens), temperature=temp, n=n, seed=0)
        th = r.get("thinking", os.environ.get("VLM_ENABLE_THINKING"))
        if th is not None and str(th) != "":
            body["chat_template_kwargs"] = {"enable_thinking": str(th).lower() in ("1", "true")}
        if temp > 0:
            body["top_p"] = float(r.get("top_p", 0.95))
        t0 = time.time()
        o = self._post("/chat/completions", body)
        dt = time.time() - t0
        raw = [c["message"].get("content") or "" for c in o["choices"]]
        texts = [_answer(t, c.get("finish_reason")) for t, c in zip(raw, o["choices"])]      # the reasoning stays in raw_texts
        u = o.get("usage", {})
        return dict(id=r["id"], texts=texts, raw_texts=raw, n_prompt_tokens=int(u.get("prompt_tokens", 0)),
                    n_new_tokens=[int(u.get("completion_tokens", 0)) // max(n, 1)] * n,
                    finish=[c.get("finish_reason") for c in o["choices"]], thinking=body.get("chat_template_kwargs", {}).get("enable_thinking"), seconds=round(dt, 3), peak_vram_gb=None)
