"""lang/backend.py -- VLM backends for the language layer.  SIMULATION / SAVED DATA ONLY.

    LiveVLM       the owner-launched file-queue daemon (world_model.v2.vlm.service; NFS files only, no sockets, no hosted API).
    MockVLM       deterministic canned responses (tests, offline runs).
    CachedVLM     wraps any backend; response cache keyed by (system, prompt, image content hash, sampling params, model tag) so a repeated
                  run / replay is reproducible and fully offline.  Every miss is also written as a prompt/response JSON (audit trail).
    make_backend  'live' | 'mock' | 'auto' (live if service.status()['up'] else mock, with the reason recorded on the backend).

ask(text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1) -> dict(texts=[...], seconds=<model s>, wall_s=<round trip>, cached=bool, backend=str)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


def _file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Backend:
    name = "base"
    model = "none"

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        raise NotImplementedError


class LiveVLM(Backend):
    """file-queue client of the running daemon.  Never starts it; `available()` says whether it is up."""
    name = "live"

    def __init__(self, timeout_s=180.0):
        from world_model.v2.vlm import service as VS
        self.VS, self.timeout_s = VS, timeout_s
        st = VS.status()
        self.model = st.get("model", "qwen3.5-9b")

    def available(self):
        return bool(self.VS.status().get("up"))

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        o = self.VS.ask(text, images=list(images), system=system, max_new_tokens=max_new_tokens, temperature=temperature, n=n, timeout_s=self.timeout_s)
        return dict(texts=list(o["texts"]), seconds=float(o.get("seconds", 0.0)), wall_s=float(o.get("wall_s", 0.0)), cached=False, backend="live",
                    n_new_tokens=o.get("n_new_tokens"))


class MockVLM(Backend):
    """rules = [(regex or callable(text, system, images)->bool, response)] first match wins; response = str | list[str] | callable(text, system, images, n) -> str | list[str].
    Falls back to `default_rules()` (a plausible deterministic mock of the three tasks) when no rule matches.  Calls are recorded in .calls."""
    name = "mock"
    model = "mock"

    def __init__(self, rules=None, latency_s=0.0, use_default=True):
        self.rules = list(rules or [])
        if use_default:
            self.rules += default_rules()
        self.calls = []
        self.latency_s = latency_s

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        self.calls.append(dict(text=text, system=system, images=list(images), n=n, temperature=temperature))
        blob = (system or "") + "\n" + text
        for pat, resp in self.rules:
            ok = pat(text, system, images) if callable(pat) else re.search(pat, blob, re.S) is not None
            if ok:
                r = resp(text, system, images, n) if callable(resp) else resp
                texts = [r] * n if isinstance(r, str) else list(r)
                return dict(texts=texts, seconds=self.latency_s, wall_s=self.latency_s, cached=False, backend="mock")
        return dict(texts=["{}"] * n, seconds=0.0, wall_s=0.0, cached=False, backend="mock")


def default_rules():
    """plausible deterministic mock: grounding -> 'object' for every mark; authoring -> the tray spec + a clear atom for every available object
    named after 'away from' / 'avoid' in the instruction; priors -> mu 0.3 +- 0.1."""
    def grounding(text, system, images, n):
        ids = [int(x) for x in re.findall(r"Marks to name: \[([0-9, ]*)\]", text) for x in x.split(",") if x.strip()]
        return "\n".join(f"{i} | object | 0.5" for i in ids)

    def authoring(text, system, images, n):
        m = re.search(r"Instruction: (.*)", text)
        ins = (m.group(1) if m else "").lower()
        avail = re.search(r"Entities: (\{.*?\})\n", text, re.S)
        names = list(json.loads(avail.group(1)).keys()) if avail else []
        spec = [dict(inside=dict(a="cup", b=dict(shrink=dict(of="tray_top", by=0.03)), margin=0.0)), dict(upright=dict(a="tray", theta_max=0.1))]
        unresolved = []
        for nm in re.findall(r"(?:away from|avoid|clear of)(?: the| my)? ([a-z_ ]+?)(?:[.,]| and |$)", ins):
            nm = nm.strip().replace(" ", "_")
            if nm in names:
                spec.append(dict(clear=dict(a="tray", b=nm, d=0.05)))
            else:
                unresolved.append(nm)
        return json.dumps(dict(spec=dict(G=dict(lo=0, hi="H", of=dict(**{"and": spec}))), unresolved=unresolved))

    def prior(text, system, images, n):
        return [json.dumps({"cup_tray_friction": {"estimate": 0.30 + 0.02 * k, "sigma": 0.10}}) for k in range(n)]

    return [(r"Marks to name", grounding), (r"cup_tray_friction", prior), (r"constraint spec", authoring)]


class CachedVLM(Backend):
    """response cache + audit log around any backend (see module docstring).  mode: 'rw' (default) | 'ro' (a miss raises KeyError: replays must be offline)."""

    def __init__(self, inner: Backend, cache_dir, mode="rw"):
        self.inner, self.dir, self.mode = inner, Path(cache_dir), mode
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = "cached(" + inner.name + ")"
        self.model = inner.model
        self.n_hit = self.n_miss = 0
        self.trace = []                    # per call: dict(key, cached, seconds, wall_s) for latency reports

    def key(self, text, images, system, max_new_tokens, temperature, n):
        h = hashlib.sha256()
        h.update(json.dumps(dict(t=text, s=system, m=max_new_tokens, T=temperature, n=n, model=self.inner.model, imgs=[_file_hash(p) for p in images]), sort_keys=True).encode())
        return h.hexdigest()[:24]

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        k = self.key(text, images, system, max_new_tokens, temperature, n)
        f = self.dir / f"{k}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rr, cur = d.get("reasoning"), getattr(self.inner, "reasoning", None)
            if rr is not None and cur is not None and rr != cur:
                # the key does not hash the reasoning level (kept so old caches replay); a record made at another level is NOT this request
                raise KeyError(f"cache record {k} was made with reasoning={rr!r}, this request asks reasoning={cur!r}")
            self.n_hit += 1
            out = dict(texts=d["response"]["texts"], seconds=d["response"].get("seconds", 0.0), wall_s=0.0, cached=True, backend=self.name,
                       orig_wall_s=d["response"].get("wall_s"))
            self.trace.append(dict(key=k, cached=True, seconds=out["seconds"], wall_s=0.0,
                                   reasoning_checked=rr is not None and cur is not None))      # False: legacy record (before 11-09) or no level
            return out
        if self.mode == "ro":
            raise KeyError(f"cache miss in read-only mode: {k}")
        t0 = time.time()
        o = self.inner.ask(text, images=images, system=system, max_new_tokens=max_new_tokens, temperature=temperature, n=n)
        self.n_miss += 1
        rec = dict(key=k, model=self.inner.model, backend=self.inner.name, reasoning=getattr(self.inner, "reasoning", None), t_unix=time.time(),
                   request=dict(text=text, system=system, max_new_tokens=max_new_tokens, temperature=temperature, n=n, image_hashes=[_file_hash(p) for p in images],
                                image_names=[Path(p).name for p in images]),
                   response=dict(texts=o["texts"], seconds=o.get("seconds"), wall_s=o.get("wall_s", time.time() - t0)))
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, f)
        o = dict(o, cached=False)
        self.trace.append(dict(key=k, cached=False, seconds=o.get("seconds"), wall_s=o.get("wall_s")))
        return o


def make_backend(kind="auto", cache_dir=None, mode="rw", mock: MockVLM | None = None, log=print):
    """kind 'live' (raises if the daemon is down) | 'mock' | 'auto'.  Wraps in CachedVLM when cache_dir is given."""
    reason = None
    if kind in ("live", "auto"):
        try:
            b = LiveVLM()
            if b.available():
                be = b
            elif kind == "live":
                raise RuntimeError("VLM daemon is not up: " + json.dumps(b.VS.status()))
            else:
                reason = "VLM daemon not up: using the mock backend"
                be = mock or MockVLM()
        except RuntimeError:
            raise
        except Exception as e:      # noqa: BLE001
            if kind == "live":
                raise
            reason = f"live backend unavailable ({type(e).__name__}): using the mock backend"
            be = mock or MockVLM()
    elif kind == "mock":
        be = mock or MockVLM()
    else:
        raise ValueError(kind)
    if reason:
        log("[lang] " + reason)
    be.fallback_reason = reason
    if cache_dir:
        be = CachedVLM(be, cache_dir, mode)
        be.fallback_reason = reason
    return be
