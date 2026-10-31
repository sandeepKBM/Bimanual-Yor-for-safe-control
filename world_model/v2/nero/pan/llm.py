"""pan/llm.py -- one swappable language-model backend for BOTH pan layers (ours: vizcoast/, baseline: vizcoast_faithful/).

    make("codex:gpt-5.6-sol:high", cache_dir)   GPT via the Codex CLI (ChatGPT login), the same call as agentchord_probe/gateway.Codex
    make("openai:<base_url>:<model>", cache_dir) any OpenAI-compatible server (the later open-weight model, e.g. vLLM)
    make("mock:<kind>")                          the scripted stand-ins (kind = the layer's mock profile; caller supplies it)

Every real backend is wrapped in house/lang/backend.CachedVLM (DIR/llm_cache: one JSON per request with model, backend, request,
response) so results can be replayed offline and say exactly which model produced them.  info(spec) gives the record that every
result must carry: model, reasoning effort, backend, caveats.

User choice 2026-10-29: the best GPT-5.6 available through Codex (gpt-5.6-sol) for VIZ-COAST, our layer and AgentChord for now;
all three move to one open-weight model for the final comparison.  Codex is a live cloud service: it gets the prompt text and the
camera image only; it never gets a shell (read-only sandbox, empty working dir) and never touches the robot or the sim.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

from world_model.v2.house.lang.backend import Backend, CachedVLM

DEFAULT = "codex:gpt-5.6-sol:high"


def info(spec):
    kind, *rest = spec.split(":")
    if kind == "codex":
        from world_model.v2.agentchord_probe.gateway import CODEX_CAVEAT, CODEX_PREFACE
        return dict(spec=spec, backend="codex", model=rest[0], reasoning=rest[1] if len(rest) > 1 else "codex default",
                    caveats=CODEX_CAVEAT + "; the gateway preface is prepended to every prompt: " + CODEX_PREFACE.strip())
    if kind == "openai":
        return dict(spec=spec, backend="openai", model=rest[-1], reasoning="server default", caveats="")
    return dict(spec=spec, backend="mock", model="scripted", reasoning="-", caveats="scripted responses, not a model")


class CodexVLM(Backend):
    name = "codex"

    def __init__(self, model, reasoning=None, timeout_s=1800.0):
        from world_model.v2.agentchord_probe.gateway import Codex
        self.model, self.reasoning = model, reasoning
        self.c = Codex(model, timeout_s=timeout_s, reasoning=reasoning)

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        msgs = [dict(role="system", content=system)] if system else []
        parts = [dict(type="text", text=text)]
        for p in images:
            mime = "image/jpeg" if str(p).lower().endswith((".jpg", ".jpeg")) else "image/png"
            parts.append(dict(type="image_url", image_url=dict(url=f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode())))
        msgs.append(dict(role="user", content=parts))
        texts, t0 = [], time.time()
        for _ in range(n):                                   # temperature is not settable through Codex; n calls = n samples
            texts.append(self.c(dict(messages=msgs))[0])
        dt = time.time() - t0
        return dict(texts=texts, seconds=dt, wall_s=dt, cached=False, backend=f"codex:{self.model}:{self.reasoning}")


class OpenAIVLM(Backend):
    name = "openai"

    def __init__(self, url, model, max_tokens=4096):
        from world_model.v2.agentchord_probe.gateway import OpenAIUpstream
        self.u = OpenAIUpstream(url, model, max_tokens=max_tokens)
        self.model = self.u.model()

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        msgs = ([dict(role="system", content=system)] if system else []) + [dict(role="user", content=text)]
        t0 = time.time()
        texts = [self.u(dict(messages=msgs, temperature=temperature, max_tokens=max_new_tokens))[0] for _ in range(n)]
        dt = time.time() - t0
        return dict(texts=texts, seconds=dt, wall_s=dt, cached=False, backend=f"openai:{self.model}")


def make(spec, cache_dir=None, mock=None):
    """spec -> backend.  mock: a callable kind -> Backend for 'mock:<kind>' (each layer has its own scripted profiles)."""
    kind, *rest = spec.split(":")
    if kind == "mock":
        return mock(rest[0])
    if kind == "codex":
        inner = CodexVLM(rest[0], rest[1] if len(rest) > 1 else None)
    elif kind == "openai":
        inner = OpenAIVLM(":".join(rest[:-1]), rest[-1])
    else:
        raise ValueError(f"unknown backend spec {spec!r}")
    return CachedVLM(inner, cache_dir) if cache_dir else inner
