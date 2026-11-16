"""pan/llm.py -- one swappable language-model backend for BOTH pan layers (ours: vizcoast/, baseline: vizcoast_faithful/).

    make("codex:gpt-5.6-sol:high", cache_dir)   GPT via the Codex CLI (ChatGPT login), the same call as agentchord_probe/gateway.Codex
    make("claude:claude-sonnet-5-5:high", cache_dir) Claude via the local Claude Code CLI (`claude -p`, the user's own login), see ClaudeCodeVLM
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
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from world_model.v2.house.lang.backend import Backend, CachedVLM

DEFAULT = "codex:gpt-5.6-sol:high"
CLAUDE_DEFAULT_MODEL = "claude-sonnet-5-5"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")      # `claude --help` (2.1.323): --effort <level>
CLAUDE_BIN = "claude"
CLAUDE_CAVEAT = ("Claude Code CLI in print mode from an empty temp dir; no tools (--tools ''), no MCP, no slash commands/skills, no session persistence, "
                 "--setting-sources '' (no user/project/local settings or hooks), CLAUDE.md auto-discovery and auto-memory switched off by env; the Claude Code default system "
                 "prompt is REPLACED by our system text; the Codex gateway preface is prepended to the user message (parity); temperature not settable; "
                 "no seed; subscription login (rate limits); the CLI still injects (canary 11-12) the account e-mail as a userEmail context block and an environment block (cwd = the temp dir, platform, OS, date, git flag) to Anthropic")


def info(spec):
    kind, *rest = spec.split(":")
    if kind == "codex":
        from world_model.v2.agentchord_probe.gateway import CODEX_CAVEAT, CODEX_PREFACE
        return dict(spec=spec, backend="codex", model=rest[0], reasoning=rest[1] if len(rest) > 1 else "codex default",
                    caveats=CODEX_CAVEAT + "; the gateway preface is prepended to every prompt: " + CODEX_PREFACE.strip())
    if kind == "claude":
        from world_model.v2.agentchord_probe.gateway import CODEX_PREFACE
        return dict(spec=spec, backend="claude", model=rest[0] if rest else CLAUDE_DEFAULT_MODEL, reasoning=rest[1] if len(rest) > 1 else "high",
                    caveats=CLAUDE_CAVEAT + "; the gateway preface is prepended to every prompt: " + CODEX_PREFACE.strip())
    if kind == "openai":
        return dict(spec=spec, backend="openai", model=rest[-1], reasoning="server default", caveats="")
    return dict(spec=spec, backend="mock", model="scripted", reasoning="-", caveats="scripted responses, not a model")


def _codex_error_text(msg):
    """11-12 B7: the Codex failure text with the CLI's stderr FIRST and a short class ("usage_limit", "timeout", "rate_limit", "auth", "error").
    gateway.Codex raises 'codex exit N: <stderr tail>'; the reader of a truncated log then sees the cause, not the exit code."""
    m = re.match(r"codex exit (\S+): (.*)$", msg, re.S)
    err, code = (m.group(2).strip(), m.group(1)) if m else (msg, "?")
    low = err.lower()
    cls = ("usage_limit" if ("usage limit" in low or "usage_limit" in low or "hit your usage" in low or "quota" in low) else
           "rate_limit" if ("rate limit" in low or "429" in low) else "timeout" if "timed out" in low or "timeout" in low else
           "auth" if ("unauthorized" in low or "401" in low or "log in" in low or "login" in low) else "error")
    return f"codex {cls}: {err} (exit {code})"


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
            try:
                texts.append(self.c(dict(messages=msgs))[0])
            except RuntimeError as e:
                raise RuntimeError(_codex_error_text(str(e))) from e
        dt = time.time() - t0
        return dict(texts=texts, seconds=dt, wall_s=dt, cached=False, backend=f"codex:{self.model}:{self.reasoning}")


class ClaudeCodeVLM(Backend):
    """Claude through the local Claude Code CLI (`claude -p`), as a plain chat-completion backend.  No tools, empty temp cwd, our system prompt REPLACES Claude Code's.
    Images go as base64 image content blocks over --input-format stream-json.  A reply cut at the token cap (stop_reason max_tokens) is NO answer: it raises.
    Per call the CLI's usage / cost / stop reason / duration are kept in self.last and appended to self.calls (CachedVLM does not store them; the compare script does)."""
    name = "claude"

    def __init__(self, model=CLAUDE_DEFAULT_MODEL, reasoning="high", timeout_s=1800.0, max_output_tokens=None, claude_bin=None):
        if reasoning not in CLAUDE_EFFORTS:
            raise ValueError(f"effort {reasoning!r} not in {CLAUDE_EFFORTS}")
        self.model, self.reasoning, self.timeout_s, self.max_output_tokens = model, reasoning, timeout_s, max_output_tokens
        self.bin = claude_bin or CLAUDE_BIN
        self.last, self.calls = {}, []

    def command(self, system):
        cmd = [self.bin, "-p", "--model", self.model, "--effort", self.reasoning, "--tools", "", "--no-session-persistence", "--strict-mcp-config",
               "--disable-slash-commands", "--setting-sources", "", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
        if system:
            cmd += ["--system-prompt", system]
        return cmd

    def env(self):
        e = dict(os.environ, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
        if self.max_output_tokens:
            e["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(self.max_output_tokens))
        return e

    @staticmethod
    def stdin_message(text, images=()):
        from world_model.v2.agentchord_probe.gateway import CODEX_PREFACE
        blocks = [dict(type="text", text=CODEX_PREFACE + text)]
        for p in images:
            mime = "image/jpeg" if str(p).lower().endswith((".jpg", ".jpeg")) else "image/png"
            blocks.append(dict(type="image", source=dict(type="base64", media_type=mime, data=base64.b64encode(Path(p).read_bytes()).decode())))
        return json.dumps(dict(type="user", message=dict(role="user", content=blocks))) + "\n"

    @staticmethod
    def parse(stdout):
        """stream-json lines -> (text, meta).  Raises RuntimeError for an error result or a reply cut at the token cap (no answer)."""
        res, stop = None, None
        for ln in stdout.splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                o = json.loads(ln)
            except ValueError:
                continue
            if o.get("type") == "assistant":
                stop = (o.get("message") or {}).get("stop_reason") or stop
            if o.get("type") == "result":
                res = o
        if res is None:
            raise RuntimeError("claude: no result line in the output")
        meta = dict(stop_reason=res.get("stop_reason") or stop, usage=res.get("usage"), cost_usd=res.get("total_cost_usd"), duration_ms=res.get("duration_ms"),
                    duration_api_ms=res.get("duration_api_ms"), num_turns=res.get("num_turns"), subtype=res.get("subtype"), is_error=res.get("is_error"), session_id=res.get("session_id"))
        if res.get("is_error") or str(res.get("subtype", "success")) != "success":
            raise RuntimeError(f"claude error result: {res.get('subtype')}: {str(res.get('result'))[:500]}")
        if meta["stop_reason"] == "max_tokens":
            raise RuntimeError("claude: reply cut at the token cap (stop_reason max_tokens): no answer")
        text = res.get("result")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("claude: empty reply")
        return text.strip(), meta

    def _one(self, text, images, system):
        with tempfile.TemporaryDirectory(prefix="ac_claude_") as d:
            r = subprocess.run(self.command(system), input=self.stdin_message(text, images), capture_output=True, text=True, timeout=self.timeout_s, cwd=d, env=self.env())
        if r.returncode != 0 and '"type":"result"' not in r.stdout.replace(' ', ''):
            raise RuntimeError(f"claude exit {r.returncode}: {(r.stderr or r.stdout or '')[-800:]}")
        return self.parse(r.stdout)

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        texts, t0 = [], time.time()
        for _ in range(n):                                   # temperature is not settable; n calls = n samples
            t1 = time.time()
            reply, meta = self._one(text, list(images), system)
            meta["wall_s"] = round(time.time() - t1, 2)
            self.last = meta
            self.calls.append(meta)
            texts.append(reply)
        dt = time.time() - t0
        return dict(texts=texts, seconds=dt, wall_s=dt, cached=False, backend=f"claude:{self.model}:{self.reasoning}", cli=self.last)


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
    elif kind == "claude":
        inner = ClaudeCodeVLM(rest[0] if rest else CLAUDE_DEFAULT_MODEL, rest[1] if len(rest) > 1 else "high")
    elif kind == "openai":
        inner = OpenAIVLM(":".join(rest[:-1]), rest[-1])
    else:
        raise ValueError(f"unknown backend spec {spec!r}")
    return CachedVLM(inner, cache_dir) if cache_dir else inner
