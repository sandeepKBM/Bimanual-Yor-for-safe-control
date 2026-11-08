"""agentchord_probe/nero/plan.py -- run THEIR TaskAgent -> RecoveryAgent -> CompileAgent once per task variant through the loopback gateway
and cache everything.  SIMULATION ONLY.  The only network path is 127.0.0.1 (gateway); the `codex` backend (owner-run) is the only thing that
reaches out, and only inside gateway.Codex.  Tests use `canned` / `replay` (no model call).

Mirrors BaseAgentEnv.generate_graph_for_actions (base_agent_env.py:183-227) with recovery=True.  Prompt files are THEIR prompts/*.txt unchanged;
only task_prompt.txt is ours (task text + facts, tasks.task_prompt).  Writes into OUT:
  task_prompt.txt agent_config.json agent_obs.png llm_log.jsonl agent_task_graph.json agent_recovery_spec.json agent_compiled_graph.json plan.json
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import threading
import time

from . import tasks as T

PROMPT_FILES = ("task_prompt", "basic_background", "atom_actions", "error_functions", "monitor_functions", "recovery_rules")
SETUP_DEVIATIONS = [
    "image: MuJoCo torso render (contents_eval.VIEWS['torso'], arms at home pose) instead of their cam_high",
    "task_prompt.txt is ours (task text + the VLM facts text); all other prompt files and all agent code are theirs, unchanged",
    "coordinates in the prompt are world-frame metres; their prompt calls the frame 'robot base frame' (base frame not modelled)",
    "atoms: coupled two-claw carry under a rigid weld (see atoms.py); rotate_eef / orient_eef / one-arm edges are rejected, not executed",
    "LLM backend: " + "see backend/model/caveats fields",
]


# USER 11-05: executed plans are authored by Codex gpt-5.6-sol at high reasoning.  LEGACY = the 10-27 choice (gpt-5.6-terra, Codex's own reasoning default).
DEFAULT_MODEL, DEFAULT_REASONING = "gpt-5.6-sol", "high"
LEGACY_MODEL, LEGACY_REASONING = "gpt-5.6-terra", None


def resolve_proposer(backend, model=None, reasoning=None, legacy=False):
    """-> (model, reasoning) for the codex backend.  Explicit model/reasoning win; `legacy` = the 10-27 pair; reasoning 'default' = Codex's own."""
    dm, dr = (LEGACY_MODEL, LEGACY_REASONING) if legacy else (DEFAULT_MODEL, DEFAULT_REASONING)
    m = model or dm
    r = dr if reasoning is None else (None if reasoning == "default" else reasoning)
    return (m, r) if backend == "codex" else (model, reasoning)


class Canned:
    """gateway backend answering from a list of reply strings, in order (tests / dry runs; no model)."""
    name, caveat = "canned", "canned replies (no model)"

    def __init__(self, replies):
        self.replies, self.i = list(replies), 0

    def model(self):
        return "canned"

    def __call__(self, body):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r, "stop", {}


def _backend(a):
    from world_model.v2.agentchord_probe import gateway as gw
    if a["backend"] == "canned":
        return Canned(json.loads(pathlib.Path(a["canned"]).read_text()))
    if a["backend"] == "replay":
        return gw.Replay()
    if a["backend"] == "codex":
        m, r = resolve_proposer("codex", a.get("model"), a.get("reasoning"), a.get("legacy", False))
        return gw.Codex(m, reasoning=r)
    raise SystemExit(f"unknown backend {a['backend']}")


def prompt_files(out):
    """write the per-task prompt + agent_config.json (theirs, with task_prompt.txt in OUT)."""
    cfg = {"TaskAgent": {"prompt_name": "generate_task_graph"}, "RecoveryAgent": {"prompt_name": "augment_task_graph"},
           "CompileAgent": {"prompt_name": "compile_agent_graph"},
           "Agent": {"prompt_kwargs": {k: {"type": "text", "name": f"{k}.txt"} for k in PROMPT_FILES}}}
    (out / "agent_config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


def make_plan(task, out, backend="canned", model=None, reasoning=None, replay_log=None, canned=None, regenerate=False, recovery=True,
              rgb=None, port=0, legacy=False):
    """-> plan.json dict.  rgb: uint8 RGB image override (tests); default = MuJoCo render of the representative world."""
    from world_model.v2.agentchord_probe import gateway as gw
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w = T.rep_world(task)
    text, facts = T.task_prompt(task, w)
    (out / "task_prompt.txt").write_text(text)
    cfg = prompt_files(out)
    if rgb is None:
        rgb = T.render_rgb(w)
    import cv2
    bgr = rgb[..., ::-1].copy()          # their encode_image is cv2.imencode (BGR in); RGB in -> colour-swapped PNG otherwise
    cv2.imwrite(str(out / "agent_obs.png"), bgr)

    log_path = str(out / "llm_log.jsonl")
    be = _backend(dict(backend=backend, model=model, reasoning=reasoning, canned=canned, legacy=legacy))
    pm, pr = resolve_proposer(backend, model, reasoning, legacy)
    srv, log = gw.serve(be, log_path, port, reuse=False, concurrency=1)
    if backend == "replay" and replay_log:
        src = gw.Log(replay_log)
        log.by_key.update(src.by_key)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["LLM_URL"] = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "offline-gateway")
    from . import shim
    shim.install(scratch_dir=out / "_unused_db")
    t0 = time.time()
    try:
        from embodichain.agents.hierarchy.compile_agent import CompileAgent
        from embodichain.agents.hierarchy.llm import create_llm       # their factory; the module-level instances bind LLM_URL at first import
        from embodichain.agents.hierarchy.recovery_agent import RecoveryAgent
        from embodichain.agents.hierarchy.task_agent import TaskAgent
        import copy
        cfgp = str(out / "agent_config.json")
        task_llm, recovery_llm, compile_llm = (create_llm(temperature=0.0, model="gpt-5") for _ in range(3))   # same args as llm.py:59-61
        mk = lambda cls, llm, key: cls(llm, **copy.deepcopy(cfg["Agent"]), **cfg[key], task_name=f"nero_{task}", config_dir=cfgp)
        ta, ra, ca = mk(TaskAgent, task_llm, "TaskAgent"), mk(RecoveryAgent, recovery_llm, "RecoveryAgent"), mk(CompileAgent, compile_llm, "CompileAgent")
        kw = dict(log_dir=str(out))
        ti = ta.get_composed_observations(env=None, regenerate=regenerate, observations={"rgb": bgr}, **kw)
        task_graph = ta.generate(**ti)
        recovery_spec = None
        if recovery:
            ri = ra.get_composed_observations(env=None, regenerate=regenerate, task_graph=task_graph, **kw)
            recovery_spec = ra.generate(**ri)
        ci = ca.get_composed_observations(env=None, regenerate=regenerate, task_graph=task_graph, recovery_spec=recovery_spec,
                                          recovery_enabled=recovery, **kw)
        graph_path, _, _ = ca.generate(**ci)
    finally:
        srv.shutdown()
    rows = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    meta = dict(task=task, plan_id=T.sha(pathlib.Path(graph_path).read_text()), graph=str(graph_path), backend=backend,
                models=sorted({(r.get("backend"), r.get("model")) for r in rows if r.get("reply") is not None}),
                reasoning=(pr or "codex default") if backend == "codex" else None,
                proposer=dict(backend=backend, model=(pm if backend == "codex" else be.model() if hasattr(be, "model") else model), reasoning=(pr or "codex default") if backend == "codex" else None,
                              legacy=bool(legacy)),
                caveats=sorted({r.get("caveat", "") for r in rows}), n_llm_calls=len(rows), n_errors=sum(1 for r in rows if r.get("error")),
                facts_text=facts, task_prompt=text, prompt_hash=T.sha(text), recovery=recovery, regenerate=regenerate,
                plan_wall_s=round(time.time() - t0, 1), setup_deviations=SETUP_DEVIATIONS, date=time.strftime("%Y-%m-%d"),
                codex_note="Codex adds its own agent system prompt; temperature not settable (their llm.py asks T=0)" if backend == "codex" else None)
    (out / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta
