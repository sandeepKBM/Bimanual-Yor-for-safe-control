"""stack/proposer/propose.py -- ask the proposer model for the task plan (and re-ask when the probe set is outside B).  SIMULATION ONLY.   (11-05)

    propose(task, scene_facts, image=None, backend_spec="codex:gpt-5.6-sol:high", cache_dir=DIR, offline=False)  -> (plan, record)
    reask(task, plan, probe_set, scene_facts=None, backend_spec=..., cache_dir=DIR, offline=False)              -> (plan, record)

The backend is pan/llm.py's (spec string; CachedVLM stores EVERY request with its response in cache_dir, one JSON per request).  offline=True (or env
BIMANUAL_NO_NETWORK=1): the backend is read-only: a cache hit replays, a miss raises ProposerCacheMiss -- the network is never touched.  A mock spec ("mock:fixture")
never touches the network either.  Codex is a live cloud service: it gets the prompt text and the optional camera image only (pan/llm.py).

plan   = the validated reply as a dict (status "ok": belief, steps, recoveries, fallback, task, proposer) or {"status": "plan_rejected", stage, reasons, task}
record = bounds rule 9: spec, model, reasoning level, backend, caveats, request hash (CachedVLM key), raw reply path (the cache JSON), prompt hash, plan id, status.
A bad reply is never repaired (schema.validate); the rejection is the result and is counted by the caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from world_model.v2.house.lang.backend import Backend, CachedVLM
from world_model.v2.nero.pan import llm as LLM
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S

DEFAULT_SPEC = LLM.DEFAULT                  # "codex:gpt-5.6-sol:high" (USER 11-05)
MAX_NEW_TOKENS = 4096                       # part of the cache key; Codex ignores it (its own limits)


class ProposerCacheMiss(RuntimeError):
    """offline replay asked for a request that is not in the cache.  Raised, never swallowed: the run must not fall back to the network."""


class _ReplayOnly(Backend):
    name = "replay-only"

    def __init__(self, model):
        self.model = model

    def ask(self, *a, **k):                                   # pragma: no cover - CachedVLM(mode="ro") raises before this
        raise ProposerCacheMiss("offline: a cache miss would need the network")


def _offline(offline, kind="codex"):
    """read-only cache: offline=True always; env BIMANUAL_NO_NETWORK=1 for every real backend (a mock never touches the network, so it may still fill its cache)."""
    return bool(offline) or (kind != "mock" and os.environ.get("BIMANUAL_NO_NETWORK") == "1")


def make_backend(spec, cache_dir, offline=False):
    kind, *rest = spec.split(":")
    off = _offline(offline, kind)
    if kind == "mock":
        inner = MK.MockProposer(rest[0] if rest else "fixture")
    elif off:
        if not cache_dir:
            raise ValueError("offline replay needs a cache_dir")
        inner = _ReplayOnly(rest[0] if rest else "unknown")
    else:
        inner = LLM.make(spec)                                # raw backend: no network call happens before .ask
        if not cache_dir:
            raise ValueError("a live proposer must have a cache_dir (every request is stored)")
    if not cache_dir:
        return inner
    return CachedVLM(inner, cache_dir, mode="ro" if off else "rw")


def _ask(be, system, user, image):
    imgs = [image] if image else []
    key = be.key(user, imgs, system, MAX_NEW_TOKENS, 0.0, 1) if isinstance(be, CachedVLM) else None
    t0 = time.time()
    try:
        out = be.ask(user, images=imgs, system=system, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, n=1)
    except KeyError as e:
        if isinstance(be, CachedVLM):
            raise ProposerCacheMiss(f"offline: request {key} is not in the cache {be.dir} ({e})") from e
        raise
    return out["texts"][0], out, key, time.time() - t0


def _record(kind, task, spec, be, key, out, wall, system, user, image, plan, reply):
    info = LLM.info(spec)
    cache_file = str(Path(be.dir) / f"{key}.json") if isinstance(be, CachedVLM) and key else None
    return dict(kind=kind, task=task, spec=spec, model=info["model"], reasoning=info["reasoning"], backend=info["backend"], caveats=info["caveats"],
                request_hash=key, raw_reply_path=cache_file, prompt_sha=hashlib.sha256((system + "\n" + user).encode()).hexdigest()[:16],
                image=str(image) if image else None, cached=bool(out.get("cached")), wall_s=round(float(wall), 2), status=plan["status"],
                plan_id=plan.get("plan_id"), reasons=plan.get("reasons"), stage=plan.get("stage"), reply_head=reply[:300],
                note="the plan is the model's reply as validated by stack/proposer/schema.py; no repair, no defaults")


def _finish(kind, task, spec, be, key, out, wall, system, user, image, reply, supports, record_extra=None):
    try:
        tp = S.validate_reply(reply, task, supports)
        d = tp.to_dict()
        d.update(task=task, plan_id=S.plan_id(d))
        info = LLM.info(spec)
        d["proposer"] = dict(spec=spec, model=info["model"], reasoning=info["reasoning"])
    except S.PlanRejected as e:
        d = dict(e.to_dict(), task=task)
    rec = _record(kind, task, spec, be, key, out, wall, system, user, image, d, reply)
    rec.update(record_extra or {})
    return d, rec


def propose(task, scene_facts=None, image=None, backend_spec=DEFAULT_SPEC, cache_dir=None, offline=False, world=None):
    """-> (plan dict, record).  scene_facts default = prompt.scene_facts(task) (no hidden truth).  image = optional camera image path (the only extra input)."""
    f = scene_facts if scene_facts is not None else PR.scene_facts(task, world)
    be = make_backend(backend_spec, cache_dir, offline)
    user = PR.user_prompt(task, f)
    reply, out, key, wall = _ask(be, PR.SYSTEM, user, image)
    return _finish("plan", task, backend_spec, be, key, out, wall, PR.SYSTEM, user, image, reply, tuple(f["task"]["supports"]))


def reask(task, plan, probe_set, scene_facts=None, backend_spec=DEFAULT_SPEC, cache_dir=None, offline=False, image=None, world=None):
    """Rule 2: `plan` (a status-ok plan dict) was written for B; the probe set `probe_set` = {param: [lo, hi]} is not inside it.  Ask for a complete new plan that covers the
    uncovered part.  The prompt carries the probe's numbers only.  -> (new plan dict, record); record["probe_set"], record["uncovered"] say what was asked."""
    if plan.get("status") != "ok":
        raise ValueError("reask needs a plan that validated (status ok)")
    f = scene_facts if scene_facts is not None else PR.scene_facts(task, world)
    be = make_backend(backend_spec, cache_dir, offline)
    user = PR.reask_prompt(task, plan, probe_set, f)
    reply, out, key, wall = _ask(be, PR.SYSTEM, user, image)
    return _finish("reask", task, backend_spec, be, key, out, wall, PR.SYSTEM, user, image, reply, tuple(f["task"]["supports"]),
                   dict(probe_set=probe_set, uncovered=PR.uncovered(plan["belief"], probe_set), previous_plan_id=plan.get("plan_id")))


def main(argv=None):
    """python -m world_model.v2.stack.proposer.propose --task box --cache DIR [--spec codex:gpt-5.6-sol:high] [--offline] --out plan.json
    Prints the prompt with --print-prompt (no model call)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=S.TASKS)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--cache", help="response cache dir (required except for --print-prompt)")
    ap.add_argument("--offline", action="store_true", help="replay from the cache only; a miss fails (no network)")
    ap.add_argument("--print-prompt", action="store_true")
    ap.add_argument("--out", help="write {plan, record} JSON here")
    a = ap.parse_args(argv)
    if a.print_prompt:
        print(PR.SYSTEM + "\n\n" + PR.user_prompt(a.task))
        return
    plan, rec = propose(a.task, backend_spec=a.spec, cache_dir=a.cache, offline=a.offline)
    print(json.dumps(dict(plan=plan, record=rec), indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(plan=plan, record=rec), indent=1, default=str))


if __name__ == "__main__":
    main()
