"""eval/demo_runner.py (moved from nero/contents 11-12: nero/contents may not import agentchord_*, dependency rule nero_contents_no_forbidden) -- ONE harness: any world list x any set of arms for the table-behind demos.  SIMULATION ONLY.   (Stage 0d, plan golden-soaring-perlis)

Built from forced_admit.py (shards, resume, offline first plan, phase-tagged U).  No network, no LLM call: every plan is replayed from the offline caches (a cache miss raises).

python -m world_model.v2.eval.demo_runner worlds  --out DIR                       # world lists (existing 20 + 20, the D1 box grid)
python -m world_model.v2.eval.demo_runner run     --worlds FILE --arms A,B,... --out DIR [--shard k/K] [--procs N] [--only id,id] [--dry]
python -m world_model.v2.eval.demo_runner summary  --out DIR

World list JSON (frozen: its sha256 is written to OUT/WORLDS.sha256 and a later run on another list is refused):
  {"name": ..., "assumptions": {...}, "worlds": [spec, ...]}
  spec = {"id", "task", "i"}                                    existing list (nero/base/smoke.behind_world(task, i): the world of run_n20_v2b)
       | {"id", "task": "box_behind", "contents_kg", "contents_xy": [x, y]}     explicit; built exactly as carry_cert.sample_world builds a box (scenes.BoxWorld(contents_kg,
                                                                 contents_xy), then the task's world_kw: shelf=False); seed = f(id) (probe noise only).  Tray: NotImplementedError.
Arms (string ids; "kind[:LEVEL][:key=value ...]"):
  ours[:probe=<method>][:reask=off]    today's v2 path (behind_job.v2_job -> probe_select.run_world): probe -> Theta -> coverage -> static rows + base level -> carry / refuse.  Default probe
                                       poslift; any other name goes to probe.make_probe (unknown -> raises).  reask=off: Rule 2 re-asks are not run (a refusal ends the run).  Default: on for `ours`, OFF for
                                       the other ours-family arms (their re-ask evidence differs from every cached request).  A re-ask replays from the plan cache; an offline cache
                                       miss RAISES (ProposerCacheMiss -> an error record, retried on resume); the network is never touched.
  oracle_mass                          NO probe; Theta = the TRUE contents point [c, c]; the load position is unknown (COM set = the whole box interior).  Coverage in report mode.
  oracle_theta                         NO probe; true mass AND true load position: the COM set P of the static rows and of the base rows is the ONE point scenes.object_truth(...)["contents_com"]
                                       (objects.boxed(com=), probe.interval_spec(com=), behind_job._static(com=)).  Box only.  An ORACLE: its admission is for this world, not a deployable bound.
  no_probe                             NO probe; Theta = Theta_0 (the prior); static rows + base level on the whole prior (generalises the "no_probe_needed" path: coverage in report mode).
  estimate                             the probe runs, then the admission runs on the POINT [mid, mid] of its interval (guarantee: none -- a point estimate; if the probe gave only a lower
                                       bound the midpoint is taken of [lower bound, prior hi], recorded in rec["estimate"]["lower_bound_only"]).
  checker_off:L1|L2|L3                 no admission: the probe runs (Theta recorded), belief = the TRUE contents mass, a_plan = A_PLAN (ASSUMED), base level forced; as forced_admit "true:Lk".
  agentchord:L1|L2|L3:plan=<dir>       AgentChord's cached graph (agentchord_probe.nero.execute.same_world_trial / behind.run_world) with base speed cap = level.
All ours-family arms and checker_off use the SAME offline plan cache (PLAN_CACHE, spec PLAN_SPEC) = the proposer's accepted plan of the source run; the record names the plan id and the proposer.

Record per (world, arm): everything the job record carries (v2 / checker_off / agentchord fields) plus world_id, arm, spec, verdicts on Theta / the true mass / theta* (recorded for every arm, obeyed
only by the arm that should), base level used, U_by_phase (probe | ramp_hold | nominal | backup | refused_fallback -> kind -> {n, t_first}), base_flags_by_phase, FL-MPC counters summed over the
carry segments, plan id + proposer model / reasoning / caveats (bounds rule 9), track["viol_all"] if the episode keeps it.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path("/common/users/ss5772/bimanual_data/contents/table_behind_same_2026-11-09")
SRC = DATA / "run_n20_v2b"
PLAN_CACHE = DATA / "ours_plans_v2"
PLAN_SPEC = "codex:gpt-5.6-sol:high"
DEFAULT_PROBE = "poslift"
A_PLAN = 0.5                     # ASSUMED (N10): the a_admit the static cell gave every admitted world of run_n20_v2b (7/7); the checker gave none on a refused Theta.  As forced_admit.
KINDS = ("ours", "oracle_mass", "oracle_theta", "no_probe", "estimate", "checker_off", "agentchord")
OURS_FAMILY = ("ours", "oracle_mass", "oracle_theta", "no_probe", "estimate")
THETA_MODE = dict(ours=None, oracle_mass="oracle_mass", oracle_theta="oracle_theta", no_probe="no_probe", estimate="estimate")
PHASES = ("probe", "ramp_hold", "nominal", "backup", "refused_fallback")
NOMINAL_PHASES = ("probe", "ramp_hold", "nominal")        # "nominal path": everything but the backup rollouts (and the let-go of a refusal)

# D1 grid (N1: ASSUMED, to be raised with the user; plan Stage 1)
D1_TOTAL_KG = (1.0, 1.75, 2.25, 2.5, 2.75, 3.0, 3.5, 4.5)      # ASSUMED (N1): total mass (box + contents)
D1_OFFSETS = ("centre", "half", "wall")                         # ASSUMED (N1): load offset along the claw-to-claw axis (y of the box frame) = 0, half, full interior reach

# U kind -> group (plan Stage 1 Metrics)
U_CONTENTS = ("obj_tilt", "obj_drop", "spill", "cup_tipped", "cup_off_tray", "lid_off", "lid_slide", "load_world_contact", "collision")
U_BASE = ("wheel_liftoff", "base_world_contact", "route_unfinished")
U_RATING = ("claw_payload", "tau_sat", "claw_twist")


def u_group(kind):
    """contents | base | rating | other.  load_world_contact is listed under contents (plan Stage 1) although the base monitor also raises it."""
    k = str(kind)
    if k in U_CONTENTS:
        return "contents"
    if k in U_BASE:
        return "base"
    if k in U_RATING:
        return "rating"
    return "other"


# ------------------------------------------------------------------------------------------------------------------------------ world list
def seed_of(wid):
    """deterministic seed of an explicit world from its id (probe noise rng only; the world itself is explicit)."""
    return int.from_bytes(hashlib.sha256(str(wid).encode()).digest()[:4], "big")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_worlds(path):
    d = json.loads(Path(path).read_text())
    ids = [w["id"] for w in d["worlds"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate world ids {[k for k, v in Counter(ids).items() if v > 1]}")
    for w in d["worlds"]:
        validate_spec(w)
    return d


def validate_spec(spec):
    if "id" not in spec or "task" not in spec:
        raise ValueError(f"world spec needs id and task: {spec}")
    if "i" in spec:
        return
    if spec["task"] != "box_behind":
        raise NotImplementedError(f"explicit worlds are built for box_behind only (not {spec['task']!r}): tray fills are not wired yet")
    if "contents_kg" not in spec or "contents_xy" not in spec:
        raise ValueError(f"explicit world spec needs contents_kg and contents_xy: {spec}")


def build_world(spec):
    """-> (seed, world).  Existing: smoke.behind_world(task, i).  Explicit box_behind: BoxWorld(contents_kg, contents_xy) then the task's world_kw (shelf=False), as carry_cert.sample_world +
    smoke.task_world build it; the contents must lie inside the box interior (the block must fit in the crate)."""
    validate_spec(spec)
    from world_model.v2.nero.base import smoke as SM
    if "i" in spec:
        return SM.behind_world(spec["task"], int(spec["i"]))
    from world_model.v2.nero.contents import scenes as SC
    from world_model.v2.stack import tasks as TK
    kg = float(spec["contents_kg"])
    x, y = (float(v) for v in spec["contents_xy"])
    lo, hi = TK.prior_box("box")
    if not (lo <= kg <= hi):
        raise ValueError(f"{spec['id']}: contents {kg} kg outside the prior box {lo}..{hi}")
    b0 = SC.BoxWorld()
    hx = b0.box_outer[0] / 2 - b0.wall_t - b0.contents_half[0]
    hy = b0.box_outer[1] / 2 - b0.wall_t - b0.contents_half[1]
    if abs(x) > hx + 1e-9 or abs(y) > hy + 1e-9:
        raise ValueError(f"{spec['id']}: contents_xy {(x, y)} outside the interior reach (+-{hx:.4f}, +-{hy:.4f})")
    w = dataclasses.replace(SC.BoxWorld(contents_kg=kg, contents_xy=(x, y)), **dict(TK.task("box_behind").world_kw))
    return seed_of(spec["id"]), w


def d1_grid():
    """the D1 box grid: total mass x load offset along the claw axis -> list of explicit specs.  The claw-to-claw axis of the box carry is y of the box frame (scenes.py shelf comment: claws at
    |dy| = 0.186); a test checks the claw sites.  Offsets: 0, half and the full interior reach hy (block touching the wall)."""
    from world_model.v2.nero.contents import objects as OB, scenes as SC
    b0 = SC.BoxWorld()
    hy = b0.box_outer[1] / 2 - b0.wall_t - b0.contents_half[1]
    off = dict(centre=0.0, half=0.5 * hy, wall=hy)
    return [dict(id=f"box_behind:m{tot:g}_{name}", task="box_behind", contents_kg=round(tot - OB.BOX["mass"], 4), contents_xy=[0.0, round(off[name], 5)], total_kg=tot, offset=name)
            for tot in D1_TOTAL_KG for name in D1_OFFSETS]


def write_worlds(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ex = lambda t: [dict(id=f"{t}:{i}", task=t, i=i) for i in range(20)]
    ass = dict(N1="ASSUMED (to be raised with the user): D1 grid = total mass {1.0, 1.75, 2.25, 2.5, 2.75, 3.0, 3.5, 4.5} kg x load offset {centre, half, wall} along the claw-to-claw axis "
                  "(y of the box frame): 0, 0.5 hy and hy (hy = box_outer_y/2 - wall_t - contents_half_y, the block touching the wall); contents = total - box mass; x = 0",
               existing="the 20 worlds of run_n20_v2b: smoke.behind_world(task, i), i < 20 (full prior box)")
    files = {"worlds_existing40.json": ex("box_behind") + ex("tray6_behind"), "worlds_d1_grid24.json": d1_grid(), "worlds_d1_box44.json": ex("box_behind") + d1_grid()}
    for fn, ws in files.items():
        (out / fn).write_text(json.dumps(dict(name=fn[:-5], assumptions=ass, worlds=ws), indent=1) + "\n")
        print(f"{out / fn}: {len(ws)} worlds  sha256 {file_sha256(out / fn)[:12]}")


# ------------------------------------------------------------------------------------------------------------------------------ arms
@dataclasses.dataclass(frozen=True)
class Arm:
    raw: str
    kind: str
    level: str | None = None
    probe: str | None = None
    reask: bool = True
    plan: str | None = None

    @property
    def theta_mode(self):
        return THETA_MODE.get(self.kind)


_OPTS = dict(ours=("probe", "reask"), estimate=("probe", "reask"), oracle_mass=("reask",), oracle_theta=("reask",), no_probe=("reask",), checker_off=("probe",), agentchord=("plan",))


def parse_arm(s):
    s = s.strip()
    toks = s.split(":")
    kind = toks[0]
    if kind not in KINDS:
        raise ValueError(f"unknown arm kind {kind!r} in {s!r}; kinds: {KINDS}")
    level, opts = None, {}
    for t in toks[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            if k not in _OPTS[kind]:
                raise ValueError(f"arm {s!r}: option {k!r} not valid for {kind} (valid: {_OPTS[kind]})")
            if k in opts:
                raise ValueError(f"arm {s!r}: option {k!r} twice")
            opts[k] = v
        elif kind in ("checker_off", "agentchord") and level is None and t in ("L1", "L2", "L3"):
            level = t
        else:
            raise ValueError(f"arm {s!r}: cannot read token {t!r}")
    if kind in ("checker_off", "agentchord") and level is None:
        raise ValueError(f"arm {s!r}: a level L1|L2|L3 is required")
    if kind not in ("checker_off", "agentchord") and level is not None:
        raise ValueError(f"arm {s!r}: no level for {kind}")
    if kind == "agentchord" and not opts.get("plan"):
        raise ValueError(f"arm {s!r}: agentchord needs plan=<dir>")
    if opts.get("reask", "on") not in ("on", "off"):
        raise ValueError(f"arm {s!r}: reask=on|off")
    reask = (kind == "ours") if "reask" not in opts else opts["reask"] == "on"          # default: only `ours` re-asks (its evidence = the source run's, so the cache can hold the reply)
    return Arm(raw=s, kind=kind, level=level, probe=opts.get("probe"), reask=reask, plan=opts.get("plan"))


def parse_arms(spec):
    arms = [parse_arm(a) for a in spec.split(",") if a.strip()]
    ids = [a.raw for a in arms]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate arms {ids}")
    return arms


def check_probe_method(name):
    """unknown probe method -> ValueError (probe.METHODS is the factory's own list; methods built on other branches appear there when merged)."""
    from world_model.v2.nero.contents import probe as P
    if name is not None and name not in P.METHODS:
        raise ValueError(f"probe method {name!r} not in {P.METHODS}")


# ------------------------------------------------------------------------------------------------------------------------------ phase tracker
class PhaseTracker:
    """Phase-tagged U (forced_admit wrapped ep._viol to count every kind; this adds the phase).  attach(ep) wraps four instance attributes of the CarryEpisode (no change to carry.py):
      _viol       every U entry -> U[phase][kind] = {n, t_first}
      restore     a restore() not followed by store() is a backup rollout (executor.run_segments: restore(branch) -> reset_track -> run_phases(backup)); the NEXT run_phases call is tagged "backup"
      store       a segment start: clears the pending flag (the closing restore(end) is followed by the next segment's store)
      run_phases  tags the call "backup" when pending; the base monitor's flagged set is read at the end of each backup call
    The phase is otherwise set by set(name): probe_select calls it through Ctx.on_phase at probe | ramp_hold | nominal | refused_fallback; other jobs call it directly."""

    def __init__(self, phase="probe"):
        self.phase = phase
        self.U = {}
        self.flags = defaultdict(set)
        self.ep = None
        self._pending = False
        self.n_backup_calls = 0

    def _flagged(self):
        b = getattr(self.ep, "bmon", None)
        return set(getattr(b, "flagged", ()) or ()) if b is not None else set()

    def set(self, name):
        if name not in PHASES:
            raise ValueError(f"phase {name!r} not in {PHASES}")
        self.flags[self.phase] |= self._flagged()
        self.phase = name

    def note(self, kind, t, phase=None):
        e = self.U.setdefault(phase or self.phase, {}).setdefault(str(kind), dict(n=0, t_first=round(float(t), 3)))
        e["n"] += 1

    def attach(self, ep):
        self.ep = ep
        orig_viol, orig_restore, orig_store, orig_run = ep._viol, ep.restore, ep.store, ep.run_phases

        def _viol(kind, t, **kw):
            self.note(kind, t)
            return orig_viol(kind, t, **kw)

        def restore(sn):
            self._pending = True
            return orig_restore(sn)

        def store(*a, **k):
            self._pending = False
            return orig_store(*a, **k)

        def run_phases(*a, **k):
            if not self._pending:
                return orig_run(*a, **k)
            saved, self._pending = self.phase, False
            self.phase = "backup"
            self.n_backup_calls += 1
            try:
                return orig_run(*a, **k)
            finally:
                self.flags["backup"] |= self._flagged()
                self.phase = saved

        ep._viol, ep.restore, ep.store, ep.run_phases = _viol, restore, store, run_phases
        return ep

    def result(self):
        self.flags[self.phase] |= self._flagged()
        return ({p: self.U[p] for p in PHASES if p in self.U}, {p: sorted(self.flags[p]) for p in PHASES if self.flags.get(p)})


_ACTIVE = [None]          # the tracker the next CarryEpisode of this process attaches to (one job at a time per process)


def install_tracking():
    """wrap carry.CarryEpisode (on top of same_worlds._init_worker's strict_claw wrapper) so that every episode built while _ACTIVE[0] is set gets the tracker.  Idempotent."""
    from world_model.v2.nero.contents import carry as CR, same_worlds as SW
    SW._init_worker()
    base = CR.CarryEpisode
    if getattr(base, "_demo_tracked", False):
        return

    class _Tracked(base):
        _demo_tracked = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if _ACTIVE[0] is not None:
                _ACTIVE[0].attach(self)
    CR.CarryEpisode = _Tracked


# ------------------------------------------------------------------------------------------------------------------------------ plans (offline)
PLAN_ITEMS = {}           # task -> same_worlds._proposer_plans item; filled in the parent before the pool forks


def load_plans(tasks, out, plan_cache=None, plan_spec=None):
    """the proposer's accepted plan per task from the OFFLINE cache (same_worlds._proposer_plans with offline=True: a miss raises, never the network)."""
    from world_model.v2.nero.contents import same_worlds as SW
    ns = argparse.Namespace(plan_spec=plan_spec or PLAN_SPEC, plan_cache=str(plan_cache or PLAN_CACHE), offline=True, outside_b="refuse", reask_limit=None)
    need = [t for t in tasks if t not in PLAN_ITEMS]
    if need:
        Path(out).mkdir(parents=True, exist_ok=True)
        PLAN_ITEMS.update(SW._proposer_plans(need, ns, Path(out)))
    for t in tasks:
        if PLAN_ITEMS[t].get("rejected"):
            raise RuntimeError(f"{t}: proposer plan rejected: {PLAN_ITEMS[t]['rejected']}")
    return {t: PLAN_ITEMS[t] for t in tasks}


def calib_eps(method):
    from world_model.v2.nero.contents import probe as P
    cal = P.load_calib(P.default_calib(method), method)["eps"]
    return float(cal["EPS_LO"]), float(cal["EPS_FREE"])


# ------------------------------------------------------------------------------------------------------------------------------ verdicts
_VERDICTS = {}


def _verdict(task, w, wp, c_lo, c_hi, com=None):
    from world_model.v2.adapters import base_envelope as BE
    from world_model.v2.nero.contents import behind_job as BJ, carry as CR, probe as P
    bt = BJ.base_task(task)
    st = BJ._static(task, c_lo, c_hi, w, wp, com=com)
    lvl = None
    if st["a_admit"] is not None:
        lvl, _ = BE.pick_level(task, P.interval_spec(bt, c_lo, c_hi, com=com), BJ.LEVELS, BJ.SOURCE)
    return CR._jsonable(dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], base_level=lvl, admit=bool(st["a_admit"] is not None and lvl is not None), contents=[c_lo, c_hi]))


def _world_com(task, w):
    """true contents COM [x, y, z] (object frame) of the world, from a fresh episode (no physics stepped)."""
    from world_model.v2.nero.contents import behind_job as BJ, carry as CR, carry_cert as CC
    from world_model.v2.stack import tasks as TK
    ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=PLAN_ITEMS[task]["plan"]["wp"], base="swerve", goal_support=TK.task(task).goal_support)
    return list(BJ.true_com(ep))


def verdicts(spec, task, w, wp, theta_total, M_true):
    """the checker's verdict on (a) the arm's Theta (prior COM set), (b) the TRUE contents mass (prior COM set), (c) theta* = true mass AND the true COM point (box only).  On the plan the
    ours-arms run (wp); recorded for every arm, obeyed only by the arm that should.  (b) and (c) are cached per world within a process."""
    from world_model.v2.nero.contents import behind_job as BJ, probe as P
    bt = BJ.base_task(task)
    e = P.empty_mass(bt)
    c_true = M_true - e
    key = spec["id"]
    if key not in _VERDICTS:
        com = _world_com(task, w) if bt == "box" else None
        _VERDICTS[key] = dict(true_mass=_verdict(task, w, wp, c_true, c_true), theta_star=(_verdict(task, w, wp, c_true, c_true, com=com) if com is not None else None), true_com=com)
    out = dict(_VERDICTS[key])
    out["theta"] = None
    if theta_total is not None and all(np.isfinite(theta_total)):
        out["theta"] = _verdict(task, w, wp, max(theta_total[0] - e, 0.0), theta_total[1] - e)
    out["note"] = ("static rows (arm + support) + base level on the proposer's first accepted plan; COM set = the prior box interior except theta_star (the true point); "
                   "guarantee: proof under MODEL for the static rows, sampled for the base rows")
    return out


# ------------------------------------------------------------------------------------------------------------------------------ arm runs
def _fl(segments):
    fk = ("fl_steps", "fl_n_infeasible", "fl_n_sat_steps", "fl_n_tau_clipped", "fl_n_speed_binding")
    return {k: int(sum(int((s.get("fl") or {}).get(k, 0) or 0) for s in segments)) for k in fk}


def run_ours_family(arm, spec, seed, w, tr, dry=False):
    from world_model.v2.nero.contents import behind_job as BJ
    task = spec["task"]
    method = arm.probe or DEFAULT_PROBE
    check_probe_method(method)
    eps_lo, eps_free = calib_eps(method)
    item = PLAN_ITEMS[task]
    plan = dict(item["plan"])
    plan["proposer"] = dict(plan["proposer"], **({} if arm.reask else dict(spec=None)))          # reask=off: no proposer spec -> a refusal ends the run (reask_unavailable)
    mode = arm.theta_mode
    extra = dict(theta_mode=mode, on_phase=tr.set, dry=dry, coverage_mode="report" if mode in ("no_probe", "oracle_mass", "oracle_theta") else "enforce")
    i = int(spec["i"]) if "i" in spec else -1
    r = BJ.v2_job(task, i, seed, w, eps_lo, eps_free, plan, method=method, adaptive=True, ctx_extra=extra, oracle_com=(arm.kind == "oracle_theta"))
    if "error" in r:
        return r
    r.update(method="ours", seed_e2e=seed, plan_source=item.get("plan_source"), plan_id=item.get("plan_id"), plan_record=item.get("plan_record"), probe_method_arm=method, reask_arm=arm.reask)
    return r


def run_checker_off(arm, spec, seed, w, tr, dry=False):
    """forced_admit.job's body on an arbitrary world: the probe runs (Theta recorded), no admission, belief = the true contents mass, a_plan = A_PLAN, base level forced."""
    from world_model.v2.adapters import outcome_carry as OCC
    from world_model.v2.nero.contents import behind_job as BJ, carry as CR, carry_cert as CC, probe as P
    from world_model.v2.stack import tasks as TK
    from world_model.v2.stack.proposer import compile as PC
    task, level = spec["task"], arm.level
    method = arm.probe or DEFAULT_PROBE
    check_probe_method(method)
    bt = BJ.base_task(task)
    eps_lo, eps_free = calib_eps(method)
    item = PLAN_ITEMS[task]
    plan, wp0 = item["plan"]["plan_dict"], item["plan"]["wp"]
    ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp0, base="swerve", goal_support=TK.task(task).goal_support)       # the tracker attaches in __init__
    tr.set("probe")
    e = P.empty_mass(bt)
    M = P.true_mass(ep)
    c_true = M - e
    cm = BJ.c_admit_max(task, wp0)
    pr = P.make_probe(ep, bt, eps_lo, eps_free, rng=np.random.default_rng(seed), noise=True, method=method, adaptive=True)
    rec = dict(task=task, i=spec.get("i"), seed=seed, belief_kind="true", level=level, a_plan=A_PLAN, plan_id=item.get("plan_id"), plan_record=item.get("plan_record"), method="checker_off",
               M_true=M, c_true=c_true, probe_method=P.method_of(pr), probe_method_arm=method, base_level=level)
    if dry:
        rec.update(outcome="probe_only", dry=True)
        return rec
    why = pr.run(M_stop=None if cm is None else e + cm)
    probe_viol = CR._jsonable(ep.track["viol"])
    rec.update(Theta=[pr.M_lo, pr.M_hi], why=why, peak_u=round(getattr(pr, "peak_u", 0.0), 4), probe_viol=probe_viol, theta_sound=bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9))
    tr.set("ramp_hold")
    ep.wp, ep.cc = PC.compile_plan(plan, task, build=False, level=level).wp, None
    ep.set_belief(c_true, a_plan=A_PLAN)
    m_nom = ep.m_nom
    ep.ctl.mass = pr.m_ff
    pr._ramp(m_nom, 0.3)
    ep.run_phases([("hold", 0.4)])
    ep.ctl.mass = m_nom
    tr.set("nominal")
    carry = CR.run_segments(ep, dict(segments=[]), branch_backups=True)
    any_U = bool(probe_viol or ep.track["viol"] or any(s["nominal"]["viol"] for s in carry["segments"]))
    outcome = "carried" if (carry["end_in_R"][0] and not any(s["nominal"]["viol"] for s in carry["segments"])) else "carry_failed"
    rec.update(belief=c_true, carry=dict(segments=carry["segments"], end_in_R=carry["end_in_R"], pre_hold_viol=carry["pre_hold_viol"]), outcome=outcome, any_U=any_U)
    rec.update(OCC.measure(ep, task, True, any_U, carry["pre_hold_viol"], refused=False, reason=outcome, end_in_R=tuple(carry["end_in_R"])))
    bl = getattr(ep, "base_log", [])
    rec["base"] = dict(level=level, n_base_edges=len(bl), n_aborted=sum(1 for b in bl if b.get("aborted")), route_end_err_mm=[b.get("err_mm") for b in bl],
                       U_flags={k: (k in set(ep.bmon.flagged)) for k in BJ.BASE_FLAGS}, base_end_pose=[float(v) for v in ep.sb.pose()])
    rec["first_U"] = BJ._first_viol(ep, probe_viol)
    return rec


def run_agentchord(arm, spec, seed, w, tr, dry=False):
    """AgentChord's cached graph.  Existing worlds: execute.same_world_trial (their entry, as run_n20_v2b); explicit worlds: behind.run_world(dict(world=(seed, w), ...)) + the
    same_world_trial wrapper fields."""
    if dry:
        raise NotImplementedError("agentchord has no dry mode")
    from world_model.v2.agentchord_probe.nero import behind as BH, execute as AX
    from world_model.v2.stack import tasks as TK
    tr.set("nominal")                                          # no probe / ramp: their episode is one nominal run (pre-hold + graph)
    task = spec["task"]
    if "i" in spec:
        return AX.same_world_trial(task, int(spec["i"]), arm.plan, level=arm.level)
    pj = Path(arm.plan)
    pj = pj if pj.suffix == ".json" else pj / "plan.json"
    lo, hi = TK.prior_box(task)
    r = BH.run_world(dict(plan=str(pj), task=task, i=-1, level=arm.level, world=(seed, w), m_hat_kg=0.5 * (lo + hi), mon_period=0.1, edge_settle=0.3))
    fu = None
    if r.get("any_U"):
        if r.get("pre_hold_viol"):
            fu = dict(seg="pre_hold", kind=r["pre_hold_viol"].get("kind"))
        else:
            ed = next((e for e in r.get("edges", []) if (e.get("track") or {}).get("viol")), None)
            v = (ed["track"]["viol"] if ed else None) or (r.get("track") or {}).get("viol") or {}
            fu = dict(seg=ed["edge_id"] if ed else None, kind=v.get("kind"))
    pr = r.get("proposer") or {}
    r.update(method="theirs", seed_e2e=seed, first_U=fu, plan_model=pr.get("model"), plan_reasoning=pr.get("reasoning"), base_level=arm.level)
    return r


def job(args):
    """(spec, arm string, dry) -> record (never raises)."""
    spec, armstr, dry = args
    t0 = time.time()
    base = dict(world_id=spec["id"], arm=armstr, spec=spec, task=spec.get("task"))
    try:
        from world_model.v2.nero.contents import carry as CR
        arm = parse_arm(armstr)
        install_tracking()
        tr = PhaseTracker()
        _ACTIVE[0] = tr
        task = spec["task"]
        seed, w = build_world(spec)
        fn = dict(checker_off=run_checker_off, agentchord=run_agentchord).get(arm.kind, run_ours_family)
        r = fn(arm, spec, seed, w, tr, dry)
        if r is None or "error" in r:
            return dict(base, error=(r or {}).get("error", "no record"), tb=(r or {}).get("tb"), wall_s=round(time.time() - t0, 1))
        rec = dict(base, **r)
        rec["world_id"], rec["arm"], rec["arm_kind"] = spec["id"], armstr, arm.kind
        if not dry:
            rec["U_by_phase"], rec["base_flags_by_phase"] = tr.result()
            va = getattr(getattr(tr, "ep", None), "track", None) or {}
            if "viol_all" in va:
                rec["viol_all"] = va["viol_all"]
            segs = (rec.get("carry") or {}).get("segments") or []
            if segs:
                rec["fl"] = _fl(segs)
            rec["n_backup_rollouts"] = tr.n_backup_calls
        if "M_true" not in rec and "truth_total_kg" in rec:                      # agentchord records
            rec["M_true"] = rec["truth_total_kg"]
        item = PLAN_ITEMS.get(task)
        if item is not None and rec.get("M_true") is not None:
            rec["verdicts"] = verdicts(spec, task, w, item["plan"]["wp"], rec.get("Theta"), float(rec["M_true"]))
            rec["verdicts"]["plan_id"] = item.get("plan_id")
        rec["assumptions"] = assumptions()
        rec["wall_s"] = round(time.time() - t0, 1)
        return CR._jsonable(rec)
    except Exception as ex:
        return dict(base, error=repr(ex), tb=traceback.format_exc()[-1500:], wall_s=round(time.time() - t0, 1))
    finally:
        _ACTIVE[0] = None


def assumptions():
    return dict(A_PLAN=f"{A_PLAN} ASSUMED (N10, checker_off only)", oracle="oracle_* arms use the TRUE contents (mass; theta*: also the COM point): not deployable, reference only",
                estimate="estimate arm: admission on the probe midpoint as a point, no guarantee", base_source="measured (base card, sampled)", u_admit="MODEL 1.0 (rated limits, USER 11-09)",
                probe_calib="MEASURED poslift CALIB n=20/task (probe.default_calib)", plan="offline cache replay, no LLM call", grid="D1 grid N1 ASSUMED")


# ------------------------------------------------------------------------------------------------------------------------------ run
def code_sha():
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception as ex:
        return f"unknown ({ex!r})"


def read_done(out):
    done = set()
    for f in sorted(Path(out).glob("trials_shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r and "world_id" in r:
                done.add((r["world_id"], r["arm"]))
    return done


def freeze_worlds(path, out):
    """copy the world list to OUT/worlds.json and write its sha256; a different list in an existing run dir is refused."""
    out = Path(out)
    h = file_sha256(path)
    hf = out / "WORLDS.sha256"
    if hf.exists() and hf.read_text().split()[0] != h:
        raise SystemExit(f"{out}: WORLDS.sha256 {hf.read_text().split()[0][:12]} differs from {path} ({h[:12]}): the world list is frozen per run dir")
    if not hf.exists():
        (out / "worlds.json").write_bytes(Path(path).read_bytes())
        hf.write_text(f"{h}  {Path(path).name}\n")
    return h


def run(a):
    import multiprocessing as mp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    worlds = load_worlds(a.worlds)["worlds"]
    arms = parse_arms(a.arms)
    for arm in arms:
        check_probe_method(arm.probe)
    wh = freeze_worlds(a.worlds, out)
    only = set(a.only.split(",")) if a.only else None
    k, K = (int(x) for x in a.shard.split("/"))
    mine = [(w, ar.raw) for w in worlds for ar in arms if only is None or w["id"] in only][k::K]
    done = read_done(out)
    todo = [(w, ar, bool(a.dry)) for w, ar in mine if (w["id"], ar) not in done]
    load_plans(sorted({w["task"] for w, ar, _ in todo if parse_arm(ar).kind != "agentchord"}), out, a.plan_cache, a.plan_spec)
    cfg = dict(host=os.uname().nodename, code_sha=code_sha(), worlds_file=str(a.worlds), worlds_sha256=wh, n_worlds=len(worlds), arms=[x.raw for x in arms], shard=f"{k}/{K}",
               n_jobs=len(todo), skipped_done=len(mine) - len(todo), procs=a.procs, dry=bool(a.dry), plan_cache=str(a.plan_cache or PLAN_CACHE), plan_spec=a.plan_spec or PLAN_SPEC,
               plan_ids={t: it.get("plan_id") for t, it in PLAN_ITEMS.items()}, agentchord_plans={x.raw: x.plan for x in arms if x.kind == "agentchord"}, assumptions=assumptions(), offline=True,
               threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / f"CONFIG_shard{k}of{K}.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"[demo] shard {k}/{K}: {len(todo)} jobs ({len(mine) - len(todo)} already done), procs {a.procs}", flush=True)
    pool = None
    if a.procs <= 1:
        results = map(job, todo)
    else:
        pool = mp.get_context("fork").Pool(a.procs, maxtasksperchild=3)
        results = pool.imap_unordered(job, todo)
    with open(out / f"trials_shard{k}of{K}.jsonl", "a") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            x = r.get("result") or {}
            print(f"[demo] {r.get('world_id')} {r.get('arm')} -> {r.get('outcome', r.get('error'))} success={x.get('success')} unsafe={x.get('unsafe')} "
                  f"U={ {p: list(v) for p, v in (r.get('U_by_phase') or {}).items()} } wall {r.get('wall_s')}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


# ------------------------------------------------------------------------------------------------------------------------------ summary
def is_refused(r):
    x = r.get("result") or {}
    return bool(x.get("refused")) or str(r.get("outcome", "")).startswith("refused") or r.get("outcome") == "plan_rejected"


def u_kinds(r, phases):
    """U kinds of the record in `phases` (U_by_phase + base flags); records without U_by_phase (older runs) fall back to first_U / any_U in 'nominal'."""
    ub = r.get("U_by_phase")
    if ub is None:
        if "nominal" in phases and r.get("any_U"):
            return {(r.get("first_U") or {}).get("kind") or "unknown"}
        return set()
    ks = set()
    for p in phases:
        ks |= set((ub.get(p) or {}).keys())
        ks |= set((r.get("base_flags_by_phase") or {}).get(p, []))
    return ks


def classify(r):
    """per-record facts the summary counts."""
    nom = u_kinds(r, NOMINAL_PHASES)
    bk = u_kinds(r, ("backup",)) - nom
    x = r.get("result") or {}
    return dict(refused=is_refused(r), success=bool(x.get("success")), safe_success=bool(x.get("safe_success")), unsafe_nominal=bool(nom), nominal_kinds=sorted(nom),
                groups=sorted({u_group(k) for k in nom}), backup_only_kinds=sorted(bk), error="error" in r)


def refusal_class(r, by_arm, ref_level="L1"):
    """why a refusal happened (plan Stage 1 Metrics), by joining the arms on the world:
       necessary             oracle_theta also refuses (and checker_off:L1 shows a contents / base U, or did not run)
       model_only            oracle_theta refuses but checker_off:L1 ran with no contents / base U
       unnecessary_mass      oracle_mass admits (the mass alone would have been enough)
       unnecessary_position  oracle_theta admits, oracle_mass refuses (the load position was needed)
       undetermined          the needed oracle record is missing."""
    ot, om = by_arm.get("oracle_theta"), by_arm.get("oracle_mass")
    if ot is None or "error" in ot:
        return "undetermined"
    if is_refused(ot):
        co = by_arm.get(f"checker_off:{ref_level}")
        if co is not None and "error" not in co and not is_refused(co):
            if not ({u_group(k) for k in u_kinds(co, NOMINAL_PHASES)} & {"contents", "base"}):
                return "model_only"
        return "necessary"
    if om is not None and "error" not in om and not is_refused(om):
        return "unnecessary_mass"
    if om is None or "error" in om:
        return "undetermined"
    return "unnecessary_position"


def true_mass_of(recs):
    for r in recs:
        if r.get("M_true") is not None:
            return float(r["M_true"])
    return None


def load_records(out):
    recs = []
    for f in sorted(Path(out).glob("trials_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def _new_cell():
    return dict(n=0, errors=0, success=0, safe_success=0, refused=0, admitted=0, unsafe=0, groups=Counter(), kinds=Counter(), backup_only=0, backup_kinds=Counter(), refusal=Counter(),
                wrong_admit=0, admit_theta_star_refuses=0, wrong_admit_worlds=[])


def build_summary(recs):
    """-> (cells {task: {arm: counts}}, by_world {world_id: {arm: record}}).  A re-run record replaces the earlier one (last wins); error records are counted."""
    last = {}
    for r in recs:
        last[(r.get("world_id"), r.get("arm"))] = r
    by_world = defaultdict(dict)
    for (wid, arm), r in last.items():
        by_world[wid][arm] = r
    cells = defaultdict(lambda: defaultdict(_new_cell))
    for wid, arms in by_world.items():
        for arm, r in arms.items():
            c = cells[r.get("task")][arm]
            c["n"] += 1
            f = classify(r)
            if f["error"]:
                c["errors"] += 1
                continue
            c["success"] += f["success"]
            c["safe_success"] += f["safe_success"]
            c["refused"] += f["refused"]
            c["admitted"] += not f["refused"]
            c["unsafe"] += f["unsafe_nominal"]
            c["groups"].update(f["groups"])
            c["kinds"].update(f["nominal_kinds"])
            c["backup_only"] += bool(f["backup_only_kinds"])
            c["backup_kinds"].update(f["backup_only_kinds"])
            if r.get("arm_kind") in OURS_FAMILY:
                if f["refused"]:
                    c["refusal"][refusal_class(r, arms)] += 1
                else:
                    if f["unsafe_nominal"]:
                        c["wrong_admit"] += 1
                        c["wrong_admit_worlds"].append(wid)
                    ts = (r.get("verdicts") or {}).get("theta_star")
                    if ts and not ts.get("admit"):
                        c["admit_theta_star_refuses"] += 1
    return cells, by_world


def _plain(x):
    if isinstance(x, (Counter, defaultdict)):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def summary(a):
    out = Path(a.out)
    recs = load_records(out)
    cells, by_world = build_summary(recs)
    cfgs = [json.loads(p.read_text()) for p in sorted(out.glob("CONFIG_shard*.json"))]
    sj = dict(cells=_plain(cells), n_records=len(recs), worlds_sha256=(cfgs[0]["worlds_sha256"] if cfgs else None), code_sha=sorted({c["code_sha"] for c in cfgs}),
              per_world={wid: {arm: dict(classify(r), outcome=r.get("outcome"), M_true=r.get("M_true"), base_level=r.get("base_level"), refusal=(refusal_class(r, arms) if is_refused(r) else None),
                                         spec=r.get("spec")) for arm, r in arms.items()} for wid, arms in by_world.items()})
    (out / "summary.json").write_text(json.dumps(sj, indent=1, default=str))
    L = ["# demo runner summary -- machine-written by demo_runner.summary", "",
         f"records {len(recs)}; world list sha256 {sj['worlds_sha256']}; code {sj['code_sha']}", "",
         "success = finished AND goal (stack.outcome); safe success = success AND no U anywhere.  unsafe = U on the NOMINAL path (probe | ramp_hold | nominal phases, base flags included); "
         "backup-only U = U only inside backup rollouts (counted separately).  U groups (worlds with >= 1 kind in the group): contents = " + "/".join(U_CONTENTS) + "; base = " + "/".join(U_BASE)
         + "; rating = " + "/".join(U_RATING) + "; else other.  Refusal split (ours-family arms, by joining the oracle arms on the world): necessary / unnecessary_mass / unnecessary_position / "
         "model_only / undetermined.  Wrong admit = admitted AND nominal U.  'admit, theta* refuses' = admitted although the static rows on the true point (theta*) refuse.  Guarantees: "
         "static rows = proof under MODEL, base rows = sampled; the counts are over the sampled worlds of the list.", ""]
    for task in sorted(cells, key=str):
        L += [f"## {task}", "", "| arm | n | errors | success | safe success | unsafe | refused | contents | base | rating | other | backup-only U | wrong admits | admit, theta* refuses | refusals |",
              "|" + "---|" * 15]
        for arm in sorted(cells[task]):
            c = cells[task][arm]
            g = c["groups"]
            L.append(f"| {arm} | {c['n']} | {c['errors']} | {c['success']} | {c['safe_success']} | {c['unsafe']} | {c['refused']} | {g.get('contents', 0)} | {g.get('base', 0)} | {g.get('rating', 0)} | "
                     f"{g.get('other', 0)} | {c['backup_only']} | {c['wrong_admit']} | {c['admit_theta_star_refuses']} | {dict(c['refusal']) or '-'} |")
        L.append("")
        for arm in sorted(cells[task]):
            c = cells[task][arm]
            if c["kinds"] or c["wrong_admit_worlds"]:
                L.append(f"- {arm}: nominal U kinds {dict(c['kinds'])}; backup-only kinds {dict(c['backup_kinds']) or '-'}; wrong-admit worlds {c['wrong_admit_worlds'] or '-'}")
        L.append("")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        make_figures(out, by_world)
    except Exception as ex:                                  # a missing matplotlib / empty grid must not lose the tables
        print(f"[demo] figures skipped: {ex!r}")


# ------------------------------------------------------------------------------------------------------------------------------ figures
CAT_COLORS = {"safe success": "#2a9d5c", "success, U": "#e9a23b", "unsafe": "#d1495b", "failed, no U": "#8d6fb5", "refused": "#9aa0a6", "missing": "#ffffff", "error": "#000000"}


def cell_category(r):
    if r is None:
        return "missing"
    f = classify(r)
    if f["error"]:
        return "error"
    if f["refused"]:
        return "refused"
    if f["safe_success"]:
        return "safe success"
    if f["unsafe_nominal"]:
        return "unsafe"
    if f["success"]:
        return "success, U"
    return "failed, no U"


def _world_order(by_world, task):
    rows = []
    for wid, arms in by_world.items():
        recs = list(arms.values())
        if recs[0].get("task") != task:
            continue
        m = true_mass_of(recs)
        off = (recs[0].get("spec") or {}).get("contents_xy")
        rows.append((m if m is not None else 99.0, abs(off[1]) if off else 0.0, wid))
    return [w for _, _, w in sorted(rows)]


def make_figures(out, by_world):
    """F4 outcome matrix, F2 level staircase, F1 map (explicit grid worlds only) per task; F3 is a stub.  PNG via matplotlib Agg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out = Path(out)
    for task in sorted({list(a.values())[0].get("task") for a in by_world.values()}, key=str):
        wids = _world_order(by_world, task)
        arms = sorted({arm for w in wids for arm in by_world[w]})
        if not wids or not arms:
            continue
        ms = {w: true_mass_of(list(by_world[w].values())) for w in wids}
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(arms) + 2), max(3, 0.28 * len(wids) + 1.5)))
        for yi, w in enumerate(wids):
            for xi, arm in enumerate(arms):
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1, color=CAT_COLORS[cell_category(by_world[w].get(arm))], ec="white", lw=0.8))
        ax.set_xlim(0, len(arms))
        ax.set_ylim(len(wids), 0)
        ax.set_xticks([i + 0.5 for i in range(len(arms))])
        ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([i + 0.5 for i in range(len(wids))])
        ax.set_yticklabels([f"{w.split(':', 1)[-1]}  {ms[w]:.2f} kg" if ms[w] is not None else w for w in wids], fontsize=6)
        ax.set_title(f"F4 outcome matrix, {task} (worlds sorted by true total mass)", fontsize=9)
        ax.legend(handles=[Patch(color=c, label=n) for n, c in CAT_COLORS.items() if n != "missing"], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"F4_outcome_matrix_{task}.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3.5))                 # F2: the base level each arm ran at (0 = refused); checker_off: its forced level if it ran without U, else 0
        xs = list(range(len(wids)))
        for arm in arms:
            if arm.startswith("agentchord"):
                continue
            ys = []
            for w in wids:
                r = by_world[w].get(arm)
                if r is None or "error" in r:
                    ys.append(np.nan)
                elif arm.startswith("checker_off"):
                    ys.append(int(arm.split(":")[1][-1]) if not u_kinds(r, NOMINAL_PHASES) else 0)
                else:
                    lv = r.get("base_level")
                    ys.append(int(str(lv)[-1]) if lv and not is_refused(r) else 0)
            ax.step(xs, ys, where="mid", label=arm, lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{ms[w]:.2f}" if ms[w] is not None else "" for w in wids], rotation=90, fontsize=6)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["none", "L1", "L2", "L3"])
        ax.set_xlabel("worlds sorted by true total mass [kg]")
        ax.set_title(f"F2 base level by world, {task} (checker_off: its level only if it ran without U)", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"F2_level_staircase_{task}.png", dpi=150)
        plt.close(fig)
        grid = [w for w in wids if (list(by_world[w].values())[0].get("spec") or {}).get("contents_xy")]
        if grid:                                                  # F1: total mass x |load offset|, one panel per arm
            fig, axs = plt.subplots(1, len(arms), figsize=(2.6 * len(arms), 3), sharey=True, squeeze=False)
            for ax, arm in zip(axs[0], arms):
                for w in grid:
                    sp = list(by_world[w].values())[0]["spec"]
                    ax.scatter([ms[w]], [abs(sp["contents_xy"][1])], s=40, c=CAT_COLORS[cell_category(by_world[w].get(arm))], edgecolors="k", linewidths=0.4)
                ax.set_title(arm, fontsize=6)
                ax.set_xlabel("total mass [kg]", fontsize=7)
            axs[0][0].set_ylabel("|load offset| along claw axis [m]", fontsize=7)
            fig.suptitle(f"F1 map, {task} (colours as F4)", fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"F1_map_{task}.png", dpi=150)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.text(0.5, 0.5, "F3 bound vs measured per row kind: stub\n(needs the per-row margins of the grid run; to be drawn from summary.json)", ha="center", va="center", fontsize=8)
    ax.axis("off")
    fig.savefig(out / "F3_bound_vs_measured_stub.png", dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------ cli
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "worlds"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", help="world list JSON (run)")
    ap.add_argument("--arms", help="comma-separated arm ids (run)")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default=None, help="world ids, comma-separated")
    ap.add_argument("--plan-cache", default=None)
    ap.add_argument("--plan-spec", default=None)
    ap.add_argument("--dry", action="store_true", help="stop at the admission (ours-family arms) / after construction (checker_off): smoke only")
    a = ap.parse_args(argv)
    if a.cmd == "worlds":
        return write_worlds(a.out)
    if a.cmd == "summary":
        return summary(a)
    if not a.worlds or not a.arms:
        raise SystemExit("run needs --worlds and --arms")
    run(a)


if __name__ == "__main__":
    main()
