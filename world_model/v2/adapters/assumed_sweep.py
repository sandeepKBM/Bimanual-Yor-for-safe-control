"""adapters/assumed_sweep.py -- vary the ASSUMED checker / planner values on a near-miss cell and report, per value, the full verdict.  SIMULATION ONLY, offline, no VLM.   (USER 11-16)

Why: a case that is probably a success but is refused by a few rows with small margins.  For every ASSUMED value those rows depend on (KNOBS below) this tool runs the
checks at a grid of values and reports the BREAK-EVEN value (the least change from today's value, in the lenient direction, at which the refusing rows pass) and the NEXT
blocker at that value.  The output is "the value needed", NOT an accepted value: every swept value stays ASSUMED (tag "ASSUMED (break-even, not measured)") and each knob names
the measurement that could turn it MEASURED (`justify`).

Levels (per knob, registry field `level`; `--level` overrides):
  static    adapters/z3_oracle_contents with the DEFAULT plan values (recomputed after the patch: e.g. CLEAR_M moves the default lift height) and all_stages=True -> every row of
            stages a..e (plan, static admission, motion + collision rows, Rule 1 per step, tilt) with margins
  episode   n seeds of the executed closed_chain carry as nero/contents/tilt_measure runs it (carry.world_trial, nominal, no backups) -> pseudo rows:
            episode_U (no stop; margin = min object-world clearance - U_OBJ_CLEAR_M when the run measured one), episode_tilt (U tilt limit - max tilt)
  both      static + episode rows together
  rule4     STOP_K_MAX, STOP_T_S_R4: adapters/rule4_contents.run_trap on the cell (its consumer is the Rule 4 transfer rows, not the oracle stages)
Each (knob, value, cell) runs in a FRESH process (one `--run-one` subprocess per run; owner 11-16, was a spawn Pool): every alias of the knob is set, then the checks are built.  The EFFECTIVE value is read back
(every alias after the run; for PATH_MARGIN_MIN the margin the joint_pos row was actually built with; for the d_min knobs the table collision_rows actually used); if it differs
from the requested value the record gets status "patch_failed" and NO rows are reported.

    python -m world_model.v2.adapters.assumed_sweep --task box --cells 1.0-1.5,2.5-3.0 --knobs PATH_MARGIN_MIN,CLEARANCE_MIN --workers 3 --out DIR
    ... --combo "PATH_MARGIN_MIN=0.02,CLEARANCE_MIN=0.002"      one joint setting (full verdict), no grid
    ... --near-miss oracle_box_1.0-1.5_default.json [...]        pick near-miss rows, print the sweep commands (does not run them)
    ... --list-knobs                                             print the registry
Env for runs: BIMANUAL_PLANNER=closed_chain (set here); thread pinning to 1 (set here).  The planner's geometry cache is a COPY under OUT/plan_cache (keys include the patched values).

Guesses (see also the build report): leniency directions are ASSUMED per knob (registry `lenient`; 0 = both directions are searched by |change|); `rows` patterns are substrings
tested as a PREFIX of the row name (and "collision" also against the failing sub-row text of a rule1[...] row); CLEARANCE_MIN is patched at collision_rows.d_min_table ONLY (the hold-pose
selection of nyu_hold reads the same constant for a different purpose and stays at 5 mm).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

TAG = "ASSUMED (break-even, not measured)"
M_CARRY = "world_model.v2.nero.contents.carry"
M_CC = "world_model.v2.stack.planners.closed_chain"
M_RC = "world_model.v2.adapters.rule1_contents"
M_R4 = "world_model.v2.adapters.rule4_contents"
M_TYPES = "world_model.v2.stack.types"
M_PROBE = "world_model.v2.nero.contents.probe"
M_ADMIT = "world_model.v2.stack.checks.admit"
M_CERT = "world_model.v2.nero.contents.carry_cert"
M_MOTION = "world_model.v2.stack.checks.motion"
M_COLROWS = "world_model.v2.adapters.collision_rows"

# alias descriptors:  ("attr", module, name) | ("item", module, name, key) | ("kwdefault", module, func, arg) | ("dmin", module, func, keys) | ("dyrange", module, name)
_MARGINS = [round(0.005 * i, 3) for i in range(0, 13)]                      # 0 .. 0.06 rad
_DY_RANGES = [(-0.04, 0.06), (-0.06, 0.08), (-0.08, 0.10)]                   # today's DY_GRID range first, then widened

KNOBS = {
    "PATH_MARGIN_MIN": dict(
        aliases=[("attr", M_CARRY, "PATH_MARGIN_MIN")], grid=_MARGINS, lenient=-1, rows=["joint_pos"], level="static", unit="rad",
        justify="J7 / joint-limit tracking error of the executed path vs the planned path (measured max over episodes), plus the URDF limit vs the firmware soft limit",
        consumers="adapters/motion_rows._limits (joint_pos row margin), nero/contents/path_reach.py, nyu_hold path check (call-time reads)",
        note="checker margin only; the planner's own target is MARGIN_WANT (a separate copy)"),
    "MARGIN_WANT": dict(
        aliases=[("attr", M_CC, "MARGIN_WANT")], grid=_MARGINS, lenient=-1, rows=["joint_pos", "plan"], level="static", unit="rad",
        justify="same as PATH_MARGIN_MIN (J7 tracking error): the planner target should be the checker margin",
        consumers="closed_chain.py:503-504 (screening of the lift offset dy), plan key params (closed_chain.py:159)", note="planner target only; the checker margin is carry.PATH_MARGIN_MIN"),
    "PATH_MARGIN_BOTH": dict(
        aliases=[("attr", M_CARRY, "PATH_MARGIN_MIN"), ("attr", M_CC, "MARGIN_WANT")], grid=_MARGINS, lenient=-1, rows=["joint_pos", "plan"], level="static", unit="rad",
        justify="J7 / joint-limit tracking error (one number for the checker margin and the planner target)", consumers="both above", note="the two copies moved together"),
    "PSI_STEP_DEG": dict(
        aliases=[("attr", M_CC, "PSI_STEP_DEG")], grid=[3.0, 2.0, 1.0], lenient=-1, rows=["joint_pos", "reach", "plan"], level="static", unit="deg",
        justify="not a physical value: a planner search resolution (finer = more IK arm-angle candidates); no measurement needed, only run time",
        consumers="closed_chain._psi_grid (call-time), plan key params", note="finer grid = more room for J7 at the cost of a longer DP"),
    "DY_RANGE": dict(
        aliases=[("dyrange", M_CC, "DY_GRID")], grid=_DY_RANGES, lenient=+1, rows=["joint_pos", "reach", "plan"], level="static", unit="m", order_key=lambda v: v[1] - v[0],
        justify="the free lateral space at the counter / shelf (measured scene layout) bounding how far the lift may be shifted sideways",
        consumers="closed_chain._dy_list / _screen_dy (call-time), plan key params", note="value = (lo, hi) of the lateral lift offset search grid [m], step 0.005 (today -0.04..0.06)"),
    "CLEARANCE_MIN": dict(
        aliases=[("dmin", M_COLROWS, "d_min_table", ("arm_arm", "arm_world"))], grid=[round(0.0005 * i, 4) for i in range(0, 13)], lenient=-1, rows=["collision"], level="static", unit="m",
        justify="the pose / calibration error of the arm links vs the world (hand-eye + FK error), measured on the robot, plus the model-vs-mesh distance error",
        consumers="adapters/collision_rows.d_min_table (collision_rows rows and the rule1 recovery collision rows; both call RCOL.d_min_table at call time)",
        note="NOT patched: nero/clearance.CLEARANCE_MIN itself (used by nyu_hold hold-pose selection, base_route, base_sweep) and nyu_hold.BODY_CLEAR_MIN (copy)"),
    "CLEAR_M_BOX": dict(
        aliases=[("item", M_CC, "CLEAR_M", "box")], grid=[round(0.02 + 0.005 * i, 3) for i in range(0, 9)], lenient=0, rows=["collision", "joint_pos", "reach", "plan"], level="static", unit="m",
        justify="the measured object-pose error at the lift / carry (perception + tracking) over the shelf edge",
        consumers="closed_chain.object_path, _lift_collision (call-time dict reads), base_route, z3_oracle default_vals (default zc moves with it)",
        note="trade-off (more clearance helps the shelf, hurts the lift reach): both directions searched by |change|"),
    "U_OBJ_CLEAR_M": dict(
        aliases=[("attr", M_CARRY, "U_OBJ_CLEAR_M")], grid=[round(0.0005 * i, 4) for i in range(0, 17)], lenient=-1, rows=["collision", "episode_U"], level="both", unit="m",
        justify="the object pose + shape error vs the world geometry (perception / calibration), measured on the robot",
        consumers="carry.CarryEpisode._obj_world_check + U monitor (call-time), adapters/collision_rows.d_min_table (obj_world)",
        note="NOT patched: stack/tasks.py:137 (hard-coded 0.005 U bound of the stack monitor, not on the world_trial path), z3_watcher_contents.py:431 (sets 0.0 itself)"),
    "SUPPORT_APPROACH_DEG": dict(
        aliases=[("attr", M_CARRY, "SUPPORT_APPROACH_DEG")], grid=[15.0, 30.0, 45.0], lenient=+1, rows=["collision", "episode_U"], level="both", unit="deg",
        justify="the geometry that defines 'resting on / approaching a support top' (a definition; justify by the measured approach direction of the real set-down)",
        consumers="carry._obj_world_check (call-time), adapters/collision_rows (cosl)", note=""),
    "TAU_NOISE": dict(
        aliases=[("attr", M_PROBE, "TAU_NOISE")], grid=[0.01, 0.02, 0.03, 0.05, 0.08], lenient=-1, rows=["rule1["], level="static", unit="Nm",
        justify="NERO joint-torque read noise: measure the read at rest / constant hold on the real arm (sd and max over n samples per joint)",
        consumers="rule1_contents.Ctx.read_nb (call-time: nb = TAU_NOISE * sum|pinv(J^T)_z| / g, max over path samples) -> claw_read_spec noise",
        note="nero/pan/cert/probe.TAU_NOISE (pan project) is a separate constant, not patched"),
    "BIN_WIDTH_KG": dict(
        aliases=[("attr", M_RC, "BIN_WIDTH_KG")], grid=[0.1, 0.25, 0.5, 1.0], lenient=-1, rows=["rule1["], level="static", unit="kg",
        justify="not a physical value: the reading-bin design choice; the finest bin the claw-load read can resolve = its measured noise (EPS_FREE + nb)",
        consumers="rule1_contents.read_program (call-time, n bins = ceil(range / width))", note=""),
    "A_BACKUP": dict(
        aliases=[("attr", M_CARRY, "A_BACKUP"), ("attr", M_RC, "A_BACKUP"), ("attr", M_R4, "R1_A_BACKUP")], grid=[0.1, 0.15, 0.25, 0.4], lenient=-1, rows=["rule1["], level="static", unit="m/s^2",
        justify="the retrace acceleration the real arms + gripper hold the load at without slip (measured slip / torque margin on hardware)",
        consumers="rule1_contents (stored backup + read program cap), rule4_contents.R1_A_BACKUP (import-time copy), stack/executor (CR.A_BACKUP, call-time), closed_chain retrace args",
        note="the stored backup's motion depends on it, so the Rule 1 rows (not the nominal plan) move"),
    "A_MIN": dict(
        aliases=[("attr", M_ADMIT, "A_MIN"), ("attr", M_CERT, "A_MIN"), ("attr", M_RC, "A_MIN")], grid=[0.02, 0.05], lenient=-1, rows=["static_action", "rule1["], level="static", unit="m/s^2",
        justify="a task-timing choice (the slowest acceptable carry); no measurement, a user ruling",
        consumers="checks/admit.admit (call-time), carry_cert.static_cell, rule1_contents.A_MIN (import-time copy), rule4_contents (RC.A_MIN call-time)", note=""),
    "EPS_CC_POS": dict(
        aliases=[("attr", M_MOTION, "EPS_CC_POS"), ("kwdefault", M_MOTION, "closed_chain_row", "eps_pos")], grid=[0.001, 0.002, 0.003, 0.005], lenient=+1, rows=["closed_chain"], level="static", unit="m",
        justify="the stiffness / compliance of the real weld grasp + the arms' FK error: the object-pose disagreement the real gripper pair can absorb (measured internal force vs offset)",
        consumers="motion.closed_chain_row (DEFAULT ARGUMENT bound at def time -> the kwdefault alias)", note=""),
    "EPS_CC_ROT": dict(
        aliases=[("attr", M_MOTION, "EPS_CC_ROT"), ("kwdefault", M_MOTION, "closed_chain_row", "eps_rot")], grid=[0.005, 0.01, 0.02], lenient=+1, rows=["closed_chain"], level="static", unit="rad",
        justify="same as EPS_CC_POS (rotation disagreement the weld absorbs)", consumers="motion.closed_chain_row (default argument)", note=""),
    "STOP_K_MAX": dict(
        aliases=[("attr", M_R4, "STOP_K_MAX")], grid=[2, 4, 6], lenient=+1, rows=["rule4["], level="rule4", unit="doublings",
        justify="the longest stop time the task tolerates (0.3 s * 2^k); a task-timing ruling, plus the measured stop deceleration the arms track",
        consumers="rule4_contents.build_transfer / transfer_row (call-time); consumer = Rule 4 transfer rows, NOT the oracle stages (run via rule4_contents.run_trap)", note="the trap scenario of rule4_contents (box 2.5-3.0, carry, t=0.75)"),
    "STOP_T_S": dict(
        aliases=[("attr", M_TYPES, "STOP_T_S")], grid=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5], lenient=0, rows=["rule1["], level="static", unit="s",
        justify="the stop time the real arms track at the stop deceleration (measured stop distance / tracking vs commanded Hermite stop; constant deceleration v / T)",
        consumers="stack/types.STOP_T_S read at call time by closed_chain.backup_phases_for, planners/scripted.backup_phases, rule1_contents (Ctx._stop_T, MotionRef.stop_T), rule4_contents._stop_t0; "
                  "the executor stop phase lasts the phase's T (finish rule: a stop in the last T s of a move lets it finish)",
        note="trade-off: a shorter T = a larger deceleration v / T (torque rows) but a shorter overshoot 0.5 v T; both directions searched by |change|. Level static = the Rule 1 rows of the oracle (stage d); the Rule 4 transfer rows: STOP_T_S_R4"),
    "STOP_T_S_R4": dict(
        aliases=[("attr", M_TYPES, "STOP_T_S")], grid=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5], lenient=0, rows=["rule4["], level="rule4", unit="s",
        justify="same as STOP_T_S", consumers="rule4_contents._stop_t0 (ladder base: STOP_T_S * 2^k, k <= STOP_K_MAX; call-time) via rule4_contents.run_trap",
        note="the trap scenario of rule4_contents (box 2.5-3.0, carry, t=0.75); the same patched attribute as STOP_T_S"),
}
NOT_SWEPT = {
    "rule1.K": "not swept: no consumer. rule1.K and rule1.NODE_CAP are only RECORDED (rule1.py:153-154 assumptions, rule1_contents.py:621/643 `worst`, rule4.py:81); "
               "no search reads them (the R_K witness is the depth-1 planner admission). Raising them cannot change a verdict.",
    "rule1.NODE_CAP": "not swept: no consumer (see rule1.K)",
    "rule4.MAX_ROUNDS / RETRACE_HALVINGS": "not requested here; rule4_contents module globals, add to KNOBS if a Rule 4 near-miss needs them",
}


def _lenient_key(name):
    k = KNOBS[name]
    ok = k.get("order_key", lambda v: v)
    return ok


def fmt_val(name, v):
    if isinstance(v, (tuple, list)):
        return f"{v[0]}..{v[1]}"
    u = KNOBS.get(name, {}).get("unit", "")
    if u == "m" and abs(float(v)) < 0.02:
        return f"{float(v) * 1e3:g} mm"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


# ----------------------------------------------------------------------------------------------------------------------- patching (worker side)
class _Alias:
    def __init__(self, desc):
        import importlib
        self.desc = desc
        self.kind, modname = desc[0], desc[1]
        self.mod = importlib.import_module(modname)
        self.name = desc[2]
        self.extra = desc[3] if len(desc) > 3 else None

    def label(self):
        return ":".join(str(x) for x in (self.kind, self.desc[1].split(".v2.")[-1], self.name) + ((self.extra,) if self.extra is not None and not isinstance(self.extra, tuple) else ()))

    def apply(self, v):
        import inspect
        import numpy as np
        if self.kind == "attr":
            setattr(self.mod, self.name, v)
        elif self.kind == "item":
            getattr(self.mod, self.name)[self.extra] = v
        elif self.kind == "dyrange":
            lo, hi = v
            setattr(self.mod, self.name, tuple(np.round(np.arange(lo, hi + 1e-4, 0.005), 4)))
        elif self.kind == "kwdefault":
            fn = getattr(self.mod, self.name)
            sig = inspect.signature(fn)
            names = [p.name for p in sig.parameters.values()]
            defs = list(fn.__defaults__)
            n_nodef = len(names) - len(defs)
            i = names.index(self.extra)
            assert i >= n_nodef, f"{self.name}.{self.extra} has no default"
            defs[i - n_nodef] = v
            fn.__defaults__ = tuple(defs)
        elif self.kind == "dmin":
            orig = getattr(self.mod, self.name)
            keys = self.extra
            rec = self.__dict__.setdefault("_used", [])

            def wrapped(*a, **kw):
                d = dict(orig(*a, **kw))
                for k in keys:
                    d[k] = v
                rec.append({k: d[k] for k in keys})
                return d
            wrapped.__wrapped_orig__ = orig
            setattr(self.mod, self.name, wrapped)
            self._v = v
        else:
            raise ValueError(self.kind)

    def read(self):
        import inspect
        if self.kind == "attr":
            return getattr(self.mod, self.name)
        if self.kind == "item":
            return getattr(self.mod, self.name)[self.extra]
        if self.kind == "dyrange":
            g = getattr(self.mod, self.name)
            return (float(g[0]), float(g[-1]))
        if self.kind == "kwdefault":
            return inspect.signature(getattr(self.mod, self.name)).parameters[self.extra].default
        if self.kind == "dmin":
            used = self.__dict__.get("_used")
            if used:                                              # the table the rows were actually built with
                vals = {x for u in used for x in u.values()}
                return vals.pop() if len(vals) == 1 else tuple(sorted(vals))
            fn = getattr(self.mod, self.name)
            return getattr(self, "_v", None) if hasattr(fn, "__wrapped_orig__") else fn()[self.extra[0]]
        raise ValueError(self.kind)


def _same(a, b):
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        try:
            return len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b))
        except Exception:
            return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        return a == b


def _install_recorders():
    """wrap motion.joint_pos_row to record the margin it is built with (read-back of PATH_MARGIN_MIN as the ROW used it)."""
    from world_model.v2.stack.checks import motion as M
    rec = dict(joint_pos_margin=set())
    orig = M.joint_pos_row

    def wrapped(q, t, q_lo, q_hi, margin=0.0, name="joint_pos"):
        rec["joint_pos_margin"].add(round(float(margin), 9))
        return orig(q, t, q_lo, q_hi, margin, name)
    M.joint_pos_row = wrapped                                  # motion_rows calls M.joint_pos_row at call time
    return rec


def _rows_json(rows):
    return [dict(name=r["name"], ok=r["ok"], margin=r.get("margin"), guarantee=r.get("guarantee", ""), why=r.get("why", "")) for r in rows]


def _static_rows(task, cell):
    from world_model.v2.adapters import z3_oracle_contents as Z
    from world_model.v2.stack.z3plan.types import Theta
    orc = Z.make_oracle(task, all_stages=True)
    vals = Z.default_vals(task)
    res = orc(vals, Theta(float(cell[0]), float(cell[1])))
    return dict(rows=_rows_json(res.rows), ok=bool(res.ok), failed_stage=getattr(res, "failed_stage", None), failed_stages=getattr(res, "failed_stages", None),
                tilt_missing=getattr(res, "tilt_missing", None), stage_wall_s=getattr(res, "stage_wall_s", None), vals=vals)


def _episode_rows(task, cell, n, a_plan):
    import numpy as np
    from world_model.v2.nero.contents import carry as CR, carry_cert as CC, tilt_measure as TM
    ul = TM.u_limit_deg(task)
    eps = []
    for i in range(n):
        seed = CC.cell_seed(task, cell, i)
        w = CC.sample_world(task, cell, np.random.default_rng(seed))
        c = CC.cell_spec(task, cell).contents
        t0 = time.time()
        r = CR.world_trial(w, f"sweep_{task}_{cell[0]}-{cell[1]}_{i}", a_plan=a_plan, m_hat=0.5 * (c.m_lo + c.m_hi), branch_backups=False)
        segs = r["segments"]
        tilt = max((s["nominal"]["obj_tilt_max"] for s in segs), default=None)
        clr = [s["nominal"].get("obj_world_clear_min") for s in segs]
        clr = [x for x in clr if x is not None and math.isfinite(x)]
        viol = next(((s["seg"], s["nominal"]["viol"]) for s in segs if s["nominal"]["viol"]), None)
        va = {}
        for s in segs:
            for k, v in (s["nominal"].get("viol_all") or {}).items():
                va[k] = va.get(k, 0) + (v.get("n", 1) if isinstance(v, dict) else 1)
        eps.append(dict(seed=seed, i=i, tilt_max=tilt, min_obj_world_clear=min(clr) if clr else None, stopped_at=[viol[0], viol[1].get("kind") if isinstance(viol[1], dict) else str(viol[1])] if viol else None,
                        viol_first=viol[1] if viol else None, viol_all=va, pre_hold_viol=r.get("pre_hold_viol"), wall_s=round(time.time() - t0, 1)))
    n_stop = sum(1 for e in eps if e["stopped_at"])
    kinds = sorted({e["stopped_at"][1] for e in eps if e["stopped_at"]})
    clr = [e["min_obj_world_clear"] for e in eps if e["min_obj_world_clear"] is not None]
    tl = [e["tilt_max"] for e in eps if e["tilt_max"] is not None]
    from world_model.v2.nero.contents import carry as CR2
    uc = float(CR2.U_OBJ_CLEAR_M)
    rows = [dict(name="episode_U", ok=(n_stop == 0), margin=(min(clr) - uc) if clr else None, guarantee="sampled",
                 why=("" if n_stop == 0 else f"{n_stop}/{n} episodes stopped; U group(s) fired: {kinds}; min obj-world clearance {min(clr) if clr else None}")),
            dict(name="episode_tilt", ok=(max(tl) < ul) if tl else None, margin=(ul - max(tl)) if tl else None, guarantee="sampled", why=f"max tilt {max(tl) if tl else None} deg vs U {ul}")]
    return dict(rows=rows, episodes=eps, n_stopped=n_stop, u_groups=kinds, min_obj_world_clear=min(clr) if clr else None, max_tilt=max(tl) if tl else None, n=n, a_plan=a_plan)


def _rule4_rows(task, cell):
    from world_model.v2.adapters import rule4_contents as R4
    try:
        acc, log, tr = R4.run_trap(True, task=task, cell=tuple(cell), seg="carry", t=0.75)
    except Exception as e:
        return dict(rows=[dict(name="rule4[trap]", ok=False, margin=None, guarantee="missing", why=f"trap not buildable: {e!r}"[:400])], accepted=None)
    rows = [dict(name="rule4[accepted]", ok=acc is not None, margin=None, guarantee="proof", why="" if acc is not None else "no target accepted")]
    for e in log:
        for v in e["verdicts"]:
            rows.append(dict(name=f"rule4[{v['target']}@round{e['round']}]", ok=bool(v["ok"]), margin=None, guarantee=str(v.get("guarantee", "")), why="; ".join(str(f) for f in v.get("failing", [])[:3])[:400]))
    return dict(rows=rows, accepted=None if acc is None else acc.name)


def _run_job(job):
    """in the fresh process: patch, build, run, read back."""
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    pc = job.get("plan_cache")
    if pc:
        os.environ["BIMANUAL_PLAN_CACHE"] = pc
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    sys.path[:0] = [p for p in job.get("syspath", []) if p not in sys.path]
    task, cell = job["task"], tuple(job["cell"])
    settings = job["settings"]                       # {knob: value}
    # today's values of EVERY knob (before any patch), for the summary
    today = {}
    for name, k in KNOBS.items():
        try:
            today[name] = _Alias(k["aliases"][0]).read() if k["aliases"][0][0] != "dmin" else _dmin_today(k["aliases"][0])
        except Exception as e:                       # noqa: BLE001
            today[name] = f"unreadable: {e!r}"
    aliases = []
    for name, v in settings.items():
        for d in KNOBS[name]["aliases"]:
            a = _Alias(d)
            a.apply(v)
            aliases.append((name, v, a))
    rec = _install_recorders()
    out = dict(today=today)
    levels = job["level"]
    rows = []
    if levels in ("static", "both"):
        s = _static_rows(task, cell)
        out["static"] = {k: v for k, v in s.items() if k != "rows"}
        rows += s["rows"]
    if levels in ("episode", "both"):
        e = _episode_rows(task, cell, job["seeds"], job["a_plan"])
        out["episode"] = {k: v for k, v in e.items() if k != "rows"}
        rows += e["rows"]
    if levels == "rule4":
        r = _rule4_rows(task, cell)
        out["rule4"] = {k: v for k, v in r.items() if k != "rows"}
        rows += r["rows"]
    # ---- read-back
    effective, bad = {}, []
    for name, v, a in aliases:
        got = a.read()
        effective.setdefault(name, []).append(dict(alias=a.label(), value=got))
        if not _same(got, v):
            bad.append(f"{name}: {a.label()} reads {got!r}, requested {v!r}")
    used = {}
    if "PATH_MARGIN_MIN" in settings or "PATH_MARGIN_BOTH" in settings:
        want = settings.get("PATH_MARGIN_MIN", settings.get("PATH_MARGIN_BOTH"))
        used["joint_pos_margin_used"] = sorted(rec["joint_pos_margin"])
        if levels in ("static", "both") and rec["joint_pos_margin"] and any(abs(x - float(want)) > 1e-9 for x in rec["joint_pos_margin"]):
            bad.append(f"joint_pos row was built with margin {sorted(rec['joint_pos_margin'])}, requested {want}")
        if levels in ("static", "both") and not rec["joint_pos_margin"] and out.get("static", {}).get("failed_stage") not in ("a_plan",):
            bad.append("joint_pos row never built (cannot confirm the margin used)")
    out.update(effective=effective, used=used)
    if bad:
        out.update(status="patch_failed", patch_errors=bad, rows=None)
    else:
        out.update(status="ok", rows=rows)
    return out


def _dmin_today(desc):
    import importlib
    m = importlib.import_module(desc[1])
    return getattr(m, desc[2])()[desc[3][0]]


def _worker(job):
    t0 = time.time()
    base = dict(knob=job["knob"], value=job["value"] if not isinstance(job["value"], dict) else job["value"], settings=job["settings"], cell=list(job["cell"]), task=job["task"], level=job["level"])
    try:
        out = _run_job(job)
    except Exception:                                # noqa: BLE001
        out = dict(status="error", error=traceback.format_exc()[-1800:], rows=None)
    rows = out.get("rows")
    verdict = None
    if rows is not None:
        bad = [r for r in rows if r["ok"] is False]
        verdict = dict(ok=(len(bad) == 0 and any(r["ok"] for r in rows)), n_rows=len(rows), failing=[dict(name=r["name"], margin=r.get("margin")) for r in bad])
    return dict(base, **out, verdict=verdict, wall_s=round(time.time() - t0, 1))


def _run_one_subprocess(job, out):
    """run one job in a fresh interpreter (same env); the result comes back through a pickle file.  A crash or timeout becomes status=error."""
    t0 = time.time()
    d = pathlib.Path(tempfile.mkdtemp(prefix="run_", dir=str(out)))
    jp, rp = d / "job.pkl", d / "res.pkl"
    with open(jp, "wb") as fh:
        pickle.dump(job, fh)
    try:
        p = subprocess.run([sys.executable, "-m", "world_model.v2.adapters.assumed_sweep", "--run-one", str(jp), str(rp)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=RUN_ONE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        p = subprocess.CompletedProcess(e.cmd, -9, stdout=f"timeout after {RUN_ONE_TIMEOUT_S} s")
    if p.returncode == 0 and rp.exists():
        with open(rp, "rb") as fh:
            r = pickle.load(fh)
        shutil.rmtree(d, ignore_errors=True)
        return r
    base = dict(knob=job["knob"], value=job["value"], settings=job["settings"], cell=list(job["cell"]), task=job["task"], level=job["level"])
    return dict(base, status="error", error=f"subprocess rc={p.returncode}: " + (p.stdout or "")[-1800:], rows=None, verdict=None,
                wall_s=round(time.time() - t0, 1), run_dir=str(d))


RUN_ONE_TIMEOUT_S = 4 * 3600   # s, a wall-time guard per run (tool setting, not a checker bound)


# ----------------------------------------------------------------------------------------------------------------------- jobs, summary
def _parse_val(name, s):
    if name == "DY_RANGE":
        lo, hi = s.split("..")
        return (float(lo), float(hi))
    x = float(s)
    return int(x) if name == "STOP_K_MAX" else x


def _jkey(j):
    return json.dumps([j["knob"], j["settings"], list(j["cell"]), j["level"]], sort_keys=True, default=list)


def _match(row, pats):
    nm = row["name"]
    for p in pats:
        if nm.startswith(p):
            return True
        if nm.startswith("rule1[") and p == "collision" and "collision" in (row.get("why") or ""):       # a rule1 step refused by a recovery collision sub-row
            return True
    return False


def _lin_estimate(knob, today, row, brows, runs, d):
    """ESTIMATE (not a run): where the first target row's margin crosses 0, linear through today's margin and the run values (numeric knobs only; extrapolated from the two most
    lenient points when no sign change was seen).  None when not computable."""
    if isinstance(today, (str, tuple, list)) or d == 0:
        return None
    pts = {float(today): next((r.get("margin") for r in brows if r["name"] == row), None)}
    for r in runs:
        if isinstance(r["value"], (int, float)):
            m = next((x.get("margin") for x in r["rows"] if x["name"] == row), None)
            pts[float(r["value"])] = m
    pts = sorted(((v, m) for v, m in pts.items() if m is not None), key=lambda x: x[0] * d)       # least lenient first
    for (v0, m0), (v1, m1) in zip(pts, pts[1:]):
        if m0 < 0 <= m1:
            return round(v0 + (v1 - v0) * (0 - m0) / (m1 - m0), 6) if m1 != m0 else None
    if len(pts) >= 2 and pts[-1][1] < 0 and pts[-1][1] != pts[-2][1]:
        (v0, m0), (v1, m1) = pts[-2], pts[-1]
        x = v0 + (v1 - v0) * (0 - m0) / (m1 - m0)
        return round(x, 6) if (x - v1) * d > 0 else None
    return None


def summarize(records, task, out_dir):
    by = {}
    for r in records:
        by.setdefault((r["knob"], tuple(r["cell"])), []).append(r)
    base = {tuple(r["cell"]): r for r in records if r["knob"] == "BASELINE"}
    summ = []
    for (knob, cell), rs in sorted(by.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        if knob in ("BASELINE", "COMBO"):
            continue
        b = base.get(cell)
        k = KNOBS[knob]
        ent = dict(knob=knob, cell=list(cell), tag=TAG, justify=k["justify"], unit=k["unit"], lenient=k["lenient"], rows=k["rows"], level=k["level"])
        if b is None or b.get("rows") is None:
            ent.update(status="no usable baseline run", break_even=None)
            summ.append(ent)
            continue
        today = b["today"].get(knob)
        ent["today"] = today
        brows = b["rows"]
        ent["baseline_verdict"] = b["verdict"]
        target = [r["name"] for r in brows if r["ok"] is False and _match(r, k["rows"])]
        ent["target_rows"] = target
        ent["target_margins"] = {r["name"]: r.get("margin") for r in brows if r["name"] in target}
        runs = sorted((r for r in rs if r["rows"] is not None), key=lambda r: _lenient_key(knob)(r["value"]))
        ent["values"] = []
        for r in runs:
            rm = {x["name"]: x for x in r["rows"]}
            tp = [rm[n]["ok"] if n in rm else None for n in target]
            ent["values"].append(dict(value=r["value"], effective=[e["value"] for e in next(iter(r["effective"].values()))][0] if r.get("effective") else None, ok=r["verdict"]["ok"],
                                      target_pass=(all(x is True for x in tp) if target else None), n_failing=len(r["verdict"]["failing"]),
                                      failing=r["verdict"]["failing"], wall_s=r["wall_s"]))
        bad_runs = [r for r in rs if r["rows"] is None]
        ent["unusable_runs"] = [dict(value=r["value"], status=r.get("status"), why=(r.get("patch_errors") or r.get("error") or "")[:300]) for r in bad_runs]
        if not target:
            ent.update(status="knob's rows pass today (no target row)" if b["verdict"]["ok"] else "knob's rows pass today; the baseline fails on other rows", break_even=None)
        else:
            lk = _lenient_key(knob)
            tk = lk(today) if not isinstance(today, str) else None
            d = k["lenient"]
            if d == 0:
                cands = sorted((v for v in ent["values"] if not _same(v["value"], today)), key=lambda v: abs(lk(v["value"]) - tk))
            else:
                cands = [v for v in ent["values"] if (lk(v["value"]) - tk) * d > 1e-12]
                cands.sort(key=lambda v: (lk(v["value"]) - tk) * d)
            be = next((v for v in cands if v["target_pass"]), None)
            if be is None:
                ent.update(status="target rows fail", break_even="none on grid", next_blocker=None)
            else:
                run = next(r for r in runs if r["value"] == be["value"])
                ent["linear_estimate"] = _lin_estimate(knob, today, target[0], brows, runs, d)
                ent.update(status="ok", break_even=be["value"], break_even_label=fmt_val(knob, be["value"]), next_blocker=[f for f in be["failing"]],
                           full_verdict_at_break_even=dict(ok=be["ok"], failing=be["failing"]), change_from_today=(None if isinstance(be["value"], (tuple, list)) else float(be["value"]) - float(today)))
        summ.append(ent)
    combos = [r for r in records if r["knob"] == "COMBO"]
    pathlib.Path(out_dir, "summary.json").write_text(json.dumps(dict(task=task, tag=TAG, entries=summ, combos=combos, not_swept=NOT_SWEPT), indent=1, default=str))
    md = [f"# ASSUMED-value sweep, task {task}", "", f"Every swept value stays **{TAG}**. A break-even is the value that would be needed, not an accepted one. Baseline = today's values.", ""]
    for cell in sorted({tuple(e["cell"]) for e in summ}):
        bl = base.get(cell)
        md.append(f"## cell {cell[0]}-{cell[1]}")
        if bl and bl.get("verdict"):
            md.append(f"baseline: ok={bl['verdict']['ok']}, failing rows: " + (", ".join(f"{f['name']} (margin {f['margin']})" for f in bl["verdict"]["failing"]) or "none"))
        md += ["", "| knob | today | target rows (margin) | break-even | change | next failing row(s) at break-even | justify (measurement that would make it MEASURED) |", "|---|---|---|---|---|---|---|"]
        for e in (x for x in summ if tuple(x["cell"]) == cell):
            if e.get("status") == "ok":
                nb = "; ".join(f"{f['name']} ({f['margin'] if f['margin'] is None else round(f['margin'], 4)})" for f in e["next_blocker"]) or "none: admits end to end"
                tg = ", ".join(f"{n} ({None if m is None else round(m, 4)})" for n, m in e["target_margins"].items())
                ch = "" if e["change_from_today"] is None else f"{e['change_from_today']:+g} {e['unit']}"
                md.append(f"| {e['knob']} | {fmt_val(e['knob'], e['today'])} | {tg} | {e['break_even_label']}{'' if 'mm' in e['break_even_label'] else ' ' + e['unit']}{'' if e.get('linear_estimate') is None else f' (linear estimate of the zero crossing {e["linear_estimate"]:g}, not a run)'} | {ch} | {nb} | {e['justify']} |")
            else:
                tg = ", ".join(e.get("target_rows", [])) or "-"
                md.append(f"| {e['knob']} | {fmt_val(e['knob'], e['today']) if 'today' in e else '?'} | {tg} | {e.get('break_even', e.get('status'))} | | {e.get('status', '')} | {e['justify']} |")
        md.append("")
    if combos:
        md += ["## combos", ""]
        for c in combos:
            v = c.get("verdict")
            md.append(f"- {c['settings']} cell {c['cell']}: " + (f"ok={v['ok']}, failing: " + (", ".join(f"{f['name']} ({f['margin']})" for f in v['failing']) or "none") if v else f"status {c.get('status')}: {c.get('patch_errors') or c.get('error')}") + f"  [{TAG}]")
        md.append("")
    md += ["## not swept", ""] + [f"- {k}: {v}" for k, v in NOT_SWEPT.items()]
    pathlib.Path(out_dir, "SUMMARY.md").write_text("\n".join(md) + "\n")
    return summ


# ----------------------------------------------------------------------------------------------------------------------- near-miss
def _collect_rows(obj, acc):
    if isinstance(obj, dict):
        if "name" in obj and "ok" in obj and ("margin" in obj or "why" in obj):
            acc.append(obj)
        for v in obj.values():
            _collect_rows(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_rows(v, acc)


def near_miss(files, tol, task="box", cells=None, out="OUT"):
    """RULE: a row is a near-miss when ok is False and it has a finite margin with |margin| <= tol (tol in the ROW'S OWN unit: rad / m / m/s^2 / kg-equivalent; default 0.05).
    Failing rows without a margin are listed but not ranked.  Row -> knobs: the registry `rows` patterns (substring of the row name, or of the failing sub-row text for rule1 rows)."""
    lines = []
    for f in files:
        txt = pathlib.Path(f).read_text()
        d = [json.loads(l) for l in txt.splitlines() if l.strip()] if str(f).endswith(".jsonl") else json.loads(txt)
        rows = []
        _collect_rows(d, rows)
        cell = (d.get("theta") or d.get("cell")) if isinstance(d, dict) else None
        cs = f"{cell[0]}-{cell[1]}" if cell else (cells or "CELL")
        bad = [r for r in rows if r.get("ok") is False]
        near = [r for r in bad if r.get("margin") is not None and math.isfinite(float(r["margin"])) and abs(float(r["margin"])) <= tol]
        nomargin = [r for r in bad if r not in near and (r.get("margin") is None)]
        far = [r for r in bad if r not in near and r.get("margin") is not None]
        lines.append(f"== {f}: {len(bad)} failing rows; near-miss (ok False, |margin| <= {tol}): {len(near)}; no margin: {len(nomargin)}; larger margin: {len(far)}")
        ks = {}
        for r in near + nomargin:
            hit = [n for n, k in KNOBS.items() if _match(dict(name=r["name"], why=r.get("why", "")), k["rows"])]
            lines.append(f"   {r['name']}: margin {r.get('margin')} -> knobs {hit or 'none in the registry'}")
            for n in hit:
                ks[n] = None
        if ks:
            lines.append(f"   would run: python -m world_model.v2.adapters.assumed_sweep --task {task} --cells {cs} --knobs {','.join(ks)} --workers 3 --out {out}")
        for n in ks:
            if KNOBS[n]["level"] != "static":
                lines.append(f"   note: {n} is level {KNOBS[n]['level']} (episode seeds / rule4 trap are run by the same command)")
    return "\n".join(lines)


def list_knobs():
    rows = ["| knob | aliases | grid | lenient | rows | level | consumers | not patched / note |", "|---|---|---|---|---|---|---|---|"]
    for n, k in KNOBS.items():
        al = "; ".join(":".join(str(x) for x in (d[0], d[1].split('.v2.')[-1], d[2]) + (() if len(d) < 4 else ((d[3],) if not isinstance(d[3], tuple) else d[3]))) for d in k["aliases"])
        g = ", ".join(fmt_val(n, v) for v in k["grid"])
        rows.append(f"| {n} | {al} | {g} | {k['lenient']:+d} | {','.join(k['rows'])} | {k['level']} | {k['consumers']} | {k['note']} |")
    return "\n".join(rows) + "\n\nNot swept:\n" + "\n".join(f"- {k}: {v}" for k, v in NOT_SWEPT.items())


# ----------------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="box")
    ap.add_argument("--cells", default="1.0-1.5", help="comma list lo-hi")
    ap.add_argument("--knobs", default="", help="comma list of knob names (see --list-knobs); default: none")
    ap.add_argument("--values", default="", help="override the grid of the (single) knob: comma list (DY_RANGE: lo..hi)")
    ap.add_argument("--combo", default="", help='one joint setting "A=v,B=v"')
    ap.add_argument("--level", default="", choices=("", "static", "episode", "both", "rule4"), help="override the knob's level")
    ap.add_argument("--seeds", type=int, default=3, help="episode level: seeds per run")
    ap.add_argument("--a", type=float, default=0.5, help="episode level: a_plan")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--near-miss", nargs="*", default=None, metavar="FILE", help="oracle / rule1 result JSONs: print the sweep commands (does not run them)")
    ap.add_argument("--near-tol", type=float, default=0.05)
    ap.add_argument("--list-knobs", action="store_true")
    ap.add_argument("--run-one", nargs=2, default=None, metavar=("JOB_PKL", "OUT_PKL"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.run_one:
        with open(a.run_one[0], "rb") as fh:
            job = pickle.load(fh)
        r = _worker(job)
        with open(a.run_one[1] + ".tmp", "wb") as fh:
            pickle.dump(r, fh)
        os.replace(a.run_one[1] + ".tmp", a.run_one[1])
        return
    if a.list_knobs:
        print(list_knobs())
        return
    if a.near_miss is not None:
        print(near_miss(a.near_miss, a.near_tol, a.task, a.cells, a.out or "OUT"))
        return
    if not a.out:
        raise SystemExit("--out DIR is required")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    here = pathlib.Path(__file__).resolve()
    root = here.parents[3]
    pc = out / "plan_cache"
    if not pc.exists():
        src = here.parents[1] / "stack" / "planners" / "plan_cache"
        if src.exists():
            shutil.copytree(src, pc)
        else:
            pc.mkdir()
    syspath = [str(root), str(root / "experiments" / "ab_smooth_vs_bangbang"), str(root / "experiments" / "realtime")]
    cells = [tuple(float(x) for x in c.split("-")) for c in a.cells.split(",") if c]
    names = [n for n in a.knobs.split(",") if n]
    for n in names:
        if n not in KNOBS:
            raise SystemExit(f"unknown knob {n!r}; known: {sorted(KNOBS)}; not swept: {sorted(NOT_SWEPT)}")
    common = dict(task=a.task, seeds=a.seeds, a_plan=a.a, plan_cache=str(pc), syspath=syspath)
    jobs = []
    for c in cells:
        jobs.append(dict(common, knob="BASELINE", value=None, settings={}, cell=c, level="static"))
    for n in names:
        k = KNOBS[n]
        vals = [_parse_val(n, x) for x in a.values.split(",") if x] if (a.values and len(names) == 1) else k["grid"]
        lv = a.level or k["level"]
        for c in cells:
            for v in vals:
                jobs.append(dict(common, knob=n, value=v, settings={n: v}, cell=c, level=lv))
    if a.combo:
        st = {}
        for kv in a.combo.split(","):
            n, x = kv.split("=")
            n = n.strip()
            if n not in KNOBS:
                raise SystemExit(f"unknown knob in --combo: {n!r}")
            st[n] = _parse_val(n, x)
        lvls = {KNOBS[n]["level"] for n in st}
        lv = a.level or ("both" if lvls & {"both", "episode"} else ("rule4" if lvls == {"rule4"} else "static"))
        for c in cells:
            jobs.append(dict(common, knob="COMBO", value=st, settings=st, cell=c, level=lv))
    # the baseline must run at a level that covers every swept knob's rows
    for j in jobs:
        if j["knob"] == "BASELINE":
            lvls = {KNOBS[n]["level"] for n in names} | ({KNOBS[n]["level"] for n in st} if a.combo else set())
            j["level"] = a.level or ("both" if lvls & {"both", "episode"} else "static")
    runs_p = out / "runs.jsonl"
    done = {}
    if runs_p.exists():
        for l in runs_p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done[_jkey(dict(knob=r["knob"], settings=r["settings"], cell=r["cell"], level=r["level"]))] = r
    todo = [j for j in jobs if _jkey(j) not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done; workers {a.workers}", flush=True)
    records = list(done.values())
    if todo:
        # owner fix 11-16: threads + one fresh `python -m ... --run-one` subprocess per run instead of a multiprocessing Pool.  A Pool keeps
        # its semaphores in /dev/shm; on ilab3 they vanished mid-run at 03:25 (another job of this user ended on the node) and the Pool
        # looped on respawn errors for 1.7 h without a result.  A subprocess per run needs no shared semaphore and is a fresh process (as before).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max(1, a.workers)) as tp, open(runs_p, "a") as f:
            futs = [tp.submit(_run_one_subprocess, j, out) for j in todo]
            for fu in as_completed(futs):
                r = fu.result()
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                records.append(r)
                v = r.get("verdict")
                print(f"{r['knob']}={fmt_val(r['knob'], r['value']) if r['knob'] not in ('COMBO', 'BASELINE') else r['settings']} cell {r['cell']} [{r['level']}] status={r.get('status')} "
                      f"ok={None if v is None else v['ok']} failing={[x['name'] for x in (v or {}).get('failing', [])][:4]} {r['wall_s']}s", flush=True)
    wanted = {_jkey(j) for j in jobs}
    records = [r for r in records if _jkey(dict(knob=r["knob"], settings=r["settings"], cell=r["cell"], level=r["level"])) in wanted or r["knob"] == "BASELINE"]
    summarize(records, a.task, out)
    print("wrote", out / "SUMMARY.md", out / "summary.json", out / "runs.jsonl", flush=True)


if __name__ == "__main__":
    main()
