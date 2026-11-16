"""adapters/z3_oracle_contents.py -- the physics ORACLE of the Z3 planner prototype (box, closed_chain).  SIMULATION ONLY, offline, no VLM.   (11-16)

The Z3 loop (stack/z3plan/) proposes a plan (PlanVals) and a contents interval Theta; this module answers pass / fail with the failing row and the constraints that
are TRUE about the answer (stack/z3plan/types.Learned).  No new physics: it only calls the existing checkers, cheapest first, and stops at the first failing stage:
  a  plan      closed_chain geometry_for (IK / reach / lift-vs-shelf collision: independent of a and theta) then plan_for(inp, a) (TOPP-RA timing, retreats)
  b  static    carry_cert.static_cell(..., wp=wp, planner="closed_chain") on [theta.lo, theta.hi]; refused, or a > a_admit, fails
  c  motion    adapters/motion_rows.motion_rows + adapters/collision_rows.collision_rows on the plan
  d  rule1     adapters/rule1_contents.run_cell(..., wp=wp, fallback=False), one row per step, judged on the recovery program the plan names (vals["recovery"])
  e  tilt      stack/checks/tilt.tilt_row on nero/results/tilt_table_2026-11-16/box_closed_chain/TABLE.json, ONLY for the default plan on a measured cell; else a
               MISSING row (reported, NOT counted in `ok`; `tilt_missing=True` on the result)
Knobs (domains()): zc (lift height), place_dx / place_dy (offset of the place target from the default goal), a (plan acceleration), recovery.

Learned constraints (only what is true; every one carries its reason):
  block_all_theta_a        geometry failure (IK / reach / lift collision, a SAMPLE distance below d_min, joint_pos / reach row): every knob except a fixed
  block_theta_superset     static refusal, a above a_admit, a stored_backup step refused, a tilt row refuted: the plan also fails on any Theta containing this one
                           (admission is "for all theta in the set").  NOT used for a failing read_then_retrace step: its bins depend on theta, so a superset's
                           bins are different sets and a finer bin could pass where the coarser failed.
  block_exact              everything else, and always added on a failure (the loop contract: at least this one)
  block_a_ge               ONLY when a stage-b torque_payload row fails (static refusal on a torque row, or a above a_admit when no open_content limit exists):
                           MEASURED non-decreasing in a on the default box path (nero/results/z3_stage0_2026-11-16), ASSUMED for other paths / between grid points.
                           Never for open_content, support, collision, or Rule 1 rows.

Known limits (guesses, see the report of the build):
  * stage d runs rule1_contents.run_cell as is: its path model is at the STATIC ADMISSION's accelerations (a_admit of the cell), not at vals["a"].  a therefore enters
    stages a, b (a <= a_admit), c and e only.
  * a read_then_retrace step is judged on the program's rows; that program is not executable today (rule1_contents marks executable=False).
  * the place rule checks (compile.py box rules) are re-implemented here for a box WaypointPlan (compile_plan needs a full proposer reply).

    python -m world_model.v2.adapters.z3_oracle_contents --task box --cells 1.0-1.5,4.5-5.0 --check-default --out DIR
    ... [--plan zc=0.18,place_dx=0.01,place_dy=0,a=0.25,recovery=stored_backup]   (a non-default plan on each cell, extra)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import shutil
import tempfile
import time

import numpy as np

from world_model.v2.stack.z3plan.types import Domain, Learned, OracleResult, PlanVals, ReadingModel, Theta, plan_key

TASKS = ("box",)
A_GRID = (0.05, 0.1, 0.25, 0.5)                # m/s^2, ASSUMED: A_MIN .. A_NOM of checks/admit; 0.1 / 0.25 / 0.5 are the measured tilt levels
RECOVERIES = ("stored_backup", "read_then_retrace")
ZC_STEP = 0.02                                 # m, ASSUMED grid of the lift height
PLACE_STEP = 0.005                             # m, ASSUMED grid of the place offset
PLACE_WINDOW = 0.01                            # m, ASSUMED: |place_dx|, |place_dy| <= this (keeps the grid <= 2000 plans); domains(window_m=...) widens it
TILT_TABLE = pathlib.Path(__file__).resolve().parents[1] / "nero" / "results" / "tilt_table_2026-11-16" / "box_closed_chain" / "TABLE.json"
EPS_A = 1e-9
A_GE_REASON = ("torque/payload utilisation non-decreasing in a: MEASURED on the default box path (z3_stage0_2026-11-16), "
               "ASSUMED for other paths and between the 0.025 grid points")


class PlanNotBuildable(ValueError):
    """the plan values violate a compile rule (place over the support, lowering, clearance, motion length, schema range)."""

    def __init__(self, reasons):
        self.reasons = [str(r) for r in reasons]
        super().__init__("; ".join(self.reasons))


# ----------------------------------------------------------------------------------------------------------------------- the plan space
def _scene(task, world=None):
    from world_model.v2.nero.contents import carry_cert as CC, scenes as SC
    w = world if world is not None else CC.world_default(task)
    return w, SC.support_tops(w)


def default_vals(task="box", world=None) -> dict:
    """the default plan as knob values: lift to z_c = max(goal top, start top) + CLEAR_M - start top (closed_chain.object_path), goal offset GOAL_DXY (offset 0), a = A_NOM, stored backup."""
    from world_model.v2.stack.planners import closed_chain as CCP
    _require(task)
    w, tops = _scene(task, world)
    t0 = tops["counter"]["top"]
    zc = max(tops["shelf"]["top"], t0) + CCP.CLEAR_M[task] - t0
    return dict(zc=round(float(zc), 4), place_dx=0.0, place_dy=0.0, a=A_GRID[-1], recovery="stored_backup")


def _require(task):
    if task not in TASKS:
        raise ValueError(f"z3_oracle_contents: built for {TASKS} only, not {task!r}")


def domains(task="box", world=None, window_m=PLACE_WINDOW) -> list:
    """the knob grids.  zc: schema lift dz range 0.02..0.20 step 0.02 (stack/proposer/schema.STEP_SKILLS['lift'], ASSUMED) + the default.  place_dx / place_dy: offset of the place
    target from the default goal (stack/tasks.GOAL_DXY), step 0.005, within +-window_m and so that the object centre stays over the shelf (compile rule), 0 included.
    a: A_GRID.  recovery: the two Rule 1 programs (D4)."""
    from world_model.v2.stack.proposer import schema as S
    _require(task)
    w, tops = _scene(task, world)
    d0 = default_vals(task, world)
    lo, hi = S.STEP_SKILLS["lift"]["dz"]
    zs = sorted({round(float(z), 4) for z in np.arange(lo, hi + 1e-9, ZC_STEP)} | {d0["zc"]})
    gx, gy = _goal_dxy(task)
    sh = tops["shelf"]
    n = int(round(window_m / PLACE_STEP))
    offs = [round(i * PLACE_STEP, 4) for i in range(-n, n + 1)]
    dxs = [o for o in offs if sh["x"][0] <= w.obj_xy[0] + gx + o <= sh["x"][1]]
    dys = [o for o in offs if sh["y"][0] <= w.obj_xy[1] + gy + o <= sh["y"][1]]
    return [Domain("zc", tuple(zs), "m", f"ASSUMED grid {ZC_STEP} m inside the proposer schema lift range [{lo}, {hi}] (schema.STEP_SKILLS); default {d0['zc']} included"),
            Domain("place_dx", tuple(dxs), "m", f"ASSUMED: offset of the place target from the default goal (x), step {PLACE_STEP}, window +-{window_m} (ASSUMED), centre kept over the shelf x {tuple(sh['x'])} (MODEL)"),
            Domain("place_dy", tuple(dys), "m", f"ASSUMED: offset of the place target from the default goal (y), step {PLACE_STEP}, window +-{window_m} (ASSUMED), centre kept over the shelf y {tuple(round(v, 4) for v in sh['y'])} (MODEL)"),
            Domain("a", A_GRID, "m/s^2", "ASSUMED: A_MIN (checks/admit) .. A_NOM; 0.1 / 0.25 / 0.5 = the measured tilt levels"),
            Domain("recovery", RECOVERIES, "", "Rule 1 programs of adapters/rule1_contents (D4); read_then_retrace is not executable today")]


def _goal_dxy(task):
    from world_model.v2.stack import tasks as TK
    return tuple(float(x) for x in TK.GOAL_DXY[TK.task(task).kind])


COMPILE_READS = ("zc", "place_dx", "place_dy")        # the knobs the compile rules read (not a, not recovery)


def compile_errors(task, vals, world=None, scene=None) -> list:
    """the box compile rules (stack/proposer/compile.py box branch, same constants), pure arithmetic on zc / place_dx / place_dy.  -> [(rule_id, text)], [] = buildable.
    scene = (world, tops) from _scene(); given, nothing is rebuilt."""
    from world_model.v2.stack.proposer import compile as CMP, schema as S
    _require(task)
    w, tops = scene if scene is not None else _scene(task, world)
    t0, sup = tops["counter"]["top"], tops["shelf"]
    zc, pdx, pdy = float(vals["zc"]), float(vals["place_dx"]), float(vals["place_dy"])
    gx, gy = _goal_dxy(task)
    dx, dy = gx + pdx, gy + pdy
    dzp = float(sup["top"]) - (t0 + zc)
    errs = []
    lo, hi = S.STEP_SKILLS["lift"]["dz"]
    if not (lo - 1e-9 <= zc <= hi + 1e-9):
        errs.append(("lift_range", f"lift dz {zc} outside the schema range [{lo}, {hi}]"))
    mx = S.STEP_SKILLS["move"]
    if not (mx["dx"][0] <= dx <= mx["dx"][1] and mx["dy"][0] <= dy <= mx["dy"][1]):
        errs.append(("move_range", f"move ({dx:.3f}, {dy:.3f}) outside the schema range"))
    cx, cy = w.obj_xy[0] + dx, w.obj_xy[1] + dy
    if not (sup["x"][0] <= cx <= sup["x"][1] and sup["y"][0] <= cy <= sup["y"][1]):
        errs.append(("place_over_support", f"place(shelf): the object centre ({cx:.3f}, {cy:.3f}) is not over the support (x {tuple(sup['x'])}, y {tuple(sup['y'])})"))
    if dzp > -CMP.MIN_MOVE_M:
        errs.append(("lowering", f"place(shelf): the object underside is at {t0 + zc:.3f} m, the shelf top at {sup['top']:.3f} m; nothing to lower (needs >= {CMP.MIN_MOVE_M} m)"))
    if zc < CMP.MIN_CLEAR_M - CMP.Z_EPS:
        errs.append(("clearance", f"move: horizontal motion with the object {zc:.3f} m above its start support; needs >= {CMP.MIN_CLEAR_M} m (no dragging)"))
    if zc < CMP.MIN_MOVE_M or math.hypot(dx, dy) < CMP.MIN_MOVE_M:
        errs.append(("min_move", "a motion is shorter than MIN_MOVE_M"))
    return errs


def to_waypoint(task, vals, world=None, scene=None):
    """vals -> closed_chain.WaypointPlan: s0_lift (0,0,zc) -> s1_move (goal dx + place_dx, goal dy + place_dy, 0) -> s2_place (lower onto the shelf), set_down backups (= the default backups).
    Compile rules: compile_errors() (lift dz and move dx, dy inside the schema ranges; every motion >= MIN_MOVE_M; horizontal motion >= MIN_CLEAR_M above the start support;
    place: object centre over the shelf, lowering >= MIN_MOVE_M).  -> PlanNotBuildable(reasons) otherwise."""
    from world_model.v2.stack.planners import closed_chain as CCP
    _require(task)
    errs = compile_errors(task, vals, world, scene)
    if errs:
        raise PlanNotBuildable([t for _, t in errs])
    w, tops = scene if scene is not None else _scene(task, world)
    t0, sup = tops["counter"]["top"], tops["shelf"]
    zc, pdx, pdy = float(vals["zc"]), float(vals["place_dx"]), float(vals["place_dy"])
    gx, gy = _goal_dxy(task)
    dx, dy = gx + pdx, gy + pdy
    dzp = float(sup["top"]) - (t0 + zc)
    return CCP.WaypointPlan(("s0_lift", "s1_move", "s2_place"), ((0.0, 0.0, round(zc, 9)), (round(dx, 9), round(dy, 9), 0.0), (0.0, 0.0, round(dzp, 9))),
                            ("set_down", "set_down", "set_down"), True)


def reading_model(task="box") -> ReadingModel:
    """y = empty box mass + contents + n, |n| <= EPS_FREE (MEASURED, probe_calib_claw_guard) + nb (ASSUMED: TAU_NOISE through |pinv(J^T)_z|, max over the default path's samples)."""
    from world_model.v2.adapters import rule1_contents as RC
    from world_model.v2.stack.checks import admit as AD
    _require(task)
    ctx = RC.get_ctx(task, "closed_chain")
    spec = RC.claw_read_spec(task, ctx.read_nb(ctx.model(AD.A_NOM)))
    return ReadingModel(offset_kg=float(ctx.em), noise_kg=float(spec.noise), tag="ASSUMED")          # noise = EPS_FREE (MEASURED) + nb (ASSUMED): ASSUMED overall


# ----------------------------------------------------------------------------------------------------------------------- the oracle
def _row(name, ok, margin=None, guarantee="", why=""):
    return dict(name=name, ok=bool(ok) if ok is not None else None, margin=None if margin is None or not math.isfinite(float(margin)) else float(margin), guarantee=str(guarantee), why=str(why)[:600])


def _from_checker_row(r):
    return _row(r.name, r.ok, r.margin, r.guarantee, r.why)


class Oracle:
    """Oracle(vals, theta) -> OracleResult.  Cached on (plan values, theta)."""

    def __init__(self, task="box", world=None, work_dir=None, all_stages=False):
        """all_stages=True (opt-in, 11-16, for adapters/assumed_sweep): do not stop at the first failing stage b..e; run every stage that can run (a failure of stage a
        still stops: nothing to check without a plan) so the result carries every row.  failed_stage = the FIRST failing stage, `failed_stages` = all; learned = that of the first."""
        _require(task)
        self.task, self.world, self.work_dir, self.all_stages = task, world, work_dir, bool(all_stages)
        self.cache, self.n_calls = {}, 0
        self._default = default_vals(task, world)
        self._scene_cache = None

    def cheap_reject(self, vals):
        """physics-free prefilter for stack/z3plan loop.solve(prefilter=...): None if no cheap rule applies, else (True, "compile[rule,..]: text", reads).  Only the box compile
        rules (compile_errors: arithmetic on zc / place_dx / place_dy, no a, no recovery, no theta), the same code stage a uses."""
        if self._scene_cache is None:
            self._scene_cache = _scene(self.task, self.world)
        errs = compile_errors(self.task, vals, self.world, self._scene_cache)
        if not errs:
            return None
        return True, "compile[" + ",".join(r for r, _ in errs) + "]: " + "; ".join(t for _, t in errs), COMPILE_READS

    # -------------------------------------------------------------- helpers
    def _inputs(self, wp):
        from world_model.v2.stack.planners import closed_chain as CCP
        if self.world is None:
            return CCP._nominal_inputs(self.task, wp)
        from world_model.v2.stack.proposer import compile as CMP
        return CCP.PlanInputs(*CMP._inputs(self.task, self.world), wp=wp)

    def _is_default(self, vals):
        d = self._default
        return abs(float(vals["zc"]) - d["zc"]) < 1e-9 and abs(float(vals["place_dx"])) < 1e-9 and abs(float(vals["place_dy"])) < 1e-9

    @staticmethod
    def _finish(res: OracleResult, rows, learned, stage_wall, failed_stage, t0, tilt_missing=False):
        res.rows, res.learned, res.wall_s = rows, learned, time.time() - t0
        for k, v in (("stage_wall_s", stage_wall), ("failed_stage", failed_stage), ("tilt_missing", tilt_missing)):
            try:
                setattr(res, k, v)                                   # extra attributes (the dataclass may not declare them)
            except Exception:
                pass
        return res

    def __call__(self, vals, theta: Theta) -> OracleResult:
        key = (plan_key(dict(vals)), float(theta.lo), float(theta.hi))
        if key in self.cache:
            r = self.cache[key]
            c = dataclasses.replace(r, cached=True, wall_s=0.0)
            for k in ("stage_wall_s", "failed_stage", "failed_stages", "tilt_missing"):
                if hasattr(r, k):
                    setattr(c, k, getattr(r, k))
            return c
        self.n_calls += 1
        r = self._run(dict(vals), theta)
        self.cache[key] = r
        return r

    # -------------------------------------------------------------- the stages
    def _run(self, vals, theta: Theta) -> OracleResult:
        from world_model.v2.stack.planners import closed_chain as CCP
        t0 = time.time()
        res = OracleResult(ok=False)
        rows, wall = [], {}
        geo_plan = dict(vals)                                                        # the knobs a failure does not depend on go in Learned.free (v2), not out of the plan
        a = float(vals["a"])

        fails = []                                                                   # all_stages: (stage, learned) of every failing stage, in order

        def fail(stage, learned, tilt_missing=False, can_continue=False):
            if not any(l.kind == "block_exact" for l in learned):
                learned = learned + [Learned("block_exact", dict(vals), theta, learned[0].reason if learned else f"stage {stage} failed")]
            if self.all_stages and can_continue:
                fails.append((stage, learned))
                return None                                                          # the caller goes on with the next stage
            return self._finish(res, rows, learned, wall, stage, t0, tilt_missing)

        def finish_fails(tilt_missing=False):
            r = self._finish(res, rows, fails[0][1], wall, fails[0][0], t0, tilt_missing)
            r.failed_stages = [f[0] for f in fails]
            return r

        # ---- a. plan
        ts = time.time()
        try:
            wp = to_waypoint(self.task, vals, self.world, self._scene_cache)
        except PlanNotBuildable as e:
            wall["a_plan"] = time.time() - ts
            rows.append(_row("compile", False, None, "refuted", "; ".join(e.reasons)))
            return fail("a_plan", [Learned("block_all_theta_a", geo_plan, None, "compile rule (arithmetic on zc / place_dx / place_dy only: compile_errors reads no a, no recovery, no theta): " + "; ".join(e.reasons), free=("recovery",))])
        inp = self._inputs(wp)
        try:
            CCP.geometry_for(inp)
        except CCP.PlanRefused as e:
            wall["a_plan"] = time.time() - ts
            why = f"{e} [reason={e.info.get('reason')}]"
            rows.append(_row("plan_geometry", False, None, "refuted", why))
            return fail("a_plan", [Learned("block_all_theta_a", geo_plan, None, "planner geometry refused (IK / reach / lift vs shelf; geometry_for(inp) takes the waypoints only: no a, no theta, no recovery): " + why[:300], free=("recovery",))])
        try:
            plan = CCP.plan_for(inp, a)
        except CCP.PlanRefused as e:
            wall["a_plan"] = time.time() - ts
            why = f"{e} [reason={e.info.get('reason')}]"
            rows.append(_row("plan_timing", False, None, "refuted", why))
            return fail("a_plan", [Learned("block_all_theta", dict(vals), None, "planner timing / retreat refused at this a (plan_for(inp, a): no theta, no recovery): " + why[:300], free=("recovery",))])
        wall["a_plan"] = time.time() - ts
        rows.append(_row("plan", True, None, "built", f"closed_chain plan at a = {a}"))

        # ---- b. static admission (for all theta in [lo, hi])
        ts = time.time()
        from world_model.v2.nero.contents import carry_cert as CC, probe as PR
        from world_model.v2.stack.checks import payload as PAY
        st = CC.static_cell(self.task, (float(theta.lo), float(theta.hi)), self.world if self.world is not None else CC.world_default(self.task),
                            spec=PR.interval_spec(self.task, float(theta.lo), float(theta.hi)), wp=wp, planner="closed_chain")
        wall["b_static"] = time.time() - ts
        st_rows, bad = [], []
        for seg, rr in st["rows"].items():
            for nm, r in rr.items():
                if nm == "torque_payload":
                    m = PAY.U_ADMIT - float(r["u_max"])
                elif nm == "open_content":
                    m = float(r["a_open"]) - float(r["a_plan"])
                elif nm == "support_dest":
                    m = r.get("margin_m")
                else:
                    m = None
                x = _row(f"static[{seg}:{nm}]", r["ok"], m, r.get("guarantee", ""), "" if r["ok"] else f"{nm} fails on [{theta.lo}, {theta.hi}] kg at a_admit {st['a_admit']}")
                st_rows.append(x)
                if not x["ok"]:
                    bad.append(x)
        refused = str(st["action"]).startswith("refuse")
        a_ad = st["a_admit"]
        st_rows.insert(0, _row("static_action", not refused, None if a_ad is None else a_ad - a, "proof under MODEL (stack/checks/admit)",
                               f"action {st['action']!r}, a_admit {a_ad}, plan a {a}" + ("" if (not refused and a <= a_ad + EPS_A) else "; refused or a above a_admit")))
        rows.extend(st_rows)
        if refused:
            names = ", ".join(b["name"] for b in bad) or st["action"]
            ln = [Learned("block_theta_superset", dict(vals), theta, f"static admission refuses on [{theta.lo}, {theta.hi}] ({st['action']}): {names}; admission is for all theta, so any superset fails too. Free a: refuse = a_admit < A_MIN or support/unreachable (admit.py), computed from the path + theta, not from the plan's a. Free recovery: static_cell takes no recovery", free=("a", "recovery"))]
            if any(":torque_payload]" in b["name"] for b in bad):
                ln.append(Learned("block_a_ge", dict(vals), theta, A_GE_REASON, free=("recovery",)))
            r_ = fail("b_static", ln, can_continue=True)
            if r_ is not None:
                return r_
        elif a > a_ad + EPS_A:
            rows.append(_row("a_le_a_admit", False, a_ad - a, "proof under MODEL (bisection of a_admit assumes u monotone in a)", f"a {a} > a_admit {a_ad:.4f} on [{theta.lo}, {theta.hi}]"))
            ln = [Learned("block_theta_superset", dict(vals), theta, f"a {a} above the static a_admit {a_ad:.4f} on [{theta.lo}, {theta.hi}] (a_admit of a superset is no larger); free recovery: static_cell takes no recovery", free=("recovery",))]
            if not np.isfinite(float(st.get("a_open", np.inf))):          # a_admit is then set by the torque row alone (box: no open_content limit)
                ln.append(Learned("block_a_ge", dict(vals), theta, A_GE_REASON, free=("recovery",)))
            r_ = fail("b_static", ln, can_continue=True)
            if r_ is not None:
                return r_

        # ---- c. motion + collision rows
        ts = time.time()
        from world_model.v2.adapters import collision_rows as RCOL, motion_rows as MR
        mrows = MR.motion_rows(plan)
        crow = RCOL.collision_rows(plan)
        wall["c_motion"] = time.time() - ts
        learned = []
        for r in mrows + [crow]:
            rows.append(_from_checker_row(r))
            if r.ok:
                continue
            if r.name in ("joint_pos", "reach"):
                learned.append(Learned("block_all_theta_a", geo_plan, None, f"{r.name} row fails (geometry: the joint path / FK reach of the waypoints, no a, no recovery): {r.why[:300]}", free=("recovery",)))
            elif r.name == "collision":
                by = r.worst.get("by_class_all", {})
                samp = [(s, c, w) for s, bc in by.items() for c, w in bc.items() if w["distance"] < w["d_min"]]
                if samp:
                    s, c, w = samp[0]
                    learned.append(Learned("block_all_theta_a", geo_plan, None, f"collision sample distance {w['distance']:.4f} m < d_min {w['d_min']:.4f} m ({c} {w['pair']}, piece {s}): geometry only (collision_rows(plan) reads the plan, no recovery)", free=("recovery",)))
                else:
                    learned.append(Learned("block_all_theta", dict(vals), None, f"collision swept bound below d_min (depends on speed, so on a; collision_rows(plan) takes no theta, no recovery): {r.why[:300]}", free=("recovery",)))
            else:
                learned.append(Learned("block_all_theta", dict(vals), None, f"{r.name} row fails at this a (motion_rows(plan) takes the plan only: no theta, no recovery): {r.why[:300]}", free=("recovery",)))
        if learned:
            r_ = fail("c_motion", learned, can_continue=True)
            if r_ is not None:
                return r_

        # ---- d. Rule 1 per step, on the plan's recovery program
        ts = time.time()
        from world_model.v2.adapters import rule1_contents as RC
        if self.work_dir is not None:
            pathlib.Path(self.work_dir).mkdir(parents=True, exist_ok=True)     # owner fix 11-16: mkdtemp does not create its parent
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="z3orc_", dir=self.work_dir))
        try:
            recs = RC.run_cell(self.task, (float(theta.lo), float(theta.hi)), tmp, n_sampled=0, planner="closed_chain", fallback=False, wp=wp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        wall["d_rule1"] = time.time() - ts
        prog = vals["recovery"]
        failing = []
        for rec in recs:
            nm = f"rule1[{rec['step']}:{rec['seg']}:{prog}]"
            pp = rec.get("per_program", {})
            if "refusal" in rec and "per_program" not in rec:
                rows.append(_row(nm, False, None, "refuted", rec["refusal"]))
                failing.append(nm)
            elif prog not in pp:
                rows.append(_row(nm, False, None, "missing", f"program {prog!r} is not offered for this step (offered: {sorted(pp)})"))
                failing.append(nm)
            else:
                p = pp[prog]
                if p["ok"]:
                    m = None
                    if rec.get("admitted") and rec.get("program") == prog:
                        ms = [x.get("margin") for x in rec["admission"]["rows"] if isinstance(x.get("margin"), (int, float)) and math.isfinite(x["margin"])]
                        m = min(ms) if ms else None
                    rows.append(_row(nm, True, m, rec.get("guarantee", "")))
                else:
                    why = "; ".join(f"{f['name']}: {(f.get('why') or f['guarantee'])[:140]}" for f in p["failing"][:3])
                    rows.append(_row(nm, False, None, "refuted", f"{p['n_rows']} rows, failing: {why}"))
                    failing.append(nm)
        if failing:
            msg = f"Rule 1 refuses {', '.join(failing)} on [{theta.lo}, {theta.hi}] kg"
            if prog == "stored_backup":
                learned = [Learned("block_theta_superset", dict(vals), theta, msg + " (a fixed one-bin program: any superset fails too). Free a: run_cell takes no plan a, its path model is at the static admission's accelerations", free=("a",))]
            else:
                learned = [Learned("block_exact", dict(vals), theta, msg + " (reading bins depend on theta: no superset claim). Free a: run_cell takes no plan a", free=("a",))]
            r_ = fail("d_rule1", learned, can_continue=True)
            if r_ is not None:
                return r_

        # ---- e. tilt (default plan on a measured cell only)
        ts = time.time()
        from world_model.v2.nero.contents import carry as CR
        from world_model.v2.stack.checks import tilt as TL
        from world_model.v2.stack.checks.row import missing
        key = f"{self.task}|{float(theta.lo)}-{float(theta.hi)}"
        if not self._is_default(vals):
            tr = missing("obj_tilt", f"the tilt table ({TILT_TABLE.parent.name}) was measured on the DEFAULT plan only; this waypoint plan differs (zc {vals['zc']}, place offset ({vals['place_dx']}, {vals['place_dy']}))")
        else:
            tr = TL.tilt_row(json.loads(TILT_TABLE.read_text()), key, a, CR.U_TILT_OBJ)
        wall["e_tilt"] = time.time() - ts
        rows.append(_from_checker_row(tr))
        tmissing = tr.guarantee == "missing"
        if not tr.ok and not tmissing:
            r_ = fail("e_tilt", [Learned("block_theta_superset", dict(vals), theta, f"tilt row refuted on the measured cell {key}: {tr.why[:300]}; free recovery: tilt_row reads the table, a and the cell only", free=("recovery",))], can_continue=True)
            if r_ is not None:
                return r_
        if fails:
            return finish_fails(tmissing)
        res.ok = True
        return self._finish(res, rows, [], wall, None, t0, tmissing)


def make_oracle(task="box", world=None, work_dir=None, all_stages=False) -> Oracle:
    return Oracle(task, world, work_dir, all_stages)


# ----------------------------------------------------------------------------------------------------------------------- CLI
def result_json(vals, theta, res: OracleResult) -> dict:
    return dict(vals=vals, theta=[theta.lo, theta.hi], ok=res.ok, failed_stage=getattr(res, "failed_stage", None), tilt_missing=getattr(res, "tilt_missing", None), wall_s=res.wall_s,
                stage_wall_s=getattr(res, "stage_wall_s", None), cached=res.cached, rows=res.rows,
                learned=[dict(kind=l.kind, plan=l.plan, theta=None if l.theta is None else [l.theta.lo, l.theta.hi], reason=l.reason) for l in res.learned])


def _summ(vals, theta, res):
    bad = next((r for r in res.rows if r["ok"] is False), None)
    sw = " ".join(f"{k[0]}={v:.1f}s" for k, v in (getattr(res, "stage_wall_s", None) or {}).items())
    return (f"[{theta.lo}-{theta.hi}] zc={vals['zc']} dx={vals['place_dx']} dy={vals['place_dy']} a={vals['a']} rec={vals['recovery']}: ok={res.ok} "
            f"fail_row={None if bad is None else bad['name']} learned={[l.kind for l in res.learned]} tilt_missing={getattr(res, 'tilt_missing', None)} wall {res.wall_s:.1f}s ({sw})")


def _parse_plan(s, task):
    v = default_vals(task)
    for kv in s.split(","):
        k, x = kv.split("=")
        v[k] = x if k == "recovery" else float(x)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="box")
    ap.add_argument("--cells", required=True, help="comma list lo-hi in contents kg (carry_cert cell units)")
    ap.add_argument("--check-default", action="store_true", help="run the oracle on the DEFAULT plan values on each cell")
    ap.add_argument("--plan", action="append", default=[], help="an extra plan per cell: zc=..,place_dx=..,place_dy=..,a=..,recovery=.. (unset = default)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ.setdefault("BIMANUAL_PLANNER", "closed_chain")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    orc = make_oracle(a.task)
    plans = ([("default", default_vals(a.task))] if a.check_default else []) + [(f"plan{i}", _parse_plan(p, a.task)) for i, p in enumerate(a.plan)]
    for c in a.cells.split(","):
        lo, hi = (float(x) for x in c.split("-"))
        th = Theta(lo, hi)
        for tag, vals in plans:
            res = orc(vals, th)
            print(_summ(vals, th, res), flush=True)
            (out / f"oracle_{a.task}_{lo}-{hi}_{tag}.json").write_text(json.dumps(result_json(vals, th, res), indent=1, default=float))


if __name__ == "__main__":
    main()
