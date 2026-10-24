"""tamp/executor.py -- EXECUTION layer of the TAMP package: runs a PDDLStream plan in the MuJoCo house sim.  SIMULATION ONLY.

A plan is a list of (action_name, args) (see tamp/types.py):
    move_free(q1, q2, r) | move_holding(o, g, q1, q2, r) | pick(o, p, g, q) | place(o, p, g, q, s)

`Executor(sim, filt)` runs the plan ONE action at a time through the existing teleop loop
    obs = sim.observe(); cmd = op.step(obs); safe, info = filt.filter(sim, cmd); sim.tick(safe)
so the 50 Hz safety filter (teleop Shield or Gatekeeper; Passthrough = filter off) stays in the loop on the base twist (and on lift / jog).
`PlanOperator` is a NOISE-FREE command generator for one action; it reuses the helper stick laws of teleop.operator.SloppyOperator
(`_drive`, `_creep_stick`, `_heading_to`, `_deadzone`) and the ALIGN / LIFT / LOWER logic of its coffee program, but reads ONLY the operator-visible
observation (sim.observe()).  The tray / slab TRUTH is read in exactly one place: the physical post-condition checks (`_check_*`), never for control.

Failure semantics: run_action returns an ActionResult (status "ok" | "failed" | "timeout" | "stalled"); the sim is left alive in whatever state it
reached, so the caller can re-plan from the current state and call run_action / run_plan again (the same Executor continues the same episode).
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

import numpy as np

from ..teleop import base as BS
from ..teleop import operator as OP
from ..teleop.sim import Cmd, TICK

# ---------------------------------------------------------------------------------------------------- noise-free operator parameters
# An AUTONOMOUS executor, not a sloppy human: no stick noise / bumps / panics / aim bias / observation delay; fixed waypoint switch radius.
# Everything else (v_cruise 0.95, v_back 0.55, kw 2.2, a_human 0.9, stop_radius 0.12, lift_up_h 0.06, jog gains ...) is the SloppyOperator default.
def exec_params(**over) -> OP.OperatorParams:
    kw = dict(ou_sigma=0.0, bump_rate=0.0, jog_bump_rate=0.0, panic_rate=0.0, lat_bias_sigma=0.0, reaction_s=0.0, sw_radius=(0.7, 0.7))
    kw.update(over)
    return OP.OperatorParams(**kw)


# tolerances of the PHYSICAL post-conditions (sim truth)
SURF_POS_TOL, SURF_YAW_TOL = 0.03, 0.04          # docked conf: CREEP converges to 2 cm / 0.02 rad (stopped 0.5 s); 1 cm / 0.02 rad slack
FREE_POS_TOL, FREE_YAW_TOL = 0.20, 0.15          # free conf: _drive arrives at stop_radius + 0.05 = 0.17 m; turn ends at stop_yaw 0.12 rad
DOCK_PRE_POS, DOCK_PRE_YAW = 0.08, 0.10          # pick / place PRECONDITION: base is at the dock conf
LIFT_TOL = 0.005                                 # lift_h >= lift_up_h - 5 mm after pick
PLACE_Z_TOL = 0.015                              # tray z within 1.5 cm of slab top + tray thickness
TRAY_REST_SPEED = 0.05                           # m/s, "resting"
SEG_REST_V = 0.03                                # m/s: a refused segment route ends the action once the backup has braked the base to this
STALL_PROGRESS = 0.05                            # m: < 5 cm progress for stall_s => stalled
DEFAULT_TIMEOUT = dict(move_free=120.0, move_holding=120.0, pick=30.0, place=30.0)


def _wrap(a):
    return BS.wrap(a)


def _pose_err(pose, q):
    return float(math.hypot(pose[0] - q.x, pose[1] - q.y)), float(abs(_wrap(q.yaw - pose[2])))


# ---------------------------------------------------------------------------------------------------- the operator
class PlanOperator:
    """Executes ONE plan action.  `.step(obs) -> Cmd`, `.phase`, `.done`.  Observation-only (no truth)."""

    def __init__(self, name, args, params: OP.OperatorParams | None = None, seed=0):
        assert name in ("move_free", "move_holding", "pick", "place"), name
        self.name, self.args = name, args
        self.p = params or exec_params()
        self.h = OP.SloppyOperator(seed, "coffee", out_route=[(0.0, 0.0), (1.0, 0.0)], params=self.p)      # helper: _drive / _creep_stick / _heading_to / _deadzone
        self.t_phase = 0.0
        self.stopped_t = 0.0
        self.phase = "INIT"
        self.q2 = None
        if name == "move_free":
            q1, q2, r = args
            self._setup_move(q2, r)
        elif name == "move_holding":
            o, g, q1, q2, r = args
            self._setup_move(q2, r)
        elif name == "pick":
            self._goto("ALIGN")
        else:
            self._goto("LOWER")

    # ------------------------------------------------------------------ setup / transitions
    def _setup_move(self, q2, r):
        self.q2 = q2
        self.route = [np.array(p, float) for p in r.pts]
        self.reverse = bool(r.reverse)
        self.h.wp_i = 1
        self._goto("DRIVE")

    def _goto(self, ph):
        self.phase = ph
        self.t_phase = 0.0
        self.stopped_t = 0.0

    @property
    def done(self):
        return self.phase == "DONE"

    def _at_rest(self, obs, ok):
        """accumulate the 'stopped' timer (ok and not moving)."""
        if ok and np.linalg.norm(obs["v"]) <= 0.03:
            self.stopped_t += TICK
        else:
            self.stopped_t = 0.0

    # ------------------------------------------------------------------ one tick
    def step(self, obs) -> Cmd:
        p, h = self.p, self.h
        cmd = Cmd()
        stick = np.zeros(3)
        lift = 0.0
        jog = np.zeros(6)
        self.t_phase += TICK
        if self.phase == "DRIVE":
            stick, arrived = h._drive(self.route, self.reverse, obs)
            q2 = self.q2
            if q2.surface is not None and h.wp_i >= len(self.route) - 1 and \
                    np.linalg.norm(np.array([q2.x, q2.y]) - obs["pose"][:2]) < 0.45:
                stick, e_w0, _ = h._creep_stick(q2.xyth(), obs, 0.5)               # last 0.45 m: closed-loop creep (backs up if overshot)
                arrived = bool(np.linalg.norm(e_w0) < 0.12)
            if arrived:
                self._goto("TURN")
        elif self.phase == "TURN":
            sw, ok = h._heading_to(self.q2.yaw, obs)
            stick = np.array([0.0, 0.0, sw])
            self._at_rest(obs, ok)
            if self.stopped_t > 0.6:
                self._goto("CREEP" if self.q2.surface is not None else "DONE")
        elif self.phase == "CREEP":
            # same tolerances as the coffee program's CREEP: 2 cm / 0.02 rad, stopped for 0.5 s
            stick, e_w, e_yaw = h._creep_stick(self.q2.xyth(), obs)
            self._at_rest(obs, bool(np.linalg.norm(e_w) < 0.02 and abs(e_yaw) < 0.02))
            if self.stopped_t > 0.5:
                self._goto("DONE")
        elif self.phase == "ALIGN":
            ep, er = obs["grasp_err_p"], obs["grasp_err_r"]
            jog[:3] = np.clip(p.jog_gain[0] * np.asarray(ep), -p.jog_max[0], p.jog_max[0])
            jog[3:] = np.clip(p.jog_gain[1] * np.asarray(er), -p.jog_max[1], p.jog_max[1])
            if obs["can_grasp"] and self.t_phase > 0.5:
                cmd.grasp = True
                self._goto("LIFT")
        elif self.phase == "LIFT":
            lift = p.lift_stick * (1.0 if obs["lift_h"] < p.lift_up_h else 0.0)
            if obs["lift_h"] >= p.lift_up_h - 0.004 and self.t_phase > 0.4:
                self._goto("DONE")
        elif self.phase == "LOWER":
            lift = -p.lift_stick * (1.0 if obs["lift_h"] > 0.004 else 0.0)
            if obs["lift_h"] <= 0.006 and self.t_phase > 0.4:
                cmd.release = True
                self._goto("SETTLE")
        elif self.phase == "SETTLE":                      # let the tray come to rest on the surface (run.py tail_s = 1.0 s)
            if self.t_phase >= 1.0:
                self._goto("DONE")
        stick = h._deadzone(np.clip(stick, -1.0, 1.0))
        cmd.twist = stick * BS.V_STICK
        cmd.lift_v = lift * 0.12
        cmd.jog = jog
        return cmd


class SegPlanOperator(PlanOperator):
    """PlanOperator whose DRIVE phase is executed by a SegmentGatekeeper (tamp/seg_gate.py) following the route's certified timed segments: while the filter reports
    "active" the operator commands ZERO (the gatekeeper commits and tracks the segment twists); "done" -> the normal DRIVE law finishes the last stretch (wp_i = last
    waypoint, closed-loop creep at a surface dock) and TURN / CREEP / ... run unchanged; "refused" -> phase REFUSED (the executor returns "segment_refused").
    Non-DRIVE actions (pick / place) are never built as SegPlanOperator."""

    def __init__(self, name, args, filt, params=None, seed=0):
        super().__init__(name, args, params, seed)
        self.filt = filt
        self.seg_active = True
        filt.begin_route(args[-1])

    def step(self, obs) -> Cmd:
        if self.phase in ("REFUSED", "FALSIFIED"):
            return Cmd()
        if self.phase == "DRIVE" and self.seg_active:
            st = self.filt.seg_state()
            if st == "active":
                self.t_phase += TICK
                return Cmd()
            if st == "refused":
                self.phase = "REFUSED"
                return Cmd()
            if st == "plan_falsified":                    # opt-in plan-prediction monitor: stop committing the old route; the executor returns at once
                self.phase = "FALSIFIED"
                return Cmd()
            self.seg_active = False                       # "done": hand the final stretch to the standard DRIVE law
            self.h.wp_i = len(self.route) - 1
        return super().step(obs)


# ---------------------------------------------------------------------------------------------------- results
@dataclass
class ActionResult:
    name: str
    status: str                       # "ok" | "failed" | "timeout" | "stalled"
    t_start: float
    t_end: float
    reason: str = ""
    checks: dict = field(default_factory=dict)         # PHYSICAL post-condition checks from sim truth: {check_name: {ok, ...numbers}}
    metrics: dict = field(default_factory=dict)        # per-action episode metrics (run.summarize-style)
    filter_counters: dict = field(default_factory=dict)  # filt.counters() at the end of the action (cumulative; {} if the filter has none)
    info: dict = field(default_factory=dict)           # end pose, stall details, final phase ...

    @property
    def ok(self):
        return self.status == "ok"

    @property
    def duration(self):
        return self.t_end - self.t_start

    def to_json(self):
        return _jsonable(asdict(self))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ---------------------------------------------------------------------------------------------------- the executor
class Executor:
    def __init__(self, sim, filt, params: OP.OperatorParams | None = None, docks=None, seed=0):
        """sim: TeleopSim built as the coffee world (counter=True, desk=True).  filt: Shield | Gatekeeper | Passthrough (anything with .filter(sim, cmd)).
        docks: stand-off xy list for filt.docks (default = both h5 hand-off poses, as run_episode does for a non-carry task)."""
        self.sim, self.filt = sim, filt
        self.params = params or exec_params()
        self.seed = seed
        wp = sim.scene.waypoints
        filt.docks = list(docks) if docks is not None else [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]]
        self.rows = []                # one dict per tick (scalars only) over the whole episode
        self.results: list[ActionResult] = []

    # ------------------------------------------------------------------ main loop for one action
    def run_action(self, name, args, timeout_s=None, stall_s=10.0) -> ActionResult:
        sim, filt = self.sim, self.filt
        timeout_s = DEFAULT_TIMEOUT.get(name, 60.0) if timeout_s is None else timeout_s
        is_move = name in ("move_free", "move_holding")
        t0 = sim.t
        row0 = len(self.rows)
        obs = sim.observe()
        res = ActionResult(name, "failed", t0, t0)
        pre = self._precondition(name, args, obs)
        if pre:
            res.reason, res.checks = "precondition: " + pre, {"precondition": dict(ok=False, why=pre)}
            return self._finish(res, row0, "PRE")
        route = args[-1] if (is_move and hasattr(filt, "begin_route")) else None
        seg_mode = route is not None and bool(getattr(route, "segments", None))          # new path: certified timed segments; the legacy stick path stays the default
        op = SegPlanOperator(name, args, filt, self.params, seed=self.seed) if seg_mode else PlanOperator(name, args, self.params, seed=self.seed)
        # stall detector (moves only): reference pose; progress = 5 cm of travel or 0.2 rad of rotation since the reference
        ref_pos, ref_yaw, ref_t = obs["pose"][:2].copy(), float(obs["pose"][2]), sim.t
        lab_hist = deque(maxlen=int(max(stall_s, 1.0) / TICK))
        q2 = args[3] if name == "move_holding" else (args[1] if name == "move_free" else None)
        status, reason = None, ""
        while True:
            obs = sim.observe()
            cmd = op.step(obs)
            safe, info = filt.filter(sim, cmd)
            o = sim.tick(safe)
            self._log(o, info, cmd, safe, op)
            lab_hist.append(info.get("label", ""))
            if op.done:
                break
            if seg_mode and op.phase == "REFUSED" and float(np.linalg.norm(o["v"])) <= SEG_REST_V:
                status, reason = "segment_refused", filt.sr["reason"] if filt.sr else "segment refused"
                break
            if seg_mode and filt.seg_state() == "plan_falsified":
                # the plan-prediction monitor flagged this tick: return NOW (the committed certified trajectory keeps being followed by the filter in coast mode)
                status, reason = "plan_falsified", filt.sr["reason"] if filt.sr else "plan falsified"
                break
            if sim.t - t0 > timeout_s:
                status, reason = "timeout", f"no completion within {timeout_s:.0f} s (phase {op.phase})"
                break
            if is_move:
                pose = o["pose"]
                if np.linalg.norm(pose[:2] - ref_pos) >= STALL_PROGRESS or abs(_wrap(pose[2] - ref_yaw)) >= 0.2 or \
                        (q2 is not None and op.phase in ("CREEP",) and np.hypot(pose[0] - q2.x, pose[1] - q2.y) < 0.15):
                    ref_pos, ref_yaw, ref_t = pose[:2].copy(), float(pose[2]), sim.t       # progress (the final creep converges within a few cm: not a stall; the timeout covers it)
                elif sim.t - ref_t >= stall_s:
                    top = Counter(lab_hist).most_common(3)
                    status, reason = "stalled", f"< {STALL_PROGRESS*100:.0f} cm progress for {stall_s:.0f} s at ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg) in phase {op.phase}; filter labels {top}"
                    res.info["stall"] = dict(t=sim.t, pose=[float(x) for x in pose], phase=op.phase, filter_labels=top,
                                             min_person_distance_m=float(o["pdist"]))
                    break
        res.t_end = sim.t
        if seg_mode and not (status == "plan_falsified" and getattr(filt, "plan_continue", False)):       # a "continue" decision keeps the route alive: the ladder ends it at t_ready
            rec = filt.end_route("done" if status is None else status)
            res.info["segments"] = rec
        if status is None:
            ok, checks = self._postcondition(name, args)
            res.checks = checks
            status = "ok" if ok else "failed"
            reason = "" if ok else "post-condition failed: " + ", ".join(k for k, v in checks.items() if not v.get("ok", True))
        else:
            res.checks = self._postcondition(name, args)[1]          # still record the physical state at the failure
        res.status, res.reason = status, reason
        return self._finish(res, row0, op.phase)

    def run_plan(self, plan, timeout_s=None, stall_s=10.0, max_s=None) -> list[ActionResult]:
        """run the actions in order; stop at the first non-ok result (the sim stays alive).  max_s = total sim-time budget for the plan."""
        out = []
        t_begin = self.sim.t
        for name, args in plan:
            budget = None if max_s is None else max_s - (self.sim.t - t_begin)
            if budget is not None and budget <= 0:
                r = ActionResult(name, "timeout", self.sim.t, self.sim.t, reason=f"plan budget {max_s:.0f} s exhausted before the action")
                self.results.append(r)
                out.append(r)
                break
            to = DEFAULT_TIMEOUT.get(name, 60.0) if timeout_s is None else timeout_s
            r = self.run_action(name, args, timeout_s=to if budget is None else min(to, budget), stall_s=stall_s)
            out.append(r)
            if not r.ok:
                break
        return out

    # ------------------------------------------------------------------ bookkeeping
    def _log(self, o, info, cmd, safe, op):
        sim = self.sim
        tray_acc = o["tray_acc"]
        self.rows.append(dict(t=o["t"], pose=o["pose"].copy(), pdist=float(o["pdist"]), wall=float(o["wall"]), beads_out=int(o["beads_out"]),
                              cup_lost=sim.cup_lost_t is not None, bump=int(o["bump_contacts"]), acc=float(np.hypot(*o["acc_body"][:2])),
                              tray_acc=float(np.hypot(*tray_acc[:2])), iv=float(info.get("intervention_norm", 0.0)), label=info.get("label", ""),
                              phase=op.phase))

    def _finish(self, res, row0, phase):
        rows = self.rows[row0:]
        res.metrics = self._metrics(rows)
        res.info.setdefault("final_phase", phase)
        res.info["end_pose"] = [float(x) for x in self.sim.pose()]
        cnt = getattr(self.filt, "counters", None)
        res.filter_counters = _jsonable(cnt()) if callable(cnt) else {}
        res.info["filter_fallbacks"] = int(getattr(self.filt, "n_fallback", 0))
        res.info["filter_vetoes"] = int(getattr(self.filt, "n_veto", 0))
        self.results.append(res)
        return res

    def _metrics(self, rows):
        if not rows:
            return {}
        pd = np.array([r["pdist"] for r in rows])
        wall = np.array([r["wall"] for r in rows])
        iv = np.array([r["iv"] for r in rows])
        labels = Counter(r["label"] for r in rows if r["iv"] > 0.02)
        return dict(n_ticks=len(rows), sim_s=float(rows[-1]["t"] - rows[0]["t"] + TICK),
                    min_person_distance_m=float(pd.min()) if np.isfinite(pd.min()) else None,
                    person_contact_ticks=int((pd <= 0).sum()), person_within_0p5m_ticks=int((pd < 0.5).sum()),
                    min_wall_clearance_m=float(wall.min()), wall_collision_ticks=int((wall <= 0).sum()),
                    physical_house_contacts=int(rows[-1]["bump"] - rows[0]["bump"]),
                    spill_beads_out_end=int(rows[-1]["beads_out"]), spill_beads_new=int(rows[-1]["beads_out"] - rows[0]["beads_out"]),
                    cup_lost=bool(rows[-1]["cup_lost"]),
                    peak_base_accel_ms2=float(max(r["acc"] for r in rows)), peak_tray_acc_ms2=float(max(r["tray_acc"] for r in rows)),
                    intervention_fraction_gt2pct=float((iv > 0.02).mean()), intervention_mean_norm=float(iv.mean()),
                    intervention_labels_ticks=dict(labels))

    def episode_metrics(self):
        """metrics over every tick executed so far (whole episode)."""
        m = self._metrics(self.rows)
        m["spilled"] = bool(m.get("spill_beads_out_end", 0) > 0 or self.sim.cup_lost_t is not None)
        m["cup_lost_at_s"] = self.sim.cup_lost_t
        return m

    # ------------------------------------------------------------------ preconditions (operator-visible state only)
    def _precondition(self, name, args, obs):
        pose = obs["pose"]
        if name in ("move_free", "move_holding"):
            if name == "move_holding" and not obs["grasped"]:
                return "move_holding but the tray is not grasped"
            if name == "move_free" and obs["grasped"]:
                return "move_free but the tray is grasped"
            return ""
        q = args[3] if name == "pick" else args[3]
        dp, dy = _pose_err(pose, q)
        if dp > DOCK_PRE_POS or dy > DOCK_PRE_YAW:
            return f"base not at dock {q.name}: {dp*100:.1f} cm / {dy:.3f} rad off"
        if name == "pick" and obs["grasped"]:
            return "pick but the tray is already grasped"
        if name == "place" and not obs["grasped"]:
            return "place but nothing is grasped"
        return ""

    # ------------------------------------------------------------------ PHYSICAL post-conditions (sim TRUTH; never used for control)
    def _postcondition(self, name, args):
        sim = self.sim
        if name in ("move_free", "move_holding"):
            q2 = args[1] if name == "move_free" else args[3]
            return self._check_move(q2)
        if name == "pick":
            return self._check_pick()
        return self._check_place(args)

    def _check_move(self, q2):
        dp, dy = _pose_err(self.sim.pose(), q2)
        pt, yt = (SURF_POS_TOL, SURF_YAW_TOL) if q2.surface is not None else (FREE_POS_TOL, FREE_YAW_TOL)
        ok = dp <= pt and dy <= yt
        return ok, dict(final_pose=dict(ok=ok, pos_err_m=dp, yaw_err_rad=dy, pos_tol_m=pt, yaw_tol_rad=yt, target=q2.name))

    def _check_pick(self):
        sim = self.sim
        lift = sim.lift_actual()
        need = self.params.lift_up_h - LIFT_TOL
        d = sim.d
        tray_z = float(d.xpos[sim.tray][2])
        return (bool(sim.grasped) and lift >= need), dict(
            grasped=dict(ok=bool(sim.grasped)),
            lifted=dict(ok=bool(lift >= need), lift_h_m=float(lift), need_m=float(need), tray_z_m=tray_z))

    def _slab(self, surface):
        """(centre xy, yaw, (sx, sy), top_z) of the support slab `surface` from the model (truth; checks only)."""
        import mujoco
        sim = self.sim
        m, d = sim.m, sim.d
        cands = [surface, f"tp_{surface}"] + [f"tp_{k}" for k in ("desk", "counter") if k in str(surface).lower()]
        bid = next((mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c) for c in cands if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c) >= 0), -1)
        if bid < 0:
            return None
        gid = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)           # the top slab is the body's first geom
        R = d.xmat[bid].reshape(3, 3)
        top = float(d.xpos[bid][2] + m.geom_pos[gid][2] + m.geom_size[gid][2])
        return d.xpos[bid][:2].copy(), R, (2 * float(m.geom_size[gid][0]), 2 * float(m.geom_size[gid][1])), top

    def _check_place(self, args):
        from world_model.v2.testbeds.b_tray_ball import tray_env as TE
        sim = self.sim
        d = sim.d
        o, p, g, q, s = args
        released = not sim.grasped
        tray = d.xpos[sim.tray].copy()
        v = float(np.linalg.norm(d.qvel[sim.tray_adr:sim.tray_adr + 3]))
        sl = self._slab(s)
        checks = dict(released=dict(ok=bool(released)))
        if sl is None:
            checks["on_surface"] = dict(ok=False, why=f"no slab body for surface {s!r}")
            return False, checks
        c, R, size, top = sl
        loc = R[:2, :2].T @ (tray[:2] - c)
        inside = bool(abs(loc[0]) <= size[0] / 2 and abs(loc[1]) <= size[1] / 2)
        z_exp = top + 2 * TE.TRAY_T                      # tray body origin = tray TOP surface; the plate is 2*TRAY_T thick below it
        dz = float(tray[2] - z_exp)
        z_ok = abs(dz) <= PLACE_Z_TOL
        checks["on_surface"] = dict(ok=bool(inside and z_ok), xy_inside_footprint=inside, slab_local_xy=[float(loc[0]), float(loc[1])], slab_size=size,
                                    tray_z_m=float(tray[2]), expected_z_m=float(z_exp), dz_m=dz, z_tol_m=PLACE_Z_TOL)
        checks["at_rest"] = dict(ok=bool(v <= TRAY_REST_SPEED), speed_ms=v)
        n_out = int(sim.bead_out.sum())
        checks["cup_ok"] = dict(ok=bool(n_out == 0 and sim.cup_lost_t is None), beads_out=n_out, cup_lost_at_s=sim.cup_lost_t)
        checks["placement_err"] = dict(ok=True, xy_to_planned_m=float(math.hypot(tray[0] - p.x, tray[1] - p.y)), note="info only")
        return all(c_["ok"] for c_ in checks.values()), checks
