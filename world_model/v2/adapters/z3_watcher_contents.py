"""adapters/z3_watcher_contents.py -- runtime WATCHER of the Z3 planner prototype (box, closed_chain).  SIMULATION ONLY, offline, no VLM.   (11-16)

USER 11-16: "a high frequency watcher going on which has an expectation of a specific thing which the planner decides; if it deviates things need to be looked at".
The planner (closed_chain plan, later the one Z3 chose) states what it EXPECTS at every control sample; the watcher compares the running sim against it at every control step
and records the deviations.  Separate from the U monitors (U = unsafe; watcher = "not what the plan expected").  The watcher never stops the run and never feeds the controller.

Expectation (per control sample of a plan piece; `Expectation`):
  q          planned joint angles per side (ep._cmd, i.e. the controller's own reference)
  obj pose   object pose implied by FK of the planned q:  T_obj_s = T_fk_s(q) @ inv(T0_s) @ T_obj0   (motion_rows._fk_arrays; same frames and the same GUESS: rotation of the
             grasp transform identity)
  load       predicted vertical claw load [kg] per side and total, over the CONTENTS SET (cell -> total carried mass interval [M_lo, M_hi], rule1_contents.cell_total),
             QUASI-STATIC model scaled by the plan's own object acceleration:  L_total in [M_lo, M_hi] * (1 + a_z / g),  a_z = the plan's object acceleration pdd_z at the sample;
             per side  share * L_total,  share in 0.5 +- LOAD_SHARE_BAND (the two claws split the load; the split depends on the contents' position).  Not the static-admission
             envelope (that is the worst-case over the path and the whole COM set; it is a bound, not an expectation).
  clearance  planned min distance per class (obj_world, arm_world, arm_arm) at the plan samples (collision_rows._Scene, same distance function, same allowance and caps)
Measurement (only what a real robot has; the simulator's object pose is NEVER used to fire):
  q, qd           joint encoders (sim: d.qpos)
  object pose     FK of the measured q (both arms, the same formula as the expectation)
  claw load       joint torques (sim: the commanded torque d.qfrc_applied, as probe.Probe) -> L_s = e_z . (J_s^T)^+ (tau_s - bias_s - M_ss qdd_plan_s) / g  (the Probe.read formula;
                  the arm's own inertial term M qdd_plan is subtracted because Probe.read is quasi-static and the carry is not; M is the MJCF model, qdd_plan the plan's)
  clearance       the same distance function on the measured q (both arms) and the object placed by the grasp at the FK-implied offset (NOT the simulator's object pose)
The simulator's true object clearance (ep.track["obj_world_clear_min"]) is recorded for ANALYSIS ONLY (`true_clear_min`), never compared against a tolerance.

Tolerances (module constants, all ASSUMED, TO BE RAISED TO THE USER; each was chosen after one nominal run so that the nominal run does not fire -- they are not measured bounds):
  JOINT_TOL_RAD, OBJ_POS_TOL_M, OBJ_ROT_TOL_RAD, LOAD_DYN_MARGIN_KG, LOAD_SHARE_BAND, CLEAR_TOL_M (+ CLEAR_STRIDE, the clearance evaluation period).  Load noise tolerance
  reuses rule1_contents.claw_read_spec (EPS_FREE MEASURED + nb ASSUMED, the propagated joint-torque noise at the current pose).

Output: ep.track["watch"] (re-attached every step because carry.reset_track replaces ep.track at each segment) =
  dict(n_steps, peak={quantity: dict(value, tol, ratio, t, piece, key, expected, measured)}   # worst deviation of EVERY quantity, fired or not
       fired={quantity: dict(first=dev, n=count, worst=dev)}, clearance={piece: {class: dict(planned_min, measured_min, true_min?, end_planned, end_measured)}}, ...)
  dev = dict(t, quantity, piece, key (side / joint / class), expected, measured, dev, tol)
`Watcher.suggestions()` -> [Suggestion(learned=Learned(kind="block_exact", plan, theta, reason), quantity, margin_needed_m, implied_total_kg)]: e.g. a clearance shortfall e at the place ->
  "tracking: executed clearance {measured} vs planned {planned}; require planned >= d_min + {e}" with margin_needed_m = e, the number the Z3 encoding can use later.

    python -m world_model.v2.adapters.z3_watcher_contents --task box --cell 1.0-1.5 [--inject extra_load:1.0@2.0 | shift_obj:0.01@2.0]
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from world_model.v2.adapters import collision_rows as CRW
from world_model.v2.adapters import motion_rows as MR
from world_model.v2.stack.checks import motion as M
from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.stack.z3plan.types import Learned, Theta

SIDES = CCP.SIDES
DT = CCP.DT
G = 9.81

# ----------------------------------------------------------------------------------------------------------------------- tolerances: ALL ASSUMED (to be raised to the user)
JOINT_TOL_RAD = 0.11          # ASSUMED (to be raised to the user): |q - q_plan| per joint [rad] (6.3 deg); nominal box 1.0-1.5 seed 0 worst 0.088 (carry accel transient)
OBJ_POS_TOL_M = 0.013         # ASSUMED (to be raised to the user): FK-implied object position vs the plan [m]; nominal seed 0 worst 10.8 mm
OBJ_ROT_TOL_RAD = 0.06        # ASSUMED (to be raised to the user): FK-implied object rotation vs the plan [rad]; nominal seed 0 worst 0.047
LOAD_DYN_MARGIN_KG = 0.70     # ASSUMED (to be raised to the user): extra claw-load tolerance [kg] AT REST (planned object speed < LOAD_MOVING_MPS) for what the quasi-static model leaves out
LOAD_MOVING_MARGIN_KG = 1.60  # ASSUMED (to be raised to the user): the same while the planned object moves: the torque-read load swings by ~1.5 kg in the nominal carry (measured here, 1 run), the read cannot see < ~1.6 kg then
LOAD_MOVING_MPS = 0.01        # ASSUMED (to be raised to the user): planned object speed above which a sample is "moving" [m/s]
LOAD_SHARE_BAND = 0.15        # ASSUMED (to be raised to the user): each claw carries (0.5 +- this) of the total load
CLEAR_TOL_M = 0.007           # ASSUMED (to be raised to the user): measured clearance may be this much below the planned one [m]; nominal seed 0 worst shortfall 5.5 mm (arm_world, lift)
CLEAR_STRIDE = 10             # ASSUMED (to be raised to the user): clearance evaluated every CLEAR_STRIDE-th plan sample (50 ms); one evaluation is ~35 ms of GJK
LOAD_MIN_LIFT_M = 0.01        # ASSUMED: load is checked once the planned object offset is this far above the start (the claws carry it; before, the counter does)
CAPS = dict(arm_arm=None, arm_world=None, obj_world=None)     # filled from the scene (class caps; a capped planned value is not a deviation source)

QUANTITIES = ("joint_q", "obj_pos", "obj_rot", "claw_load_total", "claw_load_side", "clearance")


class WatchRecord(dict):
    """the record in ep.track["watch"].  The executor does float(v) on every ep.track value (executor.run_segments), so it carries __float__ = number of fired deviations (0.0 on a nominal run)."""

    def __float__(self):
        return float(sum(f["n"] for f in self.get("fired", {}).values()))


@dataclass
class Expectation:
    """what the plan expects at one control sample (piece `piece`, sample index k, plan time t_plan)."""
    piece: str
    k: int
    t_plan: float
    q: dict                         # {side: (7,)}
    obj_pos: dict                   # {side: (3,)} FK-implied object position of the planned q
    obj_R: dict                     # {side: (3,3)}
    obj_offset: np.ndarray          # (3,) the plan's object offset p_k
    a_obj: np.ndarray               # (3,) the plan's object acceleration
    qdd: dict                       # {side: (7,)} planned joint acceleration
    load_total_kg: tuple            # (lo, hi) predicted vertical claw load, both claws
    load_side_kg: dict              # {side: (lo, hi)}
    clearance: Optional[dict]       # {class: planned min distance [m]} or None (not a clearance sample)
    carrying: bool                  # the load check applies (planned object above the start by LOAD_MIN_LIFT_M)
    moving: bool = False            # the plan's object speed at this sample is above LOAD_MOVING_MPS (the load read is much less accurate then)


@dataclass
class Suggestion:
    learned: Learned
    quantity: str
    margin_needed_m: Optional[float] = None
    implied_total_kg: Optional[float] = None
    details: dict = field(default_factory=dict)


class PieceExp:
    """arrays of one plan piece (built once): planned FK object pose per sample, load intervals, clearances on the stride grid."""

    def __init__(self, plan, name, mlo, mhi, scene):
        self.name = name
        pc = plan.pieces[name]
        self.pc = pc
        self.t, self.q, self.qd, self.p = MR.piece_arrays(plan, name)
        _, _, self.To = MR._fk_arrays(plan, self.q, self.p)
        self.n = pc.p.shape[0]
        a = np.asarray(pc.pdd, float)
        fac = 1.0 + a[:, 2] / G
        self.load_lo, self.load_hi = mlo * fac, mhi * fac
        self.carry = pc.p[:, 2] >= LOAD_MIN_LIFT_M
        v = np.zeros(self.n)
        v[1:-1] = np.linalg.norm(self.p[2:] - self.p[:-2], axis=1) / (2 * DT)
        self.moving = v > LOAD_MOVING_MPS
        self.place_like = bool(self.p[-1, 2] < self.p[0, 2] - 1e-3)            # a lowering piece: the object may come to rest on its support (load -> support)
        self.qdd = {s: np.asarray(pc.qdd[s], float) for s in SIDES}
        self.a_obj = a
        self.clear = {}                                                  # k -> {class: planned min}
        for k in range(0, self.n, CLEAR_STRIDE):
            self.clear[k] = scene_min(scene, self.q["left"][k], self.q["right"][k], self.p[k])
        k = self.n - 1
        if k not in self.clear:
            self.clear[k] = scene_min(scene, self.q["left"][k], self.q["right"][k], self.p[k])

    def at(self, k, mid):
        k = int(min(max(k, 0), self.n - 1))
        lo, hi = float(self.load_lo[k]), float(self.load_hi[k])
        if self.place_like:
            lo = 0.0                                                           # touch-down on the support is the point of the piece: the claws may carry less
        return Expectation(
            piece=self.name, k=k, t_plan=k * DT, q={s: self.q[s][k] for s in SIDES},
            obj_pos={s: self.To[s][k][:3, 3] for s in SIDES}, obj_R={s: self.To[s][k][:3, :3] for s in SIDES},
            obj_offset=self.p[k], a_obj=self.a_obj[k], qdd={s: self.qdd[s][k] for s in SIDES},
            load_total_kg=(lo, hi),
            load_side_kg={s: ((0.5 - LOAD_SHARE_BAND) * lo, (0.5 + LOAD_SHARE_BAND) * hi) for s in SIDES},
            clearance=self.clear.get(k), carrying=bool(self.carry[k]), moving=bool(self.moving[k]))


def scene_min(scene, ql, qr, p):
    """{class: min distance} at one state (same treatment as collision_rows.dense_min_distance: allowed pairs counted at the cap)."""
    scene.set_state(ql, qr, p, False)
    out = {}
    for c, (D, al) in scene.distances().items():
        D = np.where(al, scene.cap[c], D)
        out[c] = float(D.min()) if len(D) else float(scene.cap[c])
    return out


def build_expectations(plan, cell, task="box"):
    """-> {piece name: PieceExp} for every piece of the plan; the contents set = the carry_cert cell (kg contents of the box prior) -> total carried mass interval."""
    from world_model.v2.adapters import rule1_contents as RC
    mlo, mhi = RC.cell_total(task, cell)
    sc = CRW._Scene(plan)
    return {n: PieceExp(plan, n, mlo, mhi, sc) for n in plan.pieces}, sc, (mlo, mhi)


class Watcher:
    """Watcher(ep, plan, expectations).  Register with ep.step_hooks.append(watcher) (run_watched does).  __call__(ep) runs after every control step."""

    def __init__(self, ep, plan, expectations, scene=None, mass_interval=None, task="box", cell=None, plan_vals=None, noise_spec=None):
        self.ep, self.plan, self.exp, self.task, self.cell, self.plan_vals = ep, plan, expectations, task, cell, plan_vals
        self.scene = scene or CRW._Scene(plan)
        self.mass = mass_interval
        self.mid = 0.5 * (plan.inp.T0["left"][:3, 3] + plan.inp.T0["right"][:3, 3])
        self.T0inv = {s: np.linalg.inv(plan.inp.T0[s]) for s in SIDES}
        self.T_obj0 = np.eye(4)
        self.T_obj0[:3, 3] = self.mid
        self.ids = {id(pc.q["left"]): n for n, pc in plan.pieces.items()}      # identity of the joint array handed to ep._add_joint_path (executor) -> piece name
        self.dmin = CRW.d_min_table()
        self.eps = self._eps(noise_spec)
        self.record = WatchRecord(n_steps=0, peak={}, fired={}, clearance={}, tol=self.tolerances(), cost_ms=dict(n=0, mean=0.0, max=0.0, clearance_n=0, clearance_mean=0.0))
        self._cost, self._cost_clear = [0.0, 0.0, 0], [0.0, 0]
        self._last_clear = None
        self.pz = None
        from world_model.v2.nero.contents import probe as PR
        self.tau_noise = PR.TAU_NOISE
        self.log = []                                                          # every firing (capped), for the suggestions
        self._trace = None                                                     # optional per-step trace (set to [] to enable)

    # ------------------------------------------------------------------------------------------------ setup
    def _eps(self, spec):
        """claw-read residual EPS_FREE (MEASURED, rule1_contents.claw_read_spec); nb is added per step.  Falls back to 0.1 kg ASSUMED when the calibration is absent."""
        try:
            from world_model.v2.adapters import rule1_contents as RC
            return float((spec or RC.claw_read_spec(self.task, 0.0)).noise)
        except Exception:
            return 0.1                                                         # ASSUMED fallback (no calibration file)

    @staticmethod
    def tolerances():
        return dict(JOINT_TOL_RAD=JOINT_TOL_RAD, OBJ_POS_TOL_M=OBJ_POS_TOL_M, OBJ_ROT_TOL_RAD=OBJ_ROT_TOL_RAD, LOAD_DYN_MARGIN_KG=LOAD_DYN_MARGIN_KG,
                    LOAD_SHARE_BAND=LOAD_SHARE_BAND, CLEAR_TOL_M=CLEAR_TOL_M, CLEAR_STRIDE=CLEAR_STRIDE, tag="ASSUMED (to be raised to the user)")

    # ------------------------------------------------------------------------------------------------ plan time lookup
    def _locate(self, t):
        """-> (piece name or None, sample index, side-0 path entry) of the left arm's path active at time t (the same rule as CarryEpisode._cmd: the last entry with t0 <= t)."""
        cur = None
        for t0, t1, wp in self.ep.paths["left"]:
            if t < t0:
                break
            cur = (t0, t1, wp)
        if cur is None:
            return None, 0
        t0, t1, wp = cur
        a = np.clip((t - t0) / (t1 - t0), 0, 1) * (len(wp) - 1)
        return self.ids.get(id(wp)), int(round(a))

    # ------------------------------------------------------------------------------------------------ recording
    def _peak(self, quantity, dev, tol, t, piece, key, expected, measured):
        pk = self.record["peak"].get(quantity)
        ratio = dev / tol if tol > 0 else (np.inf if dev > 0 else 0.0)
        if pk is None or ratio > pk["ratio"]:
            self.record["peak"][quantity] = dict(value=float(dev), tol=float(tol), ratio=float(ratio), t=round(float(t), 4), piece=piece, key=key, expected=_j(expected), measured=_j(measured))
        if dev > tol:
            d = dict(t=round(float(t), 4), quantity=quantity, piece=piece, key=key, expected=_j(expected), measured=_j(measured), dev=float(dev), tol=float(tol))
            f = self.record["fired"].get(quantity)
            if f is None:
                self.record["fired"][quantity] = dict(first=d, n=1, worst=d)
            else:
                f["n"] += 1
                if dev / tol > f["worst"]["dev"] / f["worst"]["tol"]:
                    f["worst"] = d
            if len(self.log) < 5000:
                self.log.append(d)

    # ------------------------------------------------------------------------------------------------ the hook
    def __call__(self, ep):
        t_a = time.perf_counter()
        m, d = ep.m, ep.d
        t = ep.k * DT
        self.record["n_steps"] += 1
        ep.track["watch"] = self.record
        piece, k = self._locate(t)
        pe = self.exp.get(piece) if piece else None
        ex = pe.at(k, self.mid) if pe is not None else None
        qm = {s: d.qpos[ep.addr[s]].copy() for s in SIDES}
        # 1. joint tracking (vs the controller's own reference: valid also in holds / retreats)
        for s in SIDES:
            qp = ep._cmd(s, t)
            e = np.abs(qm[s] - qp)
            j = int(np.argmax(e))
            self._peak("joint_q", float(e[j]), JOINT_TOL_RAD, t, piece, f"{s}[{j}]", float(qp[j]), float(qm[s][j]))
        if ex is None:
            self._cost_done(t_a)
            return
        # 2. object pose implied by the measured q (FK, both arms)
        Tm = {}
        for s in SIDES:
            Tm[s] = self.plan.inp.arms[s].fk(qm[s]) @ self.T0inv[s] @ self.T_obj0
            pe_ = float(np.linalg.norm(Tm[s][:3, 3] - ex.obj_pos[s]))
            re_ = float(M.rot_angle(ex.obj_R[s].T @ Tm[s][:3, :3]))
            self._peak("obj_pos", pe_, OBJ_POS_TOL_M, t, piece, s, ex.obj_pos[s], Tm[s][:3, 3])
            self._peak("obj_rot", re_, OBJ_ROT_TOL_RAD, t, piece, s, 0.0, re_)
        # 3. claw load (only while both claws hold the object and the plan has it lifted)
        welded = all(d.eq_active[ep.weld_id[s]] for s in SIDES)
        if welded and ex.carrying:
            self._load(ep, ex, t, piece)
        # 4. clearance on the stride grid
        if ex.clearance is not None and (piece, ex.k) != self._last_clear:
            self._last_clear = (piece, ex.k)
            t_c = time.perf_counter()
            self._clearance(ex, Tm, qm, t, piece)
            self._cost_clear[0] += time.perf_counter() - t_c
            self._cost_clear[1] += 1
        # true clearance of the simulator (analysis only; never fires)
        tc = ep.track.get("obj_world_clear_min")
        if tc is not None and np.isfinite(tc):
            ent = self.record["clearance"].setdefault(piece, {}).setdefault("obj_world", {})
            ent["true_min"] = min(ent.get("true_min", np.inf), float(tc))
        self._cost_done(t_a)

    def _cost_done(self, t_a):
        c = time.perf_counter() - t_a
        self._cost[0] += c
        self._cost[1] = max(self._cost[1], c)
        self._cost[2] += 1
        cm = self.record["cost_ms"]
        cm.update(n=self._cost[2], mean=1e3 * self._cost[0] / self._cost[2], max=1e3 * self._cost[1],
                  clearance_n=self._cost_clear[1], clearance_mean=(1e3 * self._cost_clear[0] / self._cost_clear[1] if self._cost_clear[1] else 0.0))

    # ------------------------------------------------------------------------------------------------ load
    def _load(self, ep, ex, t, piece):
        m, d = ep.m, ep.d
        L, nb = {}, {}
        for s in SIDES:
            dofs = ep.dofs[s]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, ep.site[s])
            Pz = np.linalg.pinv(np.vstack([jp[:, dofs], jr[:, dofs]]).T)[2]
            v = np.zeros(m.nv)
            v[dofs] = ex.qdd[s]
            Mq = np.zeros(m.nv)
            mujoco.mj_mulM(m, d, Mq, v)
            r = d.qfrc_applied[dofs] - d.qfrc_bias[dofs] - Mq[dofs] - self._fc_cmd(ep, s, t)
            L[s] = float(Pz @ r) / G
            nb[s] = float(np.abs(Pz).sum()) * self.tau_noise / G
        tot, nbt = L["left"] + L["right"], nb["left"] + nb["right"]
        lo, hi = ex.load_total_kg
        mg = LOAD_MOVING_MARGIN_KG if ex.moving else LOAD_DYN_MARGIN_KG
        tol = self.eps + nbt + mg
        self._peak("claw_load_total", max(lo - tot, tot - hi, 0.0), tol, t, piece, "total", (lo, hi), tot)
        for s in SIDES:
            slo, shi = ex.load_side_kg[s]
            self._peak("claw_load_side", max(slo - L[s], L[s] - shi, 0.0), self.eps + nb[s] + mg, t, piece, s, (slo, shi), L[s])
        self.record["last_load"] = dict(t=round(float(t), 4), total=tot, left=L["left"], right=L["right"], expected_total=(lo, hi))

    @staticmethod
    def _fc_cmd(ep, s, t):
        """the friction-compensation torque the robot itself commanded this step (carry._step_core: f tanh(vc / v_s) + ki e_int + kp e); a known command, subtracted from the read torque
        (the read is then of the load, not of the compensation).  Not a sim-truth quantity: ep.fc / e_int are the controller's own state."""
        fc = ep.fc
        if fc is None:
            return 0.0
        vc = (ep._cmd(s, t + DT) - ep._cmd(s, t)) / DT
        e = ep._cmd(s, t - DT) - ep.d.qpos[ep.addr[s]]
        return fc["f"] * np.tanh(vc / fc["v_s"]) + fc["ki"] * ep.e_int[s] + fc["kp"] * e

    # ------------------------------------------------------------------------------------------------ clearance
    def _clearance(self, ex, Tm, qm, t, piece):
        p_meas = 0.5 * (Tm["left"][:3, 3] + Tm["right"][:3, 3]) - self.mid
        meas = scene_min(self.scene, qm["left"], qm["right"], p_meas)
        cap = self.scene.cap
        rec = self.record["clearance"].setdefault(piece, {})
        last = ex.k == self.exp[piece].n - 1
        for c, pl in ex.clearance.items():
            me = meas[c]
            ent = rec.setdefault(c, {})
            ent["planned_min"] = min(ent.get("planned_min", np.inf), pl)
            ent["measured_min"] = min(ent.get("measured_min", np.inf), me)
            if last or "end_planned" not in ent:
                pass
            if last:
                ent["end_planned"], ent["end_measured"] = pl, me
            if pl >= cap[c] - 1e-9:                                           # planned "at least cap": a shortfall is not defined against a capped value
                continue
            self._peak("clearance", pl - me, CLEAR_TOL_M, t, piece, c, pl, me)

    # ------------------------------------------------------------------------------------------------ suggestions
    def suggestions(self):
        """the constraints the Z3 loop can learn from what fired.  One per fired quantity (worst deviation); plan / theta = the run's."""
        plan = dict(self.plan_vals) if self.plan_vals else _default_vals(self.task)
        theta = Theta(float(self.cell[0]), float(self.cell[1])) if self.cell is not None else None
        out = []
        for q, f in self.record["fired"].items():
            w = f["worst"]
            if q == "clearance":
                e = w["dev"]
                reason = (f"tracking: executed clearance {w['measured']:.4f} m vs planned {w['expected']:.4f} m ({w['key']}, piece {w['piece']}, t={w['t']} s); "
                          f"require planned >= d_min + {e:.4f}")
                out.append(Suggestion(Learned("block_exact", plan, theta, reason), q, margin_needed_m=float(e), details=w))
            elif q in ("obj_pos", "joint_q"):
                e = w["dev"]
                reason = (f"tracking: {q} {w['key']} deviation {e:.4f} vs tolerance {w['tol']:.4f} (piece {w['piece']}, t={w['t']} s); "
                          + ("require planned clearance >= d_min + the object position error" if q == "obj_pos" else "joint tracking error not covered by the plan's margins"))
                out.append(Suggestion(Learned("block_exact", plan, theta, reason), q, margin_needed_m=(float(e) if q == "obj_pos" else None), details=w))
            elif q in ("claw_load_total", "claw_load_side"):
                imp = None
                if q == "claw_load_total":
                    ll = self.record.get("last_load")
                    imp = float(w["measured"])
                reason = (f"tracking: claw load {w['key']} measured {w['measured']:.3f} kg outside the predicted {tuple(round(x, 3) for x in w['expected'])} kg by {w['dev']:.3f} kg "
                          f"(tol {w['tol']:.3f}, piece {w['piece']}, t={w['t']} s); the contents set does not contain the carried mass: re-run Z3 on a Theta containing it")
                out.append(Suggestion(Learned("block_exact", plan, theta, reason), q, implied_total_kg=imp, details=w))
            else:
                reason = f"tracking: {q} deviation {w['dev']:.4f} vs tolerance {w['tol']:.4f} (piece {w['piece']}, t={w['t']} s)"
                out.append(Suggestion(Learned("block_exact", plan, theta, reason), q, details=w))
        return out


def _default_vals(task):
    from world_model.v2.adapters import z3_oracle_contents as ZO
    return ZO.default_vals(task)


def _j(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        return [float(v) for v in np.asarray(x, float).ravel()]
    return float(x) if isinstance(x, (np.floating, np.integer, float, int)) else x


# ----------------------------------------------------------------------------------------------------------------------- synthetic deviations (testing)
class Injector:
    """a step hook that applies ONE synthetic deviation at sim time t.  extra_load: body_mass of the object += kg (inertia scaled, ancestors' subtree mass updated; the controller's payload model
    does not know).  shift_obj: the weld's relative pose is displaced by dx [m] along world x on both claws (the soft weld then pulls the object over; the joints and their encoders are unchanged, so FK
    of the measured q still implies the OLD pose: only the load split / tracking can see it)."""

    def __init__(self, spec):
        self.spec, self.done, self.t_fire, self.k_fire = dict(spec), False, None, None

    def __call__(self, ep):
        sp = self.spec
        if self.done or ep.k * DT < float(sp["t"]) - 1e-9:
            return
        m, d = ep.m, ep.d
        if sp["kind"] == "extra_load":
            b = int(ep.obj)
            m0 = float(m.body_mass[b])
            m.body_mass[b] = m0 + float(sp["kg"])
            m.body_inertia[b] *= m.body_mass[b] / m0
            a = b
            while a > 0:
                m.body_subtreemass[a] += float(sp["kg"])
                a = int(m.body_parentid[a])
        elif sp["kind"] == "shift_obj":
            for s in SIDES:
                i = ep.weld_id[s]
                R1 = d.xmat[m.eq_obj1id[i]].reshape(3, 3)
                m.eq_data[i, 3:6] += R1.T @ np.array([float(sp["dx"]), 0.0, 0.0])
        else:
            raise ValueError(f"inject kind {sp['kind']!r}")
        self.done, self.t_fire, self.k_fire = True, ep.k * DT, ep.k


# ----------------------------------------------------------------------------------------------------------------------- driver
def run_watched(task="box", cell=(1.0, 1.5), seed=0, plan_vals=None, inject=None, continue_through_u=False):
    """see _run_watched.  carry.U_OBJ_CLEAR_M is restored afterwards (owner fix 11-16: continue_through_u used to leave it at 0 for the rest of the process)."""
    from world_model.v2.nero.contents import carry as CR
    saved = CR.U_OBJ_CLEAR_M
    try:
        return _run_watched(task, cell, seed, plan_vals, inject, continue_through_u)
    finally:
        CR.U_OBJ_CLEAR_M = saved


def _run_watched(task="box", cell=(1.0, 1.5), seed=0, plan_vals=None, inject=None, continue_through_u=False):
    """build the world, the closed_chain episode (as carry.world_trial does), attach the watcher, run the hold and all segments (no counterfactual branches), return the watch record.
    plan_vals = z3 plan knobs (zc, place_dx, place_dy, a, recovery; z3_oracle_contents.to_waypoint) or None = the default plan.  inject = dict(kind="extra_load", kg, t) | dict(kind="shift_obj", dx, t)."""
    from world_model.v2.nero.contents import carry as CR, carry_cert as CC
    if continue_through_u:
        # executor.run_segments stops at the first segment with a U flag; this keeps the run going to collect the watcher record (the clearance U threshold is set to 0 for this process: only
        # penetration counts).  Everything else is unchanged.  Default False: the U monitors act as in every other run.
        CR.U_OBJ_CLEAR_M = 0.0
    w = CC.sample_world(task, tuple(cell), np.random.default_rng(seed))
    wp, a = None, CR.E.A_NOM
    if plan_vals is not None:
        from world_model.v2.adapters import z3_oracle_contents as ZO
        wp, a = ZO.to_waypoint(task, plan_vals, None), float(plan_vals["a"])
    ep = CR.CarryEpisode(w, a_plan=a, planner="closed_chain", wp=wp)
    plan = ep.closed_chain()
    exps, scene, mass = build_expectations(plan, cell, task)
    wt = Watcher(ep, plan, exps, scene=scene, mass_interval=mass, task=task, cell=tuple(cell), plan_vals=plan_vals)
    inj = Injector(inject) if inject else None
    if inj is not None:
        ep.step_hooks.append(inj)
    ep.step_hooks.append(wt)
    ep.run_phases([("hold", 0.4)])
    rec = dict(tag=f"{task}_{cell[0]}_{seed}", task=task, theta=CR._theta(w), a_plan=a, m_nom=ep.m_nom, segments=[])
    CR.run_segments(ep, rec, branch_backups=False)
    watch = dict(wt.record)
    watch.pop("suggestions", None)
    watch["suggestions"] = wt.suggestions()
    watch["inject"] = None if inj is None else dict(inj.spec, t_fire=inj.t_fire, k_fire=inj.k_fire)
    watch["first_fire_steps_after_inject"] = (
        {q: int(round((f["first"]["t"] - inj.t_fire) / DT)) for q, f in wt.record["fired"].items()} if inj is not None and inj.done else None)
    watch["mass_interval_kg"] = mass
    watch["pieces"] = {n: dict(n=pe.n, T=float(pe.pc.T)) for n, pe in exps.items()}
    watch["eps_claw_kg"] = wt.eps
    watch["segments_viol"] = [(s.get("mode") or s.get("seg"), (s.get("nominal") or {}).get("viol")) for s in rec.get("segments", [])]
    return watch


def _main(argv=None):
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="box")
    ap.add_argument("--cell", default="1.0-1.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject", default="")
    ap.add_argument("--through-u", action="store_true", help="keep running after a U flag (sets the obj_world U threshold to 0 in this process)")
    a = ap.parse_args(argv)
    cell = tuple(float(x) for x in a.cell.split("-"))
    inj = None
    if a.inject:
        kind, rest = a.inject.split(":")
        v, t = rest.split("@")
        inj = dict(kind=kind, t=float(t), **({"kg": float(v)} if kind == "extra_load" else {"dx": float(v)}))
    r = run_watched(a.task, cell, a.seed, None, inj, continue_through_u=a.through_u)
    r["suggestions"] = [dict(quantity=s.quantity, margin_needed_m=s.margin_needed_m, implied_total_kg=s.implied_total_kg, kind=s.learned.kind, reason=s.learned.reason) for s in r["suggestions"]]
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    _main()
