"""pick/skill.py -- planning (worker thread) + the non-blocking pick state machine (sim thread).  SIMULATION ONLY.

Control path (nothing new is invented): joint REFERENCES -> the UNMODIFIED FL-MPC (teleop.arm.ArmCtl.tau, 200 Hz, torque box + arm-arm / table-floor rows inside the QP).
  * TRANSIT  : straight-line joint-space slew of the reference (<= 0.5 rad/s, synchronised) from the current pose to the pre-grasp IK solution.
  * APPROACH / SQUEEZE / LIFT : grasp.hands.HandServo (Cartesian hand target -> DLS IK -> ctl.set_ref), the same path as the operator tray jog / GraspAssist.
  * FINGERS  : position target slewed at 0.04 m/s (the finger actuator keeps its 20 N forcerange; the pick rig uses the friction-rig gain kp = 2000 N/m, scene.FINGER_KP_GRASP).
The base never moves (Cmd.twist = 0 is the caller's job; PickSession.step does not touch the base / lift).

Single arm  (top-down rim pinch):  READY -> TRANSIT (via a pose above the pre-grasp) -> Cartesian servo onto the pre-grasp (pad centre >= 5 cm above the object top) -> APPROACH straight down
                               until the pad centre is ~0-1.6 cm from the object top (the palm / wrist-camera housing hit the object lower than that: measured by the planner's
                               grasp-height / hand-offset search) -> CLOSE (stall on the object) -> LIFT 10 cm -> HOLD.  The pads therefore grip the top ~1 cm of the object.
Bimanual    (paddle squeeze):  READY -> TRANSIT both hands (via poses 14 cm higher: the wrists re-orient above the object) to 10 cm outside the two opposite faces (approach along the
                               hand-to-hand axis, jaws vertical, fingers closed, squeeze height >= 15 cm above the table: the camera housing hangs below the hand) -> APPROACH to the
                               planned squeeze pose (fingertips 3 cm inside the ESTIMATED faces; contact declared when a hand stalls, proprioception only) -> SQUEEZE 0.6 s ->
                               LIFT both hands by the same 10 cm (relative pose kept) -> HOLD.  The fingertip boxes (link7/8, 3.6 cm beyond the ee site) are what touches the faces.
Failures (reported, never hidden): UNREACHABLE (no IK within joint limits), COLLISION (planned path / runtime contact with furniture), NO_GRASP (fingers closed on nothing / never touched),
SLIP (grip lost during lift / hold: finger gap closes, or the two hands drift together), TIMEOUT, CANCELLED.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import mujoco

from world_model.v2.house.teleop import sim as S
from world_model.v2.house.grasp.hands import HandServo
from . import ik as IK
from . import props as PP

from .propose import ISO_RATIO as PR_ISO

TICK = S.TICK
SIDES = ("left", "right")
LIFT_M = 0.10
PRE_UP = 0.09                  # single: pre-grasp height above the grasp pose
PRE_OUT = 0.10                 # bimanual: pre-grasp standoff outside the faces
V_JOINT = 0.5                  # rad/s transit
V_APPROACH = 0.05              # m/s
V_LIFT = 0.05
V_CLOSE = 0.04                 # m/s per finger
FINGER_TIP = 0.1358 - 0.1       # fingertip boxes of link7/8 end 3.6 cm beyond the ee_grasp site along the approach axis (derived model: link7 origin 0.1358, box y -0.03..0)
SQUEEZE_Z_MIN = 0.15            # [m] squeeze height above the support plane (measured: the horizontal-approach hand's wrist-camera housing hangs ~14 cm below the axis and hits the table below 0.14)
SQUEEZE = 0.030                # bimanual: commanded depth beyond the contact [m]
SQUEEZE_BIAS = 0.0             # (extra, unused)
HOLD_S = 2.0
LIFT_TOL = 0.02                # [m] z tolerance of the lift goal (a loaded hand sags 1-1.5 cm: measured 0.923 vs 0.936)
CONTACT_LAG = 0.006            # guarded move: hand lags target by this much -> contact [m]
T_MAX = dict(TRANSIT=16.0, APPROACH=16.0, CLOSE=4.0, LIFT=6.0)
TRANSIT_TOL = 0.25            # [rad] joint error accepted at the end of the joint-space transit
SITE_TO_PAD = IK.SITE_TO_PAD


def _R_cols(x, y, z):
    return np.stack([x, y, z], 1)


@dataclass
class Waypoints:
    mode: str
    arms: tuple                         # ("left",) | ("right",) | ("left","right")
    q_via: dict = field(default_factory=dict)       # optional high via-pose before the pre-grasp (arm -> q): the hand arrives from above
    q_pre: dict = field(default_factory=dict)       # arm -> q (6)
    q_grasp: dict = field(default_factory=dict)
    q_lift: dict = field(default_factory=dict)
    goal_pre: dict = field(default_factory=dict)    # arm -> (hand origin p, R) of the pre-grasp pose (HandServo converges here before the approach)
    goal_grasp: dict = field(default_factory=dict)  # arm -> (hand origin p, R) in the robot frame (HandServo convention)
    goal_lift: dict = field(default_factory=dict)
    app_dir: dict = field(default_factory=dict)     # arm -> approach axis (3,)
    width: float = 0.0
    ik: dict = field(default_factory=dict)
    note: str = ""


def _arm_geom_mask(sim):
    """geoms of the yor_root subtree (the robot itself; the tray / cup are NOT part of it).  Mocked sims without a MuJoCo model get an all-False mask."""
    try:
        m = sim.m
        root = m.body("yor_root").id
        anc = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == root:
                    anc[b] = True
                    break
                x = int(m.body_parentid[x])
        return anc[m.geom_bodyid]
    except Exception:      # noqa: BLE001
        return np.zeros(1, bool)


class PlanError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


# ====================================================================================== planning (worker thread; reads sim.m only, scratch MjData)
class Scratch:
    """collision / FK scratch on a COPY of the sim state (made on the sim thread by snapshot())."""

    def __init__(self, sim):
        self.m = sim.m
        self.d = mujoco.MjData(sim.m)
        mujoco.mj_copyData(self.d, sim.m, sim.d)
        root = sim.m.body("yor_root").id
        anc = np.zeros(sim.m.nbody, bool)
        for b in range(sim.m.nbody):
            x = b
            while x > 0:
                if x == root:
                    anc[b] = True
                    break
                x = int(sim.m.body_parentid[x])
        self.arm_geom = anc[sim.m.geom_bodyid]
        self.lq, self.rq = np.asarray(sim.ctl.lq), np.asarray(sim.ctl.rq)
        self.qL0, self.qR0 = sim.d.qpos[self.lq].copy(), sim.d.qpos[self.rq].copy()
        self.fing = {s: [int(sim.m.jnt_qposadr[sim.m.joint(f"{s}_arm_joint{k}").id]) for k in (7, 8)] for s in SIDES}

    fingers = PP.FINGER_OPEN                      # finger opening used by the collision checks (bimanual plans set 0: closed paddles)

    def contacts(self, qL, qR, allow_body=None, fingers=None):
        fingers = self.fingers if fingers is None else fingers
        m, d = self.m, self.d
        d.qpos[self.lq], d.qpos[self.rq] = qL, qR
        for s in SIDES:
            d.qpos[self.fing[s][0]], d.qpos[self.fing[s][1]] = fingers, -fingers
        mujoco.mj_forward(m, d)
        bad = []
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if not (self.arm_geom[a] or self.arm_geom[b]):
                continue
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if allow_body is not None and (m.body(ba).name == allow_body or m.body(bb).name == allow_body):
                continue
            bad.append((m.body(ba).name, m.body(bb).name, float(c.dist)))
        return bad


def path_clear(scr, qa0, qa1, side_idx, other_q, allow_body=None, step=0.04):
    """straight joint-space path of ONE arm (side_idx 0 left / 1 right) qa0->qa1 with the other arm at other_q.  -> list of colliding pairs at the first bad sample or []."""
    n = max(2, int(math.ceil(np.abs(qa1 - qa0).max() / step)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        qa = qa0 + t * (qa1 - qa0)
        qL, qR = (qa, other_q) if side_idx == 0 else (other_q, qa)
        bad = scr.contacts(qL, qR, allow_body)
        if bad:
            return bad
    return []


def path_clear_both(scr, qL0, qL1, qR0, qR1, allow_body=None, step=0.04):
    n = max(2, int(math.ceil(max(np.abs(qL1 - qL0).max(), np.abs(qR1 - qR0).max()) / step)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        bad = scr.contacts(qL0 + t * (qL1 - qL0), qR0 + t * (qR1 - qR0), allow_body)
        if bad:
            return bad
    return []


def _jaw_dir_top(yaw_b):
    """jaw axis (horizontal) for a top-down pinch across the object's narrow side whose direction is yaw_b."""
    return np.array([math.cos(yaw_b), math.sin(yaw_b), 0.0])


GRIP_ABOVE_TOP = (-0.010, -0.005, 0.0, 0.005, 0.010, 0.016)   # [m] pad-centre heights relative to the object top tried at the grasp (lowest = most pad overlap first): the wrist-camera housing / palm hit the object when the hand is too low
GRASP_OFFSETS = (0.0, -0.012, 0.012, -0.024, 0.024)     # [m] hand shift along the gripper x axis (perpendicular to the jaws): the wrist-camera housing (left/right_arm_iphone_body capsule,
                                                         # r 2 cm, 5 cm beside the fingers) otherwise touches objects wider than ~4 cm in a top-down pinch


def plan_single(scr, obj, arm_order, ik_n_random=16):
    """obj: dict(centroid, extent (a,b,h), yaw, top_z, base_z).  -> Waypoints.  Raises PlanError(UNREACHABLE | COLLISION).  Arms are tried in `arm_order`, and for each arm the
    hand offsets GRASP_OFFSETS until a pose with a collision-free transit / descent exists."""
    a, b, h = obj["extent"]
    c = np.asarray(obj["centroid"], float)
    zg0 = obj["top_z"]
    # jaws across the narrow side: its direction is yaw + 90 deg (yaw = direction of the long axis); an isotropic footprint (round / square) uses the robot y axis (the reachable one)
    yaw_b = (obj.get("yaw", 0.0) + math.pi / 2) if a >= PR_ISO * b else math.pi / 2
    jaw = _jaw_dir_top(yaw_b)
    app = np.array([0.0, 0.0, -1.0])
    xg = np.cross(jaw, app)                                    # gripper x axis (y x z)
    errs = []
    for arm in arm_order:
        si = SIDES.index(arm)
        q_now = (scr.qL0, scr.qR0)[si]
        q_oth = (scr.qR0, scr.qL0)[si]
        for (dx, dzt) in [(dx, dzt) for dzt in GRIP_ABOVE_TOP for dx in GRASP_OFFSETS]:
            zg = max(zg0 + dzt, obj["base_z"] + 0.02)
            zp = max(zg + PRE_UP, obj["top_z"] + 0.05)                  # pad centre height at the pre-grasp: the pads must clear the object's top
            site = lambda z, dx=dx: np.array([c[0], c[1], 0.0]) + dx * xg + np.array([0.0, 0.0, z - SITE_TO_PAD])
            q_g, ig = IK.solve(site(zg), app, jaw, arm, seeds_q=[q_now], n_random=ik_n_random)
            if q_g is None:
                if dx == 0.0 and dzt == GRIP_ABOVE_TOP[0]:
                    errs.append(f"{arm}: grasp pose unreachable (IK residual pos {ig['pos_err']*1000:.0f} mm)")
                if ig["pos_err"] > 0.03:
                    break                                          # clearly out of reach: do not burn IK time on the other offsets
                continue
            bad = scr.contacts(*((q_g, q_oth) if si == 0 else (q_oth, q_g)))
            if bad:
                if dx == GRASP_OFFSETS[-1] and dzt == GRIP_ABOVE_TOP[-1]:
                    errs.append(f"{arm}: grasp pose collides {bad[0][0]}<->{bad[0][1]} (tried hand offsets {GRASP_OFFSETS}, heights {GRIP_ABOVE_TOP})")
                continue
            q_p, ip = IK.solve(site(zp), app, jaw, arm, seeds_q=[q_g, q_now], n_random=8)
            q_l, il = IK.solve(site(zg + LIFT_M), app, jaw, arm, seeds_q=[q_g, q_p], n_random=8)
            if q_p is None or q_l is None:
                errs.append(f"{arm}: pre-grasp / lift pose unreachable (offset {dx*1000:.0f} mm)")
                continue
            # transit: READY -> (via pose above the pre-grasp) -> pre-grasp; falls back to the direct joint-space path when no via pose is reachable
            q_v, bad = None, []
            for dz in (0.10, 0.06, 0.03):
                qv, iv = IK.solve(site(zp + dz), app, jaw, arm, seeds_q=[q_p, q_now], n_random=8)
                if qv is None:
                    continue
                bad = path_clear(scr, q_now, qv, si, q_oth) or path_clear(scr, qv, q_p, si, q_oth)
                if not bad:
                    q_v = qv
                    break
            if q_v is None:
                bad = path_clear(scr, q_now, q_p, si, q_oth)
                if bad:
                    errs.append(f"{arm}: transit path collides {bad[0][0]}<->{bad[0][1]}")
                    continue
            bad = path_clear(scr, q_p, q_g, si, q_oth)
            if bad:
                errs.append(f"{arm}: descent collides {bad[0][0]}<->{bad[0][1]} (offset {dx*1000:.0f} mm)")
                continue
            w = Waypoints("single", (arm,), note=" | ".join(errs))
            if q_v is not None:
                w.q_via[arm] = q_v
            w.q_pre[arm], w.q_grasp[arm], w.q_lift[arm] = q_p, q_g, q_l
            for nm, q in (("pre", q_p), ("grasp", q_g), ("lift", q_l)):
                p_site, R = IK.fk(arm, q)
                {"pre": w.goal_pre, "grasp": w.goal_grasp, "lift": w.goal_lift}[nm][arm] = (p_site - 0.1 * R[:, 2], R)
            w.app_dir[arm] = app
            w.width = b
            w.ik = dict(grasp=ig, pre=ip, lift=il, hand_offset_mm=round(dx * 1000, 1), pad_above_top_mm=round(dzt * 1000, 1))
            return w
    code = "COLLISION" if any("collides" in e for e in errs) else "UNREACHABLE"
    raise PlanError(code, "; ".join(errs))


def plan_bimanual(scr, obj, axis_yaw, ik_n_random=16):
    scr.fingers = 0.0
    a, b, h = obj["extent"]
    c = np.asarray(obj["centroid"], float)
    u = np.array([math.cos(axis_yaw), math.sin(axis_yaw), 0.0])
    if u[1] < 0:
        u = -u                                          # left hand on the +y side
    # width along u: whichever of (a, b) matches the axis; the plan carries it
    width = float(obj.get("width_axis", a))
    zc = obj["base_z"] + min(max(0.5 * h, SQUEEZE_Z_MIN), h - 0.03)       # the horizontal-approach hand's lowest point (camera housing) must clear the table
    w = Waypoints("bimanual", SIDES)
    w.width = width
    face = {"left": c + u * (width / 2), "right": c - u * (width / 2)}
    app = {"left": -u, "right": u}                      # each hand points at the object
    jaw = np.array([0.0, 0.0, 1.0])
    res = {}
    for arm in SIDES:
        si = SIDES.index(arm)
        q_now = (scr.qL0, scr.qR0)[si]
        f = face[arm].copy(); f[2] = zc
        out = -app[arm]                                  # direction away from the object
        q_pre, ip = IK.solve(f + out * (PRE_OUT + FINGER_TIP), app[arm], jaw, arm, seeds_q=[q_now], n_random=ik_n_random)
        if q_pre is None:
            raise PlanError("UNREACHABLE", f"{arm}: pre-squeeze pose unreachable (pos residual {ip['pos_err']*1000:.0f} mm)")
        q_g, ig = IK.solve(f + out * (FINGER_TIP - SQUEEZE), app[arm], jaw, arm, seeds_q=[q_pre], n_random=4)
        q_l, il = IK.solve(f + out * (FINGER_TIP - SQUEEZE) + np.array([0, 0, LIFT_M]), app[arm], jaw, arm, seeds_q=[q_g], n_random=4)
        if q_g is None or q_l is None:
            raise PlanError("UNREACHABLE", f"{arm}: squeeze / lift pose unreachable")
        q_t, it = IK.solve(f + out * (0.03 + FINGER_TIP), app[arm], jaw, arm, seeds_q=[q_pre], n_random=4)       # 3 cm outside the face: last contact-free pose
        if q_t is None:
            raise PlanError("UNREACHABLE", f"{arm}: touch-standoff pose unreachable")
        res[arm] = (q_pre, q_g, q_l, ip, ig, il, q_t)
    qpL, qpR = res["left"][0], res["right"][0]
    # transit: READY -> via (the pre-squeeze poses raised by dz: the wrists re-orient above the object) -> pre-squeeze; direct path when no via is collision-free
    via, bad = None, []
    for dz in (0.14, 0.10, 0.18, 0.06):
        vq = {}
        for arm in SIDES:
            f = face[arm].copy(); f[2] = zc
            vq[arm], _ = IK.solve(f - app[arm] * (PRE_OUT + FINGER_TIP) + np.array([0, 0, dz]), app[arm], jaw, arm, seeds_q=[res[arm][0], (scr.qL0, scr.qR0)[SIDES.index(arm)]], n_random=8)
        if vq["left"] is None or vq["right"] is None:
            continue
        bad = path_clear_both(scr, scr.qL0, vq["left"], scr.qR0, vq["right"]) or path_clear_both(scr, vq["left"], qpL, vq["right"], qpR)
        if not bad:
            via = vq
            break
    if via is None:
        bad = path_clear_both(scr, scr.qL0, qpL, scr.qR0, qpR)
        if bad:
            raise PlanError("COLLISION", f"transit path collides {bad[0][0]}<->{bad[0][1]}")
    else:
        w.q_via.update(via)
    bad = path_clear_both(scr, qpL, res["left"][6], qpR, res["right"][6], step=0.02)
    if bad:
        raise PlanError("COLLISION", f"approach collides {bad[0][0]}<->{bad[0][1]} ({bad[0][2]*1000:.0f} mm)")
    for arm in SIDES:
        q_pre, q_g, q_l, ip, ig, il, _qt = res[arm]
        w.q_pre[arm], w.q_grasp[arm], w.q_lift[arm] = q_pre, q_g, q_l
        for nm, q in (("pre", q_pre), ("grasp", q_g), ("lift", q_l)):
            p_site, R = IK.fk(arm, q)
            {"pre": w.goal_pre, "grasp": w.goal_grasp, "lift": w.goal_lift}[nm][arm] = (p_site - 0.1 * R[:, 2], R)
        w.app_dir[arm] = app[arm]
        w.ik[arm] = dict(pre=ip, grasp=ig, lift=il)
    return w


def check_reach(scr, obj, plan_mode, arm=None, axis_yaw=0.0):
    """cheap reachability probe for the veto: IK of the grasp pose only (no path / collision)."""
    a, b, h = obj["extent"]
    c = np.asarray(obj["centroid"], float)
    if plan_mode == "single":
        zg = obj["top_z"] - 0.005
        yaw_b = (obj.get("yaw", 0.0) + math.pi / 2) if a >= PR_ISO * b else math.pi / 2
        q, i = IK.solve(np.array([c[0], c[1], zg - SITE_TO_PAD]), np.array([0, 0, -1.0]), _jaw_dir_top(yaw_b), arm,
                        seeds_q=[(scr.qL0, scr.qR0)[SIDES.index(arm)]], n_random=6)
        return q is not None
    return None


# ====================================================================================== the state machine (sim thread, 50 Hz, never blocks)
class PickSkill:
    """one pick.  step(sim) once per teleop tick BEFORE sim.tick(); sets joint references (ctl.set_ref) and finger targets (d.ctrl).  Pure bookkeeping otherwise: O(1) per tick."""

    def __init__(self, sim, wp: Waypoints, target_body=None, squeeze=SQUEEZE, log=None):
        self.sim, self.wp, self.target_body = sim, wp, target_body
        self.squeeze = squeeze
        self.log = log or (lambda *a: None)
        self.arms = wp.arms
        self.ai = {s: SIDES.index(s) for s in self.arms}
        of = sim.of
        self.qref = {"left": of.qL.copy(), "right": of.qR.copy()}
        self.hs = HandServo(sim)
        self.arm_geom = _arm_geom_mask(sim)
        self.hs.ki = 0.0 if wp.mode == "bimanual" else self.hs.ki
        self.state = "TRANSIT"
        self._via_done = False
        self._hist = {}
        self._settle = 0
        self._stage = "pre"
        self._stage_t = sim.t
        self.t0 = sim.t
        self.t_state = sim.t
        self.fail = None            # (code, message)
        self.fq = {s: PP.FINGER_OPEN for s in SIDES}       # finger position target magnitude per side
        self.close_target = 0.0 if wp.mode == "single" else 0.0
        self.contact_goal = {}
        self.timeline = [("TRANSIT", sim.t)]
        self.gap_hist = []
        self.gap_closed = None
        self.hold_t0 = None
        self.max_sep0 = None
        self.cancelled = False
        self.lift_goal_set = False
        self.furniture_contacts = 0
        self.min_gap = None
        self.sep0 = None
        self.info = {}

    # ---------------------------------------------------------------- helpers
    def _enter(self, st):
        self.log(f"[pick] {self.state} -> {st} (t={self.sim.t - self.t0:.2f}s)")
        self.state = st
        self.t_state = self.sim.t
        self.timeline.append((st, self.sim.t))

    def _fail(self, code, msg):
        self.fail = (code, msg)
        self.log(f"[pick] FAILED {code}: {msg}")
        self._enter("FAILED")

    def _fingers(self):
        d, m = self.sim.d, self.sim.m
        out = {}
        for s in SIDES:
            q7 = float(d.qpos[m.jnt_qposadr[m.joint(f"{s}_arm_joint7").id]])
            q8 = float(d.qpos[m.jnt_qposadr[m.joint(f"{s}_arm_joint8").id]])
            out[s] = (q7, q8)
        return out

    def gap(self, s):
        q7, q8 = self._fingers()[s]
        return q7 - q8                       # inner-face separation of the pads [m]

    def _write_fingers(self):
        d, m = self.sim.d, self.sim.m
        for s in SIDES:
            d.ctrl[m.actuator(f"{s}_arm_joint7_pos").id] = self.fq[s]
            d.ctrl[m.actuator(f"{s}_arm_joint8_pos").id] = -self.fq[s]

    def _slew_fingers(self, target, sides=None):
        dq = V_CLOSE * TICK
        for s in (sides or self.arms):
            self.fq[s] = max(target, self.fq[s] - dq) if target < self.fq[s] else min(target, self.fq[s] + dq)

    def _slew_ref(self, goal):
        """synchronised straight-line joint slew of the reference of every active arm toward goal[arm]; -> max remaining |dq|."""
        rem = 0.0
        for s in self.arms:
            e = goal[s] - self.qref[s]
            n = float(np.abs(e).max())
            lim = V_JOINT * TICK
            self.qref[s] = goal[s].copy() if n <= lim else self.qref[s] + e * (lim / n)
            rem = max(rem, n)
        self.sim.ctl.set_ref(self.qref["left"], self.qref["right"])
        self.sim.of.qL, self.sim.of.qR = self.qref["left"].copy(), self.qref["right"].copy()
        return rem

    def _arm_speed(self):
        d, ctl = self.sim.d, self.sim.ctl
        return max(float(np.abs(d.qvel[ctl.ld if s == "left" else ctl.rd]).max()) for s in self.arms)

    def _arm_err(self, s):
        d, ctl = self.sim.d, self.sim.ctl
        q = d.qpos[ctl.lq] if s == "left" else d.qpos[ctl.rq]
        return float(np.abs(q - self.qref[s]).max())

    def _servo_goal(self, which):
        for s in self.arms:
            p, R = (self.wp.goal_grasp if which == "grasp" else self.wp.goal_lift)[s]
            self.hs.set_goal(self.ai[s], p, R)

    def _check_runtime_contacts(self):
        """furniture contact of any arm geom (table / house; props and the stowed tray are not furniture)."""
        sim = self.sim
        d = sim.d
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (sim.house_geom[g1] and self.arm_geom[g2]) or (sim.house_geom[g2] and self.arm_geom[g1]):
                n += 1
        self.furniture_contacts += n
        return n

    # ---------------------------------------------------------------- the tick
    def step(self, sim=None):
        sim = sim or self.sim
        st = self.state
        wp = self.wp
        if st in ("FAILED", "DONE_RELEASED"):
            self._write_fingers()
            return
        self._check_runtime_contacts()
        el = sim.t - self.t_state
        if self.cancelled and st not in ("RELEASE", "RETREAT"):
            self._enter("RELEASE")
            st = "RELEASE"
            el = 0.0
        if st == "TRANSIT":
            self._slew_fingers(PP.FINGER_OPEN if wp.mode == "single" else 0.0)
            via = bool(wp.q_via) and not self._via_done
            rem = self._slew_ref(wp.q_via if via else wp.q_pre)
            # joint-reference slew only brings the hands NEAR the goal (the wrist joints keep ~0.1-0.15 rad of friction offset, measured); HandServo's Cartesian integral removes it in APPROACH
            settled = rem < 1e-6 and max(self._arm_err(s) for s in self.arms) < TRANSIT_TOL and self._arm_speed() < 0.03
            self._settle = self._settle + 1 if settled else 0
            if self._settle >= 8:
                self._settle = 0
                if via:
                    self._via_done = True
                else:
                    self._begin_servo()
                    self._enter("APPROACH")
            elif el > T_MAX["TRANSIT"]:
                self._fail("TIMEOUT", "transit did not converge")
        elif st == "APPROACH":
            self._slew_fingers(PP.FINGER_OPEN if wp.mode == "single" else 0.0)
            self.hs.step(TICK)
            if self._stage == "pre":                                  # Cartesian servo onto the pre-grasp pose first (removes the transit offset), then go in
                if all(self.hs.at_goal(self.ai[s], tol_p=0.004, tol_r=math.radians(3.0)) for s in self.arms) and sim.t - self._stage_t > 0.4:
                    self._stage = "down"
                    self._stage_t = sim.t
                    self._servo_goal("grasp")
            elif wp.mode == "single":
                if all(self.hs.at_goal(self.ai[s], tol_p=0.004, tol_r=math.radians(3.0)) for s in self.arms) and sim.t - self._stage_t > 0.5:
                    self._enter("CLOSE")
            else:
                self._bimanual_approach(sim.t - self._stage_t)
            if self.state == "APPROACH" and el > T_MAX["APPROACH"]:
                self._fail("NO_GRASP" if (wp.mode == "bimanual" and self._stage == "down") else "TIMEOUT", f"approach ({self._stage}) did not reach its goal / contact")
        elif st == "CLOSE":                                   # single only
            s = self.arms[0]
            self.hs.step(TICK)
            self._slew_fingers(0.0)
            g = self.gap(s)
            self.gap_hist.append(g)
            if len(self.gap_hist) > 12:
                self.gap_hist.pop(0)
            stalled = len(self.gap_hist) >= 12 and (max(self.gap_hist) - min(self.gap_hist)) < 0.0006
            if self.fq[s] <= 0.0 or stalled:
                if g < 0.004:
                    self._fail("NO_GRASP", f"fingers closed on nothing (gap {g*1000:.1f} mm)")
                elif g > self.wp.width + 0.015:
                    self._fail("NO_GRASP", f"jaws stalled at {g*1000:.0f} mm, object width {self.wp.width*1000:.0f} mm")
                elif el > 0.6:
                    self.info["gap_at_grip_mm"] = round(g * 1000, 1)
                    self.gap_closed = g
                    self.min_gap = g
                    self._servo_goal("lift")
                    self._enter("LIFT")
            if self.state == "CLOSE" and el > T_MAX["CLOSE"]:
                self._fail("NO_GRASP", "close timed out")
        elif st == "SQUEEZE":                                 # bimanual: let the force build for 0.6 s then lift
            self.hs.step(TICK)
            if el > 0.6:
                self._servo_goal("lift")
                self.sep0 = self._hand_sep()
                self._enter("LIFT")
        elif st == "LIFT":
            self.hs.step(TICK)
            if wp.mode == "single":
                self._slew_fingers(0.0)               # keep driving the jaws toward closed: grip force = kp * (jaw q - target) on the object
            self._slip_check()
            if self.state == "LIFT" and self._lift_done():
                self.hold_t0 = sim.t
                self._enter("HOLD")
            elif self.state == "LIFT" and el > T_MAX["LIFT"]:
                self._fail("TIMEOUT", "lift did not reach the goal")
        elif st == "HOLD":
            self.hs.step(TICK)
            if wp.mode == "single":
                self._slew_fingers(0.0)
            self._slip_check()
        elif st == "RELEASE":
            self.hs.step(TICK) if self.hs.active else None
            self._slew_fingers(PP.FINGER_OPEN)
            if el > 1.0:
                self._enter("RETREAT")
        elif st == "RETREAT":
            self.hs.active = False
            self._slew_fingers(PP.FINGER_OPEN)
            rem = self._slew_ref({s: np.asarray(PP.READY_Q[s], float) for s in self.arms})
            if rem < 1e-6 and el > 0.5:
                self._enter("DONE_RELEASED")
        self._write_fingers()

    # ---------------------------------------------------------------- pieces
    def _begin_servo(self):
        hs = self.hs
        hs.start_from_current()
        for k in (0, 1):
            if SIDES[k] not in self.arms:
                hs.goal[k] = None
                hs.tgt[k] = None
        hs.v_max = V_APPROACH
        self._stage, self._stage_t = "pre", self.sim.t
        for s in self.arms:
            self.contact_goal[s] = None
            p, R = self.wp.goal_pre[s]
            hs.set_goal(self.ai[s], p, R)

    def _lift_done(self):
        """z of every active hand within LIFT_TOL of the lift goal (x / y are NOT checked: a squeezed object keeps the hands short of the commanded squeeze depth)."""
        return all(abs(self.hs.hand_pose(self.ai[s])[0][2] - self.wp.goal_lift[s][0][2]) < LIFT_TOL for s in self.arms)

    def _hand_sep(self):
        a = np.array([self.hs.hand_pose(0)[0], self.hs.hand_pose(1)[0]])
        return float(np.linalg.norm(a[0] - a[1]))

    def _bimanual_approach(self, el):
        """advance both hands to the planned squeeze pose (fingertips SQUEEZE m inside the ESTIMATED faces).  Done when each hand is within 4 mm of it, or when a hand has STALLED
        (< 3 mm/s over 0.5 s, i.e. the object blocks it) within SQUEEZE + 3 cm of it: contact is declared from proprioception (hand FK) only, no force sensor is assumed."""
        hs = self.hs
        done = True
        for s in self.arms:
            k = self.ai[s]
            p, _ = hs.hand_pose(k)
            g = self.wp.goal_grasp[s][0]
            hist = self._hist.setdefault(s, [])
            hist.append((self.sim.t, p.copy()))
            while len(hist) > 26:
                hist.pop(0)
            dist = float(np.linalg.norm(g - p))
            stalled = len(hist) >= 26 and float(np.linalg.norm(hist[-1][1] - hist[0][1])) / (hist[-1][0] - hist[0][0]) < 0.003 and dist < SQUEEZE + 0.03
            if dist < 0.004 or stalled:
                self.info.setdefault(f"contact_{s}_t", round(self.sim.t - self.t0, 2))
                self.info.setdefault(f"contact_{s}_residual_mm", round(dist * 1000, 1))
            else:
                done = False
        if done and el > 0.6:
            self.sep_contact = self._hand_sep()
            self._enter("SQUEEZE")

    def _slip_check(self):
        if self.wp.mode == "single":
            s = self.arms[0]
            g = self.gap(s)
            self.min_gap = g if self.min_gap is None else min(self.min_gap, g)
            if g < 0.004:
                self._fail("SLIP", f"grip lost (gap {g*1000:.1f} mm)")
        else:
            sep = self._hand_sep()
            if self.sep0 is not None and sep < self.sep0 - 0.03:
                self._fail("SLIP", f"hands closed by {1000*(self.sep0-sep):.0f} mm during the lift (object dropped)")

    # ---------------------------------------------------------------- API
    def cancel(self):
        self.cancelled = True

    @property
    def done(self):
        return self.state in ("HOLD",) and self.hold_t0 is not None and (self.sim.t - self.hold_t0) >= HOLD_S

    def phase_info(self):
        return dict(state=self.state, elapsed_s=round(self.sim.t - self.t0, 2), fail=self.fail, gap_mm=None if self.gap_closed is None else round(self.gap_closed * 1000, 1))
