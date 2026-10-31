"""teleop/ghost.py -- operator-intent ghost ("arm shadow").  SIMULATION ONLY.

Draws a translucent copy of the robot where it WOULD be if the operator's RAW command were executed, next to the physical robot,
which executes the shield-filtered command.  The filter is teleop/shield.py (base-twist QP + tray-jog / lift governor); the arms run the
unmodified FL-MPC in both cases.  This module only reads; it never writes the sim's MjModel / MjData and never enters physics, contacts or
control.

Prediction (every tick, re-synced to the measured state, so nothing accumulates):
  From the current base pose / twist, roll the base model (base.BaseSim.advance: the sim's own slew-limited integrator) forward for
  H seconds twice: once under the operator's command u_op, once under the shield's u_safe, each held constant (zero-order hold: "if the
  operator kept the stick here").  The ghost is the physical pose composed with the difference of the two rollouts:
        ghost = pose (+) [ p_safe(H)^-1 (+) p_raw(H) ]
  So if the shield passes the command, u_op = u_safe and the ghost sits exactly on the robot (it is hidden below SHOW_POS / SHOW_ANG).
  If the shield changes it, the ghost moves away by the requested-minus-permitted motion over H.
  Lift: the same rule with the sim's lift rate model (LIFT_V_MAX, LIFT_A_HW, [0, H_MAX]).
  Arms (live only): the tray-frame jog is integrated under both commands with ObjectFrame's own clamps, the two hand targets go through
  the same damped-least-squares IK as ObjectFrame.solve (private scratch MjData), and the ghost arm joints are q_phys + (q_raw - q_safe).
  The display is first-order smoothed (TAU) so it eases back onto the robot when the input stops.

Arrows (shown only while the ghost is visible): orange = requested base motion over H (u_op), green = permitted base motion over H
(u_safe, the shield QP's output this tick).  The green arrow is the command the shield admitted under its model for this tick, not a
guarantee over the horizon.  Nothing else is drawn as a recovery direction.

Display constants (H, TAU, the show thresholds) are display choices, not safety bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import base as BS

TICK = 0.02
H_LOOK = 0.8              # display look-ahead [s] (a display choice)
TAU = 0.15                # display smoothing [s]
SHOW_POS = 0.01           # ghost hidden while it is within 1 cm ...
SHOW_ANG = math.radians(1.0)   # ... and 1 deg (and 0.02 rad on every arm joint) of the robot
FULL_POS = 0.05           # full opacity from 5 cm / 5 deg of separation
INTERV_ON = 0.02          # same "shield active" threshold as the HUDs (|du| / V_STICK)
AUX_ON = 0.005            # jog / lift governor active
LIFT_V_MAX, LIFT_A_HW, LIFT_H_MAX = 0.12, 1.5, 0.416   # = teleop/sim.py LIFT_V_MAX, LIFT_A_HW and scene.H_MAX

RGBA_OK = (0.25, 0.80, 1.00)      # cyan: shield passes / small residual
RGBA_WARN = (1.00, 0.55, 0.10)    # orange: shield is intervening
ALPHA_MAX = 0.38
ARROW_REQ = (1.00, 0.50, 0.08, 0.95)
ARROW_OK = (0.15, 0.95, 0.35, 0.95)
Z_ARROW_CHASE = 1.45     # arrows above the robot in the chase / top views (at floor level the base hides them)
Z_ARROW_EGO = 0.02       # on the floor in the head-camera view


# ------------------------------------------------------------------------------------------------------------------ SE(2)
def se2_compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], BS.wrap(a[2] + b[2])])


def se2_inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), s * a[0] - c * a[1], -a[2]])


def rollout(v0, u, H=H_LOOK, dt=TICK, a_hw=BS.A_HW):
    """base pose after H s of command u from twist v0, in the START body frame (start pose = origin)."""
    b = BS.BaseSim(0.0, 0.0, 0.0, v=np.asarray(v0, float).copy())
    for _ in range(max(1, int(round(H / dt)))):
        b.advance(u, dt, a_hw)
    return np.array([b.x, b.y, b.yaw])


def lift_rollout(h, lv, u, H=H_LOOK, dt=TICK):
    for _ in range(max(1, int(round(H / dt)))):
        lv = float(np.clip(lv + np.clip(u - lv, -LIFT_A_HW * dt, LIFT_A_HW * dt), -LIFT_V_MAX, LIFT_V_MAX))
        h = float(np.clip(h + lv * dt, 0.0, LIFT_H_MAX))
    return h


# ------------------------------------------------------------------------------------------------------------------ frame
G_LEN = 3 + 3 + 12 + 1 + 1 + 1 + 2 + 2 + 2 + 3 + 3     # see GhostFrame.to_vec


@dataclass
class GhostFrame:
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))       # physical base pose the ghost was built on (x, y, yaw)
    ghost: np.ndarray = field(default_factory=lambda: np.zeros(3))      # ghost base pose (world)
    dq: np.ndarray = field(default_factory=lambda: np.zeros(12))        # ghost - physical arm joints (left 6, right 6)
    dh: float = 0.0                                                      # ghost - physical lift height
    alpha: float = 0.0                                                   # 0 = hidden
    warn: bool = False                                                   # shield intervening
    req_end: np.ndarray = field(default_factory=lambda: np.zeros(2))     # world xy of the requested base motion end (H)
    perm_end: np.ndarray = field(default_factory=lambda: np.zeros(2))    # world xy of the permitted base motion end (H)
    gap: np.ndarray = field(default_factory=lambda: np.zeros(2))         # (metres, radians) ghost-to-robot base separation
    u_op: np.ndarray = field(default_factory=lambda: np.zeros(3))
    u_safe: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_vec(self):
        return np.concatenate([self.pose, self.ghost, self.dq, [self.dh, self.alpha, float(self.warn)], self.req_end, self.perm_end,
                               self.gap, self.u_op, self.u_safe]).astype(float)

    @staticmethod
    def from_vec(v):
        v = np.asarray(v, float)
        return GhostFrame(pose=v[0:3].copy(), ghost=v[3:6].copy(), dq=v[6:18].copy(), dh=float(v[18]), alpha=float(v[19]), warn=bool(v[20] > 0.5),
                          req_end=v[21:23].copy(), perm_end=v[23:25].copy(), gap=v[25:27].copy(), u_op=v[27:30].copy(), u_safe=v[30:33].copy())


# ------------------------------------------------------------------------------------------------------------------ prediction
class Predictor:
    """per-tick ghost state.  step() is pure in its inputs except for the display smoothing state; reset() clears it."""

    def __init__(self, H=H_LOOK, tau=TAU, dt=TICK, a_hw=BS.A_HW):
        self.H, self.tau, self.dt, self.a_hw = H, tau, dt, np.asarray(a_hw, float)
        self.reset()

    def reset(self):
        self._D = np.zeros(3)
        self._dh = 0.0
        self._dq = np.zeros(12)

    def step(self, pose, v, u_op, u_safe, interv=0.0, interv_aux=0.0, lift=None, arm_dq=None, active=True):
        """pose, v: measured base pose / twist at the tick start.  lift = (h, lift_v, lift_op, lift_safe) or None.
        arm_dq: (12,) raw-minus-safe arm joints from ArmIntent, or None.  active=False (autonomous phases, shield off): ghost eases out."""
        pose = np.asarray(pose, float)
        u_op, u_safe = np.asarray(u_op, float), np.asarray(u_safe, float)
        rr = rollout(v, u_op, self.H, self.dt, self.a_hw)
        rp = rr if np.allclose(u_op, u_safe, atol=1e-9) else rollout(v, u_safe, self.H, self.dt, self.a_hw)
        D = se2_compose(se2_inv(rp), rr) if (active and rp is not rr) else np.zeros(3)
        dh = 0.0
        if active and lift is not None and abs(lift[2] - lift[3]) > 1e-9:
            dh = lift_rollout(lift[0], lift[1], lift[2], self.H, self.dt) - lift_rollout(lift[0], lift[1], lift[3], self.H, self.dt)
        dq = np.asarray(arm_dq, float) if (active and arm_dq is not None) else np.zeros(12)
        k = 1.0 - math.exp(-self.dt / self.tau)
        self._D = self._D + k * np.r_[D[:2] - self._D[:2], BS.wrap(D[2] - self._D[2])]
        self._dh += k * (dh - self._dh)
        self._dq += k * (dq - self._dq)
        g = se2_compose(pose, self._D)
        gap_p, gap_a = float(np.hypot(*self._D[:2])), abs(float(self._D[2]))
        sep = max(gap_p / FULL_POS, gap_a / math.radians(5.0), abs(self._dh) / FULL_POS, float(np.abs(self._dq).max(initial=0.0)) / 0.1)
        shown = gap_p > SHOW_POS or gap_a > SHOW_ANG or abs(self._dh) > SHOW_POS or float(np.abs(self._dq).max(initial=0.0)) > 0.02
        alpha = float(np.clip(sep, 0.25, 1.0)) if shown else 0.0
        warn = bool(active and (interv > INTERV_ON or interv_aux > AUX_ON))
        c, s = math.cos(pose[2]), math.sin(pose[2])
        to_w = lambda p: pose[:2] + np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
        return GhostFrame(pose=pose.copy(), ghost=g, dq=self._dq.copy(), dh=float(self._dh), alpha=alpha, warn=warn,
                          req_end=to_w(rr), perm_end=to_w(rp), gap=np.array([gap_p, gap_a]), u_op=u_op.copy(), u_safe=u_safe.copy())


def track(pose, v, u_op, u_safe, interv=None, interv_aux=None, active=None, H=H_LOOK, tau=TAU, dt=TICK, a_hw=BS.A_HW):
    """replay helper: ghost vectors (n, G_LEN) from recorded per-tick arrays.  Recorded pose / v are AFTER tick k, and u_op[k] was issued
    from the state after tick k-1, so tick k predicts from (pose[k-1], v[k-1]) and the frame is then re-anchored on pose[k]."""
    n = len(pose)
    P = Predictor(H, tau, dt, a_hw)
    out = np.zeros((n, G_LEN))
    for k in range(n):
        j = max(k - 1, 0)
        f = P.step(pose[j], v[j], u_op[k], u_safe[k], 0.0 if interv is None else interv[k], 0.0 if interv_aux is None else interv_aux[k],
                   active=True if active is None else bool(active[k]))
        D = se2_compose(se2_inv(f.pose), f.ghost)
        f.ghost = se2_compose(np.asarray(pose[k], float), D)
        sh = np.asarray(pose[k], float)[:2] - f.pose[:2]
        f.req_end, f.perm_end, f.pose = f.req_end + sh, f.perm_end + sh, np.asarray(pose[k], float).copy()
        out[k] = f.to_vec()
    return out


class ArmIntent:
    """raw-vs-safe arm joints from the tray-frame jog, through ObjectFrame's own clamps and DLS IK, on a PRIVATE scratch MjData.
    Never calls ObjectFrame.jog / solve (those set the controller reference)."""

    def __init__(self, of, H=H_LOOK, dt=TICK, iters=8):
        self.of, self.H, self.dt, self.iters = of, H, dt, iters
        self.ds = mujoco.MjData(of.m)
        self.ds.qpos[:] = of.ds.qpos
        self.ds.mocap_pos[:] = of.ds.mocap_pos
        self.ds.mocap_quat[:] = of.ds.mocap_quat

    def _offset(self, jog):
        from .arm import _log_R, _rotvec_to_R
        of = self.of
        dp, dR = of.dp.copy(), of.dR.copy()
        v, w = np.asarray(jog[:3], float), np.asarray(jog[3:], float)
        for _ in range(max(1, int(round(self.H / self.dt)))):
            dp = np.clip(dp + v * self.dt, -of.box, of.box)
            dR = _rotvec_to_R(w * self.dt) @ dR
            rv = _log_R(dR)
            rv[:2] = np.clip(rv[:2], -of.tilt_cap, of.tilt_cap)
            rv[2] = np.clip(rv[2], -of.yaw_cap, of.yaw_cap)
            dR = _rotvec_to_R(rv)
        return dp, dR

    def _hands(self, qL, qR):
        of, ds = self.of, self.ds
        ds.qpos[of.ctl.lq] = qL
        ds.qpos[of.ctl.rq] = qR
        mujoco.mj_kinematics(of.m, ds)
        mujoco.mj_comPos(of.m, ds)
        return [(of.base_R.T @ (ds.xpos[b] - of.base_p), of.base_R.T @ ds.xmat[b].reshape(3, 3)) for b in (of.bL, of.bR)]

    def _jac(self, body, dofs):
        of = self.of
        jp = np.zeros((3, of.m.nv)); jr = np.zeros((3, of.m.nv))
        mujoco.mj_jacBody(of.m, self.ds, jp, jr, body)
        J = np.vstack([jp, jr])[:, dofs]
        Rb = of.base_R.T
        return np.vstack([Rb @ J[:3], Rb @ J[3:]])

    def _ik(self, dp, dR):
        from .arm import _log_R
        of = self.of
        tg = [(of.G0p + dp + dR @ (p0 - of.G0p), dR @ R0) for (p0, R0) in of.hand0]
        q = [of.qL.copy(), of.qR.copy()]
        for _ in range(self.iters):
            hands = self._hands(q[0], q[1])
            for k, (b, dofs) in enumerate(((of.bL, of.dofL), (of.bR, of.dofR))):
                p, R = hands[k]
                e = np.concatenate([tg[k][0] - p, _log_R(tg[k][1] @ R.T)])
                J = self._jac(b, dofs)
                dq = J.T @ np.linalg.solve(J @ J.T + of.damp ** 2 * np.eye(6), e)
                q[k] = np.clip(q[k] + np.clip(dq, -0.05, 0.05), of.jlo[k], of.jhi[k])
        return np.concatenate(q)

    def dq(self, jog_op, jog_safe):
        if np.allclose(jog_op, jog_safe, atol=1e-9):
            return np.zeros(12)
        return self._ik(*self._offset(jog_op)) - self._ik(*self._offset(jog_safe))


# ------------------------------------------------------------------------------------------------------------------ drawing
def _yaw_quat(y):
    return np.array([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)])


class Drawer:
    """adds the ghost to an mjvScene that update_scene() already filled from MjData d.  Uses the scene's own robot geoms as templates
    (so it shows exactly what is visible: same meshes / groups) re-posed by a ghost kinematics pass on a private MjData."""

    def __init__(self, m, root="yor_root"):
        self.m = m
        rid = m.body(root).id
        self.mocap = int(m.body_mocapid[rid])
        in_robot = np.array([m.body_rootid[b] == rid for b in range(m.nbody)])
        arm_body = np.array([(m.body(b).name or "").startswith(("left_arm", "right_arm")) for b in range(m.nbody)])
        self.mask_all = in_robot[m.geom_bodyid]
        self.mask_arms = (in_robot & arm_body)[m.geom_bodyid]
        self.arm_q = np.array([m.jnt_qposadr[m.joint(f"{s}_arm_joint{i}").id] for s in ("left", "right") for i in range(1, 7)])
        self.slider_q = [int(m.jnt_qposadr[m.joint(f"Slider_{i}").id]) for i in (1, 2)
                         if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"Slider_{i}") >= 0]
        self.ds = mujoco.MjData(m)

    def pose_data(self, d, gf: GhostFrame):
        """the ghost kinematics on the private MjData (returned)."""
        ds = self.ds
        ds.qpos[:] = d.qpos
        ds.mocap_pos[:] = d.mocap_pos
        ds.mocap_quat[:] = d.mocap_quat
        P, G = gf.pose, gf.ghost
        dy = BS.wrap(G[2] - P[2])
        c, s = math.cos(dy), math.sin(dy)
        p = d.mocap_pos[self.mocap]
        rel = p[:2] - P[:2]
        ds.mocap_pos[self.mocap] = (G[0] + c * rel[0] - s * rel[1], G[1] + s * rel[0] + c * rel[1], p[2])
        q = np.empty(4)
        mujoco.mju_mulQuat(q, _yaw_quat(dy), d.mocap_quat[self.mocap])
        ds.mocap_quat[self.mocap] = q
        ds.qpos[self.arm_q] += gf.dq
        for a in self.slider_q:
            ds.qpos[a] += gf.dh / 2.0
        mujoco.mj_kinematics(self.m, ds)
        return ds

    def add(self, scn, d, gf: GhostFrame, which="all", arrows=True, z_arrow=Z_ARROW_CHASE):
        """returns the number of geoms added (0 when the ghost is hidden or the scene is full)."""
        if gf is None or gf.alpha <= 0.0:
            return 0
        mask = self.mask_all if which == "all" else self.mask_arms
        ds = self.pose_data(d, gf)
        col = RGBA_WARN if gf.warn else RGBA_OK
        rgba = np.array([*col, ALPHA_MAX * gf.alpha], np.float32)
        n0, added = scn.ngeom, 0
        tmpl = [i for i in range(n0) if scn.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM and 0 <= scn.geoms[i].objid < len(mask)
                and mask[scn.geoms[i].objid]]
        for i in tmpl:
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[i]
            gid = g.objid
            Rp, pp = d.geom_xmat[gid].reshape(3, 3), d.geom_xpos[gid]
            Rg, pg = ds.geom_xmat[gid].reshape(3, 3), ds.geom_xpos[gid]
            Rd = Rg @ Rp.T
            mat = Rd @ np.asarray(g.mat, float).reshape(3, 3)
            pos = pg + Rd @ (np.asarray(g.pos, float) - pp)
            ng = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(ng, int(g.type), np.asarray(g.size, float), pos, mat.reshape(-1), rgba)
            ng.dataid = g.dataid
            ng.category = mujoco.mjtCatBit.mjCAT_DECOR
            ng.emission = 0.35
            scn.ngeom += 1
            added += 1
        if arrows:
            o = np.r_[gf.pose[:2], z_arrow]
            for end, c in ((gf.req_end, ARROW_REQ), (gf.perm_end, ARROW_OK)):
                e = np.r_[end, z_arrow]
                if np.linalg.norm(e - o) < 0.02 or scn.ngeom >= scn.maxgeom:
                    continue
                ng = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(ng, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.array(c, np.float32))
                mujoco.mjv_connector(ng, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, o, e)
                ng.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1
                added += 1
        return added


def describe(gf: GhostFrame):
    """short text lines for a HUD: requested vs permitted twist and the ghost separation."""
    f = lambda u: f"vx {u[0]:+.2f}  vy {u[1]:+.2f}  wz {u[2]:+.2f}"
    return dict(requested=f(gf.u_op), permitted=f(gf.u_safe),
                gap=f"ghost {100 * gf.gap[0]:.0f} cm / {math.degrees(gf.gap[1]):.0f} deg" + (f"  lift {100 * gf.dh:+.0f} cm" if abs(gf.dh) > 0.005 else "")
                + (f"  arm {math.degrees(float(np.abs(gf.dq).max())):.0f} deg" if float(np.abs(gf.dq).max()) > 0.02 else ""))
