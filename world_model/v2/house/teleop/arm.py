"""teleop/arm.py -- bimanual abstraction: the operator commands ONE object frame (the tray); both arms follow through the
UNMODIFIED FL-MPC (experiments/.../mpc_fl.py via phase1.Ctl).  SIMULATION ONLY.

  ArmCtl      phase1.Ctl + the tray tilt cone (same call as phase2.Ctl2: left_ee_body / tilt_max_rad / cup_cone_facets; copied because
              phase2.py is being edited by another agent) with a settable joint reference.
  ObjectFrame the grasp/object frame G (the tray frame, in the BASE frame).  A 6-DoF jog twist moves G; the two hand targets are
              rigidly attached to G (hand_i = G * T_i, T_i fixed at the hold pose); damped-least-squares IK (mj_jacBody on the actual
              model) turns the hand targets into joint references for the two arms, which FL-MPC then tracks.  So FL-MPC (arm-arm
              clearance, table-floor, torque box, tilt cone) stays in the loop and unchanged: it sees only a moving q_ref.
"""
from __future__ import annotations

import math

import numpy as np
import mujoco

from .. import phase1 as P1
from . import fastpath as FP

TILT_MAX = 0.10          # rad, FL-MPC tray tilt bound (same as phase2.TILT_MAX)
CONE_FACETS = 8
ENC_Q, ENC_V = 1e-4, 2e-3


class ArmCtl(P1.Ctl):
    def __init__(self, C, tilt=True):
        super().__init__(C)
        self.lee = C.m.body("left_arm_gripper_base").id
        self.tilt = tilt
        self.last_ok = True
        self.q_hold = (np.array(C.info["hold_q"]["left"]), np.array(C.info["hold_q"]["right"]))
        # opt-in speed path (env BIMANUAL_TELEOP_FAST=1; default off = the old code path, bit-identical): see fastpath.py
        self.fast = FP.fast_enabled()
        self._barriers = FP.fast_all_pair_barriers() if self.fast else None     # None = look sim_common's up at call time (old behaviour)
        self.supports_split = type(self).tau is ArmCtl.tau     # tau() honours skip_refresh (set by fastpath.begin when mj_step1 already refreshed d); subclasses with their own tau (FFArmCtl) opt out
        self.skip_refresh = False
        self.split_pending = False
        # opt-in zonotope HOCBF rows for FL-MPC (zono_rows.arm_rows; default None = the old code path, bit-identical): set_zono / clear_zono
        self.zono = None
        self.zono_tray = True          # include the held tray's rim points in the rows (the caller keeps it equal to sim.grasped)

    def set_zono(self, zonos, names=None, **kw):
        """constrain both arms (and the held tray) against perceived zonotopes through FLMPC.step(table_rows=...).  zonos = ZonoSet or [(c, G)]."""
        from ..zono_rows.arm_rows import ArmZonoRows
        self.zono = ArmZonoRows(self, zonos, names, **kw)
        return self.zono

    def clear_zono(self):
        self.zono = None

    def set_ref(self, qL, qR):
        """joint reference for the whole horizon (constant), zero velocity reference."""
        self.qe_ref[:] = qL
        self.qi_ref[:] = qR

    def tau(self, d):
        if not self.skip_refresh:
            if self.fast:
                FP.refresh_dynamics(self.m, d)
            else:
                mujoco.mj_forward(self.m, d)
        pairs = (self._barriers or self.sc.all_pair_barriers)(self.m, d, self.ld, self.rd)
        kw = dict(left_ee_body=self.lee, tilt_max_rad=TILT_MAX, cup_cone_facets=CONE_FACETS) if self.tilt else {}
        if self.zono is None:
            te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                      self.qe_ref, self.z, self.qi_ref, self.z, pairs, **kw)
        else:
            te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                      self.qe_ref, self.z, self.qi_ref, self.z, pairs, table_rows=self.zono.rows(d, tray=self.zono_tray), **kw)
        if not ok:
            self.n_infeasible += 1
        self.last_ok = ok
        return te, ti


def _rotvec_to_R(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _log_R(R):
    c = (np.trace(R) - 1.0) / 2.0
    th = math.acos(max(-1.0, min(1.0, c)))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return th * w


class ObjectFrame:
    """Tray (object) frame in the BASE frame + differential IK of the two hands.

    jog(twist6, dt): twist6 = (vx, vy, vz, wx, wy, wz) of G expressed in the BASE frame (the operator's tray-frame jog; the scripted
    operator and the keyboard both emit base-frame axes).  Position of G is clamped to a box around the hold pose (the arms' comfortable
    workspace; documented limit), the tilt of G (roll/pitch) to `tilt_cap`.
    """

    def __init__(self, ctl: ArmCtl, C, box=(0.12, 0.12, 0.10), yaw_cap=0.35, tilt_cap=0.12, damp=0.05):
        self.ctl, self.m = ctl, C.m
        self.ds = mujoco.MjData(C.m)
        self.ds.qpos[:] = C.d.qpos
        self.ds.mocap_pos[:] = C.d.mocap_pos
        self.ds.mocap_quat[:] = C.d.mocap_quat
        self.qL, self.qR = ctl.q_hold[0].copy(), ctl.q_hold[1].copy()
        self.bL, self.bR = (C.m.body(f"{s}_arm_gripper_base").id for s in ("left", "right"))
        self.dofL, self.dofR = ctl.ld, ctl.rd
        self.box, self.yaw_cap, self.tilt_cap, self.damp = np.array(box), yaw_cap, tilt_cap, damp
        self.base0 = (np.array([0.0, 0.0, 0.0]), np.eye(3))
        # base frame = mocap root pose; compute hand poses in the BASE frame at the hold reference
        self.base_p = C.d.mocap_pos[0].copy()
        self.base_R = C.d.xmat[C.m.body("yor_root").id].reshape(3, 3).copy()
        self.hand0 = self._hands(self.qL, self.qR)            # [(p, R)] in base frame (hold pose)
        self.G0p = np.array(C.info["tray_local"], float)      # tray in base frame at hold
        self.dp = np.zeros(3)                                 # G position offset from hold (base frame)
        self.dR = np.eye(3)                                   # G orientation offset from hold
        rng = lambda s: np.array([C.m.jnt_range[C.m.joint(f"{s}_arm_joint{i}").id] for i in range(1, 7)])
        self.jlo = [rng("left")[:, 0] + 0.03, rng("right")[:, 0] + 0.03]
        self.jhi = [rng("left")[:, 1] - 0.03, rng("right")[:, 1] - 0.03]
        self.nominal = {k: np.array(v) for k, v in C.info["weld_nominal"].items()}

    # forward kinematics of the two hands in the base frame for joint vectors
    def _hands(self, qL, qR):
        ds = self.ds
        ds.qpos[self.ctl.lq] = qL
        ds.qpos[self.ctl.rq] = qR
        mujoco.mj_kinematics(self.m, ds)
        mujoco.mj_comPos(self.m, ds)
        out = []
        for b in (self.bL, self.bR):
            p = self.base_R.T @ (ds.xpos[b] - self.base_p)
            R = self.base_R.T @ ds.xmat[b].reshape(3, 3)
            out.append((p, R))
        return out

    def _jac(self, body, dofs):
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.ds, jp, jr, body)
        J = np.vstack([jp, jr])[:, dofs]
        # world-frame jacobian -> base frame
        Rb = self.base_R.T
        return np.vstack([Rb @ J[:3], Rb @ J[3:]])

    def hand_targets(self):
        """hand i target = G_target * T_i with T_i = G0^-1 hand0_i : p_i = G0p + dp + dR (p_i0 - G0p)."""
        out = []
        for (p0, R0) in self.hand0:
            p = self.G0p + self.dp + self.dR @ (p0 - self.G0p)
            out.append((p, self.dR @ R0))
        return out

    def jog(self, twist6, dt):
        v = np.asarray(twist6[:3], float) * dt
        w = np.asarray(twist6[3:], float) * dt
        self.dp = np.clip(self.dp + v, -self.box, self.box)
        self.dR = _rotvec_to_R(w) @ self.dR
        rv = _log_R(self.dR)                                       # clamp tilt (roll, pitch) and yaw
        rv[:2] = np.clip(rv[:2], -self.tilt_cap, self.tilt_cap)
        rv[2] = np.clip(rv[2], -self.yaw_cap, self.yaw_cap)
        self.dR = _rotvec_to_R(rv)
        self.solve()

    def solve(self, iters=3):
        tg = self.hand_targets()
        for _ in range(iters):
            hands = self._hands(self.qL, self.qR)
            for k, (b, dofs, attr) in enumerate(((self.bL, self.dofL, "qL"), (self.bR, self.dofR, "qR"))):
                p, R = hands[k]
                e = np.concatenate([tg[k][0] - p, _log_R(tg[k][1] @ R.T)])
                J = self._jac(b, dofs)
                dq = J.T @ np.linalg.solve(J @ J.T + self.damp ** 2 * np.eye(6), e)
                q = np.clip(getattr(self, attr) + np.clip(dq, -0.05, 0.05), self.jlo[k], self.jhi[k])   # joint limits (hold pose sits at the j5 limit)
                setattr(self, attr, q)
        self.ctl.set_ref(self.qL, self.qR)

    def tilt_cmd(self):
        return float(np.linalg.norm(_log_R(self.dR)[:2]))

    def handle_errors(self, d, rig_info=None, tray="true"):
        """gripper-to-handle errors for the grasp trigger: handle_i = tray pose composed with the nominal weld relpose
        (eq_data rel / relquat captured at the nominal hold).  Returns (pos err L, pos err R, ang err L, ang err R)."""
        m = self.m
        res = []
        # tray: "true" = the tray body (privileged, default); (p, R) = an ESTIMATED world pose (BIMANUAL_PROPRIO_ONLY); None = no estimate -> inf errors
        if isinstance(tray, str):
            tray_p, Rt = d.xpos[m.body("tray").id], d.xmat[m.body("tray").id].reshape(3, 3)
        elif tray is None:
            return [(float("inf"), float("inf")), (float("inf"), float("inf"))]
        else:
            tray_p, Rt = np.asarray(tray[0], float), np.asarray(tray[1], float)
        for s, b in (("left", self.bL), ("right", self.bR)):
            rel = self.nominal[s][3:6]                 # tray origin in gripper frame at the nominal hold
            rq = self.nominal[s][6:10]                 # q_grip * rq == q_tray at nominal
            Rg = d.xmat[b].reshape(3, 3)
            p_expected = d.xpos[b] + Rg @ rel
            Rn = np.zeros(9); mujoco.mju_quat2Mat(Rn, rq)
            R_expected = Rg @ Rn.reshape(3, 3)
            res.append((float(np.linalg.norm(tray_p - p_expected)), float(np.linalg.norm(_log_R(Rt @ R_expected.T)))))
        return res

    def grasp_error_vec(self, d, tray="true"):
        """mean tray pose error w.r.t. where the two hands expect it (nominal weld relposes), expressed in the BASE frame:
        (dp[3], rotvec[3]).  Moving the hand frame G by +dp / +rotvec brings the grippers onto the handles (the operator's visual cue)."""
        m = self.m
        if isinstance(tray, str):
            tray_p, Rt = d.xpos[m.body("tray").id], d.xmat[m.body("tray").id].reshape(3, 3)
        elif tray is None:
            return np.zeros(3), np.zeros(3)                   # no estimate: no error cue (can_grasp is False through handle_errors)
        else:
            tray_p, Rt = np.asarray(tray[0], float), np.asarray(tray[1], float)
        dps, rvs = [], []
        for s, b in (("left", self.bL), ("right", self.bR)):
            rel, rq = self.nominal[s][3:6], self.nominal[s][6:10]
            Rg = d.xmat[b].reshape(3, 3)
            p_exp = d.xpos[b] + Rg @ rel
            Rn = np.zeros(9); mujoco.mju_quat2Mat(Rn, rq)
            R_exp = Rg @ Rn.reshape(3, 3)
            dps.append(tray_p - p_exp)
            rvs.append(_log_R(Rt @ R_exp.T))
        Rb = self.base_R_now(d).T
        return Rb @ np.mean(dps, 0), Rb @ np.mean(rvs, 0)

    def base_R_now(self, d):
        return d.xmat[self.m.body("yor_root").id].reshape(3, 3)
