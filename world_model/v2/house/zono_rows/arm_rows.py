"""arm_rows.py -- perceived-zonotope HOCBF rows for the flagship's ArmCtl (teleop/arm.py), fed to FLMPC.step(table_rows=...).  SIMULATION ONLY.

ZonoCtl (ctl.py) is a whole controller (phase2.Ctl2 + rows + relaxation tiers) and builds its own FL-MPC; the flagship's TeleopSim owns an ArmCtl instead.
ArmZonoRows is just the row side of ZonoCtl.zono_rows, with the same rows.py machinery (arm-link keypoints + the held tray's rim points as fixed offsets in
the gripper frame, the active facet of every nearby zonotope, hard rows in the table_rows torque convention):

    rows = ArmZonoRows(ctl, zonos, names)           # once per perception result (H-reps are built here)
    ae, be, ai, bi = rows.rows(d, tray=sim.grasped) # per control step; d = the forwarded controller data (as ArmCtl.tau has it)

The kinematic mocap base moves the arms rigidly; J qd only sees the arm joints, so the base twist enters as v_extra (as in ZonoCtl.set_base_motion).  Here it is
measured from the mocap pose between two consecutive calls (one call per physics step), i.e. the velocity of the step about to be simulated -- no change to
teleop/sim.py needed.  Base acceleration is not modelled.

Rows are HARD (no relaxation tier): FL-MPC's own infeasibility fallback (slack on its arm-arm rows, then a=0) applies exactly as for any other table row.
`stats` counts steps / rows / the smallest row margin h so the flagship can report that the arm side was actually constrained.
"""
from __future__ import annotations

import math

import numpy as np

from . import rows as R


class ArmZonoRows:
    def __init__(self, ctl, zonos, names=None, margin=0.02, alpha1=R.ALPHA1, alpha2=R.ALPHA2, d_act=0.15, max_rows=8, kp_radius=True, tray_spacing=0.05):
        self.m, self.sc = ctl.m, ctl.sc
        self.ld, self.rd = ctl.ld, ctl.rd
        self.zonos = zonos if isinstance(zonos, R.ZonoSet) else R.ZonoSet(zonos, names)
        self.margin, self.alpha1, self.alpha2, self.d_act = margin, alpha1, alpha2, d_act
        self.max_rows, self.kp_radius = max_rows, kp_radius
        self.kp_arm = {s: R.arm_keypoints(self.m, s) for s in ("left", "right")}
        try:                                    # the held tray's rim: fixed offsets in each gripper frame (weld mode: exact)
            self.kp_tray = {s: self.kp_arm[s] + R.tray_keypoints(self.m, s, spacing=tray_spacing, include_ball=False) for s in ("left", "right")}
        except KeyError:
            self.kp_tray = self.kp_arm
        self._prev = None                       # (xy, yaw) of the mocap base at the previous call
        self.dt = float(self.m.opt.timestep)
        self.n_calls = 0
        self.n_calls_with_rows = 0
        self.rows_total = 0
        self.h_min = math.inf
        self.last_info = [{}, {}]

    def base_twist(self, d):
        """(vxy (2,), wz, origin_xy) of the kinematic base from the mocap pose now vs at the previous call (zero on the first call)."""
        xy = np.array(d.mocap_pos[0][:2], float)
        q = d.mocap_quat[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))             # pure z rotation: q = (cos(y/2), 0, 0, sin(y/2))
        if self._prev is None:
            v, wz = np.zeros(2), 0.0
        else:
            v = (xy - self._prev[0]) / self.dt
            wz = ((yaw - self._prev[1] + math.pi) % (2.0 * math.pi) - math.pi) / self.dt
        self._prev = (xy, yaw)
        return v, wz, xy

    def reset_base(self):
        self._prev = None

    def rows(self, d, tray=True):
        """(a_e, b_e, a_i, b_i): left = FL-MPC ego arm, right = interacting arm.  Needs d forwarded (xpos, cdof, cdof_dot, qM, qfrc_bias, qfrc_passive)."""
        v, wz, o = self.base_twist(d)
        vx, vy = float(v[0]), float(v[1])
        vextra = lambda p: np.stack([vx - wz * (p[:, 1] - o[1]), vy + wz * (p[:, 0] - o[0]), np.zeros(len(p))], 1)
        M_full = self.sc.full_mass_matrix(self.m, d)
        kps = self.kp_tray if tray else self.kp_arm
        out, infos = [], []
        for side, da in (("left", self.ld), ("right", self.rd)):
            Ma = M_full[np.ix_(da, da)]
            c = d.qfrc_bias[da] - d.qfrc_passive[da]
            inf = {}
            out.append(R.zono_barrier_rows(self.m, d, kps[side], da, np.linalg.inv(Ma), c, self.zonos, margin=self.margin, alpha1=self.alpha1, alpha2=self.alpha2,
                                           d_act=self.d_act, kp_radius=self.kp_radius, v_extra=vextra, max_rows=self.max_rows, info=inf))
            infos.append(inf)
        self.last_info = infos
        n = len(out[0][1]) + len(out[1][1])
        self.n_calls += 1
        if n:
            self.n_calls_with_rows += 1
            self.rows_total += n
            self.h_min = min(self.h_min, *[i.get("h_min", math.inf) for i in infos])
        return out[0][0], out[0][1], out[1][0], out[1][1]

    def summary(self):
        return dict(n_calls=self.n_calls, n_calls_with_rows=self.n_calls_with_rows, rows_mean_when_active=(self.rows_total / self.n_calls_with_rows if self.n_calls_with_rows else 0.0),
                    h_min=(None if not math.isfinite(self.h_min) else float(self.h_min)), n_zonotopes=len(self.zonos), max_rows=self.max_rows, d_act=self.d_act, margin=self.margin)
