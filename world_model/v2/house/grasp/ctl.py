"""grasp/ctl.py -- ArmCtl with FL-MPC feed-forward for the DYNAMIC base (friction-grasp rig).  SIMULATION ONLY.

With a dynamic root (scene.build_rig(dynamic_base=True)) the arm links feel the base acceleration as a pseudo-force, which a
mocap root hides.  The FL-MPC torque is tau = M a + c with c = qfrc_bias - qfrc_passive; this controller hands the unmodified FL-MPC two
extra bias terms through its `bias_ego_extra` argument (`bias_int_extra` is not an FL-MPC argument, so the right arm is handled by a
post-hoc torque correction on its first-block torque -- see tau()):

  * base acceleration a_pt (world, acceleration of the tray-hold point of the base, known from the commanded base kinematics):
        tau_ff = + sum_b J_b^T m_b a_pt            (all arm + finger bodies; cancels -m_b a_pt, gravcomp bodies included)
  * held tray: each hand carries `payload_kg` at the tray centre:
        tau_ff += - J_tray^T payload_kg (g - a_pt)  (weight and inertial load of the payload, rigid-grasp assumption)

Both are FEED-FORWARD from signals the robot owns (commanded base twist, tray mass).  The FL-MPC itself (constraints, horizon) is untouched.
"""
from __future__ import annotations

import mujoco
import numpy as np

from world_model.v2.house.teleop.arm import ArmCtl, TILT_MAX, CONE_FACETS


class FFArmCtl(ArmCtl):
    def __init__(self, C, tilt=True):
        super().__init__(C, tilt=tilt)
        m = self.m
        self.a_world = np.zeros(3)              # acceleration of the base hold point, world frame (set by the sim each substep)
        self.payload = np.zeros(2)              # kg carried by (left, right) hand
        self.payload_pt = np.zeros(3)           # world point of the payload (tray centre), set by the sim
        self.ff_on = True
        self._bodies = {}
        for side in ("left", "right"):
            first = int(m.jnt_bodyid[m.joint(f"{side}_arm_joint1").id])
            sub = [b for b in range(m.nbody) if self._is_desc(m, b, first)]
            self._bodies[side] = [b for b in sub if m.body_mass[b] > 0]
        self._jp = np.zeros((3, m.nv))

    @staticmethod
    def _is_desc(m, b, anc):
        while b > 0:
            if b == anc:
                return True
            b = int(m.body_parentid[b])
        return False

    def ff_torque(self, d):
        """(tau_left, tau_right) feed-forward to ADD to the FL-MPC torque and to its bias (so that FL's M a + c cancels the load)."""
        m = self.m
        out = []
        for side, dofs, pl in (("left", self.ld, self.payload[0]), ("right", self.rd, self.payload[1])):
            extra = np.zeros(6)
            for b in self._bodies[side]:
                mujoco.mj_jacBodyCom(m, d, self._jp, None, b)
                extra += self._jp[:, dofs].T @ (m.body_mass[b] * self.a_world)
            if pl > 0:
                bt = self._tray_body_for(side)
                mujoco.mj_jac(m, d, self._jp, None, self.payload_pt, bt)
                extra -= self._jp[:, dofs].T @ (pl * (m.opt.gravity - self.a_world))
            out.append(extra)
        return out

    def _tray_body_for(self, side):
        return int(self.m.body(f"{side}_arm_gripper_base").id)

    def tau(self, d):
        mujoco.mj_forward(self.m, d)
        pairs = self.sc.all_pair_barriers(self.m, d, self.ld, self.rd)
        kw = dict(left_ee_body=self.lee, tilt_max_rad=TILT_MAX, cup_cone_facets=CONE_FACETS) if self.tilt else {}
        if self.ff_on:
            ffl, ffr = self.ff_torque(d)
        else:
            ffl = ffr = np.zeros(6)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                  self.qe_ref, self.z, self.qi_ref, self.z, pairs, bias_ego_extra=ffl, **kw)
        # the right arm has no bias_extra argument: add its load as a plain torque (tau = M a + c + ff; c's own error is the same ff)
        ti = ti + ffr
        te = te  # left ff is already inside te through c
        if not ok:
            self.n_infeasible += 1
        self.last_ok = ok
        return te, ti
