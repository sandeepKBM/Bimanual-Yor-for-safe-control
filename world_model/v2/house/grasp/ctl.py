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

import os

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
        # JOINT-FRICTION FEED-FORWARD (default ON in the friction rig).  Every arm joint has a dry friction of 0.3 Nm (MJCF frictionloss) which
        # the FL-MPC (tau = M a + c, c without friction) does not model; with the light wrist inertia (0.008-0.027 kg m^2) its tracking stiffness
        # is only ~M*140 1/s^2 => a 0.3 Nm friction stalls the wrist 0.13 rad off its reference (measured: 85 % of a 0.15 rad step missing).
        # tau_f = f_comp * frictionloss * sat((q_ref - q)/fric_e0): a smooth Coulomb-friction compensation in the direction of the commanded
        # motion (a model constant the robot owns, identified once; no extra sensing).
        self.fric_ff = os.environ.get("BIMANUAL_FRIC_FF", "1") == "1"
        self.fric_e0 = float(os.environ.get("BIMANUAL_FRIC_E0", "0.01"))
        self.fric_comp = float(os.environ.get("BIMANUAL_FRIC_COMP", "1.0"))
        # JOINT-SPACE INTEGRATOR on the reference handed to the FL-MPC (default OFF: it hunts with the stick-slip friction and wound up against the arm-arm rows; measured, see smoke --help): removes the residual gravity /
        # unmodelled-load offset of the shoulder joints (0.02-0.035 rad measured) that the friction compensation does not touch.
        self.jint_ki = float(os.environ.get("BIMANUAL_JINT_KI", "0.0"))
        self.jint_lim = 0.05
        self.jint = [np.zeros(6), np.zeros(6)]
        self.fric_L = self.m.dof_frictionloss[self.ld].copy()
        self.fric_R = self.m.dof_frictionloss[self.rd].copy()
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
        qeR, qiR = self.qe_ref, self.qi_ref
        if self.jint_ki > 0.0:
            dt = self.fl.DT
            for k, (q, ref) in enumerate(((d.qpos[self.lq], self.qe_ref[0]), (d.qpos[self.rq], self.qi_ref[0]))):
                e = ref - q
                self.jint[k] = np.clip(self.jint[k] + self.jint_ki * dt * np.where(np.abs(e) < 0.04, e, 0.0), -self.jint_lim, self.jint_lim)
            qeR, qiR = self.qe_ref + self.jint[0], self.qi_ref + self.jint[1]
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                  qeR, self.z, qiR, self.z, pairs, bias_ego_extra=ffl, **kw)
        # the right arm has no bias_extra argument: add its load as a plain torque (tau = M a + c + ff; c's own error is the same ff)
        ti = ti + ffr
        te = te  # left ff is already inside te through c
        if self.fric_ff:
            te = te + self.fric_comp * self.fric_L * np.clip((self.qe_ref[0] - d.qpos[self.lq]) / self.fric_e0, -1.0, 1.0)
            ti = ti + self.fric_comp * self.fric_R * np.clip((self.qi_ref[0] - d.qpos[self.rq]) / self.fric_e0, -1.0, 1.0)
        if not ok:
            self.n_infeasible += 1
        self.last_ok = ok
        return te, ti
