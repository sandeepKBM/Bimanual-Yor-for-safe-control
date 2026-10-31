"""democ/payload.py -- each hand's share of a box held by BOTH hands, reflected into that arm's joint space.  SIM ONLY.

For FL-MPC's payload hooks (FLMPC.step M_ego_extra/bias_ego_extra for the left arm, M_int_extra/bias_int_extra for the right).
Rigid-grasp assumption: the box moves with each gripper_base body.  Arm `side` carries a fraction `share` of the box:

    J_p = translational Jacobian of the box COM point, attached to that arm's gripper_base body (arm dofs only)
    J_r = rotational Jacobian of that gripper_base body (arm dofs only)
    M_extra    = share * (m J_p^T J_p + J_r^T I_world J_r)       (box inertia about its COM, world frame)
    bias_extra = -share * J_p^T (m g)                            (g = model.opt.gravity, so this is +share*m*|g| lifted)

Same method as sim_common.payload_reflected_terms, except the Jacobian is taken at the box COM (xipos), not its body origin,
so a COM offset is handled, and the mass is split between the hands.  share=0.5 is the symmetric split (a box pinched at both
ends with its COM midway); it is an assumption, not measured from the contact forces.
`mass` defaults to the model's box mass (sim truth).  Pass a belief value (e.g. the upper corner of the mass set) to use one.
"""
from __future__ import annotations

import mujoco
import numpy as np


def arm_payload_terms(m, d, box_id: int, grip_body_id: int, dofadr, share: float = 0.5, mass: float | None = None):
    """-> (M_extra (nv_arm, nv_arm), bias_extra (nv_arm,)) for one arm.  Needs fresh kinematics (mj_forward)."""
    mb = float(m.body_mass[box_id]) if mass is None else float(mass)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, d.xipos[box_id], grip_body_id)
    Jp, Jr = jp[:, dofadr], jr[:, dofadr]
    R = d.ximat[box_id].reshape(3, 3)
    I_w = R @ np.diag(m.body_inertia[box_id]) @ R.T
    if mass is not None and m.body_mass[box_id] > 0:          # scale the rotational inertia with the assumed mass
        I_w = I_w * (mb / float(m.body_mass[box_id]))
    M_extra = share * (mb * Jp.T @ Jp + Jr.T @ I_w @ Jr)
    bias_extra = -share * Jp.T @ (mb * m.opt.gravity)
    return M_extra, bias_extra
