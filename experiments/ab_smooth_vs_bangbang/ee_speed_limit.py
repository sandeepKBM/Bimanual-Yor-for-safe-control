"""Single-step Cartesian end-effector SPEED-LIMIT rows for the box-QP.

Motivation (2026-09-07): the reactive cooperative HOCBF-QP (v4-C*) and the
MPC-HOCBF (v7) both bound arm-arm DISTANCE but neither bounds how fast an
end-effector may move. Measured on the flagship crossing, the right arm's
scripted return-to-home ("park") reaches ~0.79 m/s at the EE and its
collision-proxy keypoints sink ~3.4 cm below the tabletop -- i.e. "the EE
moves too fast" and "the arm sinks into the table" are the same fast
release->park segment. This module adds a HARD per-step cap the QP must
respect: after applying torque `tau` this step, the resulting next-step EE
linear velocity `v_next = J_ee @ (qd0 + dt * Minv @ (tau - bias))` is bounded
`|v_next_axis| <= v_max` on each world axis. `v_next` is affine in `tau`, so
this is 3 axes x 2 signs = 6 linear rows in the SAME `a @ tau <= b` convention
sim_common's HOCBF rows use -- callers just vstack/concatenate them onto their
existing a_ineq/b_ineq. It is the constraint that actually "holds the EE
speed"; the horizon cost (MPC) still shapes the approach, this caps magnitude.

The bound is enforced on the torque that is actually applied this step (for
the MPC, the first control block), matching how the reactive filter enforces
its own rows -- a single-step guarantee, not a horizon one.
"""
from __future__ import annotations

import numpy as np


def ee_velocity_cap_rows(Jp_ee, qd0, M_inv, bias, v_max, dt):
    """Return (a_rows(6, ndof), b_rows(6,)) capping the next-step Cartesian EE
    linear speed at `v_max` [m/s] per world axis, in the `a @ tau <= b` box-QP
    row convention.

    Jp_ee : (3, ndof) linear EE Jacobian on this arm's DOFs
            (sim_common.body_jac_full(...)[0][:, dofadr]).
    qd0   : (ndof,) current joint velocity for this arm.
    M_inv : (ndof, ndof) inverse joint-space inertia for this arm (INCLUDING
            any reflected payload, so the predicted qacc matches the plant).
    bias  : (ndof,) qfrc_bias (+ reflected payload bias) for this arm.
    v_max : scalar speed cap [m/s].
    dt    : integration step [s] (sim_common.DT).

    v_next = Jp_ee @ (qd0 + dt*M_inv@(tau - bias)) = c + G @ tau, with
      G = dt * (Jp_ee @ M_inv),  c = Jp_ee @ qd0 - dt*(Jp_ee @ M_inv @ bias).
    Rows:  G[ax] @ tau <=  v_max - c[ax]   (v_next_ax <=  v_max)
          -G[ax] @ tau <=  v_max + c[ax]   (v_next_ax >= -v_max)
    """
    Jp_ee = np.asarray(Jp_ee, dtype=np.float64)
    G = dt * (Jp_ee @ M_inv)                          # (3, ndof)
    c = Jp_ee @ qd0 - dt * (Jp_ee @ (M_inv @ bias))   # (3,)
    a_rows = np.empty((6, Jp_ee.shape[1]), dtype=np.float64)
    b_rows = np.empty(6, dtype=np.float64)
    for ax in range(3):
        a_rows[2 * ax] = G[ax]
        b_rows[2 * ax] = v_max - c[ax]
        a_rows[2 * ax + 1] = -G[ax]
        b_rows[2 * ax + 1] = v_max + c[ax]
    return a_rows, b_rows
