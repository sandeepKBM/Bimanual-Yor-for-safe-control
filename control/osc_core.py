"""Operational-space control core: J^T wrench mapping + quaternion/orientation
helpers.

The quaternion/orientation helpers are ported (behaviour-preserving) from
``/common/users/ss5772/real_Cartpole/controller_core/kinematics_utils.py`` —
battle-tested pure math, reused verbatim. The force adapter there is
single-axis (``F=[Fx,0,0]``, angular task zeroed "for future 6D extensions");
here it is GENERALISED to a full 6-DOF task wrench ``[fx,fy,fz,mx,my,mz]`` and
an arbitrary joint count ``n`` (the YOR arm is 6-DOF per arm, but nothing here
hardcodes 6).

Frame convention: all Jacobians and wrenches are WORLD-frame. ``jacobian`` is
the stacked ``[J_pos; J_rot]`` (6, n) so that ``[v; omega] = J @ qd`` with the
first three rows linear — matching MuJoCo ``mj_jac`` output stacked that way.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# J^T wrench -> joint torque (generalised OSC force mapping)
# --------------------------------------------------------------------------
def cartesian_wrench_to_joint_torque(
    wrench: np.ndarray,
    jacobian: np.ndarray,
    *,
    qd: np.ndarray | None = None,
    kd_joint: np.ndarray | float = 0.0,
    gravity_torque: np.ndarray | None = None,
    tau_max: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Map a world-frame 6-vector task wrench to joint torques via ``tau = J^T w``.

    Optional joint-space damping ``-kd_joint * qd`` and feedforward
    ``+gravity_torque`` are added afterward. ``tau_max`` (shape ``(n,)``)
    saturates the result; returns ``(tau, saturated)``.

    This is the plain OSC force-transfer identity (no QP allocation — see
    ``cartesian_impedance.CartesianImpedanceController`` for the box-QP
    version). Kept as a small, directly-testable primitive.
    """
    j = np.asarray(jacobian, dtype=np.float64)
    if j.ndim != 2 or j.shape[0] != 6:
        raise ValueError(f"jacobian must be shape (6, n); got {j.shape}")
    n = j.shape[1]
    w = np.asarray(wrench, dtype=np.float64).reshape(-1)
    if w.shape[0] != 6:
        raise ValueError(f"wrench must have length 6; got {w.shape}")

    tau = j.T @ w

    if qd is not None:
        qd_arr = np.asarray(qd, dtype=np.float64).reshape(-1)
        if qd_arr.shape[0] != n:
            raise ValueError(f"qd must have length {n}; got {qd_arr.shape}")
        kd = np.asarray(kd_joint, dtype=np.float64).reshape(-1)
        if kd.shape[0] == 1:
            kd = np.full(n, float(kd[0]), dtype=np.float64)
        if kd.shape[0] != n:
            raise ValueError(f"kd_joint must be scalar or length {n}; got {kd.shape}")
        tau = tau - kd * qd_arr

    if gravity_torque is not None:
        g = np.asarray(gravity_torque, dtype=np.float64).reshape(-1)
        if g.shape[0] != n:
            raise ValueError(f"gravity_torque length must be {n}; got {g.shape}")
        tau = tau + g

    saturated = False
    if tau_max is not None:
        tau_max_arr = np.asarray(tau_max, dtype=np.float64).reshape(-1)
        if tau_max_arr.shape[0] != n:
            raise ValueError(f"tau_max length must be {n}; got {tau_max_arr.shape}")
        tau_clipped = np.clip(tau, -tau_max_arr, +tau_max_arr)
        saturated = bool(np.any(np.abs(tau - tau_clipped) > 1e-12))
        tau = tau_clipped

    return tau, saturated


# --------------------------------------------------------------------------
# Quaternion / orientation helpers (ported verbatim from real_Cartpole)
# --------------------------------------------------------------------------
def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quaternion must have length 4; got {q.shape}")
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a ``[w, x, y, z]`` quaternion."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Hamilton product with ``[w, x, y, z]`` storage."""
    w1, x1, y1, z1 = np.asarray(q_left, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q_right, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def orientation_error_vec_wxyz(quat_des: np.ndarray, quat_cur: np.ndarray) -> np.ndarray:
    """3-vector orientation error ``e = 2 * vec(conj(q_des) * q_cur)``.

    ``e`` is the small-angle rotation taking the DESIRED frame to the CURRENT
    frame (i.e. how far current is *past* desired). A **restoring** moment is
    therefore ``-kp * e`` — see ``cartesian_impedance`` (this sign is the one
    correctness fix vs the ported cartpole ``torque_task_qp``, which used
    ``+kp*e`` and never closed an orientation loop).
    """
    qd = quat_normalize_wxyz(quat_des)
    qc = quat_normalize_wxyz(quat_cur)
    q_err = quat_multiply_wxyz(quat_conj_wxyz(qd), qc)
    q_err = quat_normalize_wxyz(q_err)
    if q_err[0] < 0.0:
        q_err = -q_err
    return 2.0 * q_err[1:4]
