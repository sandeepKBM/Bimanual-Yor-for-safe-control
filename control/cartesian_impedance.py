"""Full 6-DOF Cartesian-impedance controller with box-QP torque allocation.

GENERALISED from ``real_Cartpole/controller_core/torque_task_qp.py``
(``TorqueTaskQPController``) which was single-axis (world-X transport, angular
task pinned to the start pose). Here:

  * the task is ALWAYS full 6-DOF Cartesian (position + orientation),
  * the joint count ``n`` is arbitrary (not hardcoded 6),
  * the orientation moment uses the physically-correct RESTORING sign
    ``m = -kp_rot * e_ori`` (cartpole used ``+kp_rot * e_ori`` and never closed
    an orientation loop — see ``osc_core.orientation_error_vec_wxyz``),
  * the single-axis / transport-axis machinery is removed.

Control law (per tick):
    f      = Kp_pos (p_des - p) + Kd_pos (v_des - v)
    m      = -Kp_rot * e_ori    - Kd_rot * omega          (e_ori = des->cur)
    wrench = [f; m]                                        (world frame)
    tau*   = argmin_tau  (tau - tau_des)' H (tau - tau_des)
             s.t.  tau_lo <= tau <= tau_hi
    tau_des = J^T wrench  - Kd_joint qd  + Kp_post(q_rest-q) - Kd_post qd + g
    H       = 2 (J^T W J + lam I)
Bounds intersect the per-joint torque box (tau_max * headroom) with a
velocity-implied box so the commanded torque cannot drive |qdot| past a limit.

Frame / units: world-frame Jacobian (6, n) stacked [J_pos; J_rot]; positions in
metres, orientation as [w,x,y,z]; torque in Nm. Designed to run at the physics
rate (200 Hz, dt=0.005). Targets are zero-order-held between updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .box_qp import solve_box_qp
from .osc_core import orientation_error_vec_wxyz


@dataclass
class CartesianImpedanceConfig:
    """Gains and limits for the 6-DOF Cartesian-impedance QP controller."""

    n_joints: int = 6
    # Task-space stiffness/damping in OSC (unit-mass) acceleration units:
    # a_des = kp*(x_des-x) + kd*(xd_des-xd), so kp ~ omega^2 and critical
    # damping is kd = 2*sqrt(kp). Defaults below are the Stage-2 values tuned on
    # the YOR arm (kp=225 -> ~2.4 Hz): 3.7 mm step error, no overshoot, ~7 Nm
    # peak torque (vs 100 Nm saturation for the un-shaped UR5e-seed gains).
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([225.0, 225.0, 225.0], dtype=np.float64))
    kd_pos: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0, 30.0], dtype=np.float64))
    kp_rot: float = 80.0
    kd_rot: float = 18.0
    # Joint-space regularisers (nullspace posture + joint damping).
    kp_posture: float = 1.0
    kd_posture: float = 2.0
    kd_joint: float = 0.5
    posture_regularization: float = 0.35
    # Torque box (per joint) and headroom fraction.
    tau_max_nm: np.ndarray = field(default_factory=lambda: np.full(6, 100.0, dtype=np.float64))
    torque_headroom: float = 1.0
    # Velocity-implied torque bound (optional). Off by default: OSC inertia
    # shaping already keeps torques low/critically-damped, and the bound was
    # only needed to tame the un-shaped J^T law. Available for extra safety.
    enforce_velocity_torque_bounds: bool = False
    max_joint_velocity_radps: float = 2.5
    velocity_torque_coupling_kp: np.ndarray = field(
        default_factory=lambda: np.full(6, 8.0, dtype=np.float64)
    )
    velocity_torque_coupling_kd: np.ndarray = field(
        default_factory=lambda: np.full(6, 4.0, dtype=np.float64)
    )
    # Singular-direction wrench de-weighting (0 disables).
    jacobian_singular_cond_max: float = 0.0
    # Operational-space inertia shaping (Khatib OSC). When True and a
    # ``mass_matrix`` is supplied in the state, the task gains act on a unit-mass
    # decoupled task (F = Lambda * a_des, Lambda = (J M^-1 J^T)^-1), and posture
    # is projected into the dynamically-consistent null space. This is what makes
    # per-axis gains physically meaningful and the law well-conditioned; a pure
    # J^T impedance (use_osc_inertia=False) is kept as a fallback.
    use_osc_inertia: bool = True
    osc_lambda_damping: float = 1e-3   # Tikhonov damping on (J M^-1 J^T) for singularities

    def validate(self) -> None:
        n = int(self.n_joints)
        for name, v, length in (
            ("kp_pos", self.kp_pos, 3),
            ("kd_pos", self.kd_pos, 3),
            ("tau_max_nm", self.tau_max_nm, n),
            ("velocity_torque_coupling_kp", self.velocity_torque_coupling_kp, n),
            ("velocity_torque_coupling_kd", self.velocity_torque_coupling_kd, n),
        ):
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.shape[0] != length:
                raise ValueError(f"{name} must have length {length}; got {arr.shape}")
        if np.any(np.asarray(self.tau_max_nm, dtype=np.float64) <= 0.0):
            raise ValueError("tau_max_nm must be strictly positive")


@dataclass
class CartesianImpedanceOutput:
    tau: np.ndarray            # box-QP result, clipped to tau_max
    tau_preclip: np.ndarray    # box-QP result before final clip
    wrench: np.ndarray         # [fx,fy,fz,mx,my,mz] world frame
    tau_task: np.ndarray       # J^T wrench (nominal)
    tau_posture: np.ndarray
    tau_damping: np.ndarray
    tau_gravity: np.ndarray
    pos_error: np.ndarray      # p_des - p
    orientation_error_vec: np.ndarray  # e_ori (des->cur)
    orientation_error_norm: float
    jacobian_cond: float
    saturated: np.ndarray
    task_feasible: bool


def _velocity_implied_torque_bounds(
    q: np.ndarray,
    qd: np.ndarray,
    q_rest: np.ndarray,
    *,
    kp: np.ndarray,
    kd: np.ndarray,
    qd_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear PD map ``qdot_cmd = qd + (tau - kp*e)/kd`` bounded by ``|qdot_cmd|<=qd_max``."""
    n = q.shape[0]
    qd_lim = abs(float(qd_max))
    bias = kp * (q_rest - q)
    lo = np.full(n, -np.inf, dtype=np.float64)
    hi = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        kdi = float(kd[i])
        if kdi < 1.0e-9:
            continue
        lo[i] = float(bias[i] + kdi * (-qd_lim - qd[i]))
        hi[i] = float(bias[i] + kdi * (qd_lim - qd[i]))
        if lo[i] > hi[i]:
            mid = 0.5 * (lo[i] + hi[i])
            lo[i] = mid
            hi[i] = mid
    return lo, hi


class CartesianImpedanceController:
    """6-DOF Cartesian impedance with box-QP torque allocation, ``n`` joints."""

    def __init__(self, config: CartesianImpedanceConfig | None = None) -> None:
        self.cfg = config if config is not None else CartesianImpedanceConfig()
        self.cfg.validate()
        self._n = int(self.cfg.n_joints)
        self._q_rest = np.zeros(self._n, dtype=np.float64)
        self._p0 = np.zeros(3, dtype=np.float64)
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._hold_ref_init = False
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset_from_state(self, state: dict[str, Any]) -> None:
        self._p0 = np.asarray(state["ee_pos"], dtype=np.float64).reshape(3).copy()
        self._quat0 = np.asarray(state["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(state["q"], dtype=np.float64).reshape(self._n).copy()
        self._hold_ref_init = False
        self._initialized = True

    def _desired_pose(self, state: dict[str, Any], p: np.ndarray, quat: np.ndarray):
        if bool(state.get("hold_current_pose", False)):
            if not self._hold_ref_init:
                self._p0 = p.copy()
                self._quat0 = quat.copy()
                self._q_rest = np.asarray(state["q"], dtype=np.float64).reshape(self._n).copy()
                self._hold_ref_init = True
            return self._p0.copy(), np.zeros(3, dtype=np.float64), self._quat0.copy()
        self._hold_ref_init = False
        p_des = (
            np.asarray(state["target_ee_pos"], dtype=np.float64).reshape(3).copy()
            if state.get("target_ee_pos") is not None
            else self._p0.copy()
        )
        v_des = (
            np.asarray(state["target_ee_vel"], dtype=np.float64).reshape(3).copy()
            if state.get("target_ee_vel") is not None
            else np.zeros(3, dtype=np.float64)
        )
        quat_ref = (
            np.asarray(state["target_ee_quat"], dtype=np.float64).reshape(4).copy()
            if state.get("target_ee_quat") is not None
            else self._quat0.copy()
        )
        return p_des, v_des, quat_ref

    def compute(self, state: dict[str, Any]) -> CartesianImpedanceOutput:
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        n = self._n
        cfg = self.cfg
        q = np.asarray(state["q"], dtype=np.float64).reshape(n)
        qd = np.asarray(state["qd"], dtype=np.float64).reshape(n)
        p = np.asarray(state["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(state["ee_quat"], dtype=np.float64).reshape(4)
        v = np.asarray(state["ee_lin_vel"], dtype=np.float64).reshape(3)
        omega = np.asarray(state["ee_ang_vel"], dtype=np.float64).reshape(3)
        jac = np.asarray(state["jacobian"], dtype=np.float64).reshape(6, n)

        p_des, v_des, quat_ref = self._desired_pose(state, p, quat)
        kp_pos = np.asarray(cfg.kp_pos, dtype=np.float64).reshape(3)
        kd_pos = np.asarray(cfg.kd_pos, dtype=np.float64).reshape(3)

        pos_err = p_des - p
        e_ori = orientation_error_vec_wxyz(quat_ref, quat)   # des -> cur
        # Desired task-space acceleration (unit-mass PD). Restoring signs:
        # position pulls toward p_des; orientation uses -kp_rot*e_ori.
        a_lin = kp_pos * pos_err + kd_pos * (v_des - v)
        a_rot = -cfg.kp_rot * e_ori - cfg.kd_rot * omega
        a_des = np.concatenate([a_lin, a_rot])

        cond = float(np.linalg.cond(jac)) if jac.size else 1.0
        j_t = jac.T
        gravity = (
            np.asarray(state["gravity_torque"], dtype=np.float64).reshape(n)
            if state.get("gravity_torque") is not None
            else np.zeros(n, dtype=np.float64)
        )
        tau_damping = -cfg.kd_joint * qd
        tau_posture_raw = cfg.kp_posture * (self._q_rest - q) - cfg.kd_posture * qd

        M = state.get("mass_matrix")
        if cfg.use_osc_inertia and M is not None:
            # Khatib OSC: F = Lambda a_des, Lambda = (J M^-1 J^T)^-1 (damped);
            # posture projected into the dynamically-consistent null space.
            M = np.asarray(M, dtype=np.float64).reshape(n, n)
            Minv = np.linalg.inv(M + 1e-9 * np.eye(n))
            JMinv = jac @ Minv                       # 6 x n
            Lam_inv = JMinv @ j_t                     # 6 x 6
            Lam = np.linalg.inv(Lam_inv + cfg.osc_lambda_damping * np.eye(6))
            wrench = Lam @ a_des                       # task wrench (force+moment)
            tau_task = j_t @ wrench
            Jbar = Minv @ j_t @ Lam                   # n x 6, dyn-consistent inverse
            nullspace = np.eye(n) - j_t @ Jbar.T       # n x n
            tau_posture = nullspace @ tau_posture_raw
        else:
            # Fallback: pure J^T impedance (gains act as task stiffness/damping).
            wrench = a_des
            tau_task = j_t @ wrench
            tau_posture = tau_posture_raw

        tau_des = tau_task + tau_damping + tau_posture + gravity

        # QP objective: keep tau close to tau_des subject to the box below.
        task_weights = np.diag(
            [
                max(kp_pos[0], 1e-6), max(kp_pos[1], 1e-6), max(kp_pos[2], 1e-6),
                max(cfg.kp_rot, 1e-6), max(cfg.kp_rot, 1e-6), max(cfg.kp_rot, 1e-6),
            ]
        )
        lam = float(max(cfg.posture_regularization, 1e-6))
        hessian = 2.0 * (j_t @ task_weights @ jac + lam * np.eye(n, dtype=np.float64))
        linear = -hessian @ tau_des

        tau_limit = np.asarray(cfg.tau_max_nm, dtype=np.float64).reshape(n)
        headroom = float(np.clip(cfg.torque_headroom, 0.0, 1.0))
        tau_hi = tau_limit * max(headroom, 1e-6)
        tau_lo = -tau_hi
        if cfg.enforce_velocity_torque_bounds:
            vlo, vhi = _velocity_implied_torque_bounds(
                q, qd, self._q_rest,
                kp=np.asarray(cfg.velocity_torque_coupling_kp, dtype=np.float64).reshape(n),
                kd=np.asarray(cfg.velocity_torque_coupling_kd, dtype=np.float64).reshape(n),
                qd_max=float(cfg.max_joint_velocity_radps),
            )
            tau_lo = np.maximum(tau_lo, vlo)
            tau_hi = np.minimum(tau_hi, vhi)
            bad = tau_lo > tau_hi
            if np.any(bad):
                mid = 0.5 * (tau_lo + tau_hi)
                tau_lo = np.where(bad, mid, tau_lo)
                tau_hi = np.where(bad, mid, tau_hi)

        tau_qp = solve_box_qp(hessian, linear, tau_lo, tau_hi)
        tau_clipped = np.clip(tau_qp, -tau_limit, +tau_limit)
        saturated = np.abs(tau_qp - tau_clipped) > 1e-10
        task_feasible = bool(np.all(np.abs(tau_qp) <= tau_hi + 1e-9))

        return CartesianImpedanceOutput(
            tau=tau_clipped,
            tau_preclip=tau_qp,
            wrench=wrench,
            tau_task=tau_task,
            tau_posture=tau_posture,
            tau_damping=tau_damping,
            tau_gravity=gravity,
            pos_error=pos_err,
            orientation_error_vec=e_ori,
            orientation_error_norm=float(np.linalg.norm(e_ori)),
            jacobian_cond=cond,
            saturated=saturated.astype(np.float64),
            task_feasible=task_feasible,
        )
