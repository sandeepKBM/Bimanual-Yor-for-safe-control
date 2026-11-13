"""Joint-space impedance (PD) torque law with optional feedforward bias.

Ported from ``../real_Cartpole/controller_core/joint_impedance.py``
and GENERALISED from the hardcoded 6-joint UR5e assumption to an arbitrary
joint count ``n`` (set via ``JointImpedanceConfig.n_joints``). Used for the
posture / grasp-close inner loop of the bimanual arm.

NOTE ON GAINS: the cartpole defaults were tuned for a UR5e. They are NOT valid
for the YOR/Piper arm (different inertia; per-joint torque limit is 100 Nm here,
not 15/4). Callers must pass YOR-appropriate gains; the defaults below are only
a shape-correct placeholder and are re-derived during Stage-2 tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_matrix_or_diag(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (length,):
        arr = np.diag(arr)
    if arr.shape != (length, length):
        raise ValueError(
            f"{name} must have shape ({length},) or ({length},{length}); got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _as_vector(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


@dataclass
class JointImpedanceConfig:
    """Per-joint torque gains and saturation limits for ``n_joints`` joints."""

    n_joints: int = 6
    # Placeholder gains (UR5e shape). Re-derive for the YOR arm before use.
    kp_nm_per_rad: np.ndarray = field(
        default_factory=lambda: np.array([120.0, 120.0, 100.0, 28.0, 24.0, 18.0], dtype=np.float64)
    )
    kd_nm_per_rad_s: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 12.0, 10.0, 3.2, 2.8, 2.0], dtype=np.float64)
    )
    tau_max_nm: np.ndarray = field(
        default_factory=lambda: np.full(6, 100.0, dtype=np.float64)
    )

    def validate(self) -> None:
        n = int(self.n_joints)
        _ = _as_matrix_or_diag("kp_nm_per_rad", self.kp_nm_per_rad, n)
        _ = _as_matrix_or_diag("kd_nm_per_rad_s", self.kd_nm_per_rad_s, n)
        tau_max = _as_vector("tau_max_nm", self.tau_max_nm, n)
        if np.any(tau_max <= 0.0):
            raise ValueError("tau_max_nm must be strictly positive")
        if np.any(np.diag(_as_matrix_or_diag("kp_nm_per_rad", self.kp_nm_per_rad, n)) < 0.0):
            raise ValueError("kp_nm_per_rad must be non-negative")
        if np.any(np.diag(_as_matrix_or_diag("kd_nm_per_rad_s", self.kd_nm_per_rad_s, n)) < 0.0):
            raise ValueError("kd_nm_per_rad_s must be non-negative")


@dataclass
class JointImpedanceOutput:
    tau: np.ndarray
    tau_preclip: np.ndarray
    tau_feedback: np.ndarray
    tau_feedforward: np.ndarray
    q_error: np.ndarray
    qd_error: np.ndarray
    saturated: bool


class JointImpedanceController:
    """Joint-space PD torque law with optional feedforward bias, ``n`` joints."""

    def __init__(self, config: JointImpedanceConfig | None = None) -> None:
        self.cfg = config if config is not None else JointImpedanceConfig()
        self.cfg.validate()
        n = int(self.cfg.n_joints)
        self._n = n
        self._kp = _as_matrix_or_diag("kp_nm_per_rad", self.cfg.kp_nm_per_rad, n)
        self._kd = _as_matrix_or_diag("kd_nm_per_rad_s", self.cfg.kd_nm_per_rad_s, n)
        self._tau_max = np.asarray(self.cfg.tau_max_nm, dtype=np.float64).reshape(n)

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        qd_ref: np.ndarray | None = None,
        tau_feedforward: np.ndarray | None = None,
    ) -> JointImpedanceOutput:
        n = self._n
        q = _as_vector("q", q, n)
        qd = _as_vector("qd", qd, n)
        q_ref = _as_vector("q_ref", q_ref, n)
        qd_ref_arr = np.zeros(n, dtype=np.float64) if qd_ref is None else _as_vector("qd_ref", qd_ref, n)
        tau_ff = np.zeros(n, dtype=np.float64) if tau_feedforward is None else _as_vector("tau_feedforward", tau_feedforward, n)

        q_error = q_ref - q
        qd_error = qd_ref_arr - qd
        tau_fb = self._kp @ q_error + self._kd @ qd_error
        tau_preclip = tau_fb + tau_ff
        tau = np.clip(tau_preclip, -self._tau_max, +self._tau_max)
        saturated = bool(np.any(np.abs(tau_preclip - tau) > 1e-12))
        if not np.all(np.isfinite(tau)):
            raise RuntimeError("JointImpedanceController produced a non-finite torque command")
        return JointImpedanceOutput(
            tau=tau,
            tau_preclip=tau_preclip,
            tau_feedback=tau_fb,
            tau_feedforward=tau_ff,
            q_error=q_error,
            qd_error=qd_error,
            saturated=saturated,
        )
