"""Low-pass and rate limiting for joint torque commands."""

from __future__ import annotations

import numpy as np


class TorqueCommandFilter:
    """First-order low-pass + per-joint slew rate limit on torques (Nm)."""

    def __init__(
        self,
        num_joints: int,
        lowpass_alpha: float,
        rate_limit_nm_per_sec: np.ndarray,
    ) -> None:
        self._n = int(num_joints)
        self._alpha = float(np.clip(lowpass_alpha, 1e-6, 1.0))
        self._rate = np.asarray(rate_limit_nm_per_sec, dtype=np.float64).reshape(-1)
        if self._rate.shape[0] == 1:
            self._rate = np.full(self._n, float(self._rate[0]), dtype=np.float64)
        if self._rate.shape[0] != self._n:
            raise ValueError(f"rate_limit length must be 1 or {self._n}")
        self._y = np.zeros(self._n, dtype=np.float64)
        self._initialized = False

    def reset(self) -> None:
        self._y[:] = 0.0
        self._initialized = False

    def apply(self, tau_raw: np.ndarray, dt: float) -> np.ndarray:
        filtered, _ = self.apply_with_diagnostics(tau_raw, dt)
        return filtered

    def apply_with_diagnostics(self, tau_raw: np.ndarray, dt: float) -> tuple[np.ndarray, dict[str, object]]:
        """Apply filter and return per-joint rate-limit / low-pass diagnostics."""
        tau_raw = np.asarray(tau_raw, dtype=np.float64).reshape(-1)
        if tau_raw.shape[0] != self._n:
            raise ValueError(f"tau_raw length {tau_raw.shape[0]} != {self._n}")
        dt = max(float(dt), 1e-6)
        if not self._initialized:
            self._y = tau_raw.copy()
            self._initialized = True
            return self._y.copy(), {
                "lowpass_input": tau_raw.tolist(),
                "lowpass_output": self._y.tolist(),
                "rate_limit_nm_per_sec": self._rate.tolist(),
                "delta_max_nm": (self._rate * dt).tolist(),
                "delta_requested_nm": [0.0] * self._n,
                "delta_applied_nm": [0.0] * self._n,
                "torque_rate_clipped": [False] * self._n,
                "torque_rate_nm_per_sec": [0.0] * self._n,
                "torque_rate_fraction": [0.0] * self._n,
            }
        prev_y = self._y.copy()
        x = self._alpha * tau_raw + (1.0 - self._alpha) * self._y
        delta_max = self._rate * dt
        delta_requested = x - self._y
        dx = np.clip(delta_requested, -delta_max, +delta_max)
        self._y = self._y + dx
        rate_nm_per_sec = dx / dt
        rate_fraction = np.abs(rate_nm_per_sec) / np.maximum(self._rate, 1e-12)
        return self._y.copy(), {
            "lowpass_input": x.tolist(),
            "lowpass_output": self._y.tolist(),
            "rate_limit_nm_per_sec": self._rate.tolist(),
            "delta_max_nm": delta_max.tolist(),
            "delta_requested_nm": delta_requested.tolist(),
            "delta_applied_nm": dx.tolist(),
            "torque_rate_clipped": (np.abs(delta_requested - dx) > 1e-12).astype(bool).tolist(),
            "torque_rate_nm_per_sec": rate_nm_per_sec.tolist(),
            "torque_rate_fraction": rate_fraction.tolist(),
            "tau_before_filter": prev_y.tolist(),
        }
