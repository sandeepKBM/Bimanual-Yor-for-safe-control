#!/usr/bin/env python
"""Stage-0 smoke test for the ported OSC control core (no MuJoCo, pure math).

Asserts ONE class of thing per block:
  1. cartesian_wrench_to_joint_torque: zero wrench -> zero torque; a +Z force
     with J=I maps to +Z joint torque (sign(tau)=sign(J^T w)).
  2. CartesianImpedanceController: no error -> ~zero torque; +X position error
     -> +X restoring torque; +theta_z orientation error -> RESTORING (-z) moment
     and joint torque (the sign fix vs the cartpole source).
  3. velocity-implied torque bounds actually clip when enabled.
  4. JointImpedanceController: q_ref==q -> zero; q_ref>q -> positive restoring.

Run:
  /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python scripts/smoke_osc_core.py
Exits nonzero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from control.cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    CartesianImpedanceController,
)
from control.joint_impedance import JointImpedanceConfig, JointImpedanceController  # noqa: E402
from control.osc_core import (  # noqa: E402
    cartesian_wrench_to_joint_torque,
    orientation_error_vec_wxyz,
)

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def quat_about_z(theta: float) -> np.ndarray:
    return np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], dtype=np.float64)


def test_wrench_to_torque() -> None:
    print("1) cartesian_wrench_to_joint_torque")
    J = np.eye(6)
    tau0, sat0 = cartesian_wrench_to_joint_torque(np.zeros(6), J)
    check("zero wrench -> zero torque", np.allclose(tau0, 0.0) and not sat0)

    w = np.array([0, 0, 5.0, 0, 0, 0])  # +Z force
    tau, _ = cartesian_wrench_to_joint_torque(w, J)
    check("+Z force -> +Z joint torque (J=I)", np.isclose(tau[2], 5.0) and np.allclose(np.delete(tau, 2), 0.0))

    # sign matches J^T w for a nontrivial J
    rng = np.random.default_rng(0)
    Jr = rng.standard_normal((6, 6))
    wr = rng.standard_normal(6)
    tau_r, _ = cartesian_wrench_to_joint_torque(wr, Jr)
    check("tau == J^T w (general)", np.allclose(tau_r, Jr.T @ wr))

    # saturation flag
    tau_s, sat_s = cartesian_wrench_to_joint_torque(np.array([0, 0, 1000.0, 0, 0, 0]), J, tau_max=np.full(6, 10.0))
    check("tau_max saturates & flags", np.isclose(tau_s[2], 10.0) and sat_s)


def _base_state(n: int = 6) -> dict:
    return {
        "q": np.zeros(n),
        "qd": np.zeros(n),
        "ee_pos": np.array([0.5, 0.0, 0.5]),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_lin_vel": np.zeros(3),
        "ee_ang_vel": np.zeros(3),
        "jacobian": np.eye(6),
    }


def test_cartesian_impedance() -> None:
    print("2) CartesianImpedanceController (J=I, decoupled sign/magnitude checks)")
    # No posture/damping/vel-bounds so tau == J^T wrench exactly.
    cfg = CartesianImpedanceConfig(
        kp_pos=np.array([400.0, 400.0, 400.0]),
        kd_pos=np.array([0.0, 0.0, 0.0]),
        kp_rot=30.0, kd_rot=0.0,
        kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        posture_regularization=1e-3,
        enforce_velocity_torque_bounds=False,
    )
    ctl = CartesianImpedanceController(cfg)
    st = _base_state()
    ctl.reset_from_state(st)

    out0 = ctl.compute(st)
    check("no error -> ~zero torque", np.allclose(out0.tau, 0.0, atol=1e-6), f"max|tau|={np.max(np.abs(out0.tau)):.2e}")

    st_x = _base_state()
    st_x["target_ee_pos"] = np.array([0.51, 0.0, 0.5])  # +1cm in x
    out_x = ctl.compute(st_x)
    check("+X pos error -> +X restoring torque", out_x.tau[0] > 0 and np.isclose(out_x.tau[0], 400.0 * 0.01, rtol=1e-3),
          f"tau_x={out_x.tau[0]:.3f}")

    # Orientation restoring sign: current is +theta about z from desired(identity).
    theta = 0.1
    st_r = _base_state()
    st_r["ee_quat"] = quat_about_z(theta)
    out_r = ctl.compute(st_r)
    e = orientation_error_vec_wxyz(np.array([1.0, 0, 0, 0]), quat_about_z(theta))
    check("orientation error e_z > 0 for +theta_z", out_r.orientation_error_vec[2] > 0 and e[2] > 0,
          f"e_z={out_r.orientation_error_vec[2]:.3f}")
    check("RESTORING moment m_z < 0 (sign fix)", out_r.wrench[5] < 0, f"m_z={out_r.wrench[5]:.3f}")
    check("restoring joint torque tau_z < 0 (J=I)", out_r.tau[5] < 0, f"tau_z={out_r.tau[5]:.3f}")


def test_velocity_bounds_clip() -> None:
    print("3) velocity-implied torque bounds")
    cfg = CartesianImpedanceConfig(
        kp_pos=np.array([4000.0, 4000.0, 4000.0]), kd_pos=np.zeros(3),
        kp_rot=1.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        enforce_velocity_torque_bounds=True,
        max_joint_velocity_radps=2.5,
        velocity_torque_coupling_kd=np.full(6, 4.0),  # bound ~ +/-10 Nm at rest
    )
    ctl = CartesianImpedanceController(cfg)
    st = _base_state()
    ctl.reset_from_state(st)
    st_big = _base_state()
    st_big["target_ee_pos"] = np.array([0.6, 0.0, 0.5])  # 10cm -> huge nominal tau (400 Nm)
    out = ctl.compute(st_big)
    check("large error torque clipped below tau_max by vel bound", abs(out.tau[0]) <= 10.0 + 1e-6,
          f"tau_x={out.tau[0]:.3f} (<=10)")


def test_joint_impedance() -> None:
    print("4) JointImpedanceController")
    cfg = JointImpedanceConfig(
        n_joints=6,
        kp_nm_per_rad=np.full(6, 50.0),
        kd_nm_per_rad_s=np.full(6, 5.0),
        tau_max_nm=np.full(6, 100.0),
    )
    ctl = JointImpedanceController(cfg)
    q = np.zeros(6)
    out0 = ctl.compute(q, np.zeros(6), q)
    check("q_ref==q -> zero torque", np.allclose(out0.tau, 0.0))
    q_ref = np.full(6, 0.1)
    out1 = ctl.compute(q, np.zeros(6), q_ref)
    check("q_ref>q -> positive restoring torque", np.all(out1.tau > 0) and np.allclose(out1.tau, 50.0 * 0.1),
          f"tau={out1.tau[0]:.3f}")


def main() -> int:
    print("=== Stage-0 OSC core smoke test ===")
    test_wrench_to_torque()
    test_cartesian_impedance()
    test_velocity_bounds_clip()
    test_joint_impedance()
    print()
    if _FAILS:
        print(f"RESULT: FAIL ({len(_FAILS)} failed): {_FAILS}")
        return 1
    print("RESULT: PASS (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
