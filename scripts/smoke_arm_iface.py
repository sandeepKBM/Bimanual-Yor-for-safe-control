#!/usr/bin/env python
"""Stage-1 interface validation (real robot, no grasp yet).

Loads the cup_collect scene, builds an ArmController for the LEFT arm with its
position servos disabled, commands hold_pose, and steps the physics. Validates
the whole control stack (state_adapter + cartesian_impedance + grip_force +
FL gravity feedforward + index bundle) on the real YOR arm:

  * no NaNs,
  * EE holds near its start pose against gravity (drift < 3 cm),
  * commanded torque stays finite and within TAU_MAX.

This is NOT the grasp test — it only proves the arm-control interface works on
the real model. Run:
  MUJOCO_GL=egl not needed (no render). CPU only.
  /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python scripts/smoke_arm_iface.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402
from control.arm_controller import ArmController  # noqa: E402

_FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def main() -> int:
    print("=== Stage-1 arm-control interface smoke (LEFT arm hold) ===")
    model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)

    # start from the home keyframe if present
    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    except Exception:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    arm = ArmController(model, data, "left")
    ee0 = data.xpos[arm.idx.ee_body_id].copy()
    arm.hold_pose()
    arm.set_grip_opening(1.0)  # fingers open, not grasping anything

    max_tau = 0.0
    for k in range(400):
        arm.step()
        mujoco.mj_step(model, data)
        max_tau = max(max_tau, float(np.max(np.abs(data.qfrc_applied[arm.idx.dofadr]))))
        if not np.all(np.isfinite(data.qpos)):
            check("no NaNs during hold", False, f"step {k}")
            break
    else:
        check("no NaNs during hold", True)

    ee1 = data.xpos[arm.idx.ee_body_id].copy()
    drift = float(np.linalg.norm(ee1 - ee0))
    import sim_common as sc
    check("EE holds against gravity (drift < 3cm)", drift < 0.03, f"drift={drift*100:.2f} cm")
    check("commanded torque finite & <= TAU_MAX", np.all(np.isfinite(data.qfrc_applied)) and max_tau <= sc.TAU_MAX + 1e-6,
          f"max|tau|={max_tau:.1f} Nm (TAU_MAX={sc.TAU_MAX})")

    print()
    if _FAILS:
        print(f"RESULT: FAIL ({len(_FAILS)}): {_FAILS}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
