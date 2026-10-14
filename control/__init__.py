"""bimanual.control — real-grasp control stack.

Self-contained OSC / Cartesian-impedance + grip-force controllers and a
programmatic arm-control API, ported from real_Cartpole and generalised for the
bimanual YOR arm. No dependency on ``controller_core`` (not importable here).
"""

from __future__ import annotations

from .box_qp import solve_box_qp
from .cartesian_impedance import (
    CartesianImpedanceConfig,
    CartesianImpedanceController,
    CartesianImpedanceOutput,
)
from .joint_impedance import (
    JointImpedanceConfig,
    JointImpedanceController,
    JointImpedanceOutput,
)
from .osc_core import (
    cartesian_wrench_to_joint_torque,
    orientation_error_vec_wxyz,
    quat_to_rotmat,
    rotmat_to_quat,
)

__all__ = [
    "solve_box_qp",
    "CartesianImpedanceConfig",
    "CartesianImpedanceController",
    "CartesianImpedanceOutput",
    "JointImpedanceConfig",
    "JointImpedanceController",
    "JointImpedanceOutput",
    "cartesian_wrench_to_joint_torque",
    "orientation_error_vec_wxyz",
    "quat_to_rotmat",
    "rotmat_to_quat",
]
