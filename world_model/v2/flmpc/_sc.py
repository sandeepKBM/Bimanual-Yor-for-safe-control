"""The 5 sim_common items FL-MPC uses, copied unchanged from experiments/ab_smooth_vs_bangbang/sim_common.py (2026-11-03), so FL-MPC
no longer needs sim_common (Piper-era scene helpers) or the real_Cartpole sys.path.  Values are the originals; NERO callers pass their own
DT / tau_max_vec (TAU_MAX = 100 is the Piper-class YOR default, NOT the NERO limit)."""
import numpy as np
import mujoco

from .qp.constrained_box_qp import solve_constrained_box_qp  # noqa: F401  (sim_common re-exported it from controller_core)

DT = 0.005              # = model timestep (scenes/scene_yor_canonical.xml uses 0.005); control runs at the physics rate
TAU_MAX = 100.0         # [Nm] MODEL: UNMODIFIED YOR GitHub model, piper URDF effort=100 == MJCF piper-class forcerange +-100


def body_jac_full(model, data, body_id):
    """Translational + rotational Jacobian (each 3 x nv) of a body's origin,
    full nv width (caller restricts columns)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    return jacp, jacr


def full_mass_matrix(model, data):
    m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, m, data.qM)
    return m
