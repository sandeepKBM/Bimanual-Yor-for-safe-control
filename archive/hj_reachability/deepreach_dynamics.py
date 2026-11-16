"""``BimanualArmArmAvoidance`` -- torch ``dynamics.Dynamics`` port of
``dynamics.BimanualPairAvoidance`` for the DeepReach trainer.

CANONICAL COPY: this file is byte-for-byte (module-docstring aside) what was
added to ``/common/users/ss5772/DeepReach/deepreach/dynamics/dynamics.py`` on
git branch ``bimanual-armarm`` (see that fork's ``BimanualArmArmAvoidance``
class). Kept here too so the class is reviewable/importable from this repo
without needing the DeepReach fork checked out, and so the two copies can be
diffed to catch drift.

Interface mirrors ``UR5eGraspBoundary`` (state bounds loaded from
``dataset_description.json`` under ``data_root``, ``loss_type="brt_hjivi"``,
``deepreach_model`` passthrough) but -- unlike ``UR5eGraspBoundary``, which
has no analytic dynamics -- implements the REAL analytic
``dsdt``/``hamiltonian``/``optimal_control``/``optimal_disturbance`` from
``dynamics.BimanualPairAvoidance``'s differential-game formulation, enabling
future ``training_objective: hj_pde`` (self-supervised BRT) training, not
just ``supervised_bootstrap``.

Key design point -- ∇l is NOT autograd-through-MuJoCo (impossible: MuJoCo's
FK is not a torch op). Instead ``boundary_fn`` is a custom
``torch.autograd.Function`` (``_MuJoCoBoundaryFn`` below) whose forward calls
``mujoco_fk.MuJoCoFKOracle.closest_pair_info`` per row (gives both l(x) and
its EXACT analytic gradient via the geometric Jacobian at the current argmin
sphere pair -- the same technique as
``experiments/ab_smooth_vs_bangbang/sim_common.py``'s
``hdot = n . (v_ego - v_int)``, specialized to positions instead of
velocities: ``d(dist)/dq = n^T @ J_point(q)``) and whose backward simply
returns that precomputed analytic gradient scaled by the upstream cotangent.
This was verified against central finite differences to ~1e-10 abs error
(see hj_reachability/README.md). A finite-difference fallback would also
work (bump each of the 12 joint angles +-eps and re-run FK) but is ~24x more
MuJoCo calls per boundary_fn evaluation for no accuracy benefit here, so is
documented but not implemented.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import torch


def _load_module_from_path(module_name: str, file_path: str):
    """Load a module by absolute file path under a private name, WITHOUT
    adding its directory to sys.path. Needed because this repo's
    ``dynamics.py`` (plain module) and the DeepReach fork's ``dynamics/``
    (package) both want the top-level name "dynamics" -- if both directories
    were ever on sys.path at once, whichever gets cached into
    sys.modules['dynamics'] first would silently poison the other's import
    (observed: importing this module standalone with hj_reachability/ ahead
    of the DeepReach repo on sys.path raised ``'dynamics' is not a
    package``, because plain sys.path.insert + ``import`` had already bound
    sys.modules['dynamics'] to the wrong one)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses._is_type() needs sys.modules[cls.__module__] populated
    spec.loader.exec_module(module)
    return module


_HJ_REACHABILITY_ROOT = os.path.dirname(os.path.abspath(__file__))
mujoco_fk = _load_module_from_path("bimanual_mujoco_fk", os.path.join(_HJ_REACHABILITY_ROOT, "mujoco_fk.py"))

try:
    # When copied into DeepReach/deepreach/dynamics/dynamics.py, `Dynamics`
    # is already defined earlier in the same module -- this import is only
    # needed when this file is used standalone (e.g. smoke-tested from this
    # repo without the DeepReach fork's dynamics/dynamics.py already loaded).
    from dynamics.dynamics import Dynamics, _load_bounds_from_dataset_description  # type: ignore
except ImportError:
    _DEEPREACH_ROOT = "/common/users/ss5772/DeepReach/deepreach"
    if _DEEPREACH_ROOT not in sys.path:
        sys.path.insert(0, _DEEPREACH_ROOT)  # needed for that file's own `from utils import diff_operators`
    _deepreach_dynamics_mod = _load_module_from_path(
        "bimanual_deepreach_dynamics_base", os.path.join(_DEEPREACH_ROOT, "dynamics", "dynamics.py")
    )
    Dynamics = _deepreach_dynamics_mod.Dynamics
    _load_bounds_from_dataset_description = _deepreach_dynamics_mod._load_bounds_from_dataset_description


N_JOINTS_PER_ARM = mujoco_fk.N_JOINTS_PER_ARM
STATE_DIM = 2 * N_JOINTS_PER_ARM
STATE_FEATURE_ORDER = [f"q_ego_{i}" for i in range(N_JOINTS_PER_ARM)] + [
    f"q_int_{i}" for i in range(N_JOINTS_PER_ARM)
]


class _MuJoCoBoundaryFn(torch.autograd.Function):
    """l(x) = h at the worst (min-h) FAITHFUL ~25-keypoint-per-arm proxy
    pair (mujoco_fk.MuJoCoFKOracle.closest_pair_info's `.h`, per-pair radius
    baked in -- NOT `dist - COMBINED_RADIUS`, since the faithful proxy's
    combined radius varies per keypoint pair; see mujoco_fk.py's 2026-09-06
    upgrade notes), with an exact analytic backward computed via the
    offset-aware geometric Jacobian at that argmin pair (see module
    docstring)."""

    @staticmethod
    def forward(ctx, state: torch.Tensor) -> torch.Tensor:
        orig_shape = state.shape
        flat = state.detach().to(dtype=torch.float64, device="cpu").numpy().reshape(-1, STATE_DIM)
        n = flat.shape[0]
        oracle = mujoco_fk.get_oracle()
        l_vals = np.empty(n, dtype=np.float64)
        grads = np.empty((n, STATE_DIM), dtype=np.float64)
        for i in range(n):
            q_ego = flat[i, :N_JOINTS_PER_ARM]
            q_int = flat[i, N_JOINTS_PER_ARM:]
            info = oracle.closest_pair_info(q_ego, q_int)
            l_vals[i] = info.h
            grads[i, :N_JOINTS_PER_ARM] = info.dl_dq_ego
            grads[i, N_JOINTS_PER_ARM:] = info.dl_dq_int
        ctx.save_for_backward(torch.from_numpy(grads))
        ctx.orig_shape = orig_shape
        out = torch.from_numpy(l_vals).to(dtype=state.dtype, device=state.device)
        return out.reshape(orig_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grads,) = ctx.saved_tensors
        grads = grads.to(dtype=grad_output.dtype, device=grad_output.device)
        grad_input = grad_output.reshape(-1, 1) * grads
        return grad_input.reshape(ctx.orig_shape)


class BimanualArmArmAvoidance(Dynamics):
    """Piper-PAIR bimanual arm-arm avoidance (avoid-only BRT).

    state = [q_ego(6), q_int(6)] (rad); control = q_ego_dot (ego, minimizing);
    disturbance = q_int_dot (intruder, worst-case adversary, maximizing).
    l(x) via REAL Piper FK (mujoco_fk.MuJoCoFKOracle, loaded by file path from
    this repo's mujoco_fk.py -- so it automatically tracks whatever SCENE that
    module points at; currently scenes/scene_yor_canonical.xml) -- see module
    docstring for ∇l.
    """

    def __init__(self, data_root: str, set_mode: str = "avoid", deepreach_model: str = "vanilla",
                 u_bar_ego: float = 1.5, u_bar_int: float = 1.5):
        loaded = _load_bounds_from_dataset_description(data_root, zero_centered_max_abs=False)
        if loaded is None:
            raise RuntimeError(
                f"Could not load state bounds from dataset_description.json under {data_root} "
                "(run hj_reachability/export_dataset.py first)."
            )
        state_dim, state_mean, state_var = loaded
        assert state_dim == STATE_DIM, (state_dim, STATE_DIM)
        self.data_root = data_root
        self.u_bar_ego = float(u_bar_ego)
        self.u_bar_int = float(u_bar_int)

        desc_path = os.path.join(data_root, "dataset_description.json")
        with open(desc_path, "r") as f:
            desc = json.load(f)
        state_order = (
            ((desc.get("data_format") or {}).get("trajectory_files") or {}).get("state_order")
        ) or STATE_FEATURE_ORDER
        achieved = desc.get("achieved_bounds") or {}
        self._state_labels = list(state_order)
        self._state_ranges = []
        for idx, name in enumerate(self._state_labels):
            entry = achieved.get(name, {})
            try:
                mn, mx = float(entry.get("min")), float(entry.get("max"))
            except (TypeError, ValueError):
                mn, mx = state_mean[idx] - state_var[idx], state_mean[idx] + state_var[idx]
            self._state_ranges.append([mn, mx])

        super().__init__(
            loss_type="brt_hjivi",
            set_mode=set_mode,
            state_dim=STATE_DIM,
            input_dim=STATE_DIM + 1,
            control_dim=N_JOINTS_PER_ARM,
            disturbance_dim=N_JOINTS_PER_ARM,
            state_mean=state_mean,
            state_var=state_var,
            value_mean=0.0,
            value_var=1.0,
            value_normto=1.0,
            deepreach_model=deepreach_model,
        )

    def state_test_range(self):
        return self._state_ranges

    def equivalent_wrapped_state(self, state):
        # Piper joints are finite-stop revolute joints (see mujoco_fk.JOINT_RANGES),
        # not continuously periodic -- no wrapping needed.
        return torch.clone(state)

    # -- analytic differential-game dynamics (kinematic, state-independent) --
    def dsdt(self, state, control, disturbance):
        return torch.cat((control, disturbance), dim=-1)

    # -- boundary / cost (REAL Piper FK via MuJoCo; see module docstring) ----
    def boundary_fn(self, state):
        return _MuJoCoBoundaryFn.apply(state)

    def sample_target_state(self, num_samples):
        # Avoid-only BRT: no reach target set (mirrors UR5eGraspBoundary's stub).
        return torch.zeros(num_samples, self.state_dim)

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    # -- Hamiltonian: H = min_{|u|<=u_bar_ego} dvds_ego.u + max_{|d|<=u_bar_int} dvds_int.d
    def hamiltonian(self, state, dvds):
        dvds_ego = dvds[..., :N_JOINTS_PER_ARM]
        dvds_int = dvds[..., N_JOINTS_PER_ARM:]
        return (-self.u_bar_ego * torch.norm(dvds_ego, dim=-1)
                + self.u_bar_int * torch.norm(dvds_int, dim=-1))

    def optimal_control(self, state, dvds):
        dvds_ego = dvds[..., :N_JOINTS_PER_ARM]
        n = torch.norm(dvds_ego, dim=-1, keepdim=True).clamp_min(1e-9)
        return -self.u_bar_ego * dvds_ego / n

    def optimal_disturbance(self, state, dvds):
        dvds_int = dvds[..., N_JOINTS_PER_ARM:]
        n = torch.norm(dvds_int, dim=-1, keepdim=True).clamp_min(1e-9)
        return self.u_bar_int * dvds_int / n

    def plot_config(self):
        return {
            "state_slices": self.state_mean.tolist(),
            "state_labels": self._state_labels,
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 2,
        }
