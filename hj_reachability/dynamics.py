"""Pairwise arm-arm avoidance system spec for neural HJ reachability (Piper-PAIR).

This is the numpy reference "system spec" used by ``export_dataset.py`` to
generate a bootstrap training set for a DeepReach value network, and it is
written to be a drop-in port target for a ``dynamics.Dynamics`` subclass in
the DeepReach trainer at ``/common/users/ss5772/DeepReach/deepreach``
(see ``dynamics/dynamics.py`` there, e.g. ``UR5eGraspBoundary`` /
``MultiVehicleCollision`` for the abstract interface this mirrors:
``state_test_range``, ``dsdt``, ``boundary_fn``, ``hamiltonian``,
``optimal_control``, ``optimal_disturbance``). Kept numpy-only (no torch
dependency) so it can be imported and smoke-tested from the plain bimanual
python environment without installing the DeepReach stack.

Model (first cut, differential-game framing -- mirrors
experiments/ab_smooth_vs_bangbang/sim_common.py's ego (LEFT, controlled) vs
intruder (RIGHT, treated here as a worst-case adversarial disturbance rather
than the scripted open-loop sweep used in that experiment):

  state   x = [q_ego (6,), q_int (6,)]                       (12-dim, joint angles, rad)
  control u = q_ego_dot,  disturbance d = q_int_dot            (velocity-controlled / kinematic)
  dynamics: xdot = [u, d]        (ẋ = u for ego joints, ẋ = d for intruder joints)
  control bounds (per-arm ball, not per-joint box): ||u|| <= u_bar_ego, ||d|| <= u_bar_int

  failure boundary l(x) = min_pair capsule/sphere signed distance - d_min
                          = min over (ego keypoint, intruder keypoint) pairs of
                            ||p_ego_k - p_int_j|| - (r_ego + r_int + d_min)
  Sign convention (matches DeepReach's boundary_fn / MultiVehicleCollision and
  sim_common.py's h = dist - COMBINED_RADIUS): l(x) > 0 is SAFE (clear of the
  combined-radius+margin shell), l(x) <= 0 is UNSAFE/in the avoid set. This is
  the *opposite* sign of the ur5e_benchmark DeepReach bootstrap CSV convention
  (target_value: safe=-1.0, unsafe=+1.0, "safe_when_leq_zero"); export_dataset.py
  performs that translation explicitly when writing target_value/target_is_safe.

Forward kinematics (REAL YOR/Piper geometry):
  Each 6-DoF arm's 3 keypoint positions (elbow/link3, forearm/link5,
  gripper_base -- the same EGO_BARRIER_BODIES / INT_BARRIER_BODIES proxy set
  as sim_common.py) are read from the ACTUAL MuJoCo model
  (scenes/scene_yor_canonical.xml, the canonical YOR rig -- arms 0.185m
  apart, lift-mounted; supersedes the old scene_bimanual_pickup.xml rig,
  arms 0.5m apart on the floor) via ``mujoco_fk.MuJoCoFKOracle``: set
  qpos for the sampled joint config, run ``mj_kinematics``, read back real
  link ``xpos``. This replaced an earlier placeholder minimal serial chain
  (illustrative link lengths/axes, NOT fit to the real Piper URDF/MJCF) --
  see README.md for the before/after spot-check against sim_common.py.

Hamiltonian (min_u max_d game, avoid-only BRT, single-integrator controls):
  dvds = [dvds_ego (6,), dvds_int (6,)] (value-function gradient w.r.t. state)
  H(x, dvds) = min_{||u||<=u_bar_ego} dvds_ego . u  +  max_{||d||<=u_bar_int} dvds_int . d
             = -u_bar_ego * ||dvds_ego||  +  u_bar_int * ||dvds_int||
  optimal_control(dvds_ego)     = -u_bar_ego * dvds_ego / ||dvds_ego||   (ego steers away)
  optimal_disturbance(dvds_int) = +u_bar_int * dvds_int / ||dvds_int||  (worst-case intruder closes in)
This Hamiltonian/optimal-control pair is the exact analytic form a future
``dynamics.Dynamics`` port would implement (unlike ``UR5eGraspBoundary``,
which has no analytic dsdt/hamiltonian and is bootstrap/temporal-consistency
only) -- see README.md for the hj_pde vs supervised_bootstrap tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mujoco_fk

# ---------------------------------------------------------------------------
# Constants reused from experiments/ab_smooth_vs_bangbang/sim_common.py (via
# mujoco_fk.py, which copies them verbatim) so the neural value function's
# failure set matches the analytic sphere-distance proxy it is meant to
# (eventually) replace.
# ---------------------------------------------------------------------------
# NOTE (2026-09-06 faithful-proxy upgrade): R_EGO/R_INT/COMBINED_RADIUS below
# are LEGACY, documentation-only scalars (mujoco_fk.py's own legacy constants
# for the old fixed-radius 3-point proxy) -- the faithful ~25-keypoint-per-arm
# proxy now in mujoco_fk.py varies the combined radius per (ego,int) keypoint
# pair, so l(x) is no longer `min_pair_distance(state) - combined_radius`
# (see boundary_fn below, which now delegates directly to
# mujoco_fk.MuJoCoFKOracle.boundary_fn). Kept only so external code that still
# names them does not break.
R_EGO = mujoco_fk.R_EGO      # [m] legacy sphere radius stand-in (unused by boundary_fn)
R_INT = mujoco_fk.R_INT      # [m] legacy sphere radius stand-in (unused by boundary_fn)
D_MIN = mujoco_fk.D_MIN      # [m] real per-pair clearance margin (still used, inside the oracle)
COMBINED_RADIUS = mujoco_fk.COMBINED_RADIUS  # [m] legacy scalar only, NOT used by boundary_fn any more

N_JOINTS_PER_ARM = mujoco_fk.N_JOINTS_PER_ARM

# Real per-joint ranges (rad) for each arm, from scenes/scene_yor_canonical.xml
# (left_arm_joint1..6; right arm is symmetric -- see mujoco_fk.py). Replaces the
# earlier generic +/-170 deg placeholder.
JOINT_RANGES = mujoco_fk.JOINT_RANGES

# Per-arm control bounds ||u|| <= u_bar (velocity-controlled, rad/s), a
# conservative joint-speed-derived guess pending real actuator limits.
DEFAULT_U_BAR_EGO = 1.5
DEFAULT_U_BAR_INT = 1.5


@dataclass(frozen=True)
class BimanualPairAvoidanceConfig:
    u_bar_ego: float = DEFAULT_U_BAR_EGO
    u_bar_int: float = DEFAULT_U_BAR_INT
    r_ego: float = R_EGO
    r_int: float = R_INT
    d_min: float = D_MIN

    @property
    def combined_radius(self) -> float:
        return self.r_ego + self.r_int + self.d_min


class BimanualPairAvoidance:
    """Numpy reference implementation of the pairwise arm-arm avoidance spec.

    state_dim = 12 ([q_ego(6), q_int(6)]); control_dim = 6 (ego qdot);
    disturbance_dim = 6 (intruder qdot, treated as worst-case adversary).
    """

    state_dim = 2 * N_JOINTS_PER_ARM
    control_dim = N_JOINTS_PER_ARM
    disturbance_dim = N_JOINTS_PER_ARM
    state_feature_order = [f"q_ego_{i}" for i in range(N_JOINTS_PER_ARM)] + [
        f"q_int_{i}" for i in range(N_JOINTS_PER_ARM)
    ]

    def __init__(self, config: BimanualPairAvoidanceConfig | None = None):
        self.cfg = config or BimanualPairAvoidanceConfig()
        self._oracle: mujoco_fk.MuJoCoFKOracle | None = None

    @property
    def oracle(self) -> mujoco_fk.MuJoCoFKOracle:
        """Lazily-constructed MuJoCo FK oracle (loads the MJCF once)."""
        if self._oracle is None:
            self._oracle = mujoco_fk.get_oracle()
        return self._oracle

    # -- sampling / state space -------------------------------------------------
    def state_test_range(self) -> list[list[float]]:
        """[q_ego(6) real Piper joint ranges, q_int(6) real Piper joint ranges]."""
        ranges = JOINT_RANGES.tolist()
        return ranges + ranges

    def sample_states(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        lo_hi = np.array(self.state_test_range(), dtype=float)
        lo, hi = lo_hi[:, 0], lo_hi[:, 1]
        return rng.uniform(lo, hi, size=(num_samples, self.state_dim))

    # -- kinematics / boundary (REAL Piper geometry via MuJoCo) ------------------
    def ego_keypoints(self, state: np.ndarray) -> np.ndarray:
        q_ego, q_int = state[:N_JOINTS_PER_ARM], state[N_JOINTS_PER_ARM:]
        ego_kp, _ = self.oracle.keypoints(q_ego, q_int)
        return ego_kp

    def int_keypoints(self, state: np.ndarray) -> np.ndarray:
        q_ego, q_int = state[:N_JOINTS_PER_ARM], state[N_JOINTS_PER_ARM:]
        _, int_kp = self.oracle.keypoints(q_ego, q_int)
        return int_kp

    def min_pair_distance(self, state: np.ndarray) -> float:
        """Raw Euclidean distance of the worst-h (ego keypoint, intruder
        keypoint) pair -- reporting/logging only (per-pair radii vary in the
        faithful proxy, so this is NOT itself l(x); see boundary_fn)."""
        q_ego, q_int = state[:N_JOINTS_PER_ARM], state[N_JOINTS_PER_ARM:]
        return self.oracle.min_pair_distance(q_ego, q_int)

    def boundary_fn(self, state: np.ndarray) -> float:
        """l(x): delegates directly to mujoco_fk.MuJoCoFKOracle.boundary_fn,
        the FAITHFUL ~25-keypoint-per-arm proxy (per-pair radius baked in,
        h = dist - (r_ego+r_int+d_min) at the worst pair) -- IDENTICAL to
        experiments/ab_smooth_vs_bangbang/sim_common.py's h. l>0 safe, l<=0
        unsafe. No longer `min_pair_distance(state) - self.cfg.combined_radius`
        (that single-scalar-margin form assumed the old fixed-radius 3-sphere
        proxy)."""
        q_ego, q_int = state[:N_JOINTS_PER_ARM], state[N_JOINTS_PER_ARM:]
        return self.oracle.boundary_fn(q_ego, q_int)

    # -- workspace-oversampled task-region sampling (2026-09-06) ---------------
    # Uniform sampling over the full 12-dim joint box gives only ~4.9% unsafe
    # (see README.md) -- most configs are nowhere near the shared workspace.
    # These constants are copied VERBATIM from
    # experiments/ab_smooth_vs_bangbang/sim_common.py's LEFT_GRASP_Q/
    # RIGHT_HOME_CTRL/RIGHT_INTRUDE_CTRL: LEFT_GRASP_Q is an IK solution for
    # the (ego/left) gripper grasping test_box; RIGHT_HOME_CTRL/
    # RIGHT_INTRUDE_CTRL bracket an IK sweep for the (intruder/right) gripper
    # reaching toward the SAME shared workspace point ((0.42,-0.21) in the xy
    # plane, ~0.55 height) -- i.e. exactly the "both EEs near the shared
    # workspace/boundary" region COLLISION_FIDELITY.md's task-region sampling
    # (and this export's oversampling) targets. sim_common.py is READ-ONLY for
    # this task; these are plain numeric literals, not an import of that
    # module (which pulls in real_Cartpole/controller_core unnecessarily).
    _TASK_LEFT_GRASP_Q = np.array(
        [-0.00944173, 2.69291536, -1.83582068, -0.47028722, 1.03733985, -1.30204097])
    _TASK_RIGHT_HOME_CTRL = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78])
    _TASK_RIGHT_INTRUDE_CTRL = np.array(
        [-0.15229176, 2.56294267, -1.48373594, 0.93891875, 1.01939931, 0.80796997])

    def sample_states_mixed(
        self,
        num_samples: int,
        rng: np.random.Generator,
        task_region_frac: float = 0.5,
        task_noise_std_ego: float = 0.25,
        task_noise_std_int: float = 0.25,
    ) -> np.ndarray:
        """Mixed sampler: `(1-task_region_frac)` fraction uniform over the
        full joint-state box (`state_test_range`, unchanged), and
        `task_region_frac` fraction "task-region": q_ego = LEFT_GRASP_Q +
        gaussian noise, q_int = interpolate(RIGHT_HOME_CTRL,
        RIGHT_INTRUDE_CTRL, alpha~Uniform(0,1)) + gaussian noise -- i.e. both
        EEs sampled near the shared workspace/boundary point, mirroring
        COLLISION_FIDELITY.md's task-region assessment sampling. Clipped to
        JOINT_RANGES after noise. Reproducible given `rng`."""
        n_task = int(round(num_samples * task_region_frac))
        n_uniform = num_samples - n_task
        lo_hi = np.array(self.state_test_range(), dtype=float)
        lo, hi = lo_hi[:, 0], lo_hi[:, 1]

        uniform_states = rng.uniform(lo, hi, size=(n_uniform, self.state_dim))

        alpha = rng.uniform(0.0, 1.0, size=(n_task, 1))
        q_int_center = (self._TASK_RIGHT_HOME_CTRL[None, :]
                         + alpha * (self._TASK_RIGHT_INTRUDE_CTRL - self._TASK_RIGHT_HOME_CTRL)[None, :])
        q_ego_task = (self._TASK_LEFT_GRASP_Q[None, :]
                      + rng.normal(0.0, task_noise_std_ego, size=(n_task, N_JOINTS_PER_ARM)))
        q_int_task = q_int_center + rng.normal(0.0, task_noise_std_int, size=(n_task, N_JOINTS_PER_ARM))
        task_states = np.concatenate([q_ego_task, q_int_task], axis=1)
        task_states = np.clip(task_states, lo[None, :], hi[None, :])

        states = np.concatenate([uniform_states, task_states], axis=0)
        rng.shuffle(states)
        return states

    # -- dynamics / differential game (for a future analytic DeepReach port) ----
    def dsdt(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        return np.concatenate([control, disturbance])

    def hamiltonian(self, dvds_ego: np.ndarray, dvds_int: np.ndarray) -> float:
        return float(
            -self.cfg.u_bar_ego * np.linalg.norm(dvds_ego)
            + self.cfg.u_bar_int * np.linalg.norm(dvds_int)
        )

    def optimal_control(self, dvds_ego: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(dvds_ego)
        if n < 1e-9:
            return np.zeros_like(dvds_ego)
        return -self.cfg.u_bar_ego * dvds_ego / n

    def optimal_disturbance(self, dvds_int: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(dvds_int)
        if n < 1e-9:
            return np.zeros_like(dvds_int)
        return self.cfg.u_bar_int * dvds_int / n
