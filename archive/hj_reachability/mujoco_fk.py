"""Real Piper-geometry forward-kinematics oracle (MuJoCo-as-FK-oracle).

Replaces the placeholder serial-chain FK previously in ``dynamics.py``. For a
sampled joint configuration ``q`` (per arm), we set the corresponding qpos
addresses on the SAME MuJoCo model used by the experiment
(``experiments/ab_smooth_vs_bangbang/sim_common.py``), run ``mj_kinematics``
+ ``mj_comPos`` (forward-kinematics only -- no contacts/dynamics needed), and
read back the FAITHFUL multi-keypoint proxy (see COLLISION FIDELITY note
below), NOT the old 3-sphere-per-arm proxy.

Deliberately standalone (does not import ``sim_common`` itself): that module
also imports ``controller_core`` from ``real_Cartpole`` (QP solvers etc.) at
module scope, which is unnecessary baggage for an FK-only oracle and would
tie this file (and the DeepReach fork, which loads this file by path -- see
``deepreach_dynamics.py``) to a much larger, unrelated dependency chain. All
geometry constants/logic below are REPLICATED VERBATIM (not imported) from
``experiments/ab_smooth_vs_bangbang/sim_common.py`` and MUST be kept in sync
with it by hand -- see that module's ``COLLISION_FIDELITY.md`` doc and
docstring, section 5 ("What mujoco_fk.py must change to match").

---------------------------------------------------------------------------
COLLISION FIDELITY (2026-09-06 upgrade -- supersedes the OLD 3-point proxy)
---------------------------------------------------------------------------
This module previously used a hand-copied, STALE 3-sphere-per-arm proxy
(``link3``/``link5``/``gripper_base``, fixed ``R_EGO=R_INT=0.05``,
``D_MIN=0.08``) that ``experiments/ab_smooth_vs_bangbang/COLLISION_FIDELITY.md``
measured at a **19.8% false-safe rate** (proxy says safe, real per-link
fitted-capsule/box geometry already overlaps) -- missing link0/1/2/4/6/7/8
and the wrist ``iphone_body`` mount entirely. That doc's fix (implemented in
``sim_common.py``) replaces it with a **FAITHFUL ~25-keypoint-per-arm proxy**
(0.00% false-safe across 699 true collisions, 2 seeds): every major body
(``LINK_NAMES`` below) contributes 2 endpoint spheres fit to its REAL fitted
capsule's own radius/ends (+1 midpoint sphere for capsules with half-length
> 5cm) or 1 bounding sphere per box geom (finger pads) -- radii read directly
from the compiled model's own "collision"-class (group==3) geoms, not
guessed constants. ``h`` per (ego,intruder) keypoint pair =
``dist - (r_ego + r_int + D_MIN)``, ``D_MIN=0.03``; per-pair combined radius
now varies by link instead of one fixed scalar.

This module has been upgraded to that SAME proxy (identical ``LINK_NAMES``,
``_body_keypoints``, ``barrier_keypoints``, ``D_MIN``, ``_LONG_CAPSULE_
HALFLEN_THRESH``, and the ``_patch_real_collision_geoms`` in-memory contype/
conaffinity fix at load) so ``l(x)`` computed here is IDENTICAL (float
precision) to ``sim_common.py``'s ``min_pair_distance``/``EgoController.
barrier`` -- see README.md's spot-check. ``closest_pair_info``'s analytic
gradient now queries the Jacobian at the keypoint's WORLD OFFSET point
(``point_jac_at``-equivalent), not the body origin, since most corrected
keypoints are not at body origins (offset-aware Jacobian:
``d(dist)/dq = n^T @ J_point(q, offset)``).

``EGO_BARRIER_BODIES``/``INT_BARRIER_BODIES``/``R_EGO``/``R_INT``/
``COMBINED_RADIUS`` are kept below ONLY as legacy/documentation constants
(matching sim_common.py's own pattern) -- NOT used by ``min_pair_distance``/
``boundary_fn``/``closest_pair_info`` any more. Any dataset exported against
the OLD proxy (``datasets/bimanual_yor_v1``) is STALE and superseded by
``datasets/bimanual_yor_v2`` (see README.md).

Also provides the analytic sphere-distance gradient w.r.t. joint angles via
the geometric Jacobian (``mj_jac``), the same technique
``sim_common.all_pair_barriers``/``EgoController.barrier`` use for
``hdot = n . (v_ego - v_int)`` (chain rule through the point Jacobian),
specialized here to positions instead of velocities:
``d(dist)/dq_ego = n^T @ J_ego``, ``d(dist)/dq_int = -n^T @ J_int``, where
``n = (p_ego - p_int) / dist`` is the unit vector between the closest pair
and ``J_*`` is the 3xnv translational Jacobian of that keypoint's WORLD
offset point restricted to that arm's 6 dof columns. This is the exact
analytic gradient of ``min_pair_distance`` (hence of ``h``, since the
per-pair combined radius ``r_ego+r_int+D_MIN`` is constant w.r.t. q) at the
current argmin pair -- used by ``deepreach_dynamics.py``'s custom
``torch.autograd.Function`` in place of (impossible) autograd through MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

SCENE = "/common/users/ss5772/bimanual/scenes/scene_yor_canonical.xml"

N_JOINTS_PER_ARM = 6

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 7)]

EGO_PREFIX = "left_arm_"
INT_PREFIX = "right_arm_"

# ---------------------------------------------------------------------------
# FAITHFUL multi-keypoint proxy spec -- REPLICATED VERBATIM from
# experiments/ab_smooth_vs_bangbang/sim_common.py (LINK_NAMES/_body_keypoints/
# barrier_keypoints/D_MIN/_LONG_CAPSULE_HALFLEN_THRESH). Do not hand-edit
# without also updating sim_common.py, or the two proxies will drift again.
# ---------------------------------------------------------------------------
LINK_NAMES = [
    "link0", "link1", "link2", "link3", "link4", "link5", "link6",
    "gripper_base", "link7", "link8", "iphone_body",
]
_LONG_CAPSULE_HALFLEN_THRESH = 0.05  # [m] add a 3rd midpoint sphere above this
D_MIN = 0.03  # [m] extra required clearance margin (per keypoint pair), matches sim_common.py

# Legacy/documentation-only constants (the OLD 3-point proxy this module used
# to implement, and sim_common.py's own legacy scalars for the old fixed-
# radius scheme) -- NOT used by min_pair_distance/boundary_fn/
# closest_pair_info below; kept only so external code that still names them
# does not break.
EGO_BARRIER_BODIES = ["left_arm_link3", "left_arm_link5", "left_arm_gripper_base"]
INT_BARRIER_BODIES = ["right_arm_link3", "right_arm_link5", "right_arm_gripper_base"]
R_EGO = 0.035
R_INT = 0.035
COMBINED_RADIUS = R_EGO + R_INT + D_MIN  # = 0.10, legacy scalar only

# Real Piper per-joint ranges (rad), from scenes/scene_yor_canonical.xml
# (yor_robot_derived.mjcf) left_arm_joint1..6 -- verified identical to
# right_arm_joint1..6 (symmetric), matching scenes/YOR_CANONICAL.md's
# verified joint table. Unchanged numerically from the old rig.
JOINT_RANGES = np.array([
    [-2.61, 2.61],
    [0.0, 3.13],
    [-2.965, 0.0],
    [-1.74, 1.74],
    [-1.2, 1.2],
    [-3.0, 3.0],
])

# ---------------------------------------------------------------------------
# In-memory contype/conaffinity collision fix (REPLICATED from sim_common.py
# _patch_real_collision_geoms -- see COLLISION_FIDELITY.md root-cause section).
# Not strictly required for the keypoint-distance proxy itself (which reads
# geom_pos/size directly, independent of contype/conaffinity), but kept here
# so this module's model matches sim_common.py's model byte-for-byte and any
# future ground-truth mj_geomDistance/contact cross-check stays valid. The
# scene file on disk is never touched (in-memory MjSpec patch only).
# ---------------------------------------------------------------------------
_LEFT_BIT, _RIGHT_BIT = 0b0100, 0b1000


def _patch_real_collision_geoms(spec):
    for b in spec.bodies:
        if b.name.startswith(EGO_PREFIX):
            contype, conaffinity = _LEFT_BIT, _RIGHT_BIT
        elif b.name.startswith(INT_PREFIX):
            contype, conaffinity = _RIGHT_BIT, _LEFT_BIT
        else:
            continue
        for g in b.geoms:
            if g.contype == 0 and g.conaffinity == 1:
                g.contype, g.conaffinity = contype, conaffinity


def load_model(scene_path: str = SCENE) -> "mujoco.MjModel":
    spec = mujoco.MjSpec.from_file(scene_path)
    _patch_real_collision_geoms(spec)
    return spec.compile()


def _joint_addrs(model, names):
    jids = [model.joint(n).id for n in names]
    qposadr = np.array([model.jnt_qposadr[j] for j in jids])
    dofadr = np.array([model.jnt_dofadr[j] for j in jids])
    return qposadr, dofadr


# ---------------------------------------------------------------------------
# Faithful keypoint construction -- REPLICATED VERBATIM from sim_common.py's
# _quat_rotate / _body_keypoints / barrier_keypoints / _kp_world.
# ---------------------------------------------------------------------------
def _quat_rotate(q, v):
    w, x, y, z = q
    return np.array([
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
        2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
        2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    ])


def _body_keypoints(model, body_id):
    """(offset_local(3,), radius) sphere list approximating a body's REAL
    "collision"-class (group==3) geom(s): 2 (or 3, for long capsules)
    spheres per fitted capsule at its true radius, 1 bounding sphere per box
    geom (finger pads)."""
    kps = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id or model.geom_group[gid] != 3:
            continue
        gtype = model.geom_type[gid]
        pos = model.geom_pos[gid].copy()
        size = model.geom_size[gid].copy()
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = _quat_rotate(model.geom_quat[gid], np.array([0.0, 0.0, 1.0]))
            hl, r = float(size[1]), float(size[0])
            kps.append((pos + hl * axis, r))
            kps.append((pos - hl * axis, r))
            if hl > _LONG_CAPSULE_HALFLEN_THRESH:
                kps.append((pos, r))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            kps.append((pos, float(np.linalg.norm(size))))
        else:
            kps.append((pos, float(size[0])))
    return kps


_KEYPOINT_CACHE: dict = {}


def barrier_keypoints(model, prefix):
    """List of (body_id, local_offset(3,), radius) for all LINK_NAMES bodies
    under `prefix` ("left_arm_" or "right_arm_"). Cached per model instance
    (geometry is static after compile)."""
    key = (id(model), prefix)
    cached = _KEYPOINT_CACHE.get(key)
    if cached is None:
        out = []
        for ln in LINK_NAMES:
            bid = model.body(prefix + ln).id
            for offset, r in _body_keypoints(model, bid):
                out.append((bid, offset, r))
        cached = _KEYPOINT_CACHE[key] = out
    return cached


def _kp_world(data, body_id, offset):
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset


@dataclass
class ClosestPairInfo:
    dist: float               # raw Euclidean distance of the worst-h keypoint pair
    h: float                  # l(x) = dist - (r_ego + r_int + D_MIN) for that pair (THE boundary value)
    ego_body_id: int
    int_body_id: int
    ego_offset: np.ndarray    # local (body-frame) offset of the ego keypoint, shape (3,)
    int_offset: np.ndarray    # local (body-frame) offset of the int keypoint, shape (3,)
    n: np.ndarray             # unit vector (p_ego - p_int) / dist, shape (3,)
    dl_dq_ego: np.ndarray     # d(h)/dq_ego == d(dist)/dq_ego, shape (6,)
    dl_dq_int: np.ndarray     # d(h)/dq_int == d(dist)/dq_int, shape (6,)


class MuJoCoFKOracle:
    """Real-geometry FK/Jacobian oracle over the bimanual pickup scene, using
    the FAITHFUL ~25-keypoint-per-arm proxy (see module docstring).

    Not batched (MuJoCo's Python bindings operate on one MjData at a time);
    callers loop over samples. Fine for scaffold/dataset-export-scale N; a
    much larger dataset export would want either multiple worker processes or
    (if/when available) MuJoCo's batched/MJX path -- out of scope here.
    """

    def __init__(self, scene_path: str = SCENE):
        self.model = load_model(scene_path)
        self.data = mujoco.MjData(self.model)
        self.ego_qposadr, self.ego_dofadr = _joint_addrs(self.model, LEFT_ARM_JOINTS)
        self.int_qposadr, self.int_dofadr = _joint_addrs(self.model, RIGHT_ARM_JOINTS)
        # Warm the keypoint cache now (cheap, and makes the per-call cost
        # purely the O(25*25) distance search + 2 mj_jac calls).
        barrier_keypoints(self.model, EGO_PREFIX)
        barrier_keypoints(self.model, INT_PREFIX)

    # -- core FK ---------------------------------------------------------
    def set_state(self, q_ego: np.ndarray, q_int: np.ndarray) -> None:
        assert q_ego.shape == (N_JOINTS_PER_ARM,), q_ego.shape
        assert q_int.shape == (N_JOINTS_PER_ARM,), q_int.shape
        self.data.qpos[self.ego_qposadr] = q_ego
        self.data.qpos[self.int_qposadr] = q_int
        self.data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        # mj_jac (used by closest_pair_info's analytic gradient) needs
        # subtree_com in addition to xpos/xquat/xmat -- mj_kinematics alone
        # leaves it stale/zero. mj_comPos is still far cheaper than a full
        # mj_forward (no contacts/dynamics/actuation).
        mujoco.mj_comPos(self.model, self.data)

    def keypoints(self, q_ego: np.ndarray, q_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ego_keypoints (N_ego,3), int_keypoints (N_int,3)) world
        positions for the FULL faithful proxy (~25 each, not a fixed 3)."""
        self.set_state(q_ego, q_int)
        ego_kps = barrier_keypoints(self.model, EGO_PREFIX)
        int_kps = barrier_keypoints(self.model, INT_PREFIX)
        ego_pts = np.stack([_kp_world(self.data, b, o) for b, o, _r in ego_kps], axis=0)
        int_pts = np.stack([_kp_world(self.data, b, o) for b, o, _r in int_kps], axis=0)
        return ego_pts, int_pts

    def _closest_pair(self):
        """Worst (min-h) pair over the FULL faithful proxy (~25x25=625
        pairs), operating on the CURRENT self.data state (call set_state
        first). Returns (h, dist, ego_body_id, int_body_id, n, ego_offset,
        int_offset) -- identical tie-breaking (argmin by h, not by raw dist)
        to sim_common.min_pair_distance."""
        ego_kps = barrier_keypoints(self.model, EGO_PREFIX)
        int_kps = barrier_keypoints(self.model, INT_PREFIX)
        best = None
        for eb, eoff, er in ego_kps:
            pe = _kp_world(self.data, eb, eoff)
            for ib, ioff, ir in int_kps:
                pi = _kp_world(self.data, ib, ioff)
                d = pe - pi
                dist = float(np.linalg.norm(d))
                h = dist - (er + ir + D_MIN)
                if best is None or h < best[0]:
                    n = d / max(dist, 1e-9)
                    best = (h, dist, eb, ib, n, eoff, ioff)
        return best

    def min_pair_distance(self, q_ego: np.ndarray, q_int: np.ndarray) -> float:
        """Raw distance of the worst-h keypoint pair (reporting/logging
        column only -- since per-pair radii vary, this is NOT itself l(x);
        use boundary_fn for that)."""
        self.set_state(q_ego, q_int)
        return self._closest_pair()[1]

    # -- analytic gradient via the geometric Jacobian ---------------------
    def _point_jac_at(self, body_id: int, offset: np.ndarray) -> np.ndarray:
        """Translational Jacobian (3 x nv) of an arbitrary point rigidly
        attached to `body_id`, given as a LOCAL (body-frame) offset --
        offset-aware, since faithful-proxy keypoints are generally not at
        the body origin (unlike the old 3-point proxy's raw xpos)."""
        p_world = _kp_world(self.data, body_id, offset)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, p_world, body_id)
        return jacp

    def closest_pair_info(self, q_ego: np.ndarray, q_int: np.ndarray) -> ClosestPairInfo:
        """Sets state, finds the argmin-h (ego,int) keypoint pair, and
        returns the exact analytic gradient of h w.r.t. (q_ego, q_int) at
        that pair via the offset-aware translational Jacobian (chain rule:
        d(||p_e-p_i||)/dq = n^T @ d(p_e-p_i)/dq = n^T @ (Je or -Ji); since
        the per-pair combined radius er+ir+D_MIN is constant w.r.t. q, this
        is also exactly d(h)/dq)."""
        self.set_state(q_ego, q_int)
        h, dist, eid, iid, n, eoff, ioff = self._closest_pair()
        Je = self._point_jac_at(eid, eoff)[:, self.ego_dofadr]   # 3x6
        Ji = self._point_jac_at(iid, ioff)[:, self.int_dofadr]   # 3x6
        dl_dq_ego = n @ Je
        dl_dq_int = -(n @ Ji)
        return ClosestPairInfo(dist=dist, h=h, ego_body_id=eid, int_body_id=iid,
                                ego_offset=eoff, int_offset=ioff, n=n,
                                dl_dq_ego=dl_dq_ego, dl_dq_int=dl_dq_int)

    def boundary_fn(self, q_ego: np.ndarray, q_int: np.ndarray) -> float:
        """l(x) = h at the worst (min-h) faithful-proxy keypoint pair.
        IDENTICAL (float precision) to
        experiments/ab_smooth_vs_bangbang/sim_common.min_pair_distance's h
        for the same (q_ego, q_int) on the same scene -- see README.md's
        spot-check."""
        self.set_state(q_ego, q_int)
        return self._closest_pair()[0]


_ORACLE_SINGLETON: MuJoCoFKOracle | None = None


def get_oracle() -> MuJoCoFKOracle:
    """Process-wide lazy singleton (loading the MJCF + building MjData is
    the expensive part; reuse across calls)."""
    global _ORACLE_SINGLETON
    if _ORACLE_SINGLETON is None:
        _ORACLE_SINGLETON = MuJoCoFKOracle()
    return _ORACLE_SINGLETON
