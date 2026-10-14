"""Lean, robot-only dynamics source for the real-time FL-MPC (rt_flmpc).

SIMULATION / OFFLINE ONLY -- pure MuJoCo model math; nothing here talks to hardware.

Why: the FL-MPC needs, per control step, M(q), c(q,qd) = qfrc_bias - qfrc_passive,
body Jacobians / Jacobian-derivatives and keypoint positions of the two ARMS only.
Today they are read off an mj_forward of the FULL cup-collect scene (cups, 8 balls,
6 cubes, table: collision detection + constraint solve, ~1.4 ms). Every one of those
quantities depends only on the robot's own kinematic tree (free objects are separate
trees; the cube welds / cup re-attach are equality/teleport, not tree edges), so they
can be computed on a ROBOT-ONLY copy of the model, synced from the robot joint state,
with only the pipeline stages they need:

    mj_kinematics -> mj_comPos -> mj_makeM (qM incl. armature) -> mj_comVel (cvel, cdof_dot)
    -> mj_passive (qfrc_passive incl. gravcomp) -> mj_rne(flg_acc=0) (qfrc_bias)

No collision, no constraint assembly/solve. On the real robot these same quantities
come from the robot model, so this is also the deployment path.

The robot-only model is built from the SAME scene XML via MjSpec by deleting every
worldbody child except the robot root ("base"), then its per-body/per-dof parameters
are CHECKED bit-for-bit against the scene model the controller is attached to
(raises on any mismatch -> never silently wrong).
"""
from __future__ import annotations

import numpy as np
import mujoco

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_xml():
    # scripts/cup_collect_task_utils.YOR_CUP_COLLECT_XML (imported lazily, light path shim;
    # does NOT import the demo module)
    for p in (_REPO, os.path.join(_REPO, "scripts"), os.path.join(_REPO, "experiments", "ab_smooth_vs_bangbang")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cup_collect_task_utils as cct
    return cct.YOR_CUP_COLLECT_XML

_BODY_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                "body_gravcomp", "body_dofnum", "body_jntnum")
_DOF_FIELDS = ("dof_armature", "dof_damping", "dof_frictionloss")
_JNT_FIELDS = ("jnt_type", "jnt_axis", "jnt_pos", "jnt_stiffness")


class LeanRobotDynamics:
    """Robot-only MjModel/MjData synced (by joint name) from a scene state."""

    def __init__(self, scene_model, xml_path: str | None = None, root_body: str = "base"):
        xml_path = xml_path or _default_xml()
        spec = mujoco.MjSpec.from_file(xml_path)
        for b in list(spec.worldbody.bodies):
            if b.name != root_body:
                spec.delete(b)
        for k in list(spec.keys):
            spec.delete(k)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m, sm = self.model, scene_model
        # ---- joint maps (lean <- scene), by name ----
        q_l, q_s, v_l, v_s = [], [], [], []
        for j in range(m.njnt):
            name = m.joint(j).name
            js = sm.joint(name).id
            assert m.jnt_type[j] == sm.jnt_type[js], name
            nq = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
            nv = {0: 6, 1: 3, 2: 1, 3: 1}[int(m.jnt_type[j])]
            q_l += list(range(m.jnt_qposadr[j], m.jnt_qposadr[j] + nq))
            q_s += list(range(sm.jnt_qposadr[js], sm.jnt_qposadr[js] + nq))
            v_l += list(range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nv))
            v_s += list(range(sm.jnt_dofadr[js], sm.jnt_dofadr[js] + nv))
        self.qadr_lean = np.array(q_l, np.int64)
        self.qadr_scene = np.array(q_s, np.int64)
        self.vadr_lean = np.array(v_l, np.int64)
        self.vadr_scene = np.array(v_s, np.int64)
        assert len(q_l) == m.nq and len(v_l) == m.nv
        # body map scene id -> lean id (robot bodies only)
        self.body_scene_to_lean = {}
        for b in range(1, m.nbody):
            self.body_scene_to_lean[sm.body(m.body(b).name).id] = b
        self.dof_scene_to_lean = {int(s): int(l) for s, l in zip(v_s, v_l)}
        self._verify(sm)
        self._M = np.zeros((m.nv, m.nv))

    # ------------------------------------------------------------------
    def _verify(self, sm):
        m = self.model
        bad = []
        for b in range(1, m.nbody):
            bs = sm.body(m.body(b).name).id
            for f in _BODY_FIELDS:
                if not np.array_equal(getattr(m, f)[b], getattr(sm, f)[bs]):
                    bad.append(f"{f}[{m.body(b).name}]")
            pl = m.body_parentid[b]
            ps = sm.body_parentid[bs]
            if m.body(int(pl)).name != sm.body(int(ps)).name:
                bad.append(f"parent[{m.body(b).name}]")
        for j in range(m.njnt):
            js = sm.joint(m.joint(j).name).id
            for f in _JNT_FIELDS:
                if not np.array_equal(getattr(m, f)[j], getattr(sm, f)[js]):
                    bad.append(f"{f}[{m.joint(j).name}]")
            dl, ds = m.jnt_dofadr[j], sm.jnt_dofadr[js]
            for f in _DOF_FIELDS:
                if not np.array_equal(getattr(m, f)[dl], getattr(sm, f)[ds]):
                    bad.append(f"{f}[{m.joint(j).name}]")
        for f in ("gravity", "timestep", "density", "viscosity", "wind", "magnetic", "integrator",
                  "disableflags", "enableflags"):
            if not np.array_equal(np.asarray(getattr(m.opt, f)), np.asarray(getattr(sm.opt, f))):
                bad.append(f"opt.{f}")
        if bad:
            raise RuntimeError(f"LeanRobotDynamics: robot-only model differs from the scene model: {bad[:20]}")

    def clone(self):
        """Same robot-only model, own MjData (e.g. for a solver thread)."""
        c = object.__new__(LeanRobotDynamics)
        c.__dict__.update(self.__dict__)
        c.data = mujoco.MjData(self.model)
        c._M = np.zeros_like(self._M)
        return c

    # ------------------------------------------------------------------
    def sync_from_scene(self, scene_data):
        """'Sensing': copy the robot joint state out of the scene."""
        self.data.qpos[self.qadr_lean] = scene_data.qpos[self.qadr_scene]
        self.data.qvel[self.vadr_lean] = scene_data.qvel[self.vadr_scene]

    def set_state(self, qpos_lean, qvel_lean):
        self.data.qpos[:] = qpos_lean
        self.data.qvel[:] = qvel_lean

    def forward(self):
        """Robot-only pipeline subset (see module docstring)."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_makeM(m, d)          # mj_crb + armature (+ tendon armature) -> qM, as mj_fwdPosition
        mujoco.mj_comVel(m, d)
        mujoco.mj_passive(m, d)
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)

    def full_M(self):
        mujoco.mj_fullM(self.model, self._M, self.data.qM)
        return self._M

    # map helpers -------------------------------------------------------
    def lean_body(self, scene_body_id: int) -> int:
        return self.body_scene_to_lean[int(scene_body_id)]

    def lean_dofs(self, scene_dofadr) -> np.ndarray:
        return np.array([self.dof_scene_to_lean[int(d)] for d in np.asarray(scene_dofadr).reshape(-1)], np.int64)

    def lean_kps(self, kps):
        """Re-index a keypoint list [(scene_body_id, offset, r), ...] onto lean body ids."""
        return [(self.lean_body(b), np.asarray(o, float), float(r)) for b, o, r in kps]
