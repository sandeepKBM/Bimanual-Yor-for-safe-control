"""agentchord_probe/nero/env.py -- the fake `env` their monitors/atoms talk to, backed by one nero.contents.carry.CarryEpisode.  SIMULATION ONLY.

Surface used (what THEIR monitor_functions / monitor_utils read): env.sim.get_rigid_object_uid_list(), env.sim.get_rigid_object(n).get_local_pose
(to_matrix=True) -> torch (1,4,4); env.obj_info[n]["pose"] (refreshed by update_obj_info() every monitor sample); env.get_current_qpos(name=
"left_eef"|"right_eef") -> finger joint positions of the MuJoCo model (their get_gripper_distance averages |q|).
Names the planner may use: the carried object ("tray" | "box"), "counter", "shelf" (box task), "mug_<i>" (tray task).  Poses are WORLD frame (their
prompt says "robot base frame"; the base frame is not modelled -- recorded deviation).  counter/shelf pose = centre of the top face.

Physics driving (used by atoms.drive): every control step is CarryEpisode._step() (FL-MPC torques, 5 ms control step as in carry.py); the monitors are
evaluated every `mon_period` seconds (obj_info refreshed at the same instants).
  ASSUMED mon_period = 0.1 s: THEIR env.step is one trajectory sample (10-30 samples per segment, atom_actions.plan_trajectory) and
  monitor_object_moved compares the pose with the previous SAMPLE; our controller steps are 5 ms, so one sample is mapped to mon_period.
  The 0.02 m default threshold is therefore a displacement per 0.1 s (0.2 m/s); a plan that sets a lower threshold can fire on normal motion.
"""
from __future__ import annotations

import numpy as np
import torch

from .sim_imports import CR, E, SC, mujoco   # noqa: F401  (re-export)

SIDES = ("left", "right")
MON_PERIOD_S = 0.1          # ASSUMED, see module doc
EDGE_SETTLE_S = 0.3         # MODEL: carry.segment_phases' hold after every move (same dwell as the certified plan); 0 = back-to-back edges


def _pose4(R, p):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(p, np.float32)
    return torch.from_numpy(T)


class _RigidObject:
    def __init__(self, env, name):
        self.env, self.name = env, name

    def get_local_pose(self, to_matrix=False):
        T = self.env.pose(self.name)
        return T.unsqueeze(0) if to_matrix else T[:3, 3]


class _Sim:
    def __init__(self, env):
        self.env = env

    def get_rigid_object_uid_list(self):
        return list(self.env.names)

    def get_rigid_object(self, name):
        return _RigidObject(self.env, name)


class NeroEnv:
    def __init__(self, ep, mon_period=MON_PERIOD_S, edge_settle=EDGE_SETTLE_S):
        self.ep, self.task = ep, ep.task
        self.edge_settle = float(edge_settle)
        self.carried = ep.task                                       # "tray" | "box"
        m = ep.m
        self.mugs = [f"mug_{i}" for i in range(len(ep.cups))]
        self.names = [self.carried, "counter"] + (["shelf"] if ep.task == "box" and getattr(ep.world, "shelf", False) else []) + self.mugs
        self.sim = _Sim(self)
        self.mon_period_steps = max(1, int(round(mon_period / CR.DT)))
        self.obj_info = {}
        self.init_tcp = {s: ep.tcp(s) for s in SIDES}                # "initial pose" for back_to_initial_pose (claws at the scripted grasp pose)
        self.t_steps = 0
        self._pik = {}
        self.log = []                                                # per-edge records (atoms.drive appends)
        self.monitor_errors = []
        self.ik_miss = 0.0
        self.cur_edge = (None, False)                                # (edge id, is_recovery) set by execute.py's wrappers
        self.obj_off = self.obj_pos(self.carried) - np.mean([self.init_tcp[s] for s in SIDES], axis=0)   # MODEL: constant while welded
        self.update_obj_info()

    # ---- state their monitors read
    def pose(self, name):
        ep, m, d = self.ep, self.ep.m, self.ep.d
        if name == self.carried:
            return _pose4(d.xmat[ep.obj], d.xpos[ep.obj])
        if name.startswith("mug_"):
            b = ep.cups[int(name[4:])]
            return _pose4(d.xmat[b], d.xpos[b])
        gid = m.geom(name).id                                         # "counter" / "shelf" are geom names in scenes.py
        p = d.geom_xpos[gid].copy()
        p[2] += m.geom_size[gid][2]                                   # top face centre (box geoms)
        return _pose4(d.geom_xmat[gid], p)

    def update_obj_info(self):
        self.obj_info = {n: dict(pose=self.pose(n).clone()) for n in self.names}

    def get_current_qpos(self, name):
        side = name.split("_")[0]
        m, d = self.ep.m, self.ep.d
        q = [d.qpos[m.jnt_qposadr[j]] for j in range(m.njnt) if m.joint(j).name.startswith(f"{side}_arm_gripper_joint")]
        return np.asarray(q, np.float32)

    # ---- geometry helpers for atoms
    def obj_pos(self, name):
        return self.pose(name)[:3, 3].numpy().astype(float)

    def tcp(self, side):
        return self.ep.tcp(side)

    def tcp_cmd(self, side):
        """TCP of the last COMMANDED joint configuration (end of the planned path, FK).  Atoms resolve their targets against this, not the
        measured TCP: carry.py plans every move from the commanded q, so a measured-state target would inject the tracking error (up to
        ~12 mm, 11 deg tilt in a tray carry) as a commanded step."""
        ep = self.ep
        q = ep.paths[side][-1][2][-1] if ep.paths[side] else ep.q0[side]
        if side not in self._pik:
            from world_model.v2.nero.democ import kin as K
            self._pik[side] = K.PoseIK(ep.m, side)
        return np.asarray(self._pik[side].ik.pose(q)[0], float).copy()

    def obj_nominal(self, name):
        """position of `name`; for the carried object while welded: the commanded claw midpoint + the object-minus-claw-midpoint offset of t = 0
        (rigid weld, pure translations), i.e. the NOMINAL object position without tracking error."""
        if name == self.carried and self.weld_on():
            return np.mean([self.tcp_cmd(s) for s in SIDES], axis=0) + self.obj_off
        return self.obj_pos(name)

    def site_R(self, side):
        return self.ep.d.site_xmat[self.ep.site[side]].reshape(3, 3).copy()

    def weld_on(self):
        return all(self.ep.d.eq_active[self.ep.weld_id[s]] for s in SIDES)

