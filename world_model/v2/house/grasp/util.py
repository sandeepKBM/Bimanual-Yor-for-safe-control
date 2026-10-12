"""grasp/util.py -- frame helpers shared by the assist, the tests and the smoke.  SIMULATION ONLY."""
from __future__ import annotations

import math

import mujoco
import numpy as np


def nominal_T_g_tray(of, side):
    """tray pose in the gripper_base frame at the nominal (hold) grasp: (rel (3,), R (3,3)) from the captured weld relpose."""
    rel = np.array(of.nominal[side][3:6], float)
    rq = np.array(of.nominal[side][6:10], float)
    Rn = np.zeros(9)
    mujoco.mju_quat2Mat(Rn, rq)
    return rel, Rn.reshape(3, 3)


def prior_tray_pose(of, side, grip_pos, grip_R, retreat=0.0, lift=0.0):
    """PRIOR tray pose (world) = nominal gripper-to-tray relpose composed with the gripper pose (the 'operator brought the hand here').
    retreat / lift: the known pre-grasp offset of the hand (hand pulled back by `retreat` along its own z and raised by `lift`), so the
    prior is where the tray is expected if the hand were exactly at its pre-grasp goal."""
    rel, Rn = nominal_T_g_tray(of, side)
    p = grip_pos + grip_R @ rel + grip_R[:, 2] * retreat - np.array([0.0, 0.0, lift])
    return p, grip_R @ Rn


def tray_pose(sim):
    return sim.d.xpos[sim.tray].copy(), sim.d.xmat[sim.tray].reshape(3, 3).copy()


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def perturb_tray(sim, dxy=(0.0, 0.0), dyaw=0.0):
    """rigidly move tray + cup + beads: translate by dxy (world) and yaw by dyaw about the tray centre; zero velocities."""
    m, d = sim.m, sim.d
    c = d.xpos[sim.tray].copy()
    Rz = rotz(dyaw)
    names = ["tray"] + [n for n in ("cup",)] + [f"ball_{i}" for i in range(sim.rig.n_beads)]
    for nm in names:
        j = m.joint(f"{nm}_free" if nm != "tray" else "tray_free")
        a = int(m.jnt_qposadr[j.id])
        d.qpos[a:a + 3] = c + Rz @ (d.qpos[a:a + 3] - c) + np.array([dxy[0], dxy[1], 0.0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)]), d.qpos[a + 3:a + 7].copy())
        d.qpos[a + 3:a + 7] = q
        dv = int(m.jnt_dofadr[j.id])
        d.qvel[dv:dv + 6] = 0.0
    mujoco.mj_forward(m, d)
