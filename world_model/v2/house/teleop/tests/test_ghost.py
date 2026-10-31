"""teleop/ghost.py: the ghost coincides with the robot when the shield passes the command, separates by the requested-minus-permitted
motion when it does not, eases back when the input stops, never writes the sim's model/data, and its IK never moves the controller ref."""
import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from world_model.v2.house.teleop import ghost as GH
from world_model.v2.house.teleop import base as BS


def test_pass_through_ghost_coincides_and_is_hidden():
    P = GH.Predictor()
    u = np.array([0.6, 0.0, 0.3])
    for _ in range(20):
        f = P.step([1.0, 2.0, 0.5], [0.5, 0, 0.2], u, u, interv=0.0)
    assert f.alpha == 0.0 and np.allclose(f.ghost, [1.0, 2.0, 0.5]) and not f.warn


def test_blocked_forward_command_puts_the_ghost_ahead():
    P = GH.Predictor(tau=1e-6)                      # no smoothing: the raw prediction
    f = P.step([0, 0, math.pi / 2], [0, 0, 0], [0.6, 0, 0], [0, 0, 0], interv=0.67)
    ahead = GH.rollout([0, 0, 0], [0.6, 0, 0])[0]   # forward distance over H from rest, under the sim's slew model
    assert ahead > 0.2
    assert np.allclose(f.ghost[:2], [0, ahead], atol=1e-9)          # body +x = world +y at yaw 90 deg
    assert f.alpha > 0 and f.warn
    assert np.allclose(f.perm_end, [0, 0]) and np.allclose(f.req_end, [0, ahead], atol=1e-9)


def test_ghost_eases_back_when_input_stops_and_reset_clears():
    P = GH.Predictor()
    for _ in range(25):
        f = P.step([0, 0, 0], [0, 0, 0], [0.6, 0, 0], [0.1, 0, 0], interv=0.5)
    assert f.alpha > 0
    gaps = []
    for _ in range(100):                            # 2 s of zero input
        f = P.step([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
        gaps.append(f.gap[0])
    assert all(b <= a + 1e-12 for a, b in zip(gaps, gaps[1:])) and f.alpha == 0.0
    for _ in range(5):
        P.step([0, 0, 0], [0, 0, 0], [0.6, 0, 0], [0, 0, 0])
    P.reset()
    f = P.step([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    assert f.gap[0] == 0.0 and f.alpha == 0.0


def test_inactive_phases_show_no_ghost():
    P = GH.Predictor(tau=1e-6)
    f = P.step([0, 0, 0], [0, 0, 0], [0.6, 0, 0], [0, 0, 0], interv=0.5, active=False)
    assert f.alpha == 0.0 and not f.warn


def test_lift_ghost_and_frame_vector_roundtrip():
    P = GH.Predictor(tau=1e-6)
    f = P.step([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], interv_aux=0.04, lift=(0.0, 0.0, 0.096, 0.06))
    assert f.dh > 0.01 and f.alpha > 0 and f.warn
    g = GH.GhostFrame.from_vec(f.to_vec())
    assert len(f.to_vec()) == GH.G_LEN
    assert np.allclose(g.to_vec(), f.to_vec())


def test_track_reanchors_on_the_recorded_pose():
    n = 30
    pose = np.c_[np.linspace(0, 0.3, n), np.zeros(n), np.zeros(n)]
    v = np.tile([0.5, 0, 0], (n, 1))
    uo, us = np.tile([0.9, 0, 0], (n, 1)), np.tile([0.5, 0, 0], (n, 1))
    T = GH.track(pose, v, uo, us, interv=np.full(n, 0.4))
    f = GH.GhostFrame.from_vec(T[-1])
    assert np.allclose(f.pose, pose[-1]) and f.ghost[0] > pose[-1, 0] + 0.05 and abs(f.ghost[1]) < 1e-9


# ------------------------------------------------------------------------------------------------ with the MuJoCo model
@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", grasped=True, seed=0, with_people=False)


def test_drawer_is_visualisation_only_and_reuses_the_robot_geoms(sim):
    import mujoco
    m, d = sim.m, sim.d
    q0, mp0, mq0, rgba0 = d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), m.geom_rgba.copy()
    r = mujoco.Renderer(m, 120, 160, max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (sim.base.x, sim.base.y, 0.8)
    cam.distance, cam.elevation, cam.azimuth = 2.4, -25.0, math.degrees(sim.base.yaw)
    r.update_scene(d, camera=cam)
    n0 = r.scene.ngeom
    Dr = GH.Drawer(m)
    n_tmpl = sum(1 for i in range(n0) if r.scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM and Dr.mask_all[r.scene.geoms[i].objid])
    assert n_tmpl > 10
    pose = sim.pose()
    gf = GH.GhostFrame(pose=pose, ghost=pose, alpha=1.0)            # zero offset: the ghost geoms must sit on the robot geoms
    added = Dr.add(r.scene, d, gf, arrows=False)
    assert added == n_tmpl
    src = [i for i in range(n0) if r.scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM and Dr.mask_all[r.scene.geoms[i].objid]]
    for k, i in enumerate(src):
        a, b = r.scene.geoms[i], r.scene.geoms[n0 + k]
        assert np.allclose(a.pos, b.pos, atol=1e-5) and np.allclose(a.mat, b.mat, atol=1e-5) and a.dataid == b.dataid and a.type == b.type
        assert b.rgba[3] < 0.5
    r.update_scene(d, camera=cam)
    gf = GH.GhostFrame(pose=pose, ghost=GH.se2_compose(pose, [0.3, 0, 0.2]), alpha=1.0, warn=True,
                       req_end=pose[:2] + 0.3, perm_end=pose[:2])
    added = Dr.add(r.scene, d, gf)
    assert added == n_tmpl + 1                                      # + the requested arrow (the permitted one has zero length)
    img = r.render()
    assert img.shape == (120, 160, 3)
    assert np.array_equal(d.qpos, q0) and np.array_equal(d.mocap_pos, mp0) and np.array_equal(d.mocap_quat, mq0)
    assert np.array_equal(m.geom_rgba, rgba0)
    r.close()


def test_arm_intent_never_moves_the_controller_reference(sim):
    of = sim.of
    st = (of.qL.copy(), of.qR.copy(), of.dp.copy(), of.dR.copy(), sim.d.qpos.copy())
    AI = GH.ArmIntent(of)
    assert np.allclose(AI.dq(np.zeros(6), np.zeros(6)), 0)
    dq = AI.dq(np.array([0.08, 0, 0, 0, 0, 0]), np.zeros(6))
    assert np.abs(dq).max() > 0.01
    assert all(np.array_equal(a, b) for a, b in zip(st, (of.qL, of.qR, of.dp, of.dR, sim.d.qpos)))
