"""Sim-level tests of the friction-grasp rig (one rig build ~20 s, module-scoped).  GRASP_SLOW=1 adds the end-to-end assisted grasp (~3 min).
python -m pytest -q world_model/v2/house/grasp/tests"""
import math
import os

import mujoco
import numpy as np
import pytest

from world_model.v2.house.teleop import scene as SC
from world_model.v2.house.teleop.sim import TeleopSim, Cmd, TICK
from world_model.v2.house.grasp import util as U
from world_model.v2.house.grasp.hands import HandServo
from world_model.v2.house.grasp.wrist import WristCam


@pytest.fixture(scope="module")
def fric():
    sc = SC.get_scene("h5")
    return TeleopSim("h5", counter=True, seed=0, spawn=sc.waypoints["cafe_counter_pick"], grasp_mode="friction", with_people=False)


def fric_slab_top(sim):
    m, d = sim.m, sim.d
    g = m.geom("tp_counter_top").id
    return float(d.geom_xpos[g][2] + m.geom_size[g][2])


def test_default_mode_is_weld_and_has_no_assist():
    sc = SC.get_scene("h5")
    sim = TeleopSim("h5", grasped=False, counter=True, spawn=sc.waypoints["cafe_counter_pick"], with_people=False)
    assert sim.grasp_mode == "weld" and sim.assist is None
    assert not sim.rig.info["dynamic_base"] and not sim.rig.info["finger_contact"]
    assert sim.m.body("yor_root").mocapid[0] == 0                     # kinematic mocap root, as before
    assert sim.can_grasp()                                            # hands at the hold pose are on the handles
    assert sim.grasp() and sim.grasped
    assert all(sim.d.eq_active[sim.eq[s]] == 1 for s in ("left", "right"))   # the weld path is unchanged


def test_friction_rig_flags(fric):
    m, d = fric.m, fric.d
    assert fric.grasp_mode == "friction" and fric.assist is not None and not fric.grasped
    assert not any(d.eq_active[fric.eq[s]] for s in ("left", "right"))       # nothing welded
    root = m.body("yor_root")
    assert root.mocapid[0] == -1 and m.body("yor_target").mocapid[0] == 0   # dynamic root follows mocap index 0
    pad_ids = [m.geom(f"{s}_arm_link{k}_pad").id for s in ("left", "right") for k in (7, 8)]
    tray = m.geom("tray_top").id
    slab = m.geom("tp_counter_top").id
    cup = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "cup" and m.geom_contype[g]][0]
    coll = lambda a, b: bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    for g in pad_ids:
        assert coll(g, tray)                         # pads grip the tray
        assert not coll(g, slab)                     # pads pass through the shelf
    for g in pad_ids[:2]:
        assert not coll(g, cup)                      # left arm never touches the cup (right arm keeps the stock cup collision bit)
    # finger servo: forcerange untouched, gain raised, reflected drive inertia on the slide joints
    a = m.actuator("left_arm_joint7_pos")
    assert tuple(m.actuator_forcerange[a.id]) == (-20.0, 20.0) and m.actuator_gainprm[a.id, 0] == SC.FINGER_KP_GRASP
    assert m.dof_armature[m.jnt_dofadr[m.joint("left_arm_joint7").id]] == SC.FINGER_ARMATURE
    # ASSUMED handle strips under the plate on the two short edges; shelf lowered by their thickness so the tray top stays at the hold height
    hp, hn = m.geom("handle_p").id, m.geom("handle_n").id
    for g in pad_ids[:4]:
        assert coll(g, hp) and coll(g, hn)
    assert abs(fric_slab_top(fric) - (SC.TRAY_HOLD_Z - SC.HANDLE_T - SC.HANDLE_T)) < 1e-6
    assert m.body_mass[fric.tray] == 0.3                          # tray mass unchanged by the strips


def test_dynamic_base_follows_target_and_holds_height(fric):
    d = fric.d
    for _ in range(30):
        fric.tick(Cmd())
    root = fric.m.body("yor_root").id
    assert abs(d.xpos[root][2]) < 1e-3                                  # no sag
    for _ in range(50):
        fric.tick(Cmd(twist=np.array([-0.4, 0.0, 0.0])))
    lag = np.linalg.norm(d.xpos[root][:2] - d.mocap_pos[0][:2])
    assert lag < 0.03, lag
    for _ in range(40):
        fric.tick(Cmd())
    assert np.linalg.norm(d.xpos[root][:2] - d.mocap_pos[0][:2]) < 0.005


def test_finger_force_command_is_force_not_position(fric):
    """ctrl = q -/+ F/kp gives an actuator force of -/+ F even though the jaw is only 3 mm from closed (stock kp 200 would give 0.6 N)."""
    a, d, m = fric.assist, fric.d, fric.m
    q7, q8 = a.fa["left"]
    c7, c8 = a.fact["left"]
    d.qpos[q7], d.qpos[q8] = 0.003, -0.003
    mujoco.mj_forward(m, d)
    F = 4.0                                   # needs q >= F/kp = 2 mm so the command stays inside the servo's [0, 0.035] range
    d.ctrl[c7], d.ctrl[c8] = d.qpos[q7] - F / a.kp, d.qpos[q8] + F / a.kp
    mujoco.mj_forward(m, d)
    assert abs(d.actuator_force[c7] + F) < 0.5 and abs(d.actuator_force[c8] - F) < 0.5


def test_hand_servo_reaches_goal_through_flmpc_refs(fric):
    d = fric.d
    hs = HandServo(fric)
    hs.start_from_current()
    p0, R0 = hs.hand_pose(1)
    hs.set_goal(1, p0 + np.array([-0.03, 0.0, 0.01]), R0)
    for _ in range(220):
        hs.step(TICK)
        fric.tick(Cmd())
    assert hs.goal_error(1)[0] < 0.006, hs.goal_error(1)
    assert fric.ctl.last_ok or fric.ctl.n_infeasible < 5


def test_wrist_cameras_see_the_tray_region_and_self_filter(fric):
    fric.assist.hs.active = False
    cam = WristCam(fric.m, "right")
    fr = cam.capture(fric.d, 0, seed=0)
    assert len(fr.pts) > 500 and fr.z.min() >= 0.1 - 1e-9
    robot_hit = cam.is_robot_geom[fr.geom]
    assert not robot_hit.any()                                          # the robot's own parts are filtered out
    # deterministic given (seed, tick)
    fr2 = cam.capture(fric.d, 0, seed=0)
    assert np.array_equal(fr.pts, fr2.pts)


def test_gate_holds_base_lift_jog_while_assisting(fric):
    a = fric.assist
    a.state = "PRE"
    c = a.gate(Cmd(twist=np.ones(3), lift_v=0.1, jog=np.ones(6), grasp=True))
    assert not c.twist.any() and c.lift_v == 0.0 and not c.jog.any() and c.grasp
    a.state = "HOLD"
    c = a.gate(Cmd(twist=np.ones(3), lift_v=0.1))
    assert c.lift_v == 0.1
    a.state = "IDLE"


@pytest.mark.skipif(not os.environ.get("GRASP_SLOW"), reason="end-to-end assisted grasp takes ~3 min (GRASP_SLOW=1)")
def test_assisted_grasp_end_to_end():
    from world_model.v2.house.grasp import smoke
    r = smoke.run_episode(0, seg="depth", drive_s=3.0, verbose=False)
    assert r["grasp_state"] == "HOLD", r.get("fail_reason")
    assert r["success"]
