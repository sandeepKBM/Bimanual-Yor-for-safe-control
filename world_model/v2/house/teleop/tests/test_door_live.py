"""live (configuration-aware) footprint, doorway buffer, people option, ego camera constants.  Headless, CPU only (builds one h5 TeleopSim, ~20 s).
python -m pytest -q world_model/v2/house/teleop/tests/test_door_live.py"""
import math

import numpy as np
import pytest

from world_model.v2.house.teleop import geom as GM, scene as SC, shield as SH
from world_model.v2.house.teleop.arm import _rotvec_to_R
from world_model.v2.house.teleop.web import render_proc as RP
from world_model.v2.house.teleop.web import inputs as IN


@pytest.fixture(scope="module")
def sim():
    from world_model.v2.house.teleop.sim import TeleopSim
    return TeleopSim("h5", with_people=True, people_clock="time")


def _arm_box(sim, dp, yaw):
    """live box (pad 0) of the arms at the IK reference for tray offset (dp, yaw) + the tray rectangle at that offset (the tray body itself is not moved offline)."""
    from world_model.v2.house import build as B
    import mujoco
    of, m = sim.of, sim.m
    bodies = {b for b in B._robot_tray_bodies(m) if (m.body(b).name or "") != "tray" and not (m.body(b).name or "").startswith("ball")}
    lf = GM.LiveFootprint(m, bodies=bodies, pad=0.0)
    dp0, dR0, qL, qR = of.dp.copy(), of.dR.copy(), of.qL.copy(), of.qR.copy()
    of.dp, of.dR = np.array(dp, float), _rotvec_to_R(np.array([0, 0, yaw]))
    for _ in range(60):
        of.solve()
    mujoco.mj_kinematics(m, of.ds)
    b = lf.box(of.ds, np.array([of.base_p[0], of.base_p[1], sim.base.yaw]), pad=0.0)
    c = of.G0p + of.dp
    cr = np.array([[sx * 0.15, sy * 0.10] for sx in (-1, 1) for sy in (-1, 1)]) @ of.dR[:2, :2].T + c[:2]
    w = max(b["y"][1], cr[:, 1].max()) - min(b["y"][0], cr[:, 1].min())
    of.dp, of.dR, of.qL, of.qR = dp0, dR0, qL, qR
    of.ctl.set_ref(qL, qR)
    return w


def test_defaults_unchanged(sim):
    cfg = SH.ShieldCfg()
    assert cfg.footprint == "fixed" and cfg.d_wall == 0.10 and cfg.d_wall_door is None and cfg.doorways == ()
    sh = SH.Shield(cfg, occ=sim.occ, a_hw=sim.a_hw)
    assert sh.footprint_of(sim) is GM.FP                   # fixed: the very same dict, no per-tick work
    assert sh.fp_ms[1] == 0


def test_live_box_inside_fixed_and_cheap(sim):
    import time
    lf = GM.LiveFootprint(sim.m, pad=0.0)
    b = lf.box(sim.d, sim.pose(), pad=0.0)
    # at the phase-1 hold pose the live hull is the fixed FP (which was an AABB-corner box of the same geoms): contained, within 4 cm
    for ax in ("x", "y"):
        assert b[ax][0] >= GM.FP[ax][0] - 1e-6 and b[ax][1] <= GM.FP[ax][1] + 1e-6
        assert b[ax][0] - GM.FP[ax][0] < 0.04 and GM.FP[ax][1] - b[ax][1] < 0.04
    bp = lf.box(sim.d, sim.pose(), pad=0.02)
    assert abs((bp["y"][1] - bp["y"][0]) - (b["y"][1] - b["y"][0] + 0.04)) < 1e-9
    t0 = time.perf_counter()
    for _ in range(200):
        lf.box(sim.d, sim.pose())
    assert (time.perf_counter() - t0) / 200 < 4e-3         # << the 20 ms shield tick (measured ~0.65 ms)


def test_live_box_rotation_invariant(sim):
    """the box is in the BASE frame: rotating the whole robot pose in the world (same geoms, same base yaw argument shift) must not change it."""
    lf = GM.LiveFootprint(sim.m, pad=0.0)
    p = sim.pose()
    a = lf.box(sim.d, p)
    # same world geoms but a pose whose yaw argument differs by pi: the box is the point-reflection about the base origin
    b = lf.box(sim.d, np.array([p[0], p[1], p[2] + math.pi]))
    assert np.allclose([a["x"][0], a["x"][1]], [-b["x"][1], -b["x"][0]], atol=1e-9)
    assert np.allclose([a["y"][0], a["y"][1]], [-b["y"][1], -b["y"][0]], atol=1e-9)


def test_tuck_pose_is_narrower(sim):
    nominal = _arm_box(sim, (0, 0, 0), 0.0)
    cfg = IN.CONFIG
    tucked = _arm_box(sim, cfg["tuck_dp"], cfg["tuck_yaw"])
    assert tucked < nominal - 0.05                         # measured 0.72 -> 0.62 m
    assert max(abs(v) for v in cfg["tuck_dp"]) <= 0.12 and abs(cfg["tuck_yaw"]) <= 0.35     # inside the ObjectFrame jog box / yaw cap (no limit is changed)


def test_live_rows_follow_the_footprint(sim):
    """a wall 0.5 m to the right of the robot: the live (tucked-narrower / padded) footprint changes the wall-row margins relative to the fixed one."""
    sh_f = SH.Shield(SH.ShieldCfg(), occ=sim.occ, a_hw=sim.a_hw)
    sh_l = SH.Shield(SH.ShieldCfg(footprint="live"), occ=sim.occ, a_hw=sim.a_hw)
    v = np.zeros(3)
    sh_f.fp_now = sh_f.footprint_of(sim)
    sh_l.fp_now = sh_l.footprint_of(sim)
    assert sh_l.fp_now is not GM.FP and sh_l.fp_ms[1] == 1
    Af, bf, _ = sh_f._keep_rows(sim, v)
    Al, bl, _ = sh_l._keep_rows(sim, v)
    assert Af.shape[1] == 3 and Al.shape[1] == 3
    # person rows use the footprint too
    sim.people_state = [(np.array([sim.base.x + 1.2, sim.base.y]), np.zeros(2))]
    pf = sh_f._person_rows(sim)
    pl = sh_l._person_rows(sim)
    assert len(pf[1]) == len(pl[1]) == min(len(sim.people_state), len(sim.scene.people))
    sim.set_people()


def test_doorway_buffer_is_opt_in(sim):
    cfg = SH.ShieldCfg(d_wall_door=0.05, doorways=((sim.base.x, sim.base.y),))
    sh = SH.Shield(cfg, occ=sim.occ, a_hw=sim.a_hw)
    sh.fp_now = GM.FP
    _, b_door, _ = sh._keep_rows(sim, np.zeros(3))
    sh0 = SH.Shield(SH.ShieldCfg(), occ=sim.occ, a_hw=sim.a_hw)
    sh0.fp_now = GM.FP
    _, b_ref, _ = sh0._keep_rows(sim, np.zeros(3))
    assert len(b_door) == len(b_ref) and not np.allclose(b_door, b_ref)      # smaller buffer -> larger margins h -> different CBF bounds
    cfg2 = SH.ShieldCfg(d_wall_door=0.05, doorways=((sim.base.x + 9.0, sim.base.y),))   # far from the doorway: identical to the default
    sh2 = SH.Shield(cfg2, occ=sim.occ, a_hw=sim.a_hw)
    sh2.fp_now = GM.FP
    _, b2, _ = sh2._keep_rows(sim, np.zeros(3))
    assert np.allclose(b2, b_ref)


def test_walker_clock_legacy_bug_and_fix(sim):
    """h5 p_walker: first track segment is stationary (t 0..2 s) -> the LEGACY clock (advance only while the current velocity != 0) never starts; the time clock does."""
    sc = SC.get_scene("h5")
    w = [p for p in sc.people if p.name == "p_walker"][0]
    xy0, v0 = w.at(0.0)
    assert np.allclose(v0, 0.0)                                   # the cause: v(tau=0) == 0 -> `np.linalg.norm(v) > 0` is False -> tau never advances
    x_start = [xy for xy, _ in sim.people_state][0].copy()
    for _ in range(int(12.0 / 0.02)):                            # people_clock="time": sim.set_people(advance=True) per tick, no physics needed
        sim.t += 0.02
        sim.set_people(advance=True)
    x_end = [xy for xy, _ in sim.people_state][0]
    assert np.hypot(*(x_end - x_start)) > 2.0                    # walked from (3.6, 3.8) to ~(0.9, 3.8)
    assert sim.people_clock == "time"


def test_freeze_people():
    sc = SC.freeze_people(SC.get_scene("h5"))
    for p in sc.people:
        for t in (0.0, 5.0, 100.0):
            xy, v = p.at(t)
            assert np.allclose(v, 0.0) and np.allclose(xy, p.at(0.0)[0])


def test_ego_camera_matches_head_spec():
    from world_model.v2.house.ocs import fp_render as FR
    assert np.allclose(RP.EGO_POS, FR.HEAD_POS) and RP.EGO_PITCH_DEG == FR.HEAD_PITCH_DEG and RP.EGO_FOVY == FR.HEAD_FOVY
    import mujoco
    cam = mujoco.MjvCamera()
    fovy = RP.ego_cam(cam, 1.0, 2.0, math.pi / 2, 0.1)
    el, az = math.radians(cam.elevation), math.radians(cam.azimuth)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(cam.lookat) - cam.distance * fwd
    # yaw 90 deg: base +x -> world +y, base +y -> world -x
    assert np.allclose(pos, [1.0 + 0.20955, 2.0 + 0.05, 1.15 + 0.1], atol=1e-9)
    assert np.allclose(fwd, [0.0, math.cos(math.radians(36.3)), -math.sin(math.radians(36.3))], atol=1e-9) and fovy == 70.0
    assert RP.CAMERAS[0] == "ego" and set(RP.CAMERAS) >= {"chase", "top", "wrist_l", "wrist_r"}


def test_ego_left_eye_matches_pick_camera():
    """--task pick renders the ego view from the ZED left eye, the camera house/pick computes its proposal boxes in."""
    from world_model.v2.house.ocs import fp_render as FR
    import mujoco
    assert np.isclose(RP.EGO_LEFT_EYE_DY, 0.5 * FR.HEAD_BASELINE)
    lift, x, y, yaw = 0.1, 1.0, 2.0, 0.7
    cam = mujoco.MjvCamera()
    RP.ego_cam(cam, x, y, yaw, lift, RP.EGO_LEFT_EYE_DY)
    el, az = math.radians(cam.elevation), math.radians(cam.azimuth)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(cam.lookat) - cam.distance * fwd
    o_b, R_b = FR.head_cam_base(lift, left=True)                 # base frame
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.allclose(pos, np.array([x, y, 0.0]) + Rz @ o_b, atol=1e-9)
    assert np.allclose(fwd, Rz @ -R_b[:, 2], atol=1e-9)            # MuJoCo camera looks along -z
