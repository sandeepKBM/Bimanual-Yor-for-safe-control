"""PROPRIO-ONLY grasp assist: gripper sensor model, payload estimator (synthetic arm, mj_inverse truth), RGB cup threshold, gap/effort verify logic.
python -m pytest -q world_model/v2/house/grasp/tests/test_proprio.py"""
import os

import mujoco
import numpy as np
import pytest

from world_model.v2.house.grasp import proprio as PR
from world_model.v2.house.grasp.wrist import WristCam


class _FakeSim:
    class D:
        pass

    def __init__(self):
        self.d = self.D()
        self.d.qpos = np.zeros(4)
        self.d.actuator_force = np.zeros(2)


def test_gripper_sensor_noise_and_quantisation():
    sim = _FakeSim()
    sim.d.qpos[:] = [0.0030, -0.0030, 0, 0]
    sim.d.actuator_force[:] = [-6.0, 6.0]
    g = PR.GripperSensor(sim, {"left": (0, 1)}, {"left": (0, 1)}, seed=3)
    gaps = np.array([g.gap("left") for _ in range(4000)])
    eff = np.array([g.effort("left") for _ in range(4000)])
    assert abs(gaps.mean() - 0.006) < 3e-5 and 1.5e-4 < gaps.std() < 3e-4
    assert abs(eff.mean() - 12.0) < 0.05 and 0.2 < eff.std() < 0.4 and (eff >= 0).all()
    assert np.allclose(gaps / PR.GAP_QUANT, np.round(gaps / PR.GAP_QUANT), atol=1e-6)
    sim.d.actuator_force[:] = 0.0                       # closed on nothing: effort ~ 0 (never negative)
    assert max(g.effort("left") for _ in range(200)) < 1.5


def _two_arms():
    arm = ""
    for s in ("l", "r"):
        arm += f'<body name="{s}0" pos="{0 if s == "l" else 1} 0 1">'
        for i in range(6):
            arm += f'<joint name="{s}j{i}" type="hinge" axis="{"0 1 0" if i % 2 == 0 else "1 0 0"}" damping="0.01"/><geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.6"/><body name="{s}b{i}" pos="0.15 0 0">'
        arm += f'<geom type="sphere" size="0.02" mass="0.3"/>' + "</body>" * 6 + "</body>"
    return mujoco.MjModel.from_xml_string(f'<mujoco><option gravity="0 0 -9.81"/><worldbody>{arm}</worldbody></mujoco>')


@pytest.mark.parametrize("mass", [0.0, 0.4, 0.9])
def test_payload_estimator_recovers_mass_from_torque(mass):
    m = _two_arms()
    d = mujoco.MjData(m)
    d.qpos[:] = np.random.default_rng(0).uniform(-0.6, 0.6, m.nq)
    ld = np.arange(0, 6)
    rd = np.arange(6, 12)
    tips = [m.body("lb5").id, m.body("rb5").id]
    est = PR.PayloadEstimator(m, ld, rd, tips, seed=1)
    mujoco.mj_forward(m, d)
    pt = [d.xipos[b].copy() for b in tips]
    d.xfrc_applied[:] = 0
    for b in tips:
        d.xfrc_applied[b, 2] = -mass * 9.81            # the payload weight pulls down on each tip (true physics, not the estimator's formula)
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    qf_ext = np.zeros(m.nv)
    for b in tips:                                      # J^T xfrc
        tmp = np.zeros(m.nv)
        mujoco.mj_applyFT(m, d, d.xfrc_applied[b, :3], np.zeros(3), d.xipos[b], b, tmp)
        qf_ext += tmp
    tau_needed = d.qfrc_inverse - qf_ext                # applied torque that holds qacc = 0 against gravity + payload
    d.qvel[:] = 0
    for _ in range(60):
        d.qfrc_applied[:] = tau_needed
        est.accumulate_substep(d)
        est.accumulate_substep(d)
        est.end_tick(d, np.array(pt), collect=True)
    mh, n = est.estimate()
    assert (n > 30).all()
    assert np.all(np.abs(mh - mass) < 0.05 + 0.05 * mass), (mh, mass)


def test_rgb_cup_threshold_ignores_other_colours():
    img = np.full((90, 160, 3), 0.55)
    img[30:50, 60:80] = (0.85, 0.15, 0.15)                 # cup
    img[0:20, 0:20] = (0.45, 0.45, 0.45)                   # grey robot
    img[60:80, 100:130] = (0.6, 0.45, 0.4)                 # unsaturated brown (wall/shelf): must not count
    img = np.clip(img + np.random.default_rng(0).normal(0, 0.03, img.shape), 0, 1)
    n, u, v, rng = WristCam.seg_cup_rgb(img)
    assert 380 <= n <= 420 and abs(u - 69.5) < 1 and abs(v - 39.5) < 1 and np.isnan(rng)
    assert WristCam.seg_cup_rgb((img * 255).astype(np.uint8))[0] == n
    assert WristCam.seg_cup_rgb(np.full((90, 160, 3), 0.5)) is None


# ---------------------------------------------------------------------------------------------------------------- assist logic (one friction rig build)
@pytest.fixture(scope="module")
def fric():
    from world_model.v2.house.teleop import scene as SC
    from world_model.v2.house.teleop.sim import TeleopSim
    sc = SC.get_scene("h5")
    return TeleopSim("h5", counter=True, seed=0, spawn=sc.waypoints["cafe_counter_pick"], grasp_mode="friction", with_people=False)


def _samples(a, gap_mm, eff, slope_mm_s=0.0, n=20, noise=0.2):
    rng = np.random.default_rng(0)
    t = np.arange(n) * 0.02
    out = []
    for i in range(n):
        g = (gap_mm + slope_mm_s * t[i]) * 1e-3 + rng.normal(0, noise * 1e-3)
        out.append(([g, g], [eff + rng.normal(0, 0.3)] * 2))
    return out


def test_verify_gripper_gap_effort_logic(fric):
    a = fric.assist
    a.cfg.proprio = True
    try:
        a.verify_samples = _samples(a, 11.5, 22.0)                          # plate between the pads: gap in range, stalled, effort high
        assert a._verify_gripper()[0]
        a.verify_samples = _samples(a, 0.6, 0.3)                            # closed on nothing
        assert not a._verify_gripper()[0]
        a.verify_samples = _samples(a, 11.5, 0.5)                           # gap right but no squeeze (jaws not pressing)
        assert not a._verify_gripper()[0]
        a.verify_samples = _samples(a, 12.0, 22.0, slope_mm_s=-40.0)        # still closing
        assert not a._verify_gripper()[0]
        a.verify_samples = _samples(a, 25.0, 22.0)                          # something much thicker than the plate
        assert not a._verify_gripper()[0]
    finally:
        a.cfg.proprio = False


def test_operator_cue_uses_estimate_not_tray_body(fric):
    from world_model.v2.house.teleop.arm import ObjectFrame
    sim, a = fric, fric.assist
    of = sim.of
    pt = sim.d.xpos[sim.tray].copy()
    Rt = sim.d.xmat[sim.tray].reshape(3, 3).copy()
    e_true = of.handle_errors(sim.d)
    e_est = of.handle_errors(sim.d, tray=(pt, Rt))
    assert np.allclose(e_true, e_est)                                       # an exact estimate reproduces the privileged cue
    e_shift = of.handle_errors(sim.d, tray=(pt + np.array([0.03, 0, 0]), Rt))
    assert abs(np.mean([e[0] for e in e_shift]) - np.mean([e[0] for e in e_true])) > 0.005          # the cue follows the estimate
    assert all(np.isinf(e[0]) for e in of.handle_errors(sim.d, tray=None))
    ep, er = of.grasp_error_vec(sim.d, tray=None)
    assert not ep.any() and not er.any()
    a.cfg.proprio = True
    try:
        assert sim._tray_obs_kw() == {}                                     # before the grasp the operator model keeps its own view (cameras are blind at the hold geometry)
        os.environ["BIMANUAL_PROPRIO_OPERATOR"] = "1"
        assert sim._tray_obs_kw() == dict(tray=None)                        # strict: no wrist-camera estimate -> no cue, can_grasp False
        assert not sim.can_grasp()
        assert a.tray_from_hands() is None                                  # no grasp offset before a grasp
    finally:
        os.environ.pop("BIMANUAL_PROPRIO_OPERATOR", None)
        a.cfg.proprio = False
    assert sim._tray_obs_kw() == {}                                         # default: the original privileged path
