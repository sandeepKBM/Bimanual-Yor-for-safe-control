"""skills/actions.py + skills/kin.py: the v2 action DSL plans what it says, refuses unsafe paths, and moves arms in lockstep."""
import mujoco
import numpy as np
import pytest

from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.skills import actions as A
from world_model.v2.nero.pan.skills import kin as KN

NOMINAL = dict(placement="none", hidden_mass=None, perturb=dict(dx=0.0, dy=0.0, yaw_deg=0.0, mu_scale=1.0))
PLAN = dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")


@pytest.fixture(scope="module")
def ep():
    e = S.Episode(NOMINAL, observe=None, **PLAN)
    assert not e.refused
    return e


def _site_after(ep, side, q_arm):
    d = mujoco.MjData(ep.m)
    d.qpos[:] = ep.d.qpos
    d.qpos[ep.addr[side]] = q_arm
    mujoco.mj_forward(ep.m, d)
    sid = ep.m.site(f"{side}_arm_ee_grasp").id
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def test_clearance_limit_matches_judge(ep):
    rng = np.random.default_rng(3)
    for side in S.SIDES:
        L = KN.ClearanceLimit(ep.m, [side], detect=0.05, k=50)
        for _ in range(15):
            d = mujoco.MjData(ep.m)
            d.qpos[:] = ep.d.qpos
            d.qpos[ep.addr[side]] += rng.normal(0, 0.15, len(ep.addr[side]))
            mujoco.mj_forward(ep.m, d)
            mn = min([r[0] for r in L.rows(d)], default=0.05)
            assert abs(min(mn, 0.05) - S._arm_env_clearance(ep.m, d, side)) < 1e-6


def test_moverel_reaches_target_and_moves_one_arm(ep):
    ex = A.Executor(ep)
    p0, R0 = _site_after(ep, "left", ep.d.qpos[ep.addr["left"]])
    goals, dur = ex._arm_goals([A.MoveRel("left", (0.0, 0.0, 0.03))])
    ok, why, wps, clear, err = ex.pl.plan(goals)
    assert ok, why
    assert set(wps) == {"left"}
    p1, R1 = _site_after(ep, "left", wps["left"][-1])
    assert np.linalg.norm(p1 - (p0 + [0, 0, 0.03])) < 2e-3 and A._rot_angle(R1 @ R0.T) < 2e-2


def test_plan_into_table_is_refused(ep):
    ex = A.Executor(ep)
    goals, _ = ex._arm_goals([A.MoveRel("right", (0.0, 0.0, -0.08))])     # right claw starts ~1 cm above the pan table
    ok, why, *_ = ex.pl.plan(goals)
    assert not ok and ("clearance" in why or "IK" in why)


def test_move_object_is_rigid(ep):
    ex = A.Executor(ep)
    goals, _ = ex._object_goals(A.MoveObject(delta=(0.0, 0.0, 0.04)))
    ok, why, wps, clear, err = ex.pl.plan(goals)
    d0 = [_site_after(ep, s, wps[s][0])[0] for s in S.SIDES]
    for i in range(0, len(wps["left"]), 5):
        di = [_site_after(ep, s, wps[s][i])[0] for s in S.SIDES]
        assert np.linalg.norm((di[0] - di[1]) - (d0[0] - d0[1])) < 3e-3


def test_lockstep_paths_share_times(ep):
    sn = ep.snapshot()
    try:
        ex = A.Executor(ep)
        n0 = {s: len(ep.paths[s]) for s in S.SIDES}
        ex.step([A.MoveRel("left", (0.0, 0.0, 0.01)), A.MoveRel("right", (0.0, 0.0, 0.02))])
        new = {s: ep.paths[s][n0[s]] for s in S.SIDES}
        assert ex.ok and new["left"][:2] == new["right"][:2]
    finally:
        ep.restore(sn)
