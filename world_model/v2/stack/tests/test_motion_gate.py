"""K3 gate (owner-written, 11-16): joint / reach / closed-chain rows (stack/checks/motion.py) + the planner adapter (adapters/motion_rows.py).
SIMULATION ONLY.

Pure rows on synthetic arrays with known faults (each fault refuted with the right counterexample), then the adapter on the real closed-chain plans.
Known finding pinned here (raised to the user 11-16): the box hold pose has right J7 within PATH_MARGIN_MIN of its limit, so joint_pos refutes on box.
"""
from __future__ import annotations

import numpy as np
import pytest

from world_model.v2.stack.checks import motion as M


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_joint_pos_inside_and_between_sample_widening():
    t = np.arange(0, 1.0 + 1e-9, 0.005)
    lo, hi = -np.ones(7), np.ones(7)
    q = {"left": 0.5 * np.sin(2 * np.pi * t)[:, None] * np.ones(7), "right": np.zeros((len(t), 7))}
    r = M.joint_pos_row(q, t, lo, hi, margin=0.05)
    assert r.ok and r.guarantee == "proof" and r.margin == pytest.approx(1 - 0.05 - 0.5 - (0.5 * 2 * np.pi) * 0.005 / 2, abs=2e-4)   # max|qd| = 0.5*2pi
    # every sample 1 mrad inside the margin, but moving fast: the widening (max|qd| dt / 2) must refute it
    q2 = {"left": np.zeros((len(t), 7)), "right": np.zeros((len(t), 7))}
    q2["right"][:, 6] = np.where(np.arange(len(t)) % 2 == 0, 0.949, 0.0)   # zig-zag: |qd| ~ 0.949 / 0.005 per step
    r2 = M.joint_pos_row(q2, t, lo, hi, margin=0.05)
    assert not r2.ok and r2.guarantee == "refuted" and r2.worst["side"] == "right" and r2.worst["joint"] == 6
    # a plain violation at a sample
    q3 = {"left": np.zeros((len(t), 7))}
    q3["left"][40, 2] = 1.2
    r3 = M.joint_pos_row(q3, t, lo, hi)
    assert not r3.ok and r3.worst["index"] == 40 and r3.worst["joint"] == 2 and r3.worst["value"] == pytest.approx(1.2)


def test_joint_speed():
    t = np.arange(0, 1.0, 0.005)
    qd = {"left": np.full((len(t), 7), 0.5)}
    assert M.joint_speed_row(qd, t, np.full(7, 1.0)).ok
    qd["left"][10, 4] = 1.5
    r = M.joint_speed_row(qd, t, np.full(7, 1.0))
    assert not r.ok and r.worst["joint"] == 4


def test_reach_separate_tolerances():
    p, rr = np.array([1e-5, 2e-5]), np.array([1e-5, 5e-5])
    assert M.reach_row(p, rr, 1e-4, 1e-4).ok
    r = M.reach_row(p, np.array([1e-5, 3e-4]), 1e-4, 1e-4)
    assert not r.ok and r.worst["index"] == 1 and r.worst["rot_miss"] == pytest.approx(3e-4)
    assert not M.reach_row(np.array([2e-4]), np.array([0.0]), 1e-4, 1.0).ok


def test_rot_angle_and_closed_chain():
    for a in (0.0, 0.003, 0.5, 2.0, np.pi - 1e-6):
        assert M.rot_angle(_rz(a)) == pytest.approx(a, abs=1e-9)
    n = 50
    A = np.tile(np.eye(4), (n, 1, 1))
    A[:, :3, 3] = np.random.default_rng(1).normal(size=(n, 3))
    assert M.closed_chain_row(A, A.copy()).ok and M.closed_chain_row(A, A.copy()).margin == pytest.approx(1.0)
    B = A.copy()
    B[17, :3, 3] += np.array([0.0, 0.003, 0.0])                       # 3 mm > EPS_CC_POS (2 mm)
    r = M.closed_chain_row(A, B)
    assert not r.ok and r.worst["index"] == 17 and r.worst["dpos"] == pytest.approx(0.003)
    C = A.copy()
    C[30, :3, :3] = _rz(0.02)                                          # 0.02 rad > EPS_CC_ROT (0.01)
    r = M.closed_chain_row(A, C)
    assert not r.ok and r.worst["index"] == 30 and r.worst["drot"] == pytest.approx(0.02, abs=1e-9)


@pytest.fixture(scope="module")
def plans():
    from world_model.v2.stack.planners import closed_chain as CCP
    return {task: CCP.plan_for(CCP._nominal_inputs(task, None), 0.5) for task in ("box", "tray4", "box_behind")}


def test_adapter_rows_on_planned_paths(plans):
    from world_model.v2.adapters import motion_rows as MR
    for task, plan in plans.items():
        rows = {r.name: r for r in MR.motion_rows(plan)}
        assert set(rows) == {"joint_pos", "joint_speed", "reach", "closed_chain"}, task
        for name in ("joint_speed", "reach", "closed_chain"):
            assert rows[name].ok and rows[name].guarantee == "proof", (task, name, rows[name].why)
        if task == "box":                                              # finding 11-16 (raised to the user): hold pose J7 inside the 0.05 rad margin
            r = rows["joint_pos"]
            assert not r.ok and r.worst["side"] == "right" and r.worst["joint"] == 6, r.worst
        else:
            assert rows["joint_pos"].ok, (task, rows["joint_pos"].why)
