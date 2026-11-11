"""SRSArm tests.  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_srs_ik.py"""
import mujoco
import numpy as np
import pytest

from world_model.v2.stack.kin.srs_ik import SRSArm
from world_model.v2.stack.kin.validate_srs import load_model


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.mark.parametrize("side", ["left", "right"])
def test_roundtrip_and_limits(model, side):
    arm = SRSArm(model, side)
    assert arm.shoulder_gap < 1e-6 and arm.wrist_gap < 1e-6
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(arm.lo, arm.hi)
        T = arm.fk(q)
        psi = arm.arm_angle(q)
        res = arm.ik_full(T, psi)
        assert res and all(r["pos_err"] < 1e-6 and r["rot_err"] < 1e-6 for r in res)
        dq = min(np.abs((r["q"] - q + np.pi) % (2 * np.pi) - np.pi).max() for r in res)
        assert dq < 1e-4                                          # original q is one of the branches
        for qq in arm.ik(T, psi, limits_only=True):               # limits respected
            assert np.all(qq >= arm.lo) and np.all(qq <= arm.hi)
        # POE kinematics equal MuJoCo FK
        Rp, pp, _, _ = arm._fk_poe(q)
        Tl = np.linalg.inv(arm.Tb) @ T
        assert np.allclose(pp, Tl[:3, 3], atol=1e-6) and np.allclose(Rp, Tl[:3, :3], atol=1e-6)


def test_psi_intervals_consistent(model):
    arm = SRSArm(model, "left")
    rng = np.random.default_rng(2)
    q = rng.uniform(arm.lo, arm.hi)
    T = arm.fk(q)
    psi = arm.arm_angle(q)
    found = 0
    for br in [(s4, a, b) for s4 in (0, 1) for a in (0, 1) for b in (0, 1)]:
        iv = arm.psi_intervals(T, br)
        for lo, hi in iv:                                          # midpoint of each interval is feasible
            mid = 0.5 * (lo + hi)
            r = [x for x in arm.ik_full(T, mid) if x["branch"] == br][0]
            assert r["in_limits"]
        if any(lo - 1e-6 <= ((psi - lo + np.pi) % (2 * np.pi) - np.pi) + lo <= hi + 1e-6 or lo - 1e-6 <= psi + 2 * np.pi <= hi + 1e-6
               for lo, hi in iv):
            found += 1
    assert found >= 1                                              # the source configuration's own branch contains its psi
