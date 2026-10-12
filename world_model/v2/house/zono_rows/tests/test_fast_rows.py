"""equivalence tests for the numba zonotope-row path (zono_fast.py + the FAST branches of rows.py / base_filter.py).  SIMULATION only, CPU.
    PYTHONPATH=. python -m world_model.v2.house.zono_rows.tests.test_fast_rows        (or pytest)

The reference is the original dense numpy implementation (rows.zono_rows_from_states_ref / base_filter.facet_pairs_ref and the numpy ranking in
base_rows_from_points), still in the tree and selected by ZONO_ROWS_REF=1 / rows.FAST = False.  Recorded states: tests/data/s2_states_m64_l1t1_bf1_uni.npz =
150 controller-input states (every 10th step) of the S2 run m64_l1t1_bf1_uni (base moving toward the bottle, arms + tracked tray + ball).
Criterion: max |row difference| <= 1e-9 and the identical (keypoint, zonotope) pair list, for max_rows in {8, 64, None} and 3 base-filter configs."""
from __future__ import annotations

import functools
import math
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_FLMPC_IMPL", "rt")

import numpy as np
import mujoco

from world_model.v2.house.zono_rows import base_filter as BF
from world_model.v2.house.zono_rows import rows as R
from world_model.v2.house.zono_rows import zono_fast as ZF
from world_model.v2.house.zono_rows.ctl import ZonoCtl

DATA = Path(__file__).parent / "data" / "s2_states_m64_l1t1_bf1_uni.npz"
TOL = 1e-9


# ------------------------------------------------------------------------------------------------------------ pure geometry (no MuJoCo)
def _boxes(rng, n):
    zs = []
    for _ in range(n):
        m = int(rng.integers(3, 8))
        zs.append((rng.uniform(-2, 2, 3), rng.normal(size=(3, m)) * rng.uniform(0.05, 0.7)))
    return R.ZonoSet(zs)


def test_pairs_grid_dense_and_reference_agree():
    """random zonotopes / points / radii / d_act (incl. points outside the grid, negative-ish margins): grid == dense == the numpy reference, same order."""
    rng = np.random.default_rng(7)
    n_pairs = 0
    for trial in range(40):
        zs = _boxes(rng, int(rng.integers(1, 60)))
        K = int(rng.integers(1, 120))
        p = rng.uniform(-3.5, 3.5, (K, 3))
        r = rng.choice([0.0, 0.02, 0.05, 0.1], K)
        d_act = float(rng.choice([0.05, 0.15, 0.4]))
        margin = float(rng.choice([0.0, 0.02, 0.03]))
        pk = zs.packed()
        g = ZF.facet_pairs(p, r, pk, d_act, margin, use_grid=True)
        d_ = ZF.facet_pairs(p, r, pk, d_act, margin, use_grid=False)
        kk, nrm, h, zii = BF.facet_pairs_ref(p, r, zs, margin, d_act)
        for a, b in zip(g, d_):
            assert np.array_equal(a, b)
        assert np.array_equal(g[0], kk) and np.array_equal(g[1], zii)
        assert np.abs(g[3] - nrm).max(initial=0.0) <= 1e-15 and np.abs(g[4] - h).max(initial=0.0) <= 1e-12
        n_pairs += len(kk)
    assert n_pairs > 500
    print(f"grid == dense == numpy reference on {n_pairs} (point, zonotope) pairs over 40 random sets")


def test_selection_matches_python_loop():
    """ZF.select_urgent == the original python dict loop (key = zonotope + rounded normal) on random candidate lists with duplicated normals."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        zs = _boxes(rng, 6)
        pk = zs.packed()
        R_ = int(rng.integers(5, 200))
        f = rng.integers(0, len(pk.Nall), R_)
        free = rng.normal(size=R_)
        order = np.argsort(free, kind="stable")
        mr, pf = int(rng.integers(1, 20)), int(rng.integers(1, 4))
        seen, keep = {}, []
        zi = pk.zid[f]
        for i in order:
            key = (int(zi[i]), tuple(np.round(pk.Nall[f][i], 3)))
            if seen.get(key, 0) >= pf:
                continue
            seen[key] = seen.get(key, 0) + 1
            keep.append(i)
            if len(keep) >= mr:
                break
        assert np.array_equal(np.array(keep, dtype=np.int64), ZF.select_urgent(order, pk.gid3[f], pf, mr))
    print("select_urgent == the python (zonotope, rounded normal) loop")


# ------------------------------------------------------------------------------------------------------------ recorded S2 states
@functools.lru_cache(maxsize=1)
def _scene():
    from world_model.v2.house.zed_demo import rig as Z, execute as EX
    from world_model.v2.house.zono_rows import smoke_s2 as S
    C, _gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=False)
    mujoco.mj_forward(C.m, C.d)
    zonos, names, _meta = S.load_zonos()
    zs = R.ZonoSet(zonos, names)
    ctl = ZonoCtl(C, zonos=zs, names=names, margin=0.02, alpha1=10, alpha2=10, d_act=0.15, max_rows=64, tilt=True, refine=0, tray_level=True, track_bodies=["tray", "ball"])
    bf = BF.BaseFilter(ctl, BF.BaseCfg(), origin_xy=C.d.mocap_pos[0][:2].copy(), yaw=float(C.info["yaw"]), data=C.d)
    return C, ctl, bf, zs, dict(np.load(DATA))


def _load(C, ctl, st, i, d):
    d.qpos[:] = st["qpos"][i]; d.qvel[:] = st["qvel"][i]; d.mocap_pos[:] = st["mp"][i]; d.mocap_quat[:] = st["mq"][i]
    bt = st["bt"][i]
    ctl.set_base_motion((bt[0], bt[1]), bt[2], (bt[3], bt[4]))
    mujoco.mj_forward(C.m, d)


def test_fast_rows_equal_reference_on_recorded_states():
    C, ctl, bf, zs, st = _scene()
    d = mujoco.MjData(C.m)
    worst, nrows, ncap = 0.0, 0, 0
    old = R.FAST
    try:
        for mr in (8, 64, None):
            for i in range(len(st["qpos"])):
                _load(C, ctl, st, i, d)
                R.FAST = False
                a = ctl.zono_rows(d, max_rows=mr)
                R.FAST = True
                b = ctl.zono_rows(d, max_rows=mr)
                for s in (0, 1):
                    assert a[s][0].shape == b[s][0].shape, (mr, i, a[s][0].shape, b[s][0].shape)
                    assert a[2][s].get("ids") == b[2][s].get("ids"), (mr, i, s)
                    if len(a[s][1]):
                        worst = max(worst, float(np.abs(a[s][0] - b[s][0]).max()), float(np.abs(a[s][1] - b[s][1]).max()))
                        nrows += len(a[s][1])
                        ncap += int(a[2][s].get("dropped", 0) > 0)
    finally:
        R.FAST = old
    assert nrows > 5000 and ncap > 50, (nrows, ncap)
    assert worst <= TOL, worst
    print(f"FL-MPC zonotope rows, fast vs reference: {nrows} rows, {ncap} capped sets, max|diff| {worst:.2e}, identical pair lists")


def test_fast_base_rows_equal_reference_on_recorded_states():
    C, ctl, bf, zs, st = _scene()
    d = mujoco.MjData(C.m)
    worst, nrows, ncap = 0.0, 0, 0
    old = R.FAST
    try:
        for cfgk in (dict(), dict(max_rows=12, per_facet=2), dict(max_rows=10 ** 6, per_facet=10 ** 6, d_act=0.6)):
            cfg = BF.BaseCfg(**cfgk)
            for i in range(len(st["qpos"])):
                _load(C, ctl, st, i, d)
                org = st["mp"][i][0][:2]
                R.FAST = False
                p, r, va, nf = bf.points(d, org, float(C.info["yaw"]))
                A0, b0, m0 = BF.base_rows_from_points(p, r, va, zs, org, cfg, u_ref=np.array([0.0, 0.2, 0.0]))
                R.FAST = True
                p1, r1, va1, nf1 = bf.points(d, org, float(C.info["yaw"]))
                A1, b1, m1 = BF.base_rows_from_points(p1, r1, va1, zs, org, cfg, u_ref=np.array([0.0, 0.2, 0.0]))
                assert np.abs(p - p1).max() <= TOL and np.abs(va - va1).max() <= TOL and np.array_equal(r, r1)
                assert A0.shape == A1.shape and np.array_equal(m0["kk"], m1["kk"]) and np.array_equal(m0["zii"], m1["zii"]) and m0["dropped"] == m1["dropped"]
                if len(b0):
                    worst = max(worst, float(np.abs(A0 - A1).max()), float(np.abs(b0 - b1).max()))
                    nrows += len(b0)
                    ncap += int(m0["dropped"] > 0)
    finally:
        R.FAST = old
    assert nrows > 5000 and ncap > 50, (nrows, ncap)
    assert worst <= TOL, worst
    print(f"base-filter rows, fast vs reference: {nrows} rows, {ncap} capped sets, max|diff| {worst:.2e}")


def test_tracked_velocity_cvel_equals_jacobian_form():
    C, ctl, bf, zs, st = _scene()
    d = mujoco.MjData(C.m)
    w = 0.0
    old = R.FAST
    try:
        for i in range(0, len(st["qpos"]), 5):
            _load(C, ctl, st, i, d)
            R.FAST = True
            pa, va = R.tracked_measured(C.m, d, ctl.tracked)
            R.FAST = False
            pb, vb = R.tracked_measured(C.m, d, ctl.tracked)
            w = max(w, float(np.abs(pa - pb).max()), float(np.abs(va - vb).max()))
    finally:
        R.FAST = old
    assert w <= 1e-12, w
    print(f"tracked point velocity: cvel form == mj_jac form, max|diff| {w:.1e}")


if __name__ == "__main__":
    test_pairs_grid_dense_and_reference_agree()
    test_selection_matches_python_loop()
    test_fast_rows_equal_reference_on_recorded_states()
    test_fast_base_rows_equal_reference_on_recorded_states()
    test_tracked_velocity_cvel_equals_jacobian_form()
    print("ALL PASSED")
