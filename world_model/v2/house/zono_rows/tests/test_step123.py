"""tests for the relax / tray_level / tracked-keypoint additions of zono_rows (SIMULATION only, CPU).  Run:
    PYTHONPATH=. python -m world_model.v2.house.zono_rows.tests.test_step123     (or pytest)
The finite-difference tests build the S2 desk scene once (~20 s) -- the tray is welded to both grippers there."""
from __future__ import annotations

import functools
import math
import os
import types

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import mujoco

from world_model.v2.house import phase2 as P2
from world_model.v2.house.zono_rows import rows as R
from world_model.v2.house.zono_rows.ctl import ZonoCtl
from world_model.v2.house.zono_rows.relax import relax_search

REPORT = {}
A1 = A2 = 10.0


# ------------------------------------------------------------------------------------------------------------ 1. relaxation search
def test_relax_search_logic():
    thr = 0.37
    for s_prev in (0.0, 0.2, 3.0, 40.0):
        calls = []
        res = relax_search(lambda s: calls.append(s) or s >= thr, s_prev=s_prev)
        assert res["certified"] and thr <= res["s"] <= thr * 1.125 + 1e-9, (s_prev, res)
        assert res["n_solves"] == len(calls) <= 12, (s_prev, res["n_solves"])
    # small threshold below s0: certified at the first try, absolute error <= s0 / 8
    res = relax_search(lambda s: s >= 0.003)
    assert res["certified"] and 0.003 <= res["s"] <= 0.003 + 0.05 / 8 + 1e-12, res
    # never certifies -> unrecovered, bounded number of solves, last try is s_max
    calls = []
    res = relax_search(lambda s: calls.append(s) or False)
    assert not res["certified"] and res["s"] is None and len(calls) <= 12 and max(calls) == 50.0, (res, calls)
    print("relax_search: thr 0.37 -> s in [0.37, 0.416] for s_prev in (0, 0.2, 3, 40), <= 12 solves; unrecovered path OK")


def _fake_ctl(thr):
    """ZonoCtl shell with a fake FL-MPC whose 'solve' certifies iff s >= thr; records the a_prev state seen by every solve."""
    c = object.__new__(ZonoCtl)
    fl = types.SimpleNamespace(a_ego_prev=np.array([1.0, 2.0]), a_int_prev=np.array([3.0, 4.0]), last_fallback=2, seen=[])
    c.fl, c._s_prev = fl, 0.0
    c.relax_kw = dict(s0=0.05, s_max=50.0, n_bisect=3, max_solves=12)

    def solve_at(s):
        fl.seen.append((fl.a_ego_prev.copy(), fl.a_int_prev.copy()))
        ok = s >= thr
        fl.last_fallback = 0 if ok else 2
        fl.a_ego_prev = np.full(2, 10.0 + s)                    # what a real solve would overwrite
        return np.full(6, s), np.full(6, -s), ok, fl.last_fallback

    return c, fl, solve_at


def test_relax_step_wrapper():
    pre = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    post = (np.array([-7.0, -7.0]), np.array([-8.0, -8.0]))
    c, fl, solve_at = _fake_ctl(0.37)
    r = c._relax_step(solve_at, (np.zeros(6), np.zeros(6), False, 2), pre, post)
    assert not r["unrecovered"] and 0.37 <= r["s"] <= 0.37 * 1.125 and r["fb"] == 0 and r["n_solves"] <= 12
    assert fl.last_fallback == 0 and np.allclose(fl.a_ego_prev, 10.0 + r["s"])        # status/state are the ACCEPTED solve's, not the last trial's
    assert all(np.array_equal(a, pre[0]) and np.array_equal(b, pre[1]) for a, b in fl.seen)       # state restored before every extra solve
    assert np.allclose(r["te"], r["s"])
    c, fl, solve_at = _fake_ctl(1e9)
    r = c._relax_step(solve_at, ("TE", "TI", False, 2), pre, post)
    assert r["unrecovered"] and r["te"] == "TE" and r["ti"] == "TI" and r["fb"] == 2 and r["s"] == 0.0 and r["n_solves"] <= 12
    assert np.array_equal(fl.a_ego_prev, post[0]) and np.array_equal(fl.a_int_prev, post[1]) and fl.last_fallback == 2
    print("ZonoCtl._relax_step: certified path restores a_prev per solve; unrecovered path keeps FL-MPC's own output + state")


# ------------------------------------------------------------------------------------------------------------ 2. finite differences (S2 scene)
@functools.lru_cache(maxsize=1)
def _scene():
    from world_model.v2.house.zed_demo import rig as Z, execute as EX
    C, _gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=False)
    mujoco.mj_forward(C.m, C.d)
    ctl = ZonoCtl(C, zonos=None, tray_level=True, track_bodies=["tray", "ball"], tilt=True)
    ctl.pristine = mujoco.MjData(C.m)
    mujoco.mj_copyData(ctl.pristine, C.m, C.d)
    return C, ctl, mujoco.MjData(C.m)


def _fresh(ctl):
    """a private copy of the pristine hold state (the tests must not contaminate each other)."""
    d = mujoco.MjData(ctl.m)
    mujoco.mj_copyData(d, ctl.m, ctl.pristine)
    mujoco.mj_forward(ctl.m, d)
    return d


def _arm_terms(ctl, d, da):
    M = ctl.sc.full_mass_matrix(ctl.m, d)[np.ix_(da, da)]
    return M, np.linalg.inv(M), d.qfrc_bias[da] - d.qfrc_passive[da]


def _row_hs(ctl, d, side, da):
    """(tilt info, band info, (a_tilt, b_tilt), (a_band, b_band)) at the current (forwarded) state."""
    M, Mi, c = _arm_terms(ctl, d, da)
    it, ib = {}, {}
    if side == "right":
        at, bt = R.tray_tilt_row(ctl.m, d, side, da, Mi, c, P2.TILT_MAX, A1, A2, info=it)
    else:
        at = bt = None
    ab, bb = R.grip_height_rows(ctl.m, d, side, da, Mi, c, ctl.mount, ctl.z_ref_test[side], ctl.level_delta, A1, A2, info=ib)
    return it, ib, (at, bt), (ab, bb), M, c


def _predict_hddot(a, b, tau, h, hdot):
    """HOCBF row psi2 = b - a.tau = hddot + A1 hdot + A2 (hdot + A1 h)  ->  predicted hddot."""
    return float(b - a @ tau) - A1 * hdot - A2 * (hdot + A1 * h)


def _rel(pred, fd):
    return abs(pred - fd) / max(abs(fd), 1e-4)


def test_fd_tilt_and_band_analytic():
    """(i) right tilt row and (ii) height-band rows vs central finite differences under a constant joint acceleration a (exact kinematics)."""
    C, ctl, _ = _scene()
    m, d0 = C.m, _fresh(ctl)
    rng = np.random.default_rng(0)
    da, qa = ctl.rd, ctl.rq
    d0.qpos[qa] += rng.normal(0, 0.12, 6)                       # tilt the weld-implied tray axis off vertical (u_z = 1 is a stationary point of h)
    d0.qvel[da] = rng.normal(0, 0.4, 6)
    mujoco.mj_forward(m, d0)
    ctl.z_ref_test = {s: R.tray_origin_height(m, d0, s, ctl.mount)[1] + (0.004 if s == "right" else -0.003) for s in ("left", "right")}
    a = rng.normal(0, 1.0, 6)
    qd0, q0 = d0.qvel[da].copy(), d0.qpos[qa].copy()
    it, ib, (at, bt), (ab, bb), M, c = _row_hs(ctl, d0, "right", da)
    tau = M @ a + c
    d = mujoco.MjData(m)
    eps = 1e-4

    def at_eps(e):
        mujoco.mj_copyData(d, m, d0)
        d.qpos[qa] = q0 + e * qd0 + 0.5 * e * e * a
        d.qvel[da] = qd0 + e * a
        mujoco.mj_forward(m, d)
        u = d.xmat[ctl.gb["right"]].reshape(3, 3) @ R.tray_weld(m, "right")[2][:, 2]
        return u[2] - math.cos(P2.TILT_MAX), R.tray_origin_height(m, d, "right", ctl.mount)[1]

    hp, zp = at_eps(eps); hm, zm = at_eps(-eps); h0, z0 = at_eps(0.0)
    hd_fd, hdd_fd = (hp - hm) / (2 * eps), (hp - 2 * h0 + hm) / eps ** 2
    zd_fd, zdd_fd = (zp - zm) / (2 * eps), (zp - 2 * z0 + zm) / eps ** 2
    e_tilt_d = _rel(it["tilt_hdot"], hd_fd)
    e_tilt_dd = _rel(_predict_hddot(at[0], bt[0], tau, it["tilt_h"], it["tilt_hdot"]), hdd_fd)
    # band: row 0 is h = z - (zref - delta) (hdot = +zdot), row 1 is h = (zref + delta) - z (hdot = -zdot)
    assert abs(ib["band_h"][0] - (z0 - ctl.z_ref_test["right"] + ctl.level_delta)) < 1e-12
    e_band_d = max(_rel(ib["band_hdot"][0], zd_fd), _rel(ib["band_hdot"][1], -zd_fd))
    e_band_dd = max(_rel(_predict_hddot(ab[0], bb[0], tau, ib["band_h"][0], ib["band_hdot"][0]), zdd_fd),
                    _rel(_predict_hddot(ab[1], bb[1], tau, ib["band_h"][1], ib["band_hdot"][1]), -zdd_fd))
    print(f"FD analytic (eps {eps:g}): tilt hdot {e_tilt_d:.2e}, tilt hddot {e_tilt_dd:.2e}; band hdot {e_band_d:.2e}, band hddot {e_band_dd:.2e}  "
          f"(|hdot|={abs(hd_fd):.3f}, |hddot|={abs(hdd_fd):.3f}, |zdot|={abs(zd_fd):.3f}, |zddot|={abs(zdd_fd):.3f})")
    REPORT["fd_analytic"] = dict(tilt_hdot=e_tilt_d, tilt_hddot=e_tilt_dd, band_hdot=e_band_d, band_hddot=e_band_dd)
    assert max(e_tilt_d, e_tilt_dd, e_band_d, e_band_dd) < 1e-4


def test_fd_physics_step():
    """(i)/(ii) again with a REAL mj_step under a random small torque (arm acceleration read back from the step; the weld adds constraint force,
    so tau_eff = M a_meas + c is used in the row).  Forward-difference error is O(dt) (semi-implicit Euler) -> looser tolerance."""
    C, ctl, d1 = _scene()
    m, d0 = C.m, _fresh(ctl)
    rng = np.random.default_rng(1)
    da = ctl.rd
    for _ in range(40):                                          # let a random torque tilt the grippers off vertical (u_z = 1 is stationary)
        d0.qfrc_applied[:] = 0.0
        d0.qfrc_applied[da] = 0.5 * np.sin(0.05 * _ + np.arange(6))
        mujoco.mj_step(m, d0)
    d0.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d0)
    ctl.z_ref_test = {s: R.tray_origin_height(m, d0, s, ctl.mount)[1] for s in ("left", "right")}
    mujoco.mj_copyData(d1, m, d0)
    d1.qfrc_applied[:] = 0.0
    d1.qfrc_applied[da] = rng.normal(0, 0.2, 6)
    mujoco.mj_step(m, d1)
    dt = m.opt.timestep
    a_meas = (d1.qvel[da] - d0.qvel[da]) / dt
    it0, ib0, (at, bt), (ab, bb), M, c = _row_hs(ctl, d0, "right", da)
    mujoco.mj_forward(m, d1)
    it1, ib1, *_ = _row_hs(ctl, d1, "right", da)
    tau = M @ a_meas + c
    hdd_fd = (it1["tilt_hdot"] - it0["tilt_hdot"]) / dt
    zdd_fd = (ib1["band_hdot"][0] - ib0["band_hdot"][0]) / dt
    e1 = _rel(_predict_hddot(at[0], bt[0], tau, it0["tilt_h"], it0["tilt_hdot"]), hdd_fd)
    e2 = _rel(_predict_hddot(ab[0], bb[0], tau, ib0["band_h"][0], ib0["band_hdot"][0]), zdd_fd)
    # semi-implicit Euler: h1 - h0 = dt * hdot(q1, v1)  ->  compare the prediction AT THE NEW STATE with the backward difference
    e3 = _rel(it1["tilt_hdot"], (it1["tilt_h"] - it0["tilt_h"]) / dt)
    e4 = _rel(ib1["band_hdot"][0], (ib1["band_h"][0] - ib0["band_h"][0]) / dt)
    print(f"FD physics step (dt {dt*1e3:.0f} ms, |a_meas| {np.linalg.norm(a_meas):.2f}): tilt hddot {e1:.2e}, band hddot {e2:.2e}; "
          f"tilt hdot {e3:.2e} (|hdot| {abs(it1['tilt_hdot']):.3f}), band hdot {e4:.2e}")
    REPORT["fd_physics"] = dict(tilt_hddot=e1, band_hddot=e2, tilt_hdot=e3, band_hdot=e4)
    assert max(e1, e2, e3, e4) < 0.10          # tilt |hdot| is only ~5e-3 here (u_z near its maximum) -> few-% second-order mismatch; analytic test is the tight one


def test_fd_tracked_tray_moving_base():
    """(iii) tracked tray point, mocap base moving: predicted hdot = n . v_measured vs the backward difference of h = n . p - d over one real step
    (semi-implicit Euler: p_k - p_{k-1} = dt v_k for pure translation).  Also shows that the arm-only J qd would MISS the base motion."""
    C, ctl, _ = _scene()
    m = C.m
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, ctl.pristine)
    rng = np.random.default_rng(2)
    v_base = np.array([0.30, 0.10])
    x0 = d.mocap_pos[0].copy()
    dt = m.opt.timestep
    nrm = np.array([0.6, 0.8, 0.0])
    da, gb = ctl.rd, ctl.gb["right"]
    errs, miss = [], []
    p_prev = None
    for k in range(160):
        d.mocap_pos[0] = x0 + np.array([v_base[0], v_base[1], 0.0]) * (k + 1) * dt
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = rng.normal(0, 0.3, 6); d.qfrc_applied[da] = rng.normal(0, 0.3, 6)
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        p, J, Jd, v, qd = R.tracked_states(m, d, ctl.tracked, da, gb)
        if p_prev is not None and k > 100:                                          # base velocity settled
            fd = nrm @ (p - p_prev).T / dt                                          # (K,)
            pred = v @ nrm
            errs.append(np.abs(pred - fd).max() / np.abs(fd).max())
            miss.append(np.abs(np.einsum("kij,j->ki", J, qd) @ nrm - fd).max() / np.abs(fd).max())
        p_prev = p
    e, mi = float(np.max(errs)), float(np.median(miss))
    print(f"FD tracked tray points (base {np.linalg.norm(v_base):.2f} m/s, {len(ctl.tracked)} pts): hdot rel err max {e:.2e}; arm-only J qd would be off by {mi:.2e} (median rel)")
    REPORT["fd_tracked"] = dict(hdot=e, arm_only_miss=mi)
    assert e < 0.02 and mi > 0.3


def test_tracked_matches_fixed_offsets_and_ctl_smoke():
    """tracked tray rim points sit within a few mm of the weld-offset keypoints; ZonoCtl.tau with every option on runs and returns finite torques."""
    C, ctl, _ = _scene()
    m, d = C.m, _fresh(ctl)
    fixed = R.tray_keypoints(m, "right", include_ball=False)
    pf = R.kp_states(m, d, fixed, ctl.rd)[0]
    pt = R.tracked_states(m, d, R.tracked_keypoints(m, ["tray"]), ctl.rd, ctl.gb["right"])[0]
    assert pf.shape == pt.shape and np.abs(pf - pt).max() < 5e-3, np.abs(pf - pt).max()
    b = m.body("ball").id
    rt = R.tracked_keypoints(m, ["ball"])
    assert abs(rt.radii[0] - 0.03) < 1e-9 and len(rt) == 1
    dd = mujoco.MjData(m)
    mujoco.mj_copyData(dd, m, d)
    zono = (d.xpos[m.body("tray").id] + np.array([0.0, 0.25, 0.05]), np.diag([0.04, 0.04, 0.15]))
    ctl.set_zonos([zono])
    ctl.z_ref = {}
    for _ in range(5):
        te, ti = ctl.tau(dd)
        assert np.isfinite(te).all() and np.isfinite(ti).all()
        dd.qfrc_applied[ctl.ld] = te; dd.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, dd)
    s = ctl.summary(skip=0)
    assert s["steps"] == 5
    print(f"tracked-vs-fixed rim max offset {np.abs(pf - pt).max()*1e3:.2f} mm; ball radius from geom {rt.radii[0]:.3f}; ZonoCtl.tau smoke OK ({s['rows_per_step_mean']:.1f} zono rows/step)")


if __name__ == "__main__":
    test_relax_search_logic()
    test_relax_step_wrapper()
    test_fd_tilt_and_band_analytic()
    test_fd_physics_step()
    test_fd_tracked_tray_moving_base()
    test_tracked_matches_fixed_offsets_and_ctl_smoke()
    print("ALL PASSED", REPORT)
