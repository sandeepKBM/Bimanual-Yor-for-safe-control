"""tests for the base-twist zonotope CBF (base_filter.py) and the selective per-row relaxation (ctl._selective_step).  SIMULATION only, CPU.
    PYTHONPATH=. python -m world_model.v2.house.zono_rows.tests.test_base_filter        (or pytest)
The finite-difference test builds the S2 desk scene once (~20 s).  "Old configs reproduce bit-identically" is checked against the saved S2 result
JSON in test_old_config_bit_identical (skipped when the file is absent; it re-runs a 1500-step closed loop, ~30 s)."""
from __future__ import annotations

import functools
import math
import os
import types

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import mujoco
from scipy.optimize import linprog

from world_model.v2.house.zono_rows import base_filter as BF
from world_model.v2.house.zono_rows import rows as R
from world_model.v2.house.zono_rows.ctl import ZonoCtl
from world_model.v2.house.zono_rows.relax import residual_slack, screen_hopeless

REPORT = {}
TAU = 100.0


# ------------------------------------------------------------------------------------------------------------ 1. selective relaxation
def _fake_selective_ctl(A_l, b_l, A_r, b_r, nh=0):
    """ZonoCtl shell whose fake FL-MPC is an exact feasibility oracle: fallback 0 iff  exists |tau|<=TAU: A tau <= b + s  (LP), both arms."""
    c = object.__new__(ZonoCtl)
    fl = types.SimpleNamespace(a_ego_prev=np.zeros(2), a_int_prev=np.zeros(2), last_fallback=2, calls=[])
    c.fl, c._s_prev, c.relax_frac = fl, 0.0, 0.9
    c.sc = types.SimpleNamespace(TAU_MAX=TAU)
    c.relax_kw = dict(s0=0.05, s_max=1e5, n_bisect=3, max_solves=12)

    def lp(A, b):
        """min ||tau||_1 (stand-in for FL-MPC's min-effort objective) s.t. A tau <= b, |tau| <= TAU;  tau = p - q."""
        n = A.shape[1]
        r = linprog(np.ones(2 * n), A_ub=np.hstack([A, -A]), b_ub=b, bounds=[(0, TAU)] * (2 * n), method="highs")
        return (r.x[:n] - r.x[n:] if r.status == 0 else np.zeros(n)), r.status == 0

    def solve_at(sv):
        sl, sr = sv
        fl.calls.append((np.array(sl), np.array(sr)))
        te, okl = lp(A_l, b_l + sl)
        ti, okr = lp(A_r, b_r + sr)
        fb = 0 if (okl and okr) else 2
        fl.last_fallback = fb
        return te, ti, okl and okr, fb

    return c, fl, solve_at


def test_screen_and_residual_pure():
    A = np.array([[1.0, 2.0, 0, 0, 0, 0], [0, 0, 3.0, 0, 0, 0]])
    t = TAU * np.abs(A).sum(1)                                      # best effort: a.tau >= -t
    b = np.array([-1.5 * t[0], -0.2 * t[1]])
    s = screen_hopeless(A, b, TAU, 0.9)
    assert s[1] == 0.0 and abs(s[0] - (-0.9 * t[0] - b[0])) < 1e-9 and s[0] > 0
    tau = np.array([5.0, 0, 0, 0, 0, 0])
    r = residual_slack(A, np.array([1.0, 100.0]), tau)              # row 0 violated by 4, row 1 satisfied
    assert r[1] == 0.0 and 4.0 < r[0] < 4.1
    print("screen_hopeless / residual_slack: hopeless row only; satisfied rows get 0")


def test_single_hopeless_row_does_not_relax_the_others():
    rng = np.random.default_rng(0)
    n = 6
    A_l = rng.normal(0, 1, (4, n)); A_r = rng.normal(0, 1, (3, n))
    b_l = np.array([50.0, 40.0, 60.0, 0.0]); b_r = np.array([30.0, 45.0, 20.0])
    t0 = TAU * np.abs(A_l[3]).sum()
    b_l[3] = -1.3 * t0                                              # impossible even at full torque on its own
    c, fl, solve_at = _fake_selective_ctl(A_l, b_l, A_r, b_r)
    assert not solve_at((np.zeros(4), np.zeros(3)))[2]              # the hard problem is infeasible
    fl.calls.clear()
    pre = (np.zeros(2), np.zeros(2))
    res = c._selective_step(solve_at, (A_l, b_l), (A_r, b_r), 0, 0, (np.zeros(6), np.zeros(6), False, 2), pre, pre)
    sl, sr = fl.calls[-1]
    assert not res["unrecovered"] and res["fb"] == 0 and res["n_screened"] == 1 and res["n_rows_slack"] == 1 and res["n_solves"] == 1
    assert sl[3] > 0 and (sl[:3] == 0).all() and (sr == 0).all(), (sl, sr)
    # a uniform scheme must slacken EVERY row by at least the hopeless row's need
    need = -b_l[3] - t0
    assert need > 0.2 * t0
    print(f"selective: only the hopeless row slackened (s={sl[3]:.1f}); a uniform slack would have been >= {need:.1f} on all 7 rows; solves={res['n_solves']}")
    REPORT["hopeless_row"] = dict(s_hopeless=float(sl[3]), uniform_needs_at_least=float(need))


def test_conflicting_rows_shrink_to_the_rows_that_need_it():
    """two rows that are each fine alone but contradict (a.tau <= -d and -a.tau <= -d): the uniform search certifies at some s; the residual
    shrink must give slack to at most those two rows and none to the benign rows."""
    n = 6
    a = np.zeros(n); a[0] = 1.0
    A_l = np.vstack([a, -a, np.eye(n)[1], np.eye(n)[2]])
    b_l = np.array([-5.0, -5.0, 20.0, 30.0])
    A_r = np.eye(n)[:2]; b_r = np.array([10.0, 10.0])
    c, fl, solve_at = _fake_selective_ctl(A_l, b_l, A_r, b_r)
    pre = (np.zeros(2), np.zeros(2))
    res = c._selective_step(solve_at, (A_l, b_l), (A_r, b_r), 0, 0, (np.zeros(6), np.zeros(6), False, 2), pre, pre)
    sl, sr = fl.calls[-1]
    assert not res["unrecovered"] and res["fb"] == 0 and res["n_screened"] == 0
    assert (sl[2:] == 0).all() and (sr == 0).all() and 0 < res["n_rows_slack"] <= 2, (sl, sr, res)
    assert res["s"] >= 5.0 - 1e-6                                   # the contradiction (gap 10) needs total slack >= 10 across the pair
    print(f"conflict: slack on {res['n_rows_slack']} of 6 rows (s={res['s']:.2f}), benign rows 0, {res['n_solves']} extra solves")
    REPORT["conflict"] = dict(n_rows_slack=res["n_rows_slack"], s=float(res["s"]), solves=res["n_solves"])


# ------------------------------------------------------------------------------------------------------------ 2. base rows: pure + FD
def _box_zono(c, half):
    return (np.asarray(c, float), np.diag(half))


def test_base_closed_loop_pure():
    """point cloud 'robot' driving straight at a wall zonotope with the path velocity 0.3 m/s: the filtered twist never lets h go below ~0, stops
    at the margin, and a SECOND hopeless row (an already-violated point behind a huge-g lever) neither makes the QP infeasible nor stops the base."""
    zs = R.ZonoSet([_box_zono([0.0, 1.0, 0.5], [2.0, 0.05, 1.0])])
    cfg = BF.BaseCfg(margin=0.03, d_act=0.5)
    pts0 = np.array([[0.0, 0.0, 0.5], [0.15, 0.1, 0.5], [-0.15, 0.1, 0.5]])         # robot points relative to the origin; wall face at y = 0.95
    o = np.array([0.0, 0.0]); u_prev = np.zeros(3); hmin = 9.0
    dt = 0.005
    for k in range(3000):
        p = pts0 + np.array([o[0], o[1], 0.0])
        A, b, meta = BF.base_rows_from_points(p, np.zeros(3), np.zeros((3, 3)), zs, o, cfg, u_ref=[0, 0.3, 0])
        lo = np.maximum(-np.array(cfg.v_max), u_prev - np.array(cfg.a_slew) * dt); hi = np.minimum(np.array(cfg.v_max), u_prev + np.array(cfg.a_slew) * dt)
        u, info = BF.solve_base(A, b, np.array([0.0, 0.3, 0.0]), lo, hi, 1.0 / np.array(cfg.w_scale), cfg, u_prev)
        assert info["ok"] and not info["braked"]
        u_prev = u
        o = o + u[:2] * dt
        hmin = min(hmin, float(0.95 - (pts0[:, 1].max() + o[1]) - cfg.margin))
    gap = 0.95 - (pts0[:, 1].max() + o[1])
    print(f"pure closed loop: final gap to the wall {gap*1e3:.1f} mm (margin 30 mm), min h {hmin*1e3:.2f} mm, final speed {u_prev[1]*1e3:.2f} mm/s")
    assert hmin > -1e-3 and 0.025 < gap < 0.05 and abs(u_prev[1]) < 1e-2
    REPORT["pure_loop"] = dict(final_gap_m=float(gap), h_min=float(hmin))
    # hopeless row: a point moving DOWN at 1 m/s onto a horizontal slab (vertical normal -> g = 0): the base cannot help, the row must not make the QP infeasible
    zs2 = R.ZonoSet([_box_zono([0.0, 0.0, 0.2], [2.0, 2.0, 0.05])])                   # top face at z = 0.25
    p = np.array([[0.0, 0.0, 0.30], [0.0, 0.5, 0.30]])
    v_arm = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    A, b, _ = BF.base_rows_from_points(p, np.zeros(2), v_arm, zs2, np.zeros(2), cfg)
    assert (np.abs(A[:, :]).sum(1) < 1e-9).all() and (b < 0).any()
    u, info = BF.solve_base(A, b, np.array([0.0, 0.3, 0.0]), np.array([-0.3, -0.3, -0.3]), np.array([0.3, 0.3, 0.3]), 1.0 / np.array(cfg.w_scale), cfg, np.zeros(3))
    assert info["ok"] and not info["braked"] and info["n_lift"] >= 1 and abs(u[1] - 0.3) < 1e-6, (info, u)
    print(f"hopeless-row solve: ok, lifted={info['n_lift']}, peeled={info['n_peel']}, u={u.round(3)}")


@functools.lru_cache(maxsize=1)
def _scene():
    from world_model.v2.house.zed_demo import rig as Z, execute as EX
    from world_model.v2.house.zono_rows import smoke_s2 as S
    C, _gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=False)
    mujoco.mj_forward(C.m, C.d)
    zonos, names, _meta = S.load_zonos()
    zs = R.ZonoSet(zonos, names)
    ctl = ZonoCtl(C, zonos=zs, names=names, margin=0.02, tray_level=True, track_bodies=["tray", "ball"], tilt=True, max_rows=64)
    o = C.d.mocap_pos[0][:2].copy()
    bf = BF.BaseFilter(ctl, BF.BaseCfg(max_rows=10 ** 6, per_facet=10 ** 6, d_act=0.6), origin_xy=o, yaw=math.pi / 2, data=C.d)
    pristine = mujoco.MjData(C.m)
    mujoco.mj_copyData(pristine, C.m, C.d)
    return C, ctl, bf, zs, o, pristine


def test_fd_base_twist_rows():
    """(i) arm keypoints + footprint: CBF row g.u vs a central finite difference of  n.p  when the mocap base moves with twist u (translation + yaw
    about its origin) and the arm joints are frozen -- mj_forward moves everything on the kinematic chain.  (ii) tracked tray points: g.u + n.v_arm vs the
    MEASURED point velocity under a real mj_step with the base moving."""
    C, ctl, bf, zs, o, pristine = _scene()
    m = C.m
    d0 = mujoco.MjData(m); mujoco.mj_copyData(d0, m, pristine)
    u = np.array([0.20, -0.10, 0.30])
    mujoco.mj_forward(m, d0)
    p0, r0, va0, nf = bf.points(d0, o, math.pi / 2)
    A, b, meta = BF.base_rows_from_points(p0, r0, va0, zs, o, bf.cfg)
    assert len(b) > 20
    kk, nrm = meta["kk"], meta["nrm"]
    eps = 1e-6
    pts = {}
    for sgn in (1, -1):
        d = mujoco.MjData(m); mujoco.mj_copyData(d, m, pristine)
        d.qvel[:] = 0.0
        d.mocap_pos[0] = (o[0] + sgn * eps * u[0], o[1] + sgn * eps * u[1], 0.0)
        # rotation about the mocap ORIGIN: body positions rotate about o only if the mocap pos is the rotation centre -> the model's root is a child of the mocap
        from world_model.v2.house import build as B
        d.mocap_quat[0] = B.yaw_quat(math.pi / 2 + sgn * eps * u[2])
        mujoco.mj_forward(m, d)
        pp, _r, _v, _nf = bf.points(d, o + sgn * eps * u[:2], math.pi / 2 + sgn * eps * u[2])
        pts[sgn] = pp
    n_arm = len(p0) - nf - (len(ctl.tracked) if ctl.tracked is not None else 0)
    # the tracked tray points follow the weld only under stepping: restrict (i) to arm keypoints and footprint
    tracked_lo, tracked_hi = n_arm, len(p0) - nf
    sel = np.array([i for i, k in enumerate(kk) if not (tracked_lo <= k < tracked_hi)])
    fd = np.einsum("ri,ri->r", nrm[sel], (pts[1][kk[sel]] - pts[-1][kk[sel]]) / (2 * eps))
    pred = -(A[sel] @ u)                                               # A = -g  ->  g.u = -A.u
    err = np.abs(fd - pred).max()
    scale = np.abs(fd).max()
    n_foot = int((kk[sel] >= len(p0) - nf).sum())
    print(f"FD base rows (arm kps + footprint, {len(sel)} rows, {n_foot} footprint): max |g.u - FD| = {err:.2e} (scale {scale:.3f})")
    assert err < 1e-6 * max(scale, 1.0) + 1e-6 and n_foot > 0
    # (ii) tracked tray: real stepping with the mocap base at constant twist
    d = mujoco.MjData(m); mujoco.mj_copyData(d, m, pristine)
    ctl.set_base_motion(None)
    x0, yaw0, dt = d.mocap_pos[0][:2].copy(), math.pi / 2, m.opt.timestep
    uu = np.array([0.15, 0.05, 0.10])
    errs = []
    for k in range(240):
        t = (k + 1) * dt
        d.mocap_pos[0] = (x0[0] + uu[0] * t, x0[1] + uu[1] * t, 0.0)
        d.mocap_quat[0] = B.yaw_quat(yaw0 + uu[2] * t)
        te, ti = ctl.tau(d)                                        # the FL-MPC holds the arms / tray (free arms would collapse under gravity)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te; d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        if k > 160:
            mujoco.mj_forward(m, d)
            org = np.array([x0[0] + uu[0] * t, x0[1] + uu[1] * t])
            pt, Jl, _a, vm, qdl = R.tracked_states(m, d, ctl.tracked, ctl.ld, ctl.gb["left"])
            _p, Jr, _b, _v, qdr = R.tracked_states(m, d, ctl.tracked, ctl.rd, ctl.gb["right"])
            va = 0.5 * (np.einsum("kij,j->ki", Jl, qdl) + np.einsum("kij,j->ki", Jr, qdr))
            vb = np.stack([uu[0] - uu[2] * (pt[:, 1] - org[1]), uu[1] + uu[2] * (pt[:, 0] - org[0]), np.zeros(len(pt))], 1)
            errs.append(np.abs((va + vb) - vm).max() / max(np.abs(vm).max(), 1e-6))
    e2 = float(np.median(errs))
    print(f"FD tracked tray (real mj_step, base twist {uu}): |v_arm + v_base(u) - v_measured| / |v| median {e2:.2e}")
    assert e2 < 0.1
    REPORT["fd_base"] = dict(rows_err=float(err), tracked_rel_err=e2)


def test_base_filter_step_smoke():
    """BaseFilter.step on the real scene: finite twist, rows present, slew respected, held ticks keep the twist."""
    C, ctl, bf, zs, o, pristine = _scene()
    d = mujoco.MjData(C.m); mujoco.mj_copyData(d, C.m, pristine); mujoco.mj_forward(C.m, d)
    bf2 = BF.BaseFilter(ctl, BF.BaseCfg(every=3), origin_xy=o, yaw=math.pi / 2, data=C.d)
    u1, i1 = bf2.step(d, np.array([0.0, 0.3, 0.0]), o, math.pi / 2, 0.005)
    u2, i2 = bf2.step(d, np.array([0.0, 0.3, 0.0]), o, math.pi / 2, 0.005)
    assert np.isfinite(u1).all() and i1["n_rows"] > 0 and i2.get("held") and np.allclose(u1, u2)
    assert np.all(np.abs(u1) <= np.array(bf2.cfg.a_slew) * 0.005 + 1e-12)
    print("BaseFilter.step: slew-limited, finite, held ticks reuse the twist")


# ------------------------------------------------------------------------------------------------------------ 3. bit-identity of the old configs
def test_old_config_bit_identical():
    ref = "/data/local/ss5772/molmospaces/zono_rows/s2_results_v123_m8_relax.json"
    if not os.path.exists(ref):
        print("bit-identity: reference JSON absent, skipped"); return
    import json
    import subprocess
    import sys
    out = os.environ.get("ZONO_ROWS_OUT", "/data/local/ss5772/molmospaces/zono_rows")
    tag = "_bitident_unittest"
    # ZONO_ROWS_REF=1: the original dense-numpy row path.  The numba path (default now) gives the same rows to ~1e-13, but this config is a chaotic bottle collision
    # (m8, no base filter) where 1e-13 grows to different contact histories, so only the reference path can reproduce the saved run bit-for-bit
    # (test_fast_rows.py checks fast == reference row-by-row on recorded states, and the S2 bf1 outcome is identical to all printed digits).
    env = dict(os.environ, BIMANUAL_FLMPC_IMPL="rt", ZONO_ROWS_REF="1")
    subprocess.run([sys.executable, "-m", "world_model.v2.house.zono_rows.smoke_s2", "run", "--margin", "0.02", "--alpha", "10", "--d-act", "0.15", "--max-rows", "8",
                    "--relax", "uniform", "--refine", "0", "--only-rows", "--tag", tag], check=True, env=env, stdout=subprocess.DEVNULL)
    A = json.load(open(ref))["with_rows"]; B_ = json.load(open(f"{out}/s2_results{tag}.json"))["with_rows"]
    bad = [k for k in A if k not in ("rows", "wall_s") and not k.startswith("ctl_ms") and A[k] != B_.get(k)]
    bad += ["rows." + k for k in A["rows"] if not k.startswith("ms") and A["rows"][k] != B_["rows"].get(k)]
    assert not bad, bad
    print("old config (m8, relax=uniform) reproduces the saved S2 result bit-identically (all non-timing fields)")


if __name__ == "__main__":
    test_screen_and_residual_pure()
    test_single_hopeless_row_does_not_relax_the_others()
    test_conflicting_rows_shrink_to_the_rows_that_need_it()
    test_base_closed_loop_pure()
    test_fd_base_twist_rows()
    test_base_filter_step_smoke()
    test_old_config_bit_identical()
    print("ALL PASSED", REPORT)
