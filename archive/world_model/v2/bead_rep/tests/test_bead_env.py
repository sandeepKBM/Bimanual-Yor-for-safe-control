"""Phase-0 unit tests for bead_env (SIMULATION ONLY).

  nice -n 10 python -m pytest world_model/v2/bead_rep/tests -q      (CPU; ~1 min)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

from world_model.v2.bead_rep import bead_env as B
from world_model.v2.data import contents as C
from world_model.v2.wm_audit import common as WC


def _rand_actions(rng, n_ticks, U):
    seq = np.zeros((n_ticks, 6)); t = 0
    while t < n_ticks:
        L = int(rng.integers(3, 16)); seq[t:t + L] = U[int(rng.integers(len(U)))]; t += L
    return seq


def _settled_env(n_settle=80):
    env = B.BeadEnv()
    env.reset(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0, 0]))
    for _ in range(n_settle):
        env.phys_step()
    return env


# ------------------------------------------------------------------------------- restore determinism
def test_restore_determinism_bitwise():
    U = B.fallback_U(); rng = np.random.default_rng(0)
    env = _settled_env()
    pre = _rand_actions(rng, 30, U)
    post = _rand_actions(rng, 40, U)
    for u in pre:
        env.tick(u)
    snap = env.snapshot()
    # round trip through the flat-array form that gen_nominal saves
    snap2 = B.Snapshot.from_arrays(**snap.arrays())
    assert snap2.state.shape == (env.n_state,)

    def run(e):
        out = [e.tick(u) for u in post]
        return (np.array([o["l"] for o in out]), e.d.qpos.copy(), e.d.qvel.copy(), e.status.copy(),
                np.concatenate([e.pos, e.quat, e.v]))

    ref = run(env)
    env.restore(snap)                              # same env
    for a, b in zip(ref, run(env)):
        assert np.array_equal(a, b)
    env2 = B.BeadEnv()                             # a brand-new env + arrays round trip: the snapshot is the FULL state
    env2.restore(snap2)
    for a, b in zip(ref, run(env2)):
        assert np.array_equal(a, b)


# ------------------------------------------------------------------------------- margin
def test_margin_continuity_and_consistency():
    eps = 1e-6
    # (a) over the rim vertically (inside radius): l_i = H - z, continuous through z = H
    zs = np.linspace(B.H - 0.01, B.H + 0.01, 2001)
    li = B.bead_margin_i(np.stack([np.full_like(zs, 0.02), np.zeros_like(zs), zs], 1))
    assert np.abs(np.diff(li)).max() <= (zs[1] - zs[0]) * (1 + 1e-9)
    # (b) radially out at z = H (rim crossing): continuous through rad = R
    rs = np.linspace(B.R - 0.005, B.R + 0.005, 2001)
    li = B.bead_margin_i(np.stack([rs, np.zeros_like(rs), np.full_like(rs, B.H)], 1))
    assert np.abs(np.diff(li)).max() <= (rs[1] - rs[0]) * (1 + 1e-9)
    assert abs(B.bead_margin_i(np.array([[B.R - eps, 0, B.H]]))[0]) < 1e-5
    assert abs(B.bead_margin_i(np.array([[B.R + eps, 0, B.H]]))[0]) < 1e-5
    # (c) range / clipping of the normalized margin
    lo = B.bead_margin(np.array([[0.2, 0, 0.5]]))
    hi = B.bead_margin(np.array([[0.0, 0, 0.0]]))
    assert lo == pytest.approx(B.MARGIN_LO / B.MARGIN_HI) and hi == 1.0
    # (d) sign agrees with the contents.Oracle spill test for every non-tunnel point (z >= -H_MARGIN):
    #     out <=> l_i < 0   (bottom exits z < -0.002 are tunnel artifacts, handled by the guard)
    rng = np.random.default_rng(1)
    loc = np.stack([rng.uniform(-0.05, 0.05, 20000), rng.uniform(-0.05, 0.05, 20000), rng.uniform(-0.002, 0.12, 20000)], 1)
    out = C.Oracle.out_mask(loc)
    assert np.array_equal(out, B.bead_margin_i(loc) < 0)
    # (e) in a dynamic branch the tick-to-tick margin is not chattery on average (DESIGN: p99 |dl| 0.17)
    env = _settled_env(); U = B.fallback_U()
    ls = [env.margin()]
    for u in _rand_actions(np.random.default_rng(3), 60, U):
        ls.append(env.tick(u)["l"])
    assert np.percentile(np.abs(np.diff(ls)), 99) < 0.35


# ------------------------------------------------------------------------------- tunnel guard
def test_tunnel_guard_classification():
    env = _settled_env()
    base = env.d.qpos[env.orc.idx].reshape(-1, 3).copy()
    # ball 0 -> over the rim (z 0.075, radially out), ball 1 -> through the floor (z -0.01), cup axis vertical
    p = base.copy(); p[0] = env.pos + np.array([0.032, 0.0, 0.075]); p[1] = env.pos + np.array([0.0, 0.0, -0.01])
    env.d.qpos[env.orc.idx] = p.ravel()
    loc = env.local(); env._exit_check(loc)
    assert env.status[0] == B.ST_RIM and env.status[1] == B.ST_TUNNEL
    assert env.n_spilled == 1 and env.tunnel
    assert env.margin(loc) < 0                           # rim exit counts
    p[0] = base[0]; env.d.qpos[env.orc.idx] = p.ravel(); env.status[0] = B.ST_IN   # ball 0 back inside
    loc = env.local()
    assert env.margin(loc) > 0                           # only the tunnelled ball is out -> excluded from the margin
    # NB a floor tunnel (rad <= R, z < 0) has l_i = H - z > 0: the margin cannot see it; only the guard flags it.
    assert B.bead_margin_i(loc)[1] > 0
    snap = env.snapshot(); env.restore(snap)
    assert env.status[1] == B.ST_TUNNEL                  # status survives snapshot/restore


# ------------------------------------------------------------------------------- action parity with the pendulum labeller
def _pend_cs(c0, seq):
    import torch
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)
    c = T(c0)[None]
    l, weff, cs = WC.pend_rollout_torch(-c, torch.zeros_like(c), c, T(seq[None, :, :3]), T(seq[None, :, 3:]),
                                        project=True, return_c=True)
    return cs[0].numpy(), weff[0].numpy()


def _quat_from_axis(c):
    z = np.array([0, 0, 1.0]); q = B._min_rot(z, c / np.linalg.norm(c)); return q


@pytest.mark.parametrize("case", ["random", "tilt_projection"])
def test_action_parity_with_pendulum_labeller(case):
    """No-bead sanity: the kinematic cup integrator reproduces the pendulum labeller's cup motion
    (axis per tick, realized axis rate w_eff, tilt projection, accel meaning a == cup-origin accel)."""
    U = B.fallback_U(); rng = np.random.default_rng(7)
    if case == "random":
        seq = _rand_actions(rng, 100, U)
        c0 = np.array([0.3, -0.2, 0.9]); c0 /= np.linalg.norm(c0)
    else:                                                  # drive the tilt through the 1.5 rad limit and keep pushing
        seq = np.tile(np.array([0, 0, 0, 0.0, 2.0, 0.0]), (110, 1))
        seq[90:, 3:] = np.array([-2.0, 0.0, 0.0])          # then roll about the other axis while pinned at the limit
        c0 = np.array([0.0, 0.0, 1.0])
    cs_ref, weff_ref = _pend_cs(c0, seq)
    pos = np.array([0.1, -0.2, 1.0]); v = np.array([0.3, 0.1, 0.0]); quat = _quat_from_axis(c0)
    tilt_max = 0.0
    pos_hist = [pos.copy() - v * B.DT, pos.copy()]
    for k, u in enumerate(seq):
        v_start = v.copy()
        pos, v, quat, c_a, c_b, w_eff = B.kin_tick(pos, v, quat, u)
        assert np.abs(c_b - cs_ref[k]).max() < 1e-10, k
        assert np.abs(w_eff - weff_ref[k]).max() < 1e-7, k
        assert np.abs(B.quat_axis(quat) - c_b).max() < 1e-12
        np.testing.assert_allclose(v - v_start, B.DT_WM * u[:3], atol=1e-12)      # v' = v + DT_WM a
        tilt_max = max(tilt_max, float(np.arccos(np.clip(c_b[2], -1, 1))))
    assert tilt_max <= B.TILT_LIM + 1e-9
    if case == "tilt_projection":
        assert tilt_max > B.TILT_LIM - 1e-3                # the projection was actually exercised
    # accel meaning: backward 2nd difference of the pose positions == a (semi-implicit Euler), per physics step
    p, vv, q = np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]); ps = [p.copy()]
    a = np.array([1.7, -2.2, 0.0])
    for _ in range(10):
        p, vv, q, _ = B.kin_step(p, vv, q, a, np.zeros(3)); ps.append(p.copy())
    ps = np.array(ps)
    np.testing.assert_allclose((ps[2:] - 2 * ps[1:-1] + ps[:-2]) / B.DT ** 2, np.tile(a, (9, 1)), atol=1e-9)


def test_env_tick_matches_labeller_with_beads_present():
    """The same parity through BeadEnv.tick (beads in the loop must not change the kinematic cup)."""
    U = B.fallback_U(); rng = np.random.default_rng(11)
    seq = _rand_actions(rng, 40, U)
    c0 = np.array([0.0, 0.0, 1.0])
    cs_ref, weff_ref = _pend_cs(c0, seq)
    env = _settled_env()
    for k, u in enumerate(seq):
        r = env.tick(u)
        assert np.abs(r["obs"][:3] - cs_ref[k]).max() < 1e-9
        assert np.abs(r["obs"][6:] - weff_ref[k]).max() < 1e-6
        np.testing.assert_allclose(r["act"], np.concatenate([u[:3], r["obs"][6:]]))


def test_brake_action_within_limits_and_decelerates():
    env = _settled_env()
    for _ in range(25):
        env.tick(np.array([3.0, 1.0, 0, 0.0, 1.0, 0.0]))
    sp0 = np.linalg.norm(env.v[:2]); tilt0 = np.arccos(env.axis()[2])
    for _ in range(40):
        u = B.brake_action(env.v, env.axis())
        assert np.linalg.norm(u[:3]) <= B.A_MAX + 1e-9 and np.linalg.norm(u[3:]) <= B.W_MAX + 1e-9 and u[2] == 0 and u[5] == 0
        env.tick(u)
    assert np.linalg.norm(env.v[:2]) < 1e-6 < sp0 and np.arccos(env.axis()[2]) < 1e-3 < tilt0


# ------------------------------------------------------------------------------- replay == contents.run_rig (bitwise)
def test_replay_matches_run_rig():
    from world_model.v2.latent_hj import spill_data as sd
    paths = sd.list_shards()
    if not paths:
        pytest.skip("spill_v2 not available")
    ep = sd.load_dense(paths[0])[0]
    n = 320
    pos, quat = ep["pos"][:n], ep["quat"][:n]
    env = B.BeadEnv()
    ref = C.run_rig(env.m, env.c, pos, quat)
    env.reset(pos[0], quat[0])
    cnt = []
    for k in range(n):
        env.set_kin(pos[k], quat[k], np.zeros(3) if k == 0 else (pos[k] - pos[k - 1]) / B.DT)
        loc = env.phys_step()
        cnt.append(int(C.Oracle.out_mask(loc).sum()))
    assert np.array_equal(np.array(cnt), ref["spilled"].astype(int))
    # final bead positions identical to a fresh run_rig-style drive
    import mujoco
    d = mujoco.MjData(env.m); orc = C.Oracle(env.m, env.c.n_balls)
    mujoco.mj_resetData(env.m, d); d.qpos[orc.idx] += np.tile(pos[0] - np.array([0, 0, 1.0]), env.c.n_balls)
    prev = None
    cup = env.m.body("cup").id; cj = env.m.body_jntadr[cup]; cq, cv = int(env.m.jnt_qposadr[cj]), int(env.m.jnt_dofadr[cj])
    for k in range(n):
        C.pin_cup(env.m, d, cq, cv, pos[k], quat[k], prev); prev = d.qpos.copy(); mujoco.mj_step(env.m, d)
    assert np.array_equal(d.qpos[orc.idx], env.d.qpos[orc.idx])
