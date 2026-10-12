"""selftest.py -- equivalence / determinism checks for the d_opaque_fill rig. CPU, ~20 s."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.d_opaque_fill import opaque_env as O, lib as L  # noqa: E402
from world_model.v2.wm_audit import common as WC  # noqa: E402


def test_pend_equiv():
    import torch
    P = WC.P0
    pm = O.PendModel(P.h, zeta=P.zeta, R=sp_R(), Hfree=sp_H())
    rng = np.random.default_rng(0)
    U = WC.fallback_U()
    seq = U[rng.integers(len(U), size=60)]
    n0 = np.array([0.0, 0.0, -1.0]); c0 = np.array([0.0, 0.0, 1.0])
    l_t, _ = WC.pend_rollout_torch(torch.tensor(n0)[None], torch.zeros(1, 3, dtype=torch.float64), torch.tensor(c0)[None],
                                   torch.tensor(seq[:, :3])[None], torch.tensor(seq[:, 3:])[None])
    n, nd, c = n0.copy(), np.zeros(3), c0.copy()
    l_n = []
    for u in seq:
        n, nd, c = pm.tick(n, nd, c, u)
        l_n.append(pm.margin(n, c))
    err = np.abs(np.array(l_n) - l_t[0].numpy()).max()
    print("pend numpy vs wm_audit torch: max |dl| =", err)
    assert err < 1e-5


def sp_R():
    from world_model.v2.latent_hj import slosh_pend as sp
    return sp.CUP_INNER_R


def sp_H():
    from world_model.v2.latent_hj import slosh_pend as sp
    return sp.CUP_HEIGHT


def test_restore_bitwise():
    env = O.OpaqueEnv(O.Fill(n=60))
    env.reset_settled()
    for _ in range(10):
        env.tick(np.array([2.0, 0, 0, 0, 1.0, 0]))
    s = env.snapshot()
    u = np.array([-1.0, 1.0, 0, 0.5, 0, 0])
    for _ in range(40):
        env.tick(u)
    a = env.d.qpos.copy()
    env.restore(s)
    for _ in range(40):
        env.tick(u)
    d = np.abs(a - env.d.qpos).max()
    print("restore+replay max diff =", d)
    assert d == 0.0


def test_margin_vs_exit():
    env = O.OpaqueEnv(O.Fill(n=140))
    env.reset_settled()
    ok = True
    for t in range(160):
        r = env.tick(np.array([0, 0, 0, 0.5, 0, 0]))
        if (r["l"] < 0) != (O.out_mask(env.local())[env.status != O.ST_TUNNEL].any()):
            ok = False
    print("margin<0 <=> a grain is out:", ok, "| spilled", env.n_spilled)
    assert ok


if __name__ == "__main__":
    test_pend_equiv(); test_restore_bitwise(); test_margin_vs_exit(); print("OK")
