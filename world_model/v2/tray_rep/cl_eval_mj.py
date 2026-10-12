"""cl_eval_mj.py -- closed-loop evaluation with MuJoCo as ground truth (the analogue of vdirect/cl_eval.py).
SIMULATION ONLY (MuJoCo CPU, no GPU, no rendering, no hardware, no network).

`evaluate(policy, data, oracle, ...)` restarts MuJoCo from the snapshot of each CERTIFIED-SAVABLE state (oracle label 1),
runs T = 50 ticks (1 s) closed loop with the policy, and records whether the ball left the tray.

Policy interface
    pi(obs_hist) -> u (6,)         obs_hist (n, obs_dim) float32 = the logged obs of the episode from tick 0 to the
                                   restart state, plus every observation produced since. Row -1 is "now".
    If `pi.needs_acts` is true, it is called pi(obs_hist, act_hist) where act_hist (n-1, 6) holds the actions applied
    before the last observation (act_hist[i] was applied between obs_hist[i] and obs_hist[i+1]).
    If the object has `begin(ctx)`, it is called once per state first (ctx: renv, snap, row (oracle row), hist_len).
    The action is applied as given (no velocity feedback, gain 1: closed_loop.py semantics). A learned policy should return
    an element of `renv.U` (the a_scale-scaled 85-element fallback set); this module does not enforce it.

Reference policies
    hold_still   u = 0
    OraclePolicy executes the best library sequence the oracle found for that state (privileged: it reads the oracle row;
                 it is the upper bound the certificate gives, and fails 0% by construction because the horizon of the
                 certificate (100 ticks) is longer than T)

  python -m world_model.v2.tray_rep.cl_eval_mj --data <root>/data_full --oracle <root>/data_full/oracle/test_v0.npz --T 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.oracle import SAVABLE, K_H, load_oracle, best_sequence
from world_model.v2.bead_rep import util as U


def hold_still(obs_hist):
    return np.zeros(6)


class OraclePolicy:
    """open-loop best library sequence from the oracle row (privileged reference)."""
    needs_acts = False

    def begin(self, ctx):
        renv, snap, row = ctx["renv"], ctx["snap"], ctx["row"]
        n = len(renv.U)
        sn = snap.snap
        i, dh = int(row["best_i"]), int(row["best_dh"])
        seqs = best_sequence(sn.kin[:3].copy(), sn.kin[7:10].copy(), sn.kin[3:7].copy(), renv.U, dh if dh > 0 else 10, K_H,
                             renv.a_max, renv.w_max, snap.w_slew, renv.alpha)
        self.seq = seqs[i] if 0 <= i < n + 2 else seqs[n + 1]
        self.h0 = ctx["hist_len"]

    def __call__(self, obs_hist):
        return self.seq[min(len(obs_hist) - self.h0, len(self.seq) - 1)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def evaluate(policy, data, oracle, split="test", T=50, max_states=None, a_scale=None, label=SAVABLE):
    """-> (summary dict, per-state dict). `oracle` = dict from load_oracle (or a path)."""
    if isinstance(oracle, (str, Path)):
        oracle = load_oracle(oracle)
    st = Store(data, split)
    renv = E.make_env(st.meta, a_scale=a_scale)
    sel = np.flatnonzero(oracle["label"] == label)
    if max_states is not None and max_states < len(sel):
        sel = sel[np.linspace(0, len(sel) - 1, max_states).astype(int)]
    needs_acts = bool(getattr(policy, "needs_acts", False))
    od = renv.obs_dim
    fail = np.zeros(len(sel), bool); ftk = -np.ones(len(sel), int); dmin = np.zeros(len(sel)); tun = np.zeros(len(sel), bool)
    t0 = time.perf_counter()
    for j, s in enumerate(sel):
        e, t = int(oracle["ep"][s]), int(oracle["t"][s])
        snap, hid = st.snapshot_at(e, t)
        renv.restore(snap, mu_roll=hid)
        obs = np.zeros((t + 1 + T, od), np.float32); act = np.zeros((t + T, 6), np.float32)
        obs[:t + 1] = st.rows("obs", e, 0, t + 1); act[:t] = st.rows("act", e, 0, t)
        n = t + 1
        if hasattr(policy, "begin"):
            policy.begin(dict(renv=renv, snap=snap, row={k: oracle[k][s] for k in oracle}, hist_len=n))
        dm = renv.env.dist()
        for k in range(T):
            u = policy(obs[:n], act[:n - 1]) if needs_acts else policy(obs[:n])
            r = renv.tick(np.asarray(u, np.float64))
            obs[n] = r["obs"]; act[n - 1] = r["act"]; n += 1
            dm = min(dm, r["d"])
            if r["tunnel"]:
                tun[j] = True
            if r["failed"] and not fail[j]:
                fail[j], ftk[j] = True, k
                break
        dmin[j] = dm
    res = dict(n=int(len(sel)), T=T, fail_rate=float(fail.mean()) if len(sel) else float("nan"),
               fail_ci95=wilson(int(fail.sum()), len(sel)), n_fail=int(fail.sum()), tunnel_states=int(tun.sum()),
               median_fail_tick=float(np.median(ftk[fail])) if fail.any() else None, wall_s=time.perf_counter() - t0)
    return res, dict(sel=sel, fail=fail, fail_tick=ftk, dmin=dmin, tunnel=tun)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(K.ROOT / "data_full"))
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = load_oracle(a.oracle)
    out = {}
    for name, pol in (("hold_still", hold_still), ("oracle", OraclePolicy())):
        out[name], _ = evaluate(pol, a.data, o, a.split, a.T, a.max_states)
        print(f"[cl_eval_mj] {name}: {out[name]}", flush=True)
    if a.out:
        U.jdump(out, a.out)


if __name__ == "__main__":
    main()
