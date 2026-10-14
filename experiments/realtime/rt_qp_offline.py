"""Offline check of rt_qp (Goldfarb-Idnani) on QPs captured from the current fast
path (rt_capture_qp.py). SIMULATION DATA ONLY.

For every captured hard QP: GI solution vs the dual-coordinate-ascent (old) solution
(max |dx|), active set (old: lam>0 ; GI: active general rows), feasibility decision,
KKT certificate of the GI point, and the OLD point's own constraint violation.
Also every captured relaxed QP (augmented problem) vs GI with the slack start.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import time

import numpy as np

import rt_qp


def gi_relaxed(Ha, fa, loa, hia, Aa, ba, n, max_iter=2000):
    """Solve the captured AUGMENTED relaxed QP with GI in the structured form
    (slack columns detected from A_aug)."""
    nt = Ha.shape[0]
    ns = nt - n
    A = np.ascontiguousarray(Aa[:, :n])
    m = A.shape[0]
    sidx = -np.ones(m, np.int64)
    for k in range(ns):
        rows = np.nonzero(Aa[:, n + k])[0]
        assert len(rows) == 1 and Aa[rows[0], n + k] == -1.0
        sidx[rows[0]] = k
    G = 0.5 * (Ha[:n, :n] + Ha[:n, :n].T) + 1e-8 * np.eye(n)
    J0x = rt_qp.factor(G)
    sig = np.sqrt(Ha[n, n] + 1e-8)
    ws = rt_qp.GIWorkspace(nt, m)
    q0 = rt_qp.gi_init_slack(n, ns, m, J0x, sig, fa, ws.J, ws.R, ws.act, ws.u, ws.x, ws.iact)
    st = rt_qp.gi_core(n, ns, m, A, ba, sidx, fa, loa, hia, ws.J, ws.R, ws.act, ws.u, ws.x, ws.iact,
                       ws.d, ws.z, ws.r, q0, max_iter, ws.stats, ws.rn)
    q = int(ws.stats[1])
    return ws.x[:nt].copy(), st, ws.stats.copy(), ws.act[:q].copy(), ws.u[:q].copy(), A, sidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/data/local/ss5772/bimanual_runs/rt_flmpc/capture/qp_*.pkl")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()
    rows = []
    rel = []
    for fn in sorted(glob.glob(args.glob)):
        D = pickle.load(open(fn, "rb"))
        for rec in D["qps"][:args.max]:
            c = rec["calls"][0]
            Hm, f, lo, hi, A, b = c["H"], c["f"], c["lo"], c["hi"], c["A"], c["b"]
            n, m = Hm.shape[0], A.shape[0]
            G = 0.5 * (Hm + Hm.T) + 1e-8 * np.eye(n)
            J0 = rt_qp.factor(G)
            ws = rt_qp.GIWorkspace(n, m)
            t0 = time.perf_counter()
            x, st, stats, act, u = rt_qp.solve_dense(Hm, f, lo, hi, A, b, max_iter=5000, J0=J0, ws=ws)
            dt = time.perf_counter() - t0
            viol, umin, stat = rt_qp.kkt_residuals(Hm, f, lo, hi, A, b, x, act, u) if st == 0 else (np.nan,) * 3
            act_gen = set(int(i) for i in act if i < m)
            act_old = set(np.nonzero(c["lam"] > 0)[0].tolist())
            old_viol = float(np.max(A @ c["x"] - b))
            obj = lambda z: 0.5 * z @ G @ z + f @ z
            rows.append((fn.split("/")[-1], rec["i"], n, m, int(c["feas"]), st, int(stats[0]), int(stats[1]),
                         float(np.max(np.abs(x - c["x"]))) if st == 0 else np.nan,
                         int(act_gen == act_old), len(act_gen), len(act_old), viol, umin, stat, old_viol,
                         obj(x) - obj(c["x"]) if st == 0 else np.nan, dt, c["dt"]))
            if len(rec["calls"]) > 1:
                cr = rec["calls"][1]
                y, st2, stats2, act2, u2, A2, sidx = gi_relaxed(cr["H"], cr["f"], cr["lo"], cr["hi"], cr["A"],
                                                               cr["b"], n)
                v2 = rt_qp.kkt_residuals(cr["H"], cr["f"], cr["lo"], cr["hi"], A2, cr["b"], y, act2, u2, n=n,
                                         sidx=sidx) if st2 == 0 else (np.nan,) * 3
                rel.append((fn.split("/")[-1], rec["i"], int(cr["feas"]), st2, int(stats2[0]),
                            float(np.max(np.abs(y[:n] - cr["x"][:n]))), *v2))
    R = np.array([r[2:] for r in rows], dtype=float)
    print(f"hard QPs: {len(rows)}  (n={int(R[0,0])}, m {int(R[:,1].min())}-{int(R[:,1].max())})")
    print(f"  feasibility decision old vs GI agree: {int(np.sum((R[:,2]==1) == (R[:,3]==0)))}/{len(R)};"
          f"  GI status counts: " + str({int(s): int(np.sum(R[:,3]==s)) for s in np.unique(R[:,3])}))
    ok = R[:, 3] == 0
    print(f"  GI iterations: mean {R[ok,4].mean():.2f} p50 {np.median(R[ok,4]):.0f} p99 {np.percentile(R[ok,4],99):.0f} max {R[ok,4].max():.0f};"
          f"  final active q: mean {R[ok,5].mean():.2f} max {R[ok,5].max():.0f}")
    print(f"  max|x_GI - x_old|: median {np.nanmedian(R[ok,6]):.3e} p99 {np.nanpercentile(R[ok,6],99):.3e} max {np.nanmax(R[ok,6]):.3e}")
    print(f"  active set identical (general rows): {int(R[ok,7].sum())}/{int(ok.sum())};  |act_GI| mean {R[ok,8].mean():.2f}, |act_old| mean {R[ok,9].mean():.2f}")
    print(f"  GI KKT: max primal viol {np.nanmax(R[ok,10]):.2e}, min u {np.nanmin(R[ok,11]):.2e}, max stationarity {np.nanmax(R[ok,12]):.2e}")
    print(f"  OLD point max row violation: median {np.median(R[:,13]):.2e} p99 {np.percentile(R[:,13],99):.2e} max {R[:,13].max():.2e}")
    print(f"  objective(GI) - objective(old): max {np.nanmax(R[ok,14]):.3e} min {np.nanmin(R[ok,14]):.3e}")
    print(f"  solve time GI (python wrapper incl.) ms: mean {R[:,15].mean()*1e3:.3f} p99 {np.percentile(R[:,15],99)*1e3:.3f} max {R[:,15].max()*1e3:.3f}"
          f" | old in-episode ms: mean {R[:,16].mean()*1e3:.3f} p99 {np.percentile(R[:,16],99)*1e3:.3f} max {R[:,16].max()*1e3:.3f}")
    if rel:
        Q = np.array([r[2:] for r in rel], dtype=float)
        print(f"relaxed QPs: {len(rel)}; GI status counts {({int(s): int(np.sum(Q[:,1]==s)) for s in np.unique(Q[:,1])})};"
              f" old feasible2 True: {int(Q[:,0].sum())}")
        okr = Q[:, 1] == 0
        if okr.any():
            print(f"  GI iterations: mean {Q[okr,2].mean():.1f} max {Q[okr,2].max():.0f};  max|x_GI-x_old| median {np.median(Q[okr,3]):.3e} max {Q[okr,3].max():.3e}")
            print(f"  GI KKT: max viol {np.nanmax(Q[okr,4]):.2e} min u {np.nanmin(Q[okr,5]):.2e} stationarity {np.nanmax(Q[okr,6]):.2e}")


if __name__ == "__main__":
    main()
