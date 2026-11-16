"""model_uq/residual.py -- branches (collect.py) -> residual table: MuJoCo vs the analytic ball model (reach.ball_step). SIMULATION / SAVED DATA ONLY.

A WINDOW starts at branch tick t (state = true tray-frame ball position and central-difference velocity), is rolled k ticks by the
analytic model under the TRUE tray axes and tray accelerations of the branch (so only the ball-model class is tested, no estimator,
no backup law), with the TRUE hidden mu_roll mapped through the calibrated map (boed/calib.json), and compared with MuJoCo:
    r = x_true(t+k) - x_model(t+k)    (2,) position residual [m]
    a_eq = 2 r / (k dt)^2             (2,) equivalent constant acceleration error [m/s^2]  <- what F5's eps_a (per-axis box) must cover
    y = max(|a_eq_x|, |a_eq_y|)       scalar target (box semantics)
Regime features (per window):
    lrho   log10( mean|g_in| / mean f )  over the window (model quantities, TRUE mu)   <- drive vs rolling resistance (gn/f)
    lrho_m same with f at the MID of the prior range of mu (what the filter can compute without knowing mu)
    speed  |v_rel| at window start [m/s] (central difference)
    edge   distance to the nearest tray edge at start [m]
    tilt   arccos(c_z) at start [rad]
    acc    mean |a_xy| of the tray over the window [m/s^2]
    lmu    log10 mu_roll
    disp   |model displacement| over the window [m];  tdisp  |true displacement|
Windows are kept only if the ball stays on the tray (and does not lift off) for the whole window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY

DT = RC.DT
FEATS = ("lrho", "lrho_m", "speed", "edge", "tilt", "acc", "lmu", "disp", "tdisp")
MU_MID = float(np.sqrt(5e-4 * 6e-3))


def _vcen(pos, t):
    return (pos[t + 1, :2] - pos[t - 1, :2]) / (2 * DT)


def windows(arr, k=5, wstride=2, lift_tol=0.004):
    """-> dict of arrays, one row per window."""
    pos, cc, U, nvalid, mus = arr["pos"], arr["c"], arr["u"], arr["n"], arr["mu"]
    me_mid = RC.MU_EFF(MU_MID)
    rows = []
    for b in range(len(pos)):
        n = int(nvalid[b])
        me = RC.MU_EFF(float(mus[b]))
        P, C, Ub = pos[b], cc[b], U[b]
        on = (np.abs(P[:, 0]) <= HX - 0.002) & (np.abs(P[:, 1]) <= HY - 0.002) & (P[:, 2] >= RC.R_BALL - lift_tol) \
            & (P[:, 2] <= RC.R_BALL + 6 * lift_tol)
        for t in range(1, n - k, wstride):
            if not on[t - 1:t + k + 1].all():
                continue
            v = _vcen(P, t)
            s = np.array([P[t, 0], P[t, 1], v[0], v[1]], np.float64)
            gs = fs = 0.0
            asum = 0.0
            for j in range(k):
                s, g, f = RC.ball_step(s, C[t + j], C[t + j + 1], Ub[t + j, :3], me, RC.K_IN, RC.R_BALL)
                gs += g
                fs += f
                asum += float(np.hypot(Ub[t + j, 0], Ub[t + j, 1]))
            r = P[t + k, :2] - s[:2]
            a_eq = 2.0 * r / (k * DT) ** 2
            rho = gs / max(fs, 1e-9)
            tdisp = float(np.hypot(*(P[t + k, :2] - P[t, :2])))
            disp = float(np.hypot(*(s[:2] - P[t, :2])))
            rows.append((b, t, int(arr["ep"][b]), int(arr["kind"][b]), r[0], r[1], a_eq[0], a_eq[1],
                         np.log10(max(rho, 1e-6)), np.log10(max(rho * me / me_mid, 1e-6)), float(np.hypot(*v)),
                         min(HX - abs(P[t, 0]), HY - abs(P[t, 1])), float(np.arccos(np.clip(C[t, 2], -1, 1))), asum / k,
                         np.log10(float(mus[b])), disp, tdisp))
    A = np.array(rows, np.float64)
    names = ("branch", "t", "ep", "kind", "rx", "ry", "ax", "ay") + FEATS
    out = {nm: A[:, i] for i, nm in enumerate(names)}
    for nm in ("branch", "t", "ep", "kind"):
        out[nm] = out[nm].astype(np.int64)
    out["y"] = np.maximum(np.abs(out["ax"]), np.abs(out["ay"]))
    out["k"] = np.full(len(A), k)
    return out


def openloop(arr, hs=(10, 20, 39), lift_tol=0.004):
    """free-run model error from branch tick 1 over the rest of the branch (centre error at ticks hs), per branch; NaN if invalid."""
    pos, cc, U, nvalid, mus = arr["pos"], arr["c"], arr["u"], arr["n"], arr["mu"]
    out = np.full((len(pos), len(hs)), np.nan)
    for b in range(len(pos)):
        n = int(nvalid[b])
        me = RC.MU_EFF(float(mus[b]))
        P, C, Ub = pos[b], cc[b], U[b]
        v = _vcen(P, 1)
        s = np.array([P[1, 0], P[1, 1], v[0], v[1]], np.float64)
        on = (np.abs(P[:, 0]) <= HX - 0.002) & (np.abs(P[:, 1]) <= HY - 0.002)
        for j in range(1, n):
            s, _, _ = RC.ball_step(s, C[j], C[j + 1], Ub[j, :3], me, RC.K_IN, RC.R_BALL)
            if not on[j + 1]:
                break
            for i, h in enumerate(hs):
                if j - 0 == h:
                    out[b, i] = float(np.max(np.abs(P[j + 1, :2] - s[:2])))
    return out


def load_branches(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def build(branch_path, out_path, ks=(5,), wstride=2):
    arr, meta = load_branches(branch_path)
    tabs = {k: windows(arr, k, wstride) for k in ks}
    save = {}
    for k, t in tabs.items():
        for nm, v in t.items():
            save[f"k{k}_{nm}"] = v
    save["meta"] = json.dumps(dict(meta, ks=list(ks), wstride=wstride, feats=list(FEATS)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save)
    return tabs


def load_table(path, k=5):
    z = np.load(path, allow_pickle=False)
    return {nm[len(f"k{k}_"):]: z[nm] for nm in z.files if nm.startswith(f"k{k}_")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="5,15")
    ap.add_argument("--wstride", type=int, default=2)
    a = ap.parse_args()
    tabs = build(a.branches, a.out, tuple(int(x) for x in a.ks.split(",")), a.wstride)
    for k, t in tabs.items():
        y = t["y"]
        print(f"k={k}: {len(y)} windows; |a_eq|inf quantiles 50/90/95/99/max = "
              f"{np.quantile(y, [.5, .9, .95, .99]).round(3).tolist()} {y.max():.2f}; frac<=0.08: {(y <= 0.08).mean():.3f}")
