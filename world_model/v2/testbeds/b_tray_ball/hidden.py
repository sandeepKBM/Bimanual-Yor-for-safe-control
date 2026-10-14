"""hidden.py -- hidden-parameter sensitivity and a safe-probe identifiability check (SIMULATION, CPU).
  python -m world_model.v2.testbeds.b_tray_ball.hidden [--workers 8]
(1) same 40 open-loop carries (seeds fixed) replayed under parameter variants: how many nominal failure labels flip?
(2) probe: tray tilted about y by theta (rate 1 rad/s), held 0.3 s, ball starts at the centre. Ball x(t) -> a(theta).
    Closed form (rolling, with rolling resistance): a = (g sin th - (mu_roll/r) g cos th) / (1 + k). Two angles separate
    k and mu_roll/r. A 30 Hz tracker with 2 mm noise (ASSUMED ZED figure) is simulated; the fit is least squares on x(t).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.b_tray_ball.tray_env import TrayEnv, BallParams, ST_OFF, DT_WM, G
from world_model.v2.testbeds.b_tray_ball.carry import make_carry, carry_u
from world_model.v2.testbeds.b_tray_ball.probe import merge_json

VARIANTS = {"nominal": {}, "mu_roll_0.0005": dict(mu_roll=0.0005), "mu_roll_0.006": dict(mu_roll=0.006),
            "k_0.667(shell)": dict(inertia_k=0.667), "mu_0.15": dict(mu=0.15), "mu_0.8": dict(mu=0.8),
            "radius_0.025": dict(radius=0.025), "radius_0.035": dict(radius=0.035), "mass_0.2": dict(mass=0.2)}


def carry_fail(args):
    seed, kw = args
    rng = np.random.default_rng(seed)
    env = TrayEnv(BallParams(**kw))
    env.reset(env.home, np.array([1.0, 0, 0, 0]), ball_xy=rng.normal(0, 0.005, 2))
    cr = make_carry(rng, 1.7)
    for k in range(int(cr["T_end"] / DT_WM)):
        env.tick(carry_u(cr, k * DT_WM, env.axis()))
        if env.status:
            break
    return env.status == ST_OFF


def probe_trace(kw, theta, hold=0.3, seed=0, noise=0.0):
    env = TrayEnv(BallParams(**kw))
    env.reset(env.home, np.array([1.0, 0, 0, 0]))
    for _ in range(10):
        env.tick(np.zeros(6))
    w = np.array([0.0, 1.0, 0.0])                       # rotate about +y: +x edge goes down? sign irrelevant (symmetric)
    n_up = int(round(theta / 1.0 / DT_WM))
    xs, ts, ok = [], [], True
    for k in range(n_up + int(hold / DT_WM)):
        u = np.concatenate([np.zeros(3), w if k < n_up else np.zeros(3)])
        r = env.tick(u)
        xs.append(r["loc"][0]); ts.append((k + 1) * DT_WM)
        ok &= env.status == 0
    return np.array(ts), np.array(xs), ok, n_up


def fit_a(ts, xs, n_up):
    """least squares of x = x0 + 0.5 a (t - t_up)^2 (+ v0 (t-t_up)) over the HOLD phase; returns a."""
    t = ts[n_up:] - ts[n_up - 1]
    A = np.stack([np.ones_like(t), t, 0.5 * t * t], 1)
    return np.linalg.lstsq(A, xs[n_up:], rcond=None)[0][2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    seeds = list(range(20_000, 20_040))
    out = {}
    with Pool(a.workers) as P:
        base = np.array(P.map(carry_fail, [(s, {}) for s in seeds]))
        for name, kw in VARIANTS.items():
            f = np.array(P.map(carry_fail, [(s, kw) for s in seeds]))
            out[name] = dict(fail_rate=float(f.mean()), flipped_vs_nominal=int((f != base).sum()))
            print(name, out[name], flush=True)
    merge_json("hidden_param_carry_sensitivity", out)
    # probe identifiability
    rng = np.random.default_rng(0)
    grid = [dict(inertia_k=k, mu_roll=m) for k in (0.4, 0.667) for m in (0.0005, 0.002, 0.006)]
    res = []
    for kw in grid:
        row = dict(params=kw)
        for th in (0.12, 0.30):
            ts, xs, ok, n_up = probe_trace(kw, th)
            r_ = 0.03
            row[f"a_{th}"] = float(fit_a(ts, xs, n_up)); row[f"max_x_cm_{th}"] = float(100 * np.abs(xs).max()); row[f"safe_{th}"] = bool(ok)
            row[f"a_noisy_sd_{th}"] = float(np.std([fit_a(ts[::1], xs + rng.normal(0, 0.002, xs.shape), n_up) for _ in range(200)]))
        # invert the closed form with the two measured accelerations
        a1, a2 = row["a_0.12"], row["a_0.3"]
        s1, c1, s2, c2 = np.sin(.12), np.cos(.12), np.sin(.3), np.cos(.3)
        M = np.array([[-a1, -G * c1], [-a2, -G * c2]])        # unknowns [k, mu_roll/r]: a (1+k) = g s - (mu_r/r) g c
        sol = np.linalg.solve(np.array([[a1, G * c1], [a2, G * c2]]), np.array([G * s1 - a1, G * s2 - a2]))
        row["k_hat"], row["mu_roll_hat"] = float(sol[0]), float(sol[1] * 0.03)
        res.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "params"}, kw, flush=True)
    merge_json("probe_identifiability", res)


if __name__ == "__main__":
    main()
