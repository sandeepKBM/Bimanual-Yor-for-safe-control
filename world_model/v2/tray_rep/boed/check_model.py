"""boed/check_model.py -- (1) fit the analytic-model calibration, (2) report the analytic-vs-MuJoCo gap on the probe trajectory,
(3) check the posterior (accuracy, coverage) of the probe on random mu_roll. SIMULATION / SAVED DATA ONLY. CPU only.

  python -m world_model.v2.tray_rep.boed.check_model calib   --n-start 3            # writes boed/calib.json (VAL ball starts)
  python -m world_model.v2.tray_rep.boed.check_model gap     --n 3 --out <json>       # gap per mu, noise-free, controller probe
  python -m world_model.v2.tray_rep.boed.check_model posterior --setup piper_trk --n 60 --out <json>
  python -m world_model.v2.tray_rep.boed.check_model battery   --setup piper_trk --n 200 --out <json>   # the probe's own drop rate at the worst mu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.boed import model as MD
from world_model.v2.tray_rep.boed import probe as PB

OUT = K.ROOT / "boed"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_env(setup, split="val"):
    st = Store(K.ROOT / f"data_{setup}_full", split)
    return E.env_from_meta(st.meta), st


def noise_cfg(setup):
    return (None, True) if setup == "piper" else (MD.TRK_REAL, False)


def calib(n_start=3, hold=30, n_mu=18, seed=0):
    """fit mu_eff per mu on the open-loop part of the probe (pre + `hold` ticks of +theta), noise-free obs, ball starts ~ N(0, 5 mm)."""
    renv, _ = make_env("piper")
    spec = PB.ProbeSpec(n_hold_max=hold, d_trig=9.9)               # no early trigger
    rng = np.random.default_rng(seed)
    mus = np.exp(np.linspace(np.log(E.MU_LO), np.log(E.MU_HI), n_mu))
    fine = np.exp(np.linspace(np.log(5e-5), np.log(0.008), 400))
    starts = rng.normal(0, 0.005, (n_start, 2))
    out_eff, rms0, rms1 = [], [], []
    for mu in mus:
        err = np.zeros(len(fine)); e0 = 0.0; cnt = 0
        for b in starts:
            renv.reset(ball_xy=b, mu_roll=float(mu))
            kf = PB.make_kf(None, True)
            res = PB.run_probe(renv, spec, kf)
            ctrl = res["ctrl"]
            t0, t1 = ctrl.t_pre, ctrl.t_pre + spec.n_pre + hold
            obs, loc = res["obs"][t0:t1 + 1], res["loc"][t0:t1 + 1]
            X = MD.integrate(fine, obs[:, 0:3], loc[0, :2], np.zeros(2))
            err += ((X - loc[None, :, :2].transpose(1, 0, 2)) ** 2).sum((0, 2)) if False else ((X - loc[:, None, :2]) ** 2).sum((0, 2))
            X0 = MD.integrate(np.array([mu]), obs[:, 0:3], loc[0, :2], np.zeros(2))[:, 0]
            e0 += ((X0 - loc[:, :2]) ** 2).sum(); cnt += len(loc)
        j = int(np.argmin(err))
        out_eff.append(float(fine[j])); rms1.append(float(np.sqrt(err[j] / cnt / 1.0))); rms0.append(float(np.sqrt(e0 / cnt)))
        log(f"mu {mu:.5f} -> mu_eff {fine[j]:.5f} (ratio {fine[j] / mu:.3f})  rms gap per xy sample {1000 * rms0[-1]:.2f} -> {1000 * rms1[-1]:.2f} mm")
    eff = np.maximum.accumulate(np.array(out_eff))                  # monotone map
    d = dict(mu=mus.tolist(), mu_eff=eff.tolist(), rms_gap_mm_before=[1000 * x for x in rms0], rms_gap_mm_after=[1000 * x for x in rms1],
             spec=spec.to_dict(), n_start=n_start, note="fitted on VAL-like ball starts, noise-free, open-loop pre+hold window; MuJoCo soft rolling friction")
    (Path(MD.CALIB_PATH)).write_text(json.dumps(d, indent=1))
    log("wrote", MD.CALIB_PATH)
    return d


def gap(n=4, seed=11, out=None):
    """analytic (calibrated and uncalibrated) vs MuJoCo ball path in the likelihood window, with the real probe controller, noise-free
    obs. The model is started from the true state at the window start, so the gap isolates the model (not the nuisance fit)."""
    renv, _ = make_env("piper")
    spec = PB.ProbeSpec()
    rng = np.random.default_rng(seed)
    rows = []
    for mu in MD.MU_GRID[::3]:
        g_cal, g_raw, gmax = [], [], []
        for _ in range(n):
            renv.reset(ball_xy=rng.normal(0, 0.005, 2), mu_roll=float(mu))
            res = PB.run_probe(renv, spec, PB.make_kf(None, True))
            t0, t1 = res["ctrl"].window()
            if t1 <= t0:
                continue
            obs, loc = res["obs"][t0:t1 + 1], res["loc"][t0:t1 + 1]
            j = int(np.argmin(np.abs(MD.MU_GRID - mu)))
            for arr, mu_model in ((g_cal, MD.MU_EFF[j]), (g_raw, MD.MU_GRID[j])):
                X = MD.integrate(np.array([mu_model]), obs[:, 0:3], loc[0, :2], np.zeros(2))[:, 0]
                arr.append(np.sqrt(((X - loc[:, :2]) ** 2).sum(1).mean()))
                if arr is g_cal:
                    gmax.append(np.abs(X - loc[:, :2]).max())
        rows.append(dict(mu=float(mu), rms_mm_calibrated=1000 * float(np.mean(g_cal)), rms_mm_uncalibrated=1000 * float(np.mean(g_raw)),
                         max_mm_calibrated=1000 * float(np.max(gmax))))
        log(rows[-1])
    pooled = float(np.sqrt(np.mean([r["rms_mm_calibrated"] ** 2 for r in rows])))
    res = dict(rows=rows, pooled_rms_mm_calibrated=pooled, pooled_rms_mm_uncalibrated=float(np.sqrt(np.mean([r["rms_mm_uncalibrated"] ** 2 for r in rows]))))
    log("pooled gap (rms per xy sample): calibrated %.2f mm, uncalibrated %.2f mm" % (pooled, res["pooled_rms_mm_uncalibrated"]))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


sigma_total = MD.sigma_total


def posterior(setup="piper_trk", n=40, seed=5, out=None, mus=None, spec=None):
    """probe at pickup on random mu_roll (log-uniform), no filter, belief update. Reports accuracy and coverage."""
    renv_setup, _ = make_env(setup)
    trk, nf = noise_cfg(setup)
    spec = spec or PB.ProbeSpec()
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mu = float(mus[i % len(mus)]) if mus is not None else E.sample_mu_roll(rng)
        renv = renv_setup
        renv.reset(ball_xy=rng.normal(0, 0.005, 2), mu_roll=mu, track_seed=int(rng.integers(1 << 30)))
        kf = PB.make_kf(trk, nf)
        kf.reset_prior(sx=0.006, sv=0.01)
        res = PB.run_probe(renv, spec, kf)
        ctrl = res["ctrl"]
        bel = MD.Belief()
        if not res["dropped"] and ctrl.t_hold_end > 0:
            t0, t1 = ctrl.window()
            ll = MD.probe_loglik(res["obs"], t0, t1, sigma_total(setup), 0 if nf else trk["obs_latency_ticks"], ctrl.prior, nf)
            bel.update(ll)
        rows.append(dict(mu=mu, drop=res["dropped"], dmin=res["dmin"], n=res["n"], hold=ctrl.n_hold, reason=ctrl.reason,
                         mean_log=bel.mean_log(), sd_log=bel.sd_log(), err_log=bel.mean_log() - np.log(mu), cdf=bel.cdf(mu),
                         cover90=bel.covers(mu, 0.9), cover50=bel.covers(mu, 0.5), x_end=float(res["loc"][-1, 0]), y_end=float(res["loc"][-1, 1]),
                         settled=bool(ctrl.settled_end), lower01=bel.lower_mu(0.01)))
    d = dict(setup=setup, n=n, rows=rows, spec=spec.to_dict(), sigma=sigma_total(setup))
    A = lambda k: np.array([r[k] for r in rows], float)
    mu_a = A("mu")
    d["summary"] = dict(drop_rate=float(A("drop").mean()), rmse_log=float(np.sqrt((A("err_log") ** 2).mean())), mean_sd_log=float(A("sd_log").mean()),
                        cover90=float(A("cover90").mean()), cover50=float(A("cover50").mean()), prior_sd_log=float(np.log(E.MU_HI / E.MU_LO) / np.sqrt(12)),
                        mean_probe_ticks=float(A("n").mean()), end_offset_cm=float(100 * np.mean(np.hypot(A("x_end"), A("y_end")))),
                        settled=float(A("settled").mean()),
                        by_band={f"mu {lo:.4f}-{hi:.4f}": dict(n=int(((mu_a >= lo) & (mu_a < hi)).sum()),
                                                                rmse_log=float(np.sqrt((A("err_log")[(mu_a >= lo) & (mu_a < hi)] ** 2).mean())) if ((mu_a >= lo) & (mu_a < hi)).any() else None,
                                                                sd_log=float(A("sd_log")[(mu_a >= lo) & (mu_a < hi)].mean()) if ((mu_a >= lo) & (mu_a < hi)).any() else None)
                                 for lo, hi in ((0.0005, 0.001), (0.001, 0.002), (0.002, 0.0036), (0.0036, 0.0061))})
    log(json.dumps(d["summary"], indent=1))
    if out:
        Path(out).write_text(json.dumps(d, indent=1, default=float))
    return d


def battery(setup="piper_trk", n=200, seed=3, out=None, mus=(0.0005, 0.0007, 0.001), spec=None):
    """the probe's own drop rate: n random ball starts and tracker streams at the WORST mu values (the small ones roll fastest), no filter.
    Also the largest excursion of the ball from the centre, the smallest edge distance, and where the ball is left."""
    renv, _ = make_env(setup)
    trk, nf = noise_cfg(setup)
    spec = spec or PB.ProbeSpec()
    rng = np.random.default_rng(seed)
    rows = {}
    for mu in mus:
        drops, dmin, xmax, off, ticks = 0, 1e9, 0.0, [], []
        for _ in range(n):
            renv.reset(ball_xy=rng.normal(0, 0.005, 2), mu_roll=float(mu), track_seed=int(rng.integers(1 << 30)))
            kf = PB.make_kf(trk, nf)
            kf.reset_prior(sx=0.006, sv=0.01)
            res = PB.run_probe(renv, spec, kf)
            drops += int(res["dropped"]); dmin = min(dmin, res["dmin"])
            xmax = max(xmax, float(np.abs(res["loc"][:, :2]).max())); off.append(float(np.hypot(*res["loc"][-1, :2]))); ticks.append(res["n"])
        rows[f"{mu:g}"] = dict(n=n, drops=drops, min_edge_dist_cm=100 * dmin, max_excursion_cm=100 * xmax, mean_end_offset_cm=100 * float(np.mean(off)),
                               p95_end_offset_cm=100 * float(np.quantile(off, 0.95)), mean_ticks=float(np.mean(ticks)))
        log(setup, mu, rows[f"{mu:g}"])
    d = dict(setup=setup, rows=rows, spec=spec.to_dict())
    if out:
        Path(out).write_text(json.dumps(d, indent=1))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calib", "gap", "posterior", "battery"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-start", type=int, default=3)
    ap.add_argument("--setup", default="piper_trk")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "calib":
        calib(a.n_start)
    elif a.cmd == "gap":
        gap(a.n, a.seed, a.out or None)
    elif a.cmd == "battery":
        battery(a.setup, a.n, a.seed, a.out or None)
    else:
        posterior(a.setup, a.n, a.seed, a.out or None)


if __name__ == "__main__":
    main()
