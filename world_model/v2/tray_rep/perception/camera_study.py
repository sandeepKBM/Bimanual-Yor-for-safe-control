"""camera_study.py -- which camera rig can track the ball, and when? SIMULATION ONLY (CPU, ray casting, no GL/GPU/hardware).

Over N nominal Piper-config carries (configs/piper.json, clip carry; seeds 100000.. as data_piper_full), for each rig:
  1. COLLECT: run the carry with the rig capturing EVERY tick (50 Hz, no delay) and log the per-frame measurement payload
     (sphere-fit centre, n points, covariance) plus eval-only truth (ball hits by geometry id, privileged state).
  2. ANALYSE: replay the logged payloads through the SAME pipeline code the env uses (clock, delay queue, Kalman filter) for any
     (frame rate, latency). Frame noise depends only on (seed, tick), so a replay at 30 Hz equals a live 30 Hz run (selftest checks).
Metrics, over all pre-failure ticks and over the last 25 ticks before the ball leaves the tray (failing episodes only):
  visibility = share of captured frames with a valid sphere fit;  n_points = inlier points of the fit;
  true ball hits = rays that hit the ball (geometry ids, eval only);
  position error = |Kalman xy - privileged xy|; velocity error = |Kalman v - privileged v|;
  edge error = |d(est) - d(true)|, d = signed distance to the open tray edge (this is what the failure margin uses);
  track-loss events = times 'ticks since last good measurement' first passes 10 (0.2 s).

  python -m world_model.v2.tray_rep.perception.camera_study --n-eps 200 --workers 6          (collect + analyse, ~10 min)
  python -m world_model.v2.tray_rep.perception.camera_study --analyse-only                    (re-run the analysis from tmp/)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import gen_nominal as GN
from world_model.v2.tray_rep.perception import pc_env as PC
from world_model.v2.tray_rep.perception import ball_estimator as BE
from world_model.v2.tray_rep.perception import depth_sim as D
from world_model.v2.testbeds.b_tray_ball import tray_env as TE

OUT = K.ROOT / "perception"
TMP = OUT / "tmp"
RIGS = ["head", "head_low", "head_aim", "wrist_l", "wrist_r", "wrist_both", "head_aim+wrist", "head+wrist", "zed_nav"]
MAIN = {"ideal (50 Hz, 0 ticks)": (50.0, 0), "realistic (30 Hz, 3 ticks)": (30.0, 3)}
SWEEP_RATES, SWEEP_LATS = (50.0, 30.0, 15.0), (0, 1, 2, 3, 5)
PRE = 25                                                         # ticks before failure that matter


class StudyEnv(PC.PCRepEnv):
    """PCRepEnv that also logs the tray centre every tick (eval only)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.poslog = []

    def reset(self, *a, **k):
        self.poslog = []
        r = super().reset(*a, **k)
        self.poslog.append(self.env.pos.copy())
        return r

    def tick(self, u):
        r = super().tick(u)
        self.poslog.append(self.env.pos.copy())
        return r


def collect_task(args):
    rig, seeds, cfg_name = args
    cfg = GN.load_config(cfg_name)
    renv = StudyEnv(rig, cfg["a_scale"], False, w_scale=cfg["w_scale"], alpha=cfg["alpha"], rate_hz=50.0, latency_ticks=0,
                    log_meas=True)
    out = []
    for sd in seeds:
        renv.log = []
        ep = GN.run_episode(renv, sd, cfg["intensity"], with_snaps=False, carry_mode=cfg["carry_mode"])
        T = len(ep["obs"])
        L = renv.log
        pay = np.zeros((T, BE.PAYLOAD_LEN)); nb = np.zeros(T)
        for (t, p, n_b, _) in L:
            if t < T:
                pay[t], nb[t] = p, n_b
        pos = np.array(renv.poslog[:T])
        out.append(dict(seed=sd, payload=pay, nb=nb, priv=ep["priv"], l=ep["l"], onset=ep["onset"], lens=T,
                        tray_off=np.linalg.norm(pos - TE_HOME, axis=1)))
    return rig, out, dict(renv.cost)


TE_HOME = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------------------------- analysis
def sdf(xy, lip=False):
    return np.array([TE.sdf_rect(float(x), float(y), lip) for x, y in xy])


def _stats(v, scale=1.0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return dict(mean=None, p95=None, n=0)
    return dict(mean=float(v.mean() * scale), p95=float(np.percentile(v, 95) * scale), n=int(len(v)))


def analyse_rig(eps, rate, lat, sigma_a=1.5):
    """-> dict of metrics for one rig at one (rate, latency)."""
    A = dict(pos=[], vel=[], edge=[], vis=[], nb=[], npts=[], n_cand=[])
    B = {k: [] for k in A}
    loss_all, loss_fail, n_fail = [], [], 0
    cover, offs = [], []
    for e in eps:
        T, on = e["lens"], e["onset"]
        est, qual, since, nloss = PC.replay(e["payload"], rate, lat, sigma_a)
        end = on if on >= 0 else T
        t = np.arange(end)
        pr = e["priv"]
        pe = np.linalg.norm(est[:end, :2] - pr[:end, :2], axis=1)
        ve = np.linalg.norm(est[:end, 2:] - pr[:end, 3:5], axis=1)
        d_est = sdf(est[:end, :2]); d_tru = pr[:end, 9]
        de = np.abs(np.clip(d_est, -1, 1) - d_tru)
        vis = e["payload"][:end, BE.I_VALID] > 0.5
        sets = [(A, np.ones(end, bool))]
        if on >= PRE:
            sets.append((B, t >= on - PRE))
            n_fail += 1
        for dst, m in sets:
            dst["pos"].append(pe[m]); dst["vel"].append(ve[m]); dst["edge"].append(de[m]); dst["vis"].append(vis[m])
            dst["nb"].append(e["nb"][:end][m]); dst["npts"].append(e["payload"][:end, BE.I_N][m][vis[m]])
            dst["n_cand"].append(e["payload"][:end, BE.I_CAND][m])
        fl = since[:end] > BE.T_LOSS                              # loss events BEFORE the ball leaves the tray (rising edges)
        ev = np.flatnonzero(fl[1:] & ~fl[:-1]) + 1
        loss_all.append(int(len(ev) + (1 if fl[0] else 0)))
        if on >= PRE:
            loss_fail.append(int((ev >= on - PRE).sum() + (1 if fl[on - PRE] else 0)))
        cover.append(np.mean(e["tray_off"][:end] < 0.25)); offs.append(e["tray_off"][:end])

    def pack(S):
        cat = lambda k: np.concatenate(S[k]) if S[k] else np.zeros(0)
        nb, vis = cat("nb"), cat("vis")
        npts = cat("npts")
        return dict(frames=int(len(vis)), visibility=float(vis.mean()) if len(vis) else None,
                    ball_hit_rate=float((nb > 0).mean()) if len(nb) else None,
                    median_true_ball_hits=float(np.median(nb)) if len(nb) else None,
                    median_fit_points=float(np.median(npts)) if len(npts) else None,
                    pos_err_mm=_stats(cat("pos"), 1e3), vel_err_mps=_stats(cat("vel")), edge_err_mm=_stats(cat("edge"), 1e3))
    out = dict(all=pack(A), last25=pack(B), n_eps=len(eps), n_fail_eps_with_25=n_fail,
               loss_events_per_ep=float(np.mean(loss_all)),
               frac_eps_with_loss=float(np.mean(np.array(loss_all) > 0)),
               loss_events_per_failing_ep_last25=float(np.mean(loss_fail)) if loss_fail else None,
               frac_ticks_tray_within_25cm_of_home=float(np.mean(cover)),
               tray_offset_from_home_m=dict(median=float(np.median(np.concatenate(offs))), p95=float(np.percentile(np.concatenate(offs), 95))))
    return out


def reference(eps):
    """Ball motion near failure, and what a plain delay (no filter) would cost: |p(t-L) - p(t)| over the last 25 ticks."""
    sp, d25, stale = [], [], {L: [] for L in range(0, 6)}
    for e in eps:
        on = e["onset"]
        if on < PRE + 5:
            continue
        pr = e["priv"]
        sl = slice(on - PRE, on)
        sp.append(np.linalg.norm(pr[sl, 3:5], axis=1))
        d25.append(pr[on - PRE, 9])
        for L in stale:
            stale[L].append(np.linalg.norm(pr[on - PRE - L:on - L, :2] - pr[sl, :2], axis=1))
    sp = np.concatenate(sp)
    return dict(n_failing_eps=len(d25), ball_speed_last25_mps=dict(median=float(np.median(sp)), p95=float(np.percentile(sp, 95))),
                edge_distance_25_ticks_before_failure_mm=dict(median=float(1e3 * np.median(d25)), p95=float(1e3 * np.percentile(d25, 95))),
                stale_pos_err_mm_by_latency_ticks={str(L): float(1e3 * np.concatenate(v).mean()) for L, v in stale.items()})


def by_tray_offset(eps, rate=50.0, lat=0, bins=(0.0, 0.15, 0.30, 0.50, 9.9)):
    """visibility and Kalman position error vs how far the tray centre is from its home (the nominal carry wanders)."""
    vis = [[] for _ in bins[:-1]]; err = [[] for _ in bins[:-1]]
    for e in eps:
        est, _, _, _ = PC.replay(e["payload"], rate, lat)
        end = e["onset"] if e["onset"] >= 0 else e["lens"]
        v = e["payload"][:end, BE.I_VALID] > 0.5
        pe = np.linalg.norm(est[:end, :2] - e["priv"][:end, :2], axis=1)
        off = e["tray_off"][:end]
        for i in range(len(bins) - 1):
            m = (off >= bins[i]) & (off < bins[i + 1])
            vis[i].append(v[m]); err[i].append(pe[m])
    out = {}
    for i in range(len(bins) - 1):
        v, pe = np.concatenate(vis[i]), np.concatenate(err[i])
        out[f"{bins[i]:.2f}-{bins[i + 1]:.2f}m"] = dict(ticks=int(len(v)), visibility=float(v.mean()) if len(v) else None,
                                                        pos_err_mm_mean=float(1e3 * pe.mean()) if len(v) else None,
                                                        pos_err_mm_p95=float(1e3 * np.percentile(pe, 95)) if len(v) else None)
    return out


def frame_quality(eps):
    """per captured frame (before failure): share of frames that are valid although NO ray hit the ball (false detections),
    and share of valid frames whose centre is more than 20 mm off (measurement error, before the Kalman filter)."""
    fv, tot, bad, nval = 0, 0, 0, 0
    for e in eps:
        end = e["onset"] if e["onset"] >= 0 else e["lens"]
        pay = e["payload"][:end]
        v = pay[:, BE.I_VALID] > 0.5
        err = np.linalg.norm(pay[:, 1:3] - e["priv"][:end, :2], axis=1)
        fv += int((v & (e["nb"][:end] == 0)).sum()); tot += end; nval += int(v.sum()); bad += int((v & (err > 0.02)).sum())
    return dict(false_valid_rate=fv / max(tot, 1), valid_with_err_gt_20mm=bad / max(nval, 1))


def table(res):
    L = ["| rig | setting | vis all | vis last25 | med ball hits (true) | med fit pts | pos err mm mean/p95 (all) | pos err mm mean/p95 (last25) | vel err m/s (last25) | edge err mm (last25) | loss/ep |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
    for rig, cfgs in res["rigs"].items():
        for name, r in cfgs["main"].items():
            a, b = r["all"], r["last25"]
            L.append(f"| {rig} | {name} | {f(a['visibility'])} | {f(b['visibility'])} | {f(a['median_true_ball_hits'], 0)} | "
                     f"{f(a['median_fit_points'], 0)} | {f(a['pos_err_mm']['mean'], 1)} / {f(a['pos_err_mm']['p95'], 1)} | "
                     f"{f(b['pos_err_mm']['mean'], 1)} / {f(b['pos_err_mm']['p95'], 1)} | "
                     f"{f(b['vel_err_mps']['mean'], 3)} / {f(b['vel_err_mps']['p95'], 3)} | "
                     f"{f(b['edge_err_mm']['mean'], 1)} / {f(b['edge_err_mm']['p95'], 1)} | {f(r['loss_events_per_ep'], 2)} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-eps", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=GN.SEED0)
    ap.add_argument("--config", default="piper")
    ap.add_argument("--rigs", default=",".join(RIGS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--sweep-rigs", default="head_aim,wrist_l,head_aim+wrist")
    ap.add_argument("--out", default=str(OUT / "camera_study.json"))
    a = ap.parse_args()
    K.TMP.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    rigs = [r for r in a.rigs.split(",") if r]
    seeds = [a.seed0 + i for i in range(a.n_eps)]
    cost = {}
    if not a.analyse_only:
        t0 = time.perf_counter()
        tasks = []
        for r in rigs:
            if (TMP / f"study_{r}_{a.n_eps}.pkl").exists():
                print(f"[study] {r}: cached", flush=True)
                continue
            tasks += [(r, seeds[i:i + a.chunk], a.config) for i in range(0, len(seeds), a.chunk)]
        acc = {}
        with Pool(a.workers, maxtasksperchild=2) as P:
            for rig, out, c in P.imap_unordered(collect_task, tasks):
                acc.setdefault(rig, []).append((out, c))
                n_done = sum(len(o) for o, _ in acc[rig])
                print(f"[study] {rig}: {n_done}/{a.n_eps} eps, wall {time.perf_counter() - t0:.0f}s", flush=True)
                if n_done == a.n_eps:
                    eps = sorted([e for o, _ in acc[rig] for e in o], key=lambda e: e["seed"])
                    cs = {k: sum(c[k] for _, c in acc[rig]) for k in acc[rig][0][1]}
                    with open(TMP / f"study_{rig}_{a.n_eps}.pkl", "wb") as f:
                        pickle.dump(dict(eps=eps, cost=cs), f)
    res = dict(n_eps=a.n_eps, seeds=[seeds[0], seeds[-1]], config=a.config, pre_failure_ticks=PRE,
               main_settings={k: dict(rate_hz=v[0], latency_ticks=v[1]) for k, v in MAIN.items()}, rigs={})
    for r in rigs:
        p = TMP / f"study_{r}_{a.n_eps}.pkl"
        if not p.exists():
            print(f"[study] {r}: no data", flush=True)
            continue
        d = pickle.load(open(p, "rb"))
        eps = d["eps"]
        entry = dict(main={k: analyse_rig(eps, *v) for k, v in MAIN.items()}, cost=d["cost"],
                     cost_per_frame_ms=dict(cast=1e3 * d["cost"]["t_cast"] / max(d["cost"]["frames"], 1),
                                            fit=1e3 * d["cost"]["t_fit"] / max(d["cost"]["frames"], 1),
                                            rays=d["cost"]["rays"] / max(d["cost"]["frames"], 1)),
                     frames_total=d["cost"]["frames"], ticks_total=int(sum(e["lens"] for e in eps)))
        entry["by_tray_offset_ideal"] = by_tray_offset(eps)
        entry["frame_quality"] = frame_quality(eps)
        if r in a.sweep_rigs.split(","):
            entry["sweep_last25"] = {f"{int(rt)}Hz_lat{lt}": analyse_rig(eps, rt, lt)["last25"]
                                     for rt in SWEEP_RATES for lt in SWEEP_LATS}
        res["rigs"][r] = entry
        if "reference" not in res:
            res["reference"] = reference(eps)
        print(f"[study] analysed {r}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    tab = table(res)
    (OUT / "camera_study_table.md").write_text(tab + "\n")
    print(tab, flush=True)
    print(f"[study] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
