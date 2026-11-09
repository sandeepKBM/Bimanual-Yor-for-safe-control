"""Evaluate the built observe_pose skill (pan/observe_pose.py).  SIMULATION ONLY, CPU only (osmesa depth rendering, no GPU).

    python -m world_model.v2.nero.pan.observe_eval accuracy --out DIR [--n 200] [--noise clean low mid zed zed_calib] [--procs P]
    python -m world_model.v2.nero.pan.observe_eval closed   --out DIR [--noise clean mid zed zed_calib] [--n-interior 30] [--procs P]

accuracy (E1): theta uniform in the PREREG pose box (dx, dy +-1.5 cm, yaw +-8 deg; mu x0.6-1.2, 0-0.4 kg random placement);
  per noise level one estimate (up to 3 images while fit_ok is False) vs the measured true pan pose.  No arm motion.
closed (E2): the plan our layer admitted (pinch 0.65, push 0.07, clamp, lift; under_rim.run) with observe ON, on worlds from the
  FULL PREREG box (no assumed shrink): its 32 vertices + n random interior worlds (seed 1).  One JSON per run in DIR/raw/.
Threads are pinned to 1 per worker (AGENTS.md); size --procs from the idle cores.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PLAN = dict(grasp_frac=0.65, push=0.07, grip_force=10.0)
PL = ["none", "near_handle", "far_centre", "far_left", "far_right"]


def _pin():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("MUJOCO_GL", "osmesa")


def _acc(job):
    _pin()
    import mujoco
    from . import covered as CV, scene as S, observe_pose as O, under_rim as U
    tag, th, noises, seed = job
    try:
        cp = CV.CoveredParams(placement=th["placement"])
        if th["hidden_mass"]:
            import dataclasses
            cp = dataclasses.replace(cp, hidden_mass=th["hidden_mass"], overload_mass=th["hidden_mass"])
        p = cp.pan
        m = CV.compose(cp)
        m.geom_friction[:, 0] *= th["mu_scale"]
        d = mujoco.MjData(m)
        S.settle(m, d, 1.0)
        r = mujoco.Renderer(m, O.CAM["height"], O.CAM["width"])
        tm = O.template(m, d, p, renderer=r)
        pos0, R0 = d.xpos[m.body("pan").id].copy(), d.xmat[m.body("pan").id].reshape(3, 3).copy()
        U._perturb_pose(m, d, p.pan_centre_xy, th["dx"], th["dy"], th["yaw_deg"])
        S.settle(m, d, 0.5)
        tru = O.true_pose(m, d, p, pos0, R0)
        out = []
        for nz in noises:
            rng = np.random.default_rng(seed)
            for k in range(3):
                e = O.estimate(m, d, p, tm, nz, rng, renderer=r)
                if e["fit_ok"]:
                    break
            out.append(dict(tag=tag, theta=th, noise=nz, n_images=k + 1, est=e, true=tru,
                            residual_mm=1e3 * float(np.hypot(e["dx"] - tru["dx"], e["dy"] - tru["dy"])),
                            residual_yaw_deg=float(e["yaw_deg"] - tru["yaw_deg"]), fit_ok=e["fit_ok"]))
        r.close()
        return out
    except Exception:
        return [dict(tag=tag, theta=th, error=traceback.format_exc()[-800:])]


def _closed(job):
    _pin()
    from . import under_rim as U
    try:
        from .vizcoast import evidence as E
    except ImportError as e:      # USER 11-09: moved to branch archive/pan-history-2026-11-09
        raise ImportError("observe_eval needs nero/pan/vizcoast, moved to branch archive/pan-history-2026-11-09") from e
    tag, th, nz, seed = job
    kw = dict(placement=th["placement"], hidden_mass=th["hidden_mass"] if th["hidden_mass"] > 0 else None,
              perturb=dict(dx=th["dx"], dy=th["dy"], yaw_deg=th["yaw_deg"], mu_scale=th["mu_scale"]),
              observe=dict(noise=nz, seed=seed, retries=3), **PLAN)
    try:
        r = U.run(**kw)
        r.pop("log", None)
    except Exception:
        r = dict(kw, held=False, error=traceback.format_exc()[-800:])
    r.update(tag=tag, noise=nz, theta=th, category=E.category(r))
    return r


def _random_thetas(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pl = PL[rng.integers(5)]
        out.append(dict(dx=float(rng.uniform(-.015, .015)), dy=float(rng.uniform(-.015, .015)), yaw_deg=float(rng.uniform(-8, 8)),
                        mu_scale=float(rng.uniform(0.6, 1.2)), placement=pl,
                        hidden_mass=float(rng.uniform(0, 0.4)) if pl != "none" else 0.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["accuracy", "closed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-interior", type=int, default=30)
    ap.add_argument("--noise", nargs="+", default=None)
    ap.add_argument("--procs", type=int, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = get_context("spawn")
    if a.exp == "accuracy":
        noises = a.noise or ["clean", "low", "mid", "zed", "zed_calib"]
        jobs = [(f"E{i:03d}", th, noises, 1000 + i) for i, th in enumerate(_random_thetas(a.n, 7))]
        f = out / "accuracy.jsonl"
        with ctx.Pool(a.procs, maxtasksperchild=3) as pool:
            for rows in pool.imap_unordered(_acc, jobs):
                with open(f, "a") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, default=float) + "\n")
                print(rows[0]["tag"], [(r.get("noise"), round(r.get("residual_mm", -1), 2), round(r.get("residual_yaw_deg", 0), 2))
                                       for r in rows], flush=True)
        (out / "_DONE_accuracy").write_text("ok")
        return
    try:
        from .vizcoast import domain as D
    except ImportError as e:      # USER 11-09: moved to branch archive/pan-history-2026-11-09
        raise ImportError("observe_eval needs nero/pan/vizcoast, moved to branch archive/pan-history-2026-11-09") from e
    noises = a.noise or ["clean", "mid", "zed", "zed_calib"]
    worlds = [(f"V{i:02d}", th) for i, th in enumerate(D.PREREG.vertices())]
    worlds += [(f"I{i:02d}", th) for i, th in enumerate(_random_thetas(a.n_interior, 1))]
    jobs = [(f"{tag}_{nz}", th, nz, 2000 + j) for nz in noises for j, (tag, th) in enumerate(worlds)]
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    done = {p.stem for p in raw.glob("*.json")}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"closed loop: {len(jobs)} runs", flush=True)
    with ctx.Pool(a.procs, maxtasksperchild=2) as pool:
        for r in pool.imap_unordered(_closed, jobs):
            (raw / f"{r['tag']}.json").write_text(json.dumps(r, indent=1, default=float))
            o = r.get("observe") or {}
            print(r["tag"], r["held"], r["category"], round(o.get("residual_mm", -1), 2), round(o.get("residual_yaw_deg", 0), 2), flush=True)
    (out / "_DONE_closed").write_text("ok")


if __name__ == "__main__":
    main()
