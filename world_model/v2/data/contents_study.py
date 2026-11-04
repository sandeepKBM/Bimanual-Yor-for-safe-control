#!/usr/bin/env python
"""contents_study.py -- how much do cup CONTENTS change spill for the SAME cup motion?
Plus sim cost vs particle count and two cheap WATER proxies.

SIMULATION ONLY (MuJoCo cup-only rig, offline). New file; edits nothing.

Subcommands
  sens    one-at-a-time sensitivity: nominal beads vs each parameter at its range
          ends, on a fixed bank of rig motions (experiments/slosh/rig.make_motion
          families, fixed seeds). Metrics vs nominal: P(spill), mean spilled frac,
          label-flip rate, onset shift.
  cost    ms per physics step (and real-time factor) vs number of spheres.
  liquid  (A) liquid-like particles (60/120 small soft frictionless spheres,
          optional drag relative to the cup) and (B) equivalent-mechanical first-
          slosh-mode spherical pendulum (analytic ODE driven by the scripted cup
          pose), vs nominal beads: quasi-static tilt spill angle, cost, and onset
          agreement on the same motion bank.

Example
  python world_model/v2/data/contents_study.py sens --n-motions 48 --workers 16
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contents as C  # noqa: E402

FAMS = ("translate", "shake", "tilt", "bangbang", "mixed", "vertical")
G = 9.81


def motion_bank(n, seed0=1000):
    import rig
    out = []
    for i in range(n):
        fam = FAMS[i % len(FAMS)]
        rng = np.random.default_rng(seed0 + i)
        for _ in range(50):
            pos, quat, k0, p = rig.make_motion(rng, fam)
            if rig.max_point_speed(pos, quat) <= rig.V_WALL_MAX:
                break
        out.append((fam, pos + (np.array([0, 0, 1.0]) - rig.CUP_HOME), quat, k0))
    return out


def _onset(sp, k0):
    idx = np.flatnonzero(sp[k0:] > 0)
    return float((idx[0]) * C.DT) if len(idx) else np.nan


# ------------------------------------------------------------------ workers
def _run_beads(args):
    cdict, mi, n_motions, seed0 = args
    c = C.Contents(**cdict)
    fam, pos, quat, k0 = motion_bank(n_motions, seed0)[mi]
    m = C.rig_model(c)
    r = C.run_rig(m, c, pos, quat, drag=c.drag)
    sp = r["spilled_rim"]          # genuine over-the-rim spill only (floor/wall tunnels excluded)
    return dict(cfg=cdict, mi=mi, fam=fam, spill=int(sp[k0:].max() > 0), frac=float(sp[-1]) / c.n_balls,
                onset=_onset(sp, k0), pre=int(sp[k0] > 0), tunnel=r["tunnel"], ms=r["ms_per_step"])


def pendulum_run(pos, quat, h, zeta=0.02, sub=10):
    """(B) equivalent-mechanical model of the FIRST slosh mode of a cylindrical
    tank (Abramson/Ibrahim): spherical pendulum, length L = g/w1^2 with
    w1^2 = (g*xi/R) tanh(xi*h/R), xi = 1.841; pivot on the cup axis at the free-
    surface height h (simplification), driven by the scripted cup pose (pivot
    acceleration from finite differences). Spill when the equivalent free-surface
    tilt phi (pendulum angle relative to the cup axis) reaches the rim:
    tan(phi) >= (H - h)/R  (exact in the quasi-static limit).
    Returns (phi_deg (T,), spill (T,) bool, L, f1, us_per_step)."""
    t0 = time.perf_counter()
    R, xi, H = C.CUP_INNER_R, 1.841, C.CUP_HEIGHT
    w1 = np.sqrt(G * xi / R * np.tanh(xi * h / R)); L = G / w1**2
    T = len(pos)
    Rm = np.empty((T, 3, 3))
    import mujoco
    for k in range(T):
        tmp = np.empty(9); mujoco.mju_quat2Mat(tmp, quat[k].astype(float)); Rm[k] = tmp.reshape(3, 3)
    piv = pos + Rm[:, :, 2] * h
    acc = np.zeros_like(piv); acc[1:-1] = (piv[2:] - 2 * piv[1:-1] + piv[:-2]) / C.DT**2
    # cup angular velocity (world) from R_dot R^T
    w = np.zeros((T, 3))
    for k in range(1, T):
        dR = (Rm[k] @ Rm[k - 1].T - np.eye(3)) / C.DT
        w[k] = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    g = np.array([0, 0, -G])
    n = -Rm[0][:, 2].copy(); nd = np.zeros(3)
    phi = np.zeros(T); h_sub = C.DT / sub
    for k in range(T):
        for _ in range(sub):
            f = (g - acc[k]); f = f - n * (n @ f)
            ndd = f / L - (nd @ nd) * n - 2 * zeta * w1 * (nd - np.cross(w[k], n))
            nd = nd + h_sub * ndd
            n = n + h_sub * nd; n /= np.linalg.norm(n); nd -= n * (n @ nd)
        phi[k] = np.degrees(np.arccos(np.clip(-n @ Rm[k][:, 2], -1, 1)))
    phi_star = np.degrees(np.arctan((H - h) / R))
    us = 1e6 * (time.perf_counter() - t0) / T
    return phi, phi >= phi_star, L, w1 / (2 * np.pi), us, phi_star


def _run_pend(args):
    mi, n_motions, seed0, h, zeta = args
    fam, pos, quat, k0 = motion_bank(n_motions, seed0)[mi]
    phi, sp, L, f1, us, ps = pendulum_run(pos, quat, h, zeta)
    idx = np.flatnonzero(sp[k0:])
    return dict(cfg=dict(kind="pendulum", h=h, zeta=zeta), mi=mi, fam=fam, spill=int(len(idx) > 0),
                frac=np.nan, onset=float(idx[0] * C.DT) if len(idx) else np.nan, pre=0, tunnel=False,
                ms=us / 1e3, L=L, f1=f1, phi_star=ps)


def _pool(fn, jobs, workers):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=20) as p:
        return p.map(fn, jobs, chunksize=2)


def compare(rows_ref, rows):
    ref = {r["mi"]: r for r in rows_ref}
    flips = np.mean([r["spill"] != ref[r["mi"]]["spill"] for r in rows])
    both = [(r["onset"], ref[r["mi"]]["onset"]) for r in rows if r["spill"] and ref[r["mi"]]["spill"]]
    d = np.array([a - b for a, b in both]) if both else np.array([np.nan])
    return dict(p_spill=float(np.mean([r["spill"] for r in rows])),
                frac=float(np.nanmean([r["frac"] for r in rows])),
                flip=float(flips), n_both=len(both),
                onset_shift_med=float(np.nanmedian(d)), onset_absdiff_med=float(np.nanmedian(np.abs(d))),
                ms=float(np.mean([r["ms"] for r in rows])),
                pre=int(sum(r["pre"] for r in rows)), tunnel=int(sum(r["tunnel"] for r in rows)))


def _scaled(pos, quat, s):
    sys.path.insert(0, str(C.REPO / "world_model" / "v2" / "data"))
    from gen_spill_data import quat_pow
    home = pos[0]
    return home + s * (pos - home), quat_pow(quat, s)


def _thresh_one(args):
    """Bisection on the motion amplitude scale s in [0, S_MAX]: smallest s that spills."""
    cdict, mi, n_motions, seed0, iters = args
    c = C.Contents(**cdict)
    fam, pos, quat, k0 = spill_bank(n_motions, seed0)[mi]
    m = C.rig_model(c)
    def spills(s):
        p, q = _scaled(pos, quat, s)
        return C.run_rig(m, c, p, q)["spilled_rim"][k0:].max() > 0
    lo, hi = 0.0, 1.5
    if not spills(hi):
        return dict(cfg=cdict, mi=mi, fam=fam, s=np.nan)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if spills(mid):
            hi = mid
        else:
            lo = mid
    return dict(cfg=cdict, mi=mi, fam=fam, s=hi)


def spill_bank(n, seed0=5000):
    """Spill-PRONE motions (tilt/mixed/shake/vertical) for threshold probing."""
    import rig
    fams = ("tilt", "mixed", "shake", "vertical")
    out = []
    for i in range(n):
        fam = fams[i % len(fams)]
        rng = np.random.default_rng(seed0 + i)
        for _ in range(50):
            pos, quat, k0, p = rig.make_motion(rng, fam)
            if rig.max_point_speed(pos, quat) <= rig.V_WALL_MAX:
                break
        out.append((fam, pos + (np.array([0, 0, 1.0]) - rig.CUP_HOME), quat, k0))
    return out


def cmd_thresh(a):
    nom = C.Contents(**C.NOMINAL)
    variants = [("nominal", nom)]
    for n in (2, 4, 12):
        variants.append((f"n_balls={n}", replace(nom, n_balls=n)))
    for k, vals in (("radius", (0.006, 0.010)), ("friction", (0.25, 0.8, 0.9)), ("dampratio", (0.45, 0.7)),
                    ("density", (600.0, 1400.0)), ("cup_mass", (0.015, 0.12))):
        for v in vals:
            variants.append((f"{k}={v}", replace(nom, **{k: v})))
    jobs, names = [], []
    for name, c in variants:
        cd = c.to_dict(); cd["solimp"] = tuple(cd["solimp"])
        for mi in range(a.n_motions):
            jobs.append((cd, mi, a.n_motions, a.seed0, 5)); names.append(name)
    t0 = time.perf_counter()
    res = _pool(_thresh_one, jobs, a.workers)
    print(f"[thresh] {len(jobs)} bisections in {time.perf_counter() - t0:.0f}s", flush=True)
    by = {}
    for nm, r in zip(names, res):
        by.setdefault(nm, {})[r["mi"]] = r["s"]
    ref = by["nominal"]
    print(f"{'variant':16s} {'n_spillable':>11s} {'med s*':>7s} {'med s*/s*_nom':>13s} {'IQR ratio':>16s} {'frac |ratio-1|>10%':>18s}")
    tab = {}
    for nm, _ in variants:
        rr = np.array([by[nm][i] / ref[i] for i in range(a.n_motions) if np.isfinite(ref[i]) and np.isfinite(by[nm][i])])
        ns = int(np.sum(np.isfinite(list(by[nm].values()))))
        ss = np.array([v for v in by[nm].values() if np.isfinite(v)])
        q1, q3 = (np.percentile(rr, [25, 75]) if len(rr) else (np.nan, np.nan))
        tab[nm] = dict(n=ns, med_s=float(np.median(ss)) if len(ss) else np.nan, med_ratio=float(np.median(rr)) if len(rr) else np.nan,
                       q1=float(q1), q3=float(q3), frac_10=float(np.mean(np.abs(rr - 1) > 0.1)) if len(rr) else np.nan)
        t = tab[nm]
        print(f"{nm:16s} {ns:11d} {t['med_s']:7.3f} {t['med_ratio']:13.3f} {'[%.2f, %.2f]' % (q1, q3):>16s} {t['frac_10']:18.2f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(tab, indent=1))


def _bank_run(args):
    """One (contents, motion) rig run on the rig.py bank; rim-only spill label."""
    name, cdict, mi, n_motions, seed0 = args
    fam, pos, quat, k0 = motion_bank(n_motions, seed0)[mi]
    if cdict.get("kind") == "pendulum":
        phi, sp, L, f1, us, ps = pendulum_run(pos, quat, cdict["h"], cdict["zeta"])
        idx = np.flatnonzero(sp[k0:])
        return dict(name=name, mi=mi, fam=fam, spill=int(len(idx) > 0), onset=(k0 + idx[0]) if len(idx) else -1,
                    ms=us / 1e3, tunnel=False)
    c = C.Contents(**cdict)
    m = C.rig_model(c)
    r = C.run_rig(m, c, pos, quat, drag=c.drag)
    idx = np.flatnonzero(r["spilled_rim"][k0:] > 0)
    return dict(name=name, mi=mi, fam=fam, spill=int(len(idx) > 0), onset=(k0 + idx[0]) if len(idx) else -1,
                ms=r["ms_per_step"], tunnel=r["tunnel"])


def _slow_tilt(args):
    name, cdict = args
    pos, quat, k0, th = tilt_ramp(deg=120.0, dur=12.0)   # 10 deg/s
    if cdict.get("kind") == "pendulum":
        phi, sp, *_ = pendulum_run(pos, quat, cdict["h"], cdict["zeta"])
        idx = np.flatnonzero(sp[k0:])
    else:
        c = C.Contents(**cdict); m = C.rig_model(c)
        idx = np.flatnonzero(C.run_rig(m, c, pos, quat, drag=c.drag)["spilled_rim"][k0:] > 0)
    return name, float(np.degrees(th[k0 + idx[0]])) if len(idx) else np.nan


def boundary_contents():
    nom = C.Contents(**C.NOMINAL)
    cd = lambda c: {**c.to_dict(), "solimp": tuple(c.solimp)}
    types = [
        ("beads8 nominal", cd(nom)),
        ("beads12 highfill", cd(replace(nom, n_balls=12))),
        ("beads12 r6 light lowfric", cd(replace(nom, n_balls=12, radius=0.006, density=600.0, friction=0.25))),
        ("beads8 r10 bouncy", cd(replace(nom, n_balls=8, radius=0.010, dampratio=0.45))),
        ("A60 poly liquid-like", cd(particle_contents(60, 0.0038, poly=0.3))),
        ("A120 liquid-like", cd(particle_contents(120, 0.0035))),
        ("A120 poly highfill", cd(particle_contents(120, 0.0032, poly=0.3))),
        ("B pendulum h33 z.02", dict(kind="pendulum", h=0.033, zeta=0.02)),
        ("B pendulum h55 z.02", dict(kind="pendulum", h=0.055, zeta=0.02)),
        ("B pendulum h33 z.2", dict(kind="pendulum", h=0.033, zeta=0.2)),
    ]
    return types


def cmd_boundary(a):
    import slosh_models as SM
    types = boundary_contents()
    bank = motion_bank(a.n_motions, a.seed0)
    # predictors from the SCRIPTED cup kinematics (same for every contents type)
    P = np.stack([b[1] for b in bank]); Q = np.stack([b[2] for b in bank])
    import mujoco
    N = np.zeros_like(P)
    for e in range(len(bank)):
        for k in range(P.shape[1]):
            R = np.empty(9); mujoco.mju_quat2Mat(R, Q[e, k].astype(float)); N[e, k] = R.reshape(3, 3)[:, 2]
    K = SM.kinematics(P, N)
    k0 = bank[0][3]
    a0 = K["theta0"][:, k0:].max(1); a1 = K["theta_eff"][:, k0:].max(1)
    b3 = {}
    for w in (5, 10, 20, 35, 50):
        for z in (0.2, 1.0, 2.5):
            b3[(w, z)] = SM.lti_world(K, w, z)[:, k0:].max(1)
    jobs = [(nm, cdict, mi, a.n_motions, a.seed0) for nm, cdict in types for mi in range(a.n_motions)]
    t0 = time.perf_counter()
    res = _pool(_bank_run, jobs, a.workers)
    slow = dict(_pool(_slow_tilt, [(nm, cdict) for nm, cdict in types], min(a.workers, len(types))))
    print(f"[boundary] {len(jobs)} runs + {len(types)} slow tilts in {time.perf_counter() - t0:.0f}s; "
          f"bank n={a.n_motions} (rig.py families x{a.n_motions // 6}); spill = RIM exits only", flush=True)
    print(f"{'contents':26s} {'qs-tilt':>7s} {'P(spill)':>8s} {'tun':>4s} {'AUROC a0':>8s} {'a1':>6s} {'b3 fix':>6s} "
          f"{'b3 refit':>8s} {'thr a0':>6s} {'sp@<45':>6s} {'sp@<60':>6s} {'ms/step':>7s}")
    out = {}
    deg = np.degrees
    for nm, _ in types:
        R = sorted([r for r in res if r["name"] == nm], key=lambda r: r["mi"])
        y = np.array([r["spill"] for r in R])
        tun = sum(r["tunnel"] for r in R)
        if 0 < y.sum() < len(y):
            au0 = SM.auroc(a0, y); au1 = SM.auroc(a1, y); au3 = SM.auroc(b3[(35, 2.5)], y)
            best = max(b3, key=lambda k: SM.auroc(b3[k], y)); au3r = SM.auroc(b3[best], y)
            ths = np.unique(a0); yj = [np.mean(a0[y == 1] >= t) - np.mean(a0[y == 0] >= t) for t in ths]
            thr = deg(ths[int(np.argmax(yj))])
        else:
            au0 = au1 = au3 = au3r = thr = np.nan; best = None
        sp45 = float(np.mean(deg(a0[y == 1]) < 45)) if y.sum() else np.nan
        sp60 = float(np.mean(deg(a0[y == 1]) < 60)) if y.sum() else np.nan
        ms = float(np.mean([r["ms"] for r in R]))
        out[nm] = dict(qs_tilt=slow[nm], p_spill=float(y.mean()), tunnel=int(tun), auroc_a0=au0, auroc_a1=au1,
                       auroc_b3_fixed=au3, auroc_b3_refit=au3r, b3_refit_params=best, a0_threshold_deg=thr,
                       frac_spills_below45=sp45, frac_spills_below60=sp60, ms_per_step=ms)
        print(f"{nm:26s} {slow[nm]:7.1f} {y.mean():8.2f} {tun:4d} {au0:8.3f} {au1:6.3f} {au3:6.3f} {au3r:8.3f} "
              f"{thr:6.1f} {sp45:6.2f} {sp60:6.2f} {ms:7.2f}   b3refit={best}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))


# ------------------------------------------------------------------ subcommands
def cmd_sens(a):
    nom = C.Contents(**C.NOMINAL)
    variants = [("nominal", nom)]
    for n in (2, 4, 12):
        variants.append((f"n_balls={n}", replace(nom, n_balls=n)))
    for k, vals in (("radius", (0.006, 0.010)), ("friction", (0.25, 0.8, 0.9)), ("dampratio", (0.45, 0.7)),
                    ("density", (600.0, 1400.0)), ("cup_mass", (0.015, 0.12))):
        for v in vals:
            variants.append((f"{k}={v}", replace(nom, **{k: v})))
    jobs, names = [], []
    for name, c in variants:
        assert C.layout(c) is not None, name
        cd = c.to_dict(); cd["solimp"] = tuple(cd["solimp"])
        for mi in range(a.n_motions):
            jobs.append((cd, mi, a.n_motions, a.seed0)); names.append(name)
    t0 = time.perf_counter()
    res = _pool(_run_beads, jobs, a.workers)
    print(f"[sens] {len(jobs)} rig runs in {time.perf_counter() - t0:.0f}s", flush=True)
    by = {}
    for nm, r in zip(names, res):
        by.setdefault(nm, []).append(r)
    ref = by["nominal"]
    table = {}
    print(f"{'variant':18s} {'P(spill)':>8s} {'frac':>6s} {'flip':>6s} {'n_both':>6s} {'dOnset_med[s]':>13s} {'|dOnset|_med':>12s} {'ms/step':>8s}")
    for nm, _ in variants:
        s = compare(ref, by[nm]); table[nm] = s
        print(f"{nm:18s} {s['p_spill']:8.2f} {s['frac']:6.2f} {s['flip']:6.2f} {s['n_both']:6d} "
              f"{s['onset_shift_med']:13.3f} {s['onset_absdiff_med']:12.3f} {s['ms']:8.2f}", flush=True)
    # seed-noise floor: re-run nominal with a 1e-5 m perturbation of ONE ball -> chaotic divergence floor
    if a.out:
        Path(a.out).write_text(json.dumps(dict(table=table, n_motions=a.n_motions), indent=1))


def _cost_one(args):
    kind, n, r, cd = args
    import rig
    c = C.Contents(**cd)
    rng = np.random.default_rng(7)
    pos, quat, k0, _ = rig.make_motion(rng, "shake")
    pos = pos + (np.array([0, 0, 1.0]) - rig.CUP_HOME)
    m = C.rig_model(c)
    T = 400
    res = C.run_rig(m, c, pos[:T], quat[:T], drag=c.drag)
    return dict(kind=kind, n=n, r=r, ms=res["ms_per_step"], fill=C.fill_height(c))


def particle_contents(n, r, drag=0.0, poly=0.0):
    return C.Contents(n_balls=n, radius=r, density=1000.0, friction=0.0, dampratio=1.0, condim=1,
                      solref_tc=0.01, solimp=(0.8, 0.95, 0.002), torsional=0.0, rolling=0.0,
                      drag=drag, kind="particles", poly=poly)


def cmd_cost(a):
    jobs = []
    for n, r in ((2, 0.008), (8, 0.008), (12, 0.008), (30, 0.0055), (60, 0.0045), (120, 0.0035)):
        cb = C.Contents(n_balls=n, radius=r, kind="beads")
        if C.layout(cb) is not None:
            d = cb.to_dict(); d["solimp"] = tuple(d["solimp"]); jobs.append(("beads(condim6)", n, r, d))
        cp = particle_contents(n, r)
        if C.layout(cp) is not None:
            d = cp.to_dict(); d["solimp"] = tuple(d["solimp"]); jobs.append(("particles(condim1)", n, r, d))
    res = _pool(_cost_one, jobs, min(a.workers, len(jobs)))
    print(f"{'kind':20s} {'N':>4s} {'r[mm]':>6s} {'fill[mm]':>8s} {'ms/step':>8s} {'RTF':>6s} {'s/4s-episode':>12s}")
    for r in res:
        print(f"{r['kind']:20s} {r['n']:4d} {1e3*r['r']:6.1f} {1e3*r['fill']:8.1f} {r['ms']:8.2f} "
              f"{C.DT*1e3/r['ms']:6.2f} {r['ms']*800/1e3:12.2f}", flush=True)


def tilt_ramp(deg=120.0, dur=4.0, settle=0.4):
    t = np.arange(0, settle + dur + 0.3, C.DT)
    th = np.radians(deg) * np.clip((t - settle) / dur, 0, 1)
    pos = np.tile([0, 0, 1.0], (len(t), 1))
    quat = np.stack([np.cos(th / 2), np.sin(th / 2), 0 * th, 0 * th], 1)
    return pos, quat, int(settle / C.DT), th


def _tilt_one(args):
    name, cd = args
    c = C.Contents(**cd)
    pos, quat, k0, th = tilt_ramp()
    m = C.rig_model(c)
    r = C.run_rig(m, c, pos, quat, drag=c.drag)
    sp = r["spilled_rim"]; idx = np.flatnonzero(sp[k0:] > 0)
    ang = float(np.degrees(th[k0 + idx[0]])) if len(idx) else np.nan
    half = np.flatnonzero(sp >= max(1, c.n_balls // 2))
    return name, ang, (float(np.degrees(th[half[0]])) if len(half) else np.nan), r["ms_per_step"], C.fill_height(c)


def cmd_liquid(a):
    nom = C.Contents(**C.NOMINAL)
    A60 = particle_contents(60, 0.0045); A120 = particle_contents(120, 0.0035)
    A120d = particle_contents(120, 0.0035, drag=8.0)
    A120p = particle_contents(120, 0.0032, poly=0.3)
    cfgs = [("beads8(nominal)", nom), ("A60 r4.5", A60), ("A120 r3.5", A120), ("A120 r3.5 +drag8", A120d),
            ("A120 poly30%", A120p)]
    if a.tilt_only:
        cfgs = [("A120 poly30%", A120p), ("A60 poly30%", particle_contents(60, 0.0038, poly=0.3))]
    for nm, c in cfgs:
        assert C.layout(c) is not None, nm
    def cd(c):
        d = c.to_dict(); d["solimp"] = tuple(d["solimp"]); return d
    # 1) quasi-static tilt spill angle
    tj = [(nm, cd(c)) for nm, c in cfgs]
    tres = _pool(_tilt_one, tj, len(tj))
    print("[quasi-static tilt ramp 0->120 deg over 4 s]")
    print(f"{'model':20s} {'fill[mm]':>8s} {'first-out[deg]':>14s} {'half-out[deg]':>13s} {'ms/step':>8s} {'RTF':>6s}")
    for nm, ang, half, ms, fill in tres:
        print(f"{nm:20s} {1e3*fill:8.1f} {ang:14.1f} {half:13.1f} {ms:8.2f} {C.DT*1e3/ms:6.2f}")
    if a.tilt_only:
        return
    h_liq = C.fill_height(A120)
    pos, quat, k0, th = tilt_ramp()
    phi, sp, L, f1, us, ps = pendulum_run(pos, quat, h_liq)
    idx = np.flatnonzero(sp)
    print(f"{'B pendulum(h=A120)':20s} {1e3*h_liq:8.1f} {np.degrees(th[idx[0]]) if len(idx) else np.nan:14.1f} "
          f"{'-':>13s} {us/1e3:8.3f} {C.DT*1e6/us:6.0f}   (L={1e3*L:.1f} mm, f1={f1:.2f} Hz, phi*={ps:.1f} deg)")
    # 2) dynamic agreement on the motion bank
    jobs, names = [], []
    for nm, c in cfgs:
        for mi in range(a.n_motions):
            jobs.append((cd(c), mi, a.n_motions, a.seed0)); names.append(nm)
    t0 = time.perf_counter()
    res = _pool(_run_beads, jobs, a.workers)
    pres = _pool(_run_pend, [(mi, a.n_motions, a.seed0, h_liq, 0.02) for mi in range(a.n_motions)], a.workers)
    print(f"[liquid] dynamic bank: {len(jobs) + len(pres)} runs in {time.perf_counter() - t0:.0f}s")
    by = {}
    for nm, r in zip(names, res):
        by.setdefault(nm, []).append(r)
    by["B pendulum"] = pres
    for refname in ("beads8(nominal)", "A120 r3.5"):
        print(f"-- vs reference {refname}")
        print(f"{'model':20s} {'P(spill)':>8s} {'flip':>6s} {'n_both':>6s} {'dOnset_med[s]':>13s} {'|dOnset|_med':>12s} {'ms/step':>8s}")
        for nm in list(by):
            s = compare(by[refname], by[nm])
            print(f"{nm:20s} {s['p_spill']:8.2f} {s['flip']:6.2f} {s['n_both']:6d} {s['onset_shift_med']:13.3f} "
                  f"{s['onset_absdiff_med']:12.3f} {s['ms']:8.3f}")
    fam_rates = {nm: {f: float(np.mean([r['spill'] for r in rs if r['fam'] == f])) for f in FAMS} for nm, rs in by.items()}
    print("-- P(spill) per family")
    print(f"{'model':20s} " + " ".join(f"{f:>9s}" for f in FAMS))
    for nm, d in fam_rates.items():
        print(f"{nm:20s} " + " ".join(f"{d[f]:9.2f}" for f in FAMS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sens", "cost", "liquid", "thresh", "boundary"))
    ap.add_argument("--tilt-only", action="store_true")
    ap.add_argument("--n-motions", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    dict(sens=cmd_sens, cost=cmd_cost, liquid=cmd_liquid, thresh=cmd_thresh, boundary=cmd_boundary)[a.cmd](a)


if __name__ == "__main__":
    main()
