"""E2 (estimation + calibration), E3 (prediction), E4 (constraint status) over the recorded runs.  SIMULATION ONLY, offline replay.

    python -m world_model.v2.house.ocs.exp_e234 run [--cal 0 1 2] [--test 3 4 5] [--procs 7]      # ~10 min
Pass A: perception producers with engineering PRIORS (no calibration)  -> E2 (+E4) records.
Fit  : split-conformal radii on the calibration seeds (pass A residuals).
Pass B: producers with the calibrated sets -> E2 / E3 / E4 records on all seeds; coverage is reported separately for the calibration seeds (in-sample) and
        the held-out seeds.  The flagship desk recording uses the h5 calibration of the robot-side sets (base / tray FK / cup); its obstacle sets are the
        stereo pipeline's own p95 inflation.
Outputs: <OUT>/e2_calibration.{json,md}, e3_prediction.{json,md}, e4_constraints.{json,md}, calib_h5.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from world_model.v2.house.ocs import evalcore as EC, producers as P

OUT = Path(os.environ.get("OCS_OUT", "data/molmospaces/ocs"))
REC = OUT / "rec"       # recordings; --out redirects only the results, --rec the recordings
ALPHA = 0.05
CALIB_LEGACY = False    # --calib-legacy: v3 cup calibration (fit on ALL rows with a blob reading, stale ones included)
UNDER = 0.90            # held-out coverage below this is flagged


def rec_path(seed):
    return REC / f"carry_s{seed}.pkl.gz"


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


# ====================================================================================== calibration
def _fresh_sides(r, legacy):
    """sides whose blob reading `r["raw"][side]` is FRESH in this e2 cup record (see evalcore: in the latest blob measurement and within the 0.15 s hold).
    legacy=True (or a record without `blob_fresh`) = every side that has a reading, including stale / frozen ones (the v3 behaviour)."""
    if legacy or "blob_fresh" not in r:
        return list(r["raw"])
    return [s_ for s_ in r["raw"] if r["blob_fresh"].get(s_, False)]


def _held_sides(r, legacy):
    """sides of the estimate the producer holds at this tick (mean over the sides of its latest blob measurement); legacy = all sides with a reading."""
    if legacy or "blob_sides" not in r:
        return list(r["raw"])
    return [s_ for s_ in r["raw"] if s_ in r["blob_sides"]]


def fit_calib(results_cal, legacy=False):
    """results_cal: pass-A process() outputs of the calibration seeds.
    legacy=False (default): the per-side blob affine / bias are fitted ONLY on rows whose blob reading is fresh for that side (record field blob_fresh), and the
    visible residuals use the fresh sides; the occluded residuals use the sides of the held estimate.  legacy=True (--calib-legacy): the v3 behaviour, every row
    with a reading (stale/frozen held readings included) -- kept so v3 numbers stay reproducible."""
    base = [r for res in results_cal for r in res["e2"]["base"]]
    tray = [r for res in results_cal for r in res["e2"]["tray"]]
    cup = [r for res in results_cal for r in res["e2"]["cup"]]
    per = [r for res in results_cal for r in res["e2"]["person"] if r["matched"]]
    sc = {}
    sc["base_pos"] = P.cq([r["e_xy"] for r in base], ALPHA)                                  # raw radius (reported; NOT used: episode-level bias breaks exchangeability)
    sc["base_k"] = max(2.0, P.cq([r["e_xy"] / max(r["sigma"], 1e-6) for r in base], ALPHA))   # normalised: radius = k * sigma_xy, never below the estimator's own 2 sigma
    sc["base_rot"] = P.cq([r["e_yaw"] for r in base], ALPHA)
    sc["tray_pos"] = P.cq([r["e_rel"] for r in tray], ALPHA)
    sc["tray_rot"] = P.cq([r["e_tilt"] for r in tray], ALPHA)
    # cup: per-side affine correction of the blob-ray estimate in the blob centroid pixel (u, v) (ridge least squares on the calibration seeds), then the
    # residual of the corrected mean
    aff, bias = {}, {}
    for s_ in ("left", "right"):
        fr = [r for r in cup if s_ in _fresh_sides(r, legacy)]
        rows = [(np.array([1.0, r["uv"][s_][0], r["uv"][s_][1]]), (np.array(r["raw"][s_]) - np.array(r["truth_tray"]))[:2]) for r in fr if s_ in r.get("uv", {})]
        if len(rows) >= 30:
            X = np.array([a for a, _ in rows]); Y = np.array([b for _, b in rows])
            lam = 1e-3 * len(X)
            W = np.linalg.solve(X.T @ X + lam * np.diag([0.0, 1.0, 1.0]), X.T @ Y)     # (3, 2)
            aff[s_] = W.T.tolist()
        d = [np.array(r["raw"][s_]) - np.array(r["truth_tray"]) for r in fr]
        bias[s_] = (np.median(d, 0).tolist() if d else [0.0, 0.0, 0.0])
        bias[s_][2] = 0.0
    sc["cup_affine"], sc["cup_bias"] = aff, bias
    vis_e, occ_e = [], []
    for r in cup:
        if not r["raw"]:
            continue
        isvis = r.get("blob_vis", r["visible"])       # the blob's own freshness (another source may keep the cup "visible")
        use = _fresh_sides(r, legacy) if isvis else _held_sides(r, legacy)
        ests = []
        for s_ in use:
            v = r["raw"][s_]
            if s_ in aff and s_ in r.get("uv", {}):
                ests.append(np.array(v) - np.r_[np.array(aff[s_]) @ np.array([1.0, *r["uv"][s_]]), 0.0])
            else:
                ests.append(np.array(v) - np.array(bias.get(s_, [0.0, 0.0, 0.0])))
        if not ests:
            continue
        e = float(np.linalg.norm((np.mean(ests, 0) - np.array(r["truth_tray"]))[:2]))
        (vis_e if isvis else occ_e).append(e)
    sc["cup_pos"] = P.cq(vis_e, ALPHA)
    sc["cup_pos_occ"] = P.cq(occ_e, ALPHA) if len(occ_e) >= 20 else sc["cup_pos"]
    sc["cup_rot"] = P.cq([r["e_tilt_rel"] for r in cup], ALPHA)
    # FoundationPose source (anchored_fp): split-conformal radius / tilt bound of its fresh measurements (tray frame, xy)
    fp_rows = [r for r in cup if r.get("fp") is not None]
    if len(fp_rows) >= 20:
        sc["cup_fp_pos"] = P.cq([float(np.linalg.norm((np.array(r["fp"]) - np.array(r["truth_tray"]))[:2])) for r in fp_rows], ALPHA)
        sc["cup_fp_rot"] = P.cq([r["fp_tilt"] for r in fp_rows], ALPHA)
    # people: split-conformal radius table of people_track (visibility x age x range bins), T = 0 (position error now)
    # people: Mahalanobis (normalised) split-conformal score of the ROBOT-FRAME error against the track covariance: set = ellipse(P, q) (+) disc(floor)
    sc["person_q"] = P.cq([r["mahal"] for r in per if np.isfinite(r["mahal"])], ALPHA)
    sc["person_floor"] = 0.03
    pm = None
    sc["n_cal"] = dict(base=len(base), tray=len(tray), cup=len(cup), person=len(per))
    return P.Calib(sc, pm, ALPHA)


# ====================================================================================== E2 aggregation
def cov_rows(rows, key):
    v = [r[key] for r in rows]
    return mean(v)


def agg_e2(results, cal_seeds, test_seeds, title):
    """results: process() outputs (any pass).  Returns dict kind -> stats for {all, cal seeds, test seeds}."""
    out = {}
    splits = dict(all=None, cal=set(cal_seeds), test=set(test_seeds))
    for sname, sset in splits.items():
        R = [r for r in results if r["scene"] == "h5" and (sset is None or r["seed"] in sset)]
        if not R:
            continue
        def collect(kind):
            return [x for r in R for x in r["e2"][kind]]
        d = {}
        b = collect("base")
        d["base"] = dict(n=len(b), err_xy_med=pct([x["e_xy"] for x in b], 50), err_xy_p95=pct([x["e_xy"] for x in b], 95), err_yaw_med=pct([x["e_yaw"] for x in b], 50),
                         err_yaw_p95=pct([x["e_yaw"] for x in b], 95), cov_pos=cov_rows(b, "cov_pos"), cov_rot=cov_rows(b, "cov_rot"))
        t = collect("tray")
        d["tray"] = dict(n=len(t), err_world_med=pct([x["e_world"] for x in t], 50), err_world_p95=pct([x["e_world"] for x in t], 95), err_fk_rel_med=pct([x["e_rel"] for x in t], 50),
                         err_fk_rel_p95=pct([x["e_rel"] for x in t], 95), tilt_med=pct([x["e_tilt"] for x in t], 50), tilt_p95=pct([x["e_tilt"] for x in t], 95),
                         cov_pos=cov_rows(t, "cov_pos"), cov_pos_rel=cov_rows(t, "cov_pos_rel"), cov_rot=cov_rows(t, "cov_rot"))
        c = collect("cup")
        for nm, sel in (("cup_visible", [x for x in c if x["visible"]]), ("cup_occluded", [x for x in c if not x["visible"]])):
            d[nm] = dict(n=len(sel), err_tray_med=pct([x["e_tray"] for x in sel], 50), err_tray_p95=pct([x["e_tray"] for x in sel], 95),
                         err_world_med=pct([x["e_world"] for x in sel], 50), err_world_p95=pct([x["e_world"] for x in sel], 95), tilt_med=pct([x["e_tilt"] for x in sel], 50),
                         tilt_p95=pct([x["e_tilt"] for x in sel], 95), cov_pos=cov_rows(sel, "cov_pos"), cov_pos_rel=cov_rows(sel, "cov_pos_rel"), cov_rot=cov_rows(sel, "cov_rot"))
        p = collect("person")
        for nm, sel in (("person_los", [x for x in p if x["los"]]), ("person_occluded", [x for x in p if not x["los"]])):
            m = [x for x in sel if x["matched"]]
            d[nm] = dict(n=len(sel), track_recall=len(m) / max(len(sel), 1), err_med=pct([x["e_robot"] for x in m], 50), err_p95=pct([x["e_robot"] for x in m], 95),
                         err_world_p95=pct([x["e"] for x in m], 95), cov_given_track=mean([x["cover_robot"] for x in m]),
                         cov_incl_misses=sum(x["cover_robot"] for x in m) / max(len(sel), 1), cov_world_given_track=mean([x["cover"] for x in m]),
                         infl_med=pct([x["infl"] for x in m], 50))
        for nm, sel in (("person_moving", [x for x in p if x["speed"] > 0.1]), ("person_static", [x for x in p if x["speed"] <= 0.1])):
            m = [x for x in sel if x["matched"]]
            d[nm] = dict(n=len(sel), track_recall=len(m) / max(len(sel), 1), err_med=pct([x["e_robot"] for x in m], 50), err_p95=pct([x["e_robot"] for x in m], 95),
                         cov_given_track=mean([x["cover_robot"] for x in m]), cov_incl_misses=sum(x["cover_robot"] for x in m) / max(len(sel), 1),
                         cov_world_given_track=mean([x["cover"] for x in m]))
        out[sname] = d
    # flagship obstacles
    F = [r for r in results if r["scene"] == "flagship"]
    if F:
        ob = F[0]["e2"]["obstacle"]
        out["flagship_obstacles"] = ob
        out["flagship_robot_side"] = {k: dict(n=len(F[0]["e2"][k]), cov_pos=cov_rows(F[0]["e2"][k], "cov_pos"), cov_rot=cov_rows(F[0]["e2"][k], "cov_rot") if k != "person" else None,
                                              err_med=pct([x.get("e_world", x.get("e_xy")) for x in F[0]["e2"][k]], 50), err_p95=pct([x.get("e_world", x.get("e_xy")) for x in F[0]["e2"][k]], 95))
                                      for k in ("base", "tray", "cup")}
    return out


def flag(v):
    return "**UNDER**" if (v is not None and np.isfinite(v) and v < UNDER) else ""


def md_e2(prior, cal):
    L = ["# E2 estimation + calibration (h5 carry, 6 seeds; calibration seeds 0-2, held-out 3-5)", "",
         "Coverage = fraction of ticks the TRUE pose lies in the reported `unc` set (position zonotope composed along the parent chain; rotation = body-axis tilt bound; base: yaw bound).",
         f"`prior` = engineering k-sigma sets (no calibration); `cal` = split-conformal sets (alpha {ALPHA}) fitted on seeds 0-2.  Flag: held-out coverage < {UNDER}.", ""]
    rows = [("base", "pos", "base: odometry+VIO pose (xy)"), ("tray", "pos", "tray: FK + nominal grasp offset (world)"), ("cup_visible", "pos", "cup: wrist blob, visible"),
            ("cup_occluded", "pos", "cup: held estimate (no blob > 0.15 s)"), ("person_los", "p", "person: truth line-of-sight"), ("person_occluded", "p", "person: truth occluded")]
    L += ["| kind | n (held-out) | err med / p95 [m] (persons: robot frame) | prior cov (held-out) | cal cov (in-sample) | cal cov (held-out) | flag |", "|---|---|---|---|---|---|---|"]
    for key, _, label in rows:
        def g(src, split):
            d = src.get(split, {}).get(key)
            return d
        pt, ct, cc = g(prior, "test"), g(cal, "test"), g(cal, "cal")
        if ct is None:
            continue
        if key.startswith("person"):
            e = f"{ct['err_med']:.3f} / {ct['err_p95']:.3f}"
            f = lambda d: f"{d['cov_incl_misses']:.3f} (track recall {d['track_recall']:.2f}, given track {d['cov_given_track']:.3f})" if d else "-"
            fv = ct["cov_incl_misses"]
            L.append(f"| {label} | {ct['n']} | {e} | {f(pt)} | {f(cc)} | {f(ct)} | {flag(fv)} |")
        else:
            ekey = {"base": ("err_xy_med", "err_xy_p95"), "tray": ("err_world_med", "err_world_p95")}.get(key, ("err_world_med", "err_world_p95"))
            e = f"{ct[ekey[0]]:.3f} / {ct[ekey[1]]:.3f}"
            fm = lambda d: (f"{d['cov_pos']:.3f}" + (f" (robot frame {d['cov_pos_rel']:.3f})" if "cov_pos_rel" in d else "")) if d else "-"
            L.append(f"| {label} | {ct['n']} | {e} | {fm(pt)} | {fm(cc)} | {fm(ct)} | {flag(ct['cov_pos'])} |")
    L += ["", "Rotation / tilt bound coverage (held-out) and tilt error [rad]:", "", "| kind | tilt err med / p95 | prior cov | cal cov (held-out) |", "|---|---|---|---|"]
    for key, label, ek in (("base", "base yaw", ("err_yaw_med", "err_yaw_p95")), ("tray", "tray axis", ("tilt_med", "tilt_p95")), ("cup_visible", "cup axis (world)", ("tilt_med", "tilt_p95"))):
        ct, pt = cal["test"][key], prior["test"][key]
        L.append(f"| {label} | {ct[ek[0]]:.4f} / {ct[ek[1]]:.4f} | {pt['cov_rot']:.3f} | {ct['cov_rot']:.3f} {flag(ct['cov_rot'])} |")
    L += ["", "Persons split by motion (calibrated, held-out):", "", "| group | n | track recall | err med / p95 [m] | cov given track | cov incl. misses |", "|---|---|---|---|---|---|"]
    for key in ("person_moving", "person_static"):
        d = cal["test"][key]
        L.append(f"| {key} | {d['n']} | {d['track_recall']:.2f} | {d['err_med']:.3f} / {d['err_p95']:.3f} | {d['cov_given_track']:.3f} | {d['cov_incl_misses']:.3f} {flag(d['cov_incl_misses'])} |")
    if "flagship_obstacles" in cal:
        L += ["", "Flagship desk stereo zonotopes (obstacle sets; true object boundary samples inside the union of its perceived clusters, inflated by the pipeline's p95 stereo error):", "",
              "| true object | clusters | frac of ALL surface samples inside (inflated) | frac of FRONT-facing samples inside (inflated) | front, nominal (no inflation) | centroid err [m] |", "|---|---|---|---|---|---|"]
        for o in cal["flagship_obstacles"]:
            fr = o.get('frac_front', float('nan'))
            L.append(f"| {o['key']} | {o['n_clusters']} | {o['frac']:.2f} | {fr:.2f} {flag(fr)} | {o.get('frac_front_nominal', float('nan')):.2f} | {o.get('cen_err', float('nan')):.3f} |")
        L += ["", "Flagship robot-side sets (h5 calibration transferred to the office rig):", "", "| kind | n | err med / p95 | cov pos | cov rot |", "|---|---|---|---|---|"]
        for k, d in cal["flagship_robot_side"].items():
            L.append(f"| {k} | {d['n']} | {d['err_med']:.3f} / {d['err_p95']:.3f} | {d['cov_pos']:.3f} {flag(d['cov_pos'])} | {d['cov_rot']:.3f} |")
    return "\n".join(L) + "\n"


# ====================================================================================== E3 aggregation
def agg_e3(resB, cal_seeds, test_seeds):
    recs = collections.defaultdict(list)
    for r in resB:
        if r["scene"] != "h5" or "e3" not in r:
            continue
        for x in r["e3"]:
            recs[(x["kind"], x["src"], x["planned"], x["tau"])].append((r["seed"], x))
    out = []
    for key, rows in sorted(recs.items()):
        kind, src, planned, tau = key
        cal = [x["e"] for s, x in rows if s in cal_seeds]
        test = [(x["e"], x["cov_set"], x["vis"]) for s, x in rows if s in test_seeds]
        if len(cal) < 20 or len(test) < 10:
            continue
        rad = P.cq(cal, ALPHA)
        te = np.array([t[0] for t in test])
        cs = [t[1] for t in test if t[1] is not None]
        ent = dict(kind=kind, src=src, planned=planned, tau=tau, n_cal=len(cal), n_test=len(test), err_med=pct(te, 50), err_p95=pct(te, 95), conf_radius=rad,
                   conf_cov_test=float(np.mean(te <= rad)) if np.isfinite(rad) else float("nan"), set_cov_test=mean(cs) if cs else None)
        if kind.startswith("person"):
            for nm, sel in (("visible", [t for t in test if t[2]]), ("occluded", [t for t in test if not t[2]])):
                if len(sel) >= 10:
                    ent[f"err_p95_{nm}"] = pct([t[0] for t in sel], 95)
                    ent[f"set_cov_{nm}"] = mean([t[1] for t in sel if t[1] is not None])
                    ent[f"conf_cov_{nm}"] = float(np.mean([t[0] <= rad for t in sel]))
                    ent[f"n_{nm}"] = len(sel)
        out.append(ent)
    return out


def md_e3(rows):
    L = ["# E3 prediction (rollouts vs the recorded future truth; per-kind conformal radii fitted on seeds 0-2, coverage on seeds 3-5)", "",
         "`set cov` = fraction of held-out ticks the true future position lies in the rollout's growing analytic set (dynamics.rollout); `conf r` = split-conformal "
         f"radius (alpha {ALPHA}) of the position residual; `conf cov` its held-out coverage.  Flag: any held-out coverage < {UNDER}.", ""]
    for src, planned in (("perc", False), ("perc", True), ("truth", False), ("truth", True)):
        title = {"perc": "OCS(perception)", "truth": "OCS(truth state, dynamics-model error only)"}[src] + (", oracle base plan" if planned else ", constant base twist")
        L += [f"## {title}", "", "| kind | tau [s] | n test | err med / p95 [m] | set cov | conf r [m] | conf cov | flag |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["src"] != src or r["planned"] != planned:
                continue
            sc = r["set_cov_test"]
            fl = flag(r["conf_cov_test"]) or flag(sc if sc is not None else 1.0)
            L.append(f"| {r['kind']} | {r['tau']} | {r['n_test']} | {r['err_med']:.3f} / {r['err_p95']:.3f} | " + ("-" if sc is None else f"{sc:.3f}") +
                     f" | {r['conf_radius']:.3f} | {r['conf_cov_test']:.3f} | {fl} |")
        L.append("")
    occ = [r for r in rows if r["src"] == "perc" and not r["planned"] and "set_cov_occluded" in r]
    if occ:
        L += ["## Persons: visible vs occluded track at the start of the rollout (perception, constant base twist, held-out)", "",
              "| kind | tau | n vis / occ | set cov vis | set cov occ | conf cov vis | conf cov occ | err p95 vis / occ |", "|---|---|---|---|---|---|---|---|"]
        for r in occ:
            L.append(f"| {r['kind']} | {r['tau']} | {r.get('n_visible', 0)} / {r['n_occluded']} | {r.get('set_cov_visible', float('nan')):.3f} | {r['set_cov_occluded']:.3f} {flag(r['set_cov_occluded'])} | "
                     f"{r.get('conf_cov_visible', float('nan')):.3f} | {r['conf_cov_occluded']:.3f} {flag(r['conf_cov_occluded'])} | {r.get('err_p95_visible', float('nan')):.3f} / {r['err_p95_occluded']:.3f} |")
    return "\n".join(L) + "\n"


# ====================================================================================== E4 aggregation
STATES = ("true", "unknown", "false")


def conf_matrix(rows, key="inst"):
    M = {t: {p: 0 for p in STATES} for t in ("true", "false")}
    for r in rows:
        t, p = r["truth"][0], r["perc"][0]
        if t in M:
            M[t][p] += 1
    return M


def false_safe_stats(rows, atoms=True):
    n = len(rows)
    tf = [r for r in rows if r["truth"][0] == "false"]
    fs = [r for r in tf if r["perc"][0] == "true"]
    nom_fs = [r for r in tf if np.isfinite(r["perc"][1]) and r["perc"][1] >= 0]
    d = dict(n=n, n_truth_false=len(tf), n_false_safe=len(fs), false_safe_given_violated=len(fs) / max(len(tf), 1), false_safe_overall=len(fs) / max(n, 1),
             n_nominal_false_safe=len(nom_fs), nominal_false_safe_given_violated=len(nom_fs) / max(len(tf), 1))
    if atoms and fs:
        cnt = collections.Counter()
        for r in fs:
            a = r.get("truth_atoms", {})
            if a:
                cnt[min(a, key=a.get)] += 1
        d["false_safe_by_violated_atom"] = dict(cnt)
    tt = [r for r in rows if r["truth"][0] == "true"]
    d["n_truth_true"] = len(tt)
    d["unknown_given_true"] = sum(r["perc"][0] == "unknown" for r in tt) / max(len(tt), 1)
    d["false_alarm_given_true"] = sum(r["perc"][0] == "false" for r in tt) / max(len(tt), 1)
    return d


ATOMS = ("inside", "upright", "clear_laptop", "clear_person")


def combine(stats):
    if any(x == "false" for x in stats):
        return "false"
    return "true" if all(x == "true" for x in stats) else "unknown"


def atom_stats(rows):
    """per-atom confusion (truth true/false vs perception true/unknown/false) and false-safe; plus the conjunction WITHOUT the inside atom."""
    out = {}
    def status(r, src, atom):
        d = r.get(src + "_astat", {})
        return d.get(atom, "unknown")
    for name, atoms in [(a, (a,)) for a in ATOMS] + [("all_but_inside", ("upright", "clear_laptop", "clear_person"))]:
        M = {t: {p: 0 for p in STATES} for t in ("true", "false")}
        for r in rows:
            t = combine([status(r, "truth", a) for a in atoms])
            p = combine([status(r, "perc", a) for a in atoms])
            if t in M:
                M[t][p] += 1
        nf = sum(M["false"].values())
        out[name] = dict(matrix=M, n_truth_false=nf, n_false_safe=M["false"]["true"], false_safe_given_violated=M["false"]["true"] / max(nf, 1),
                         n_truth_true=sum(M["true"].values()), certified_given_true=M["true"]["true"] / max(sum(M["true"].values()), 1))
    return out


def agg_e4(results):
    out = {}
    groups = collections.defaultdict(list)
    for r in results:
        if "e4" not in r:
            continue
        g = "flagship_desk" if r["scene"] == "flagship" else "h5_carry_all"
        groups[g].append(r)
        if r["scene"] == "h5":
            groups["h5_carry_seeds_" + ("cal" if r["seed"] < 3 else "test")].append(r)
    for g, R in groups.items():
        inst = [dict(x, seed=r["seed"]) for r in R for x in r["e4"]["inst"]]
        pred = [dict(x, seed=r["seed"]) for r in R for x in r["e4"]["pred"]]
        out[g] = dict(mapping=R[0]["e4"]["mapping"], instant=dict(matrix=conf_matrix(inst), **false_safe_stats(inst), atoms=atom_stats(inst)),
                      predictive_G1s=dict(matrix=conf_matrix(pred), **false_safe_stats(pred, atoms=False), atoms=atom_stats(pred)))
        out[g]["per_seed"] = {}
        for r in R:
            ii = r["e4"]["inst"]
            out[g]["per_seed"][r["seed"]] = dict(n=len(ii), **{k: v for k, v in false_safe_stats(ii, atoms=False).items() if k in ("n_truth_false", "n_false_safe", "false_safe_given_violated")})
    return out


def md_e4(priors, cal):
    L = ["# E4 constraint status: G(inside(cup,tray_top,0.02) AND upright(cup,0.1) AND clear(cup,laptop,0.1) AND clear(cup,person,0.5))", "",
         "3-valued IR status per tick on OCS(perception) vs OCS(truth).  Truth OCS has exact sets, so its status is true / false.  "
         "**FALSE-SAFE** = perception says `true` while truth says `false` (a violated constraint certified as satisfied).", ""]
    for g in sorted(cal):
        d = cal[g]
        L += [f"## {g}", "", f"Entity mapping (substitutions): {json.dumps(d['mapping'])}", ""]
        for part, nm in (("instant", "instantaneous (per tick)"), ("predictive_G1s", "predictive G[0, 1 s] (perception rollout vs the recorded future truth)")):
            s = d[part]
            M = s["matrix"]
            L += [f"### {nm}", "", "| truth \\ perception | true | unknown | false |", "|---|---|---|---|"]
            for t in ("true", "false"):
                L.append(f"| {t} | {M[t]['true']} | {M[t]['unknown']} | {M[t]['false']} |")
            L += ["", f"- ticks {s['n']}, truth violated {s['n_truth_false']}, **false-safe {s['n_false_safe']}** = {100 * s['false_safe_given_violated']:.2f}% of violated ticks "
                  f"({100 * s['false_safe_overall']:.2f}% of all ticks)",
                  f"- nominal-margin (sets ignored) would certify {s['n_nominal_false_safe']} violated ticks = {100 * s['nominal_false_safe_given_violated']:.2f}% of violated",
                  f"- truth satisfied: perception says unknown {100 * s['unknown_given_true']:.1f}%, false (false alarm) {100 * s['false_alarm_given_true']:.1f}%"]
            if "false_safe_by_violated_atom" in s:
                L.append(f"- false-safe ticks by the violated atom: {s['false_safe_by_violated_atom']}")
            L += ["", "Per atom (and the conjunction without `inside`):", "", "| atom | truth violated | false-safe (perc `true`) | truth satisfied | perc certifies `true` | confusion truth=true [t/u/f] | truth=false [t/u/f] |",
                  "|---|---|---|---|---|---|---|"]
            for a, v in s["atoms"].items():
                M = v["matrix"]
                L.append(f"| {a} | {v['n_truth_false']} | **{v['n_false_safe']}** ({100 * v['false_safe_given_violated']:.2f}%) | {v['n_truth_true']} | {100 * v['certified_given_true']:.1f}% | "
                         f"{M['true']['true']}/{M['true']['unknown']}/{M['true']['false']} | {M['false']['true']}/{M['false']['unknown']}/{M['false']['false']} |")
            L.append("")
        if g in priors:
            p = priors[g]["instant"]
            L.append(f"(prior, uncalibrated sets: false-safe {p['n_false_safe']} / {p['n_truth_false']} violated = {100 * p['false_safe_given_violated']:.2f}%, "
                     f"unknown given true {100 * p['unknown_given_true']:.1f}%)")
            L.append("")
    return "\n".join(L) + "\n"


# ====================================================================================== driver
def run(cal_seeds, test_seeds, procs, cup_tracker="wrist", pred_contingency="hold"):
    import multiprocessing as mp
    seeds = list(cal_seeds) + list(test_seeds)
    paths = [str(rec_path(s)) for s in seeds]
    fpath = str(REC / "flagship_desk.pkl.gz")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(procs, len(paths) + 1), maxtasksperchild=1) as pool:
        resA = pool.map(EC.process_job, [(p, None, ("e2", "e4"), 5, "A", cup_tracker, pred_contingency) for p in paths] + [(fpath, None, ("e2", "e4"), 5, "A", cup_tracker, pred_contingency)])
    print(f"[pass A] {time.time() - t0:.0f}s", flush=True)
    calib = fit_calib([r for r in resA if r["scene"] == "h5" and r["seed"] in cal_seeds], legacy=CALIB_LEGACY)
    calib.save(OUT / "calib_h5.json")
    cj = calib.to_json()
    t0 = time.time()
    with ctx.Pool(min(procs, len(paths) + 1), maxtasksperchild=1) as pool:
        resB = pool.map(EC.process_job, [(p, cj, ("e2", "e3", "e4"), 5, "B", cup_tracker, pred_contingency) for p in paths] + [(fpath, cj, ("e2", "e4"), 5, "B", cup_tracker, pred_contingency)])
    print(f"[pass B] {time.time() - t0:.0f}s", flush=True)
    import pickle
    with open(OUT / "e234_raw.pkl", "wb") as f:
        pickle.dump(dict(A=resA, B=resB, calib=cj), f, protocol=4)
    return resA, resB, calib


def report(resA, resB, calib, cal_seeds, test_seeds):
    prior = agg_e2(resA, cal_seeds, test_seeds, "prior")
    cal = agg_e2(resB, cal_seeds, test_seeds, "cal")
    (OUT / "e2_calibration.json").write_text(json.dumps(dict(calib=calib.to_json() if hasattr(calib, "to_json") else calib, prior=prior, calibrated=cal,
                                                              cal_seeds=list(cal_seeds), test_seeds=list(test_seeds), alpha=ALPHA), indent=1, default=str))
    (OUT / "e2_calibration.md").write_text(md_e2(prior, cal))
    e3 = agg_e3(resB, set(cal_seeds), set(test_seeds))
    (OUT / "e3_prediction.json").write_text(json.dumps(e3, indent=1, default=str))
    (OUT / "e3_prediction.md").write_text(md_e3(e3))
    e4p, e4c = agg_e4(resA), agg_e4(resB)
    (OUT / "e4_constraints.json").write_text(json.dumps(dict(calibrated=e4c, prior=e4p), indent=1, default=str))
    (OUT / "e4_constraints.md").write_text(md_e4(e4p, e4c))
    print((OUT / "e2_calibration.md").read_text())
    print((OUT / "e4_constraints.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "all"], default="all", nargs="?")
    ap.add_argument("--cal", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--test", type=int, nargs="*", default=[3, 4, 5])
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--out", default=None, help="results directory (default $OCS_OUT or the molmospaces/ocs dir); created if missing")
    ap.add_argument("--rec", default=None, help="recordings directory (default <default OUT>/rec, i.e. NOT redirected by --out)")
    ap.add_argument("--cup-tracker", choices=["wrist", "anchored", "anchored_fp"], default="wrist",
                    help="cup producer: 'wrist' = CupWristProducer (frozen while occluded), 'anchored' = ocs.anchored SE(3)-anchored tracker")
    ap.add_argument("--calib-legacy", action="store_true", help="cup blob calibration as in v3 (fit on every row with a reading, stale/frozen included); default = fresh rows only")
    ap.add_argument("--fp-dir", default=None, help="anchored_fp: directory of fp_s<seed>.npz (default $OCS_FP_DIR or molmospaces/ocs/fp)")
    ap.add_argument("--pred-contingency", choices=["hold", "hold+brake"], default="hold",
                    help="E4 predictive rollout contingency set: 'hold' = operator holds the current twist (default, v3 behaviour); 'hold+brake' = also the shield brake from now "
                         "(base slews to 0 at the hardware limit); perceived pred status = conservative combination (true only if true under all, false only if false under all)")
    ap.add_argument("--flagship-cup", choices=["flagship", "sim"], default="flagship",
                    help="cup shape of the flagship_desk recording: 'flagship' = flagship.perception paper-cup model (0.045/0.115), 'sim' = the MJCF cup the sim contains")
    a = ap.parse_args()
    global OUT, REC, CALIB_LEGACY
    CALIB_LEGACY = bool(a.calib_legacy)
    if a.out:
        OUT = Path(a.out)
        OUT.mkdir(parents=True, exist_ok=True)
    if a.rec:
        REC = Path(a.rec)
    os.environ["OCS_FLAGSHIP_CUP"] = a.flagship_cup           # inherited by the spawned workers
    if a.fp_dir:
        os.environ["OCS_FP_DIR"] = a.fp_dir                     # anchored_fp: where fp_pass.py wrote fp_s<seed>.npz (inherited by the workers)
    if a.what in ("run", "all"):
        resA, resB, calib = run(a.cal, a.test, a.procs, a.cup_tracker, a.pred_contingency)
    if a.what == "report":
        import pickle
        d = pickle.load(open(OUT / "e234_raw.pkl", "rb"))
        resA, resB, calib = d["A"], d["B"], P.Calib.from_json(d["calib"])
    if a.what in ("report", "all"):
        report(resA, resB, calib, a.cal, a.test)


if __name__ == "__main__":
    main()
