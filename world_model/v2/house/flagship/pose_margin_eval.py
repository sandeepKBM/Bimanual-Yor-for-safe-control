"""flagship/pose_margin_eval.py -- clearance margins of the stereo zonotopes vs TRUTH when the cloud is placed with the ESTIMATED base pose.  SIMULATION ONLY.

Cached desk perception (perception_desk.pkl, built with the TRUE camera pose) + a pose-estimate REPLAY: the camera-frame cloud is identical, so the zonotopes
built with the estimated pose are the true-pose ones moved by the planar (est - true) correction (flagship.perception.apply_pose_estimate).  Pose-error
samples come from the house/loc PoseEstimator driven along a path ending at the desk-capture pose (default LocCfg); the uncertainty fed to the inflation is
what the estimator REPORTS (sigma_xy proxy, yaw proxy), never the sample's true error.

Conditions (per pose-error sample):
  true      zonotopes as built (true pose; the privileged reference)
  est       estimated-pose frame, NO extra inflation        (the naive fix: right frame, ignores the pose error)
  est+loc_abs  estimated-pose frame + loc_delta(range, sigma_xy, sigma_yaw) with the localizer's reported ABSOLUTE uncertainty (needed by consumers that
               combine the zonotopes with map-frame constraints: walls, other captures)
  est+loc_rel  same but with only the drift accumulated between capture and use (plan length ~1 m): valid when the plan is purely relative inside the
               capture's estimated frame (common-mode absolute error cancels)
Metrics: (1) containment of the GROUND-TRUTH geometry of each perceived object (GT collision-geom sample points, mapped into the estimated frame) in the
inflated zonotope union: fraction of (sample, object) pairs fully contained, worst outside depth; (2) the desk PLAN made in that frame (zed_demo.plan.plan_task):
n_feasible, planned (claimed) min path clearance vs the TRUE min clearance of the executed path against GT-hull obstacles (displacement rotated back by the
heading error), and the number of samples whose true path clearance <= 0 (contact).

  python -m world_model.v2.house.flagship.pose_margin_eval --n 40 --path-m 12 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

from . import perception as FP
from . import scene as FS


def pose_error_samples(pose_true, n, path_m=12.0, speed=0.4, dt=0.02, seed0=0):
    """run the loc PoseEstimator over a path of path_m metres (straight legs with two 90-degree turns, ending at pose_true) -> list of (pose_est, (sigma_xy, sigma_yaw))."""
    from world_model.v2.house.loc.estimator import PoseEstimator, LocCfg
    out = []
    leg = path_m / 3.0
    xs = []
    p = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        for _ in range(int(leg / speed / dt)):
            p = p + np.array([speed * dt * math.cos(p[2]), speed * dt * math.sin(p[2]), 0.0])
            xs.append(p.copy())
        for _ in range(int(0.5 * math.pi / 0.8 / dt)):
            p = p + np.array([0, 0, 0.8 * dt]); xs.append(p.copy())
    xs = np.array(xs)
    xs[:, 2] = (xs[:, 2] + math.pi) % (2 * math.pi) - math.pi
    for sd in range(n):
        est = PoseEstimator(LocCfg(), seed=seed0 + sd, true_pose0=xs[0])
        for k, q in enumerate(xs[1:], 1):
            est.update(k * dt, q, dt)
        e = est.pose - xs[-1]
        e[2] = (e[2] + math.pi) % (2 * math.pi) - math.pi
        pe = np.asarray(pose_true, float) + e
        out.append((pe, FP.loc_sigmas(est), float(np.hypot(e[0], e[1])), float(abs(e[2]))))
    return out


class _TwinLite(FP.Twin):
    """Twin without the stereo rig (no EGL): enough for payload zonotopes + the desk planner."""

    def __init__(self):
        import mujoco
        from world_model.v2.house import phase1 as P1
        from world_model.v2.house.zed_demo import perceive as PC, plan as PL
        self.mujoco, self.PC, self.PL = mujoco, PC, PL
        self.C = FS.build_twin(30.0)
        self.m, self.d = self.C.m, self.C.d
        self.cls = P1.geom_class(self.m)
        self.keys = PC.gt_keys(self.m)
        self.base0 = np.array(self.C.info["spawn"], float)
        self.yaw0 = float(self.C.info["yaw"])
        self.tray_id = self.m.body("tray").id
        self.payload = self._payload()


def gt_points(tw, keys, roi_pose=None):
    PC = tw.PC
    out = {}
    for key in keys:
        gs = [g for g in range(tw.m.ngeom) if tw.keys[g] == key and (tw.m.geom_contype[g] or tw.m.geom_conaffinity[g]) and tw.m.geom_group[g] in (3, 4)]
        if gs:
            X = np.concatenate([PC.geom_pts(tw.m, tw.d, g) for g in gs])
            X = X[X[:, 2] <= FP.ZMAX]
            if roi_pose is not None:                                  # the pipeline's own ROI (robot frame): only surfaces it could have seen
                c, s_ = math.cos(roi_pose[2]), math.sin(roi_pose[2])
                Pr = (X[:, :2] - roi_pose[:2]) @ np.array([[c, -s_], [s_, c]])
                X = X[(Pr[:, 0] > 0.05) & (Pr[:, 0] < 1.47) & (Pr[:, 1] > -1.30) & (Pr[:, 1] < 0.80)]
            if len(X):
                out[key] = X
    return out


def unpack_res(r):
    zs = []
    for z in r["zonos"]:
        zz = dict(z)
        zz["Z"], zz["Zi"] = FP.zono_unpack(z["Z"]), FP.zono_unpack(z["Zi"])
        zs.append(zz)
    return dict(zonos=zs)


def containment(res_est, gt, bt, be, dy):
    from world_model.v2.perception_zono.zonoutil import ZUnion
    ok, worst = [], []
    for key, X in gt.items():
        zz = [z for z in res_est["zonos"] if z["kind"] == "obj" and z["gt_key"] == key]
        if not zz:
            continue
        Xe = FP.xform_points(X, bt, be, dy)
        U = ZUnion([z["Zi"] for z in zz])
        out_d = U.outside_depth(Xe)
        ok.append(float((out_d <= 1e-9).all())); worst.append(float(out_d.max()))
    return ok, worst


def true_path_clearance(tw, gt_hull, path_t, bt, dy):
    """min exact clearance of the TRUE payload (true base, build yaw) moved by the executed displacement (est-frame displacement rotated back by dy)."""
    pay = tw.payload_at(bt)
    c, s = math.cos(-dy), math.sin(-dy)
    ds = path_t[::max(len(path_t) // 25, 1)]
    best = 1e9
    for t in ds:
        tt = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]])
        best = min(best, tw.PL.exact_clearance(pay, gt_hull, tt)[0])
    return float(best)


def coverage(pose_true, n, k=2.0, path_m=12.0, ranges=(0.8, 1.2, 1.6)):
    """does the reported-uncertainty inflation cover the pose error?  Required inflation at range r = |e_xy| + r |e_yaw| (worst point displacement of a body
    at distance r from the base); provided = loc_delta(r, sigma_xy, sigma_yaw, k).  ABS: error vs truth; REL: error accumulated over the last ~0.8 m / 2 s
    (what matters between capture and use when the plan stays inside the capture's frame; provided = loc_delta with the 1 m drift sigmas)."""
    from world_model.v2.house.loc.estimator import PoseEstimator, LocCfg
    smp = pose_error_samples(pose_true, n, path_m)
    out = {}
    for r in ranges:
        a = [FP.loc_delta(r, sg[0], sg[1], k) >= e + r * y for _, sg, e, y in smp]
        out[f"abs_cover_r{r}"] = float(np.mean(a))
    out["abs_err_xy_mean_m"] = float(np.mean([s[2] for s in smp])); out["abs_err_xy_p95_m"] = float(np.percentile([s[2] for s in smp], 95))
    out["abs_err_xy_max_m"] = float(np.max([s[2] for s in smp])); out["sigma_reported_xy"] = smp[0][1][0]
    out["abs_frac_err_gt_sigma_xy"] = float(np.mean([s[2] > s[1][0] for s in smp]))
    out["abs_frac_err_gt_2sigma_xy"] = float(np.mean([s[2] > 2 * s[1][0] for s in smp]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-n", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--path-m", type=float, default=12.0)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--cache", default=str(FS.OUT / "perception" / "perception_desk.pkl"))
    ap.add_argument("--out", default="/data/local/ss5772/molmospaces/percept/people2/zono_pose_margin.json")
    a = ap.parse_args()
    r = pickle.load(open(a.cache, "rb"))
    pose = np.asarray(r["pose"], float)
    if a.coverage_n:
        cv = coverage(pose, a.coverage_n, a.k, a.path_m)
        print("COVERAGE", json.dumps(cv))
        json.dump(cv, open(a.out.replace(".json", "_coverage.json"), "w"), indent=1)
        return
    tw = _TwinLite()
    from world_model.v2.house.office import compose as CP
    CP.set_base_pose(tw.C, float(pose[0]), float(pose[1]), float(pose[2]))
    tw.mujoco.mj_forward(tw.m, tw.d)
    from world_model.v2.perception_zono.fit import fit_hull_zono
    gt = gt_points(tw, sorted({z["gt_key"] for z in r["zonos"] if z["kind"] == "obj"}), roi_pose=pose)
    gt_hull = [(f"gt:{k}", fit_hull_zono(X, order=3)) for k, X in gt.items()]
    samples = pose_error_samples(pose, a.n, a.path_m)
    print(f"{len(samples)} pose-error samples: pos err mean {np.mean([s[2] for s in samples]):.3f} m (max {np.max([s[2] for s in samples]):.3f}), "
          f"yaw err mean {np.degrees(np.mean([s[3] for s in samples])):.2f} deg; reported sigma_xy {samples[0][1][0]:.3f} sigma_yaw {samples[0][1][1]:.4f}", flush=True)
    res0 = unpack_res(r)
    summ = {c: dict(contained=[], worst=[], n_feas=[], claim=[], true=[], infl=[]) for c in ("true", "est", "est+loc_abs", "est+loc_rel")}
    tray0 = np.asarray(r["tray0"], float)
    bt = pose[:2]
    for i, (pe, sg, _e, _y) in enumerate(samples):
        for cond in ("true", "est", "est+loc_abs", "est+loc_rel"):
            if cond == "true":
                pest, sig = pose, None
            else:
                # abs: the localizer's reported absolute uncertainty; rel: only the drift accumulated between capture and use (plan length ~1 m)
                pest, sig = pe, (sg if cond == "est+loc_abs" else (0.01 + 0.02 * math.sqrt(1.0), 0.005 + 0.003 * math.sqrt(1.0)) if cond == "est+loc_rel" else None)
            rp = FP.apply_pose_estimate(r, pose, pest, sig, a.k) if cond != "true" else r
            res = unpack_res(rp)
            _, be, dy = FP.pose_correction(pose, pest)
            ok, worst = containment(res, gt, bt, be, dy)
            summ[cond]["contained"] += ok
            summ[cond]["worst"] += worst
            summ[cond]["infl"].append(float(np.mean([z.get("delta_loc", 0.0) for z in rp["zonos"]])))
            # desk plan in that frame
            sup = [z for z in res["zonos"] if z["kind"] == "support"]
            corr = None if cond == "true" else (bt, be, dy)
            plan = tw._plan_desk(res, sup, tray0, corr)
            summ[cond]["n_feas"].append(int(plan.get("n_feasible", 0)) if plan.get("ok") else 0)
            if plan.get("ok"):
                summ[cond]["claim"].append(float(plan["path_min_exact_clear_m"]))
                summ[cond]["true"].append(true_path_clearance(tw, gt_hull, np.asarray(plan["path_t"]), bt, dy if corr else 0.0))
        if i % 5 == 0:
            print("sample", i, flush=True)
    out = {}
    for c, v in summ.items():
        t = np.array(v["true"]) if v["true"] else np.array([np.nan])
        out[c] = dict(n_pairs=len(v["contained"]), contained_frac=float(np.mean(v["contained"])), worst_out_mean_m=float(np.mean(v["worst"])),
                      worst_out_p95_m=float(np.percentile(v["worst"], 95)), worst_out_max_m=float(np.max(v["worst"])),
                      mean_loc_inflation_m=float(np.mean(v["infl"])), plans_ok=len(v["true"]), n_plans=a.n if c != "true" else a.n,
                      claimed_clear_mean_m=float(np.mean(v["claim"])) if v["claim"] else None,
                      true_clear_mean_m=float(np.nanmean(t)), true_clear_min_m=float(np.nanmin(t)), true_clear_p5_m=float(np.nanpercentile(t, 5)),
                      n_true_contact=int((t <= 0.0).sum()), n_claim_gt_true_5mm=int(sum(1 for cl, tr in zip(v["claim"], v["true"]) if cl - tr > 0.005)))
    out["_samples"] = dict(n=len(samples), pos_err_mean_m=float(np.mean([s[2] for s in samples])), pos_err_max_m=float(np.max([s[2] for s in samples])),
                           yaw_err_mean_deg=float(np.degrees(np.mean([s[3] for s in samples]))), sigma_reported=samples[0][1], path_m=a.path_m, k=a.k,
                           n_gt_objects=len(gt))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    for c, v in out.items():
        print(c, json.dumps(v, default=float))


if __name__ == "__main__":
    main()
