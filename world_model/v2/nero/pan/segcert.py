"""pan/segcert.py -- segment-wise admission on the covered pan (user rule, 10-29).  SIMULATION ONLY, CPU only.

    python -m world_model.v2.nero.pan.segcert library --out DIR --set full|overload|iid --procs P [--sparse]
    python -m world_model.v2.nero.pan.segcert falsify --out DIR --placement P --pop 8 --gens 6 [--x0-from DIR]
    python -m world_model.v2.nero.pan.segcert recheck --out DIR --finds DIR ...      (dense check of falsifier finds)
    python -m world_model.v2.nero.pan.segcert summary --out OUT.md DIR ...

library: one record per world theta (segments.certify_world): the robot's policy segment by segment, every segment's backup run
         from branch points inside it (every 0.1 s, 0.5 s in holds, plus contact events) and from its end, plus the boundary ROA check.
falsify: CMA-ES over (dx, dy, yaw, mu, mass) for one placement. Objective = the smallest normalised margin to U or to R(Theta) over
         the segment and its backups (sparse: branch every 0.3 s, no ROA kicks). Values < 0 are counterexamples. The search is
         adversarial and its samples are not i.i.d., so they are NOT counted in the failure-rate bound.
summary: per segment: admitted (every library theta in the box passes), success rate, failures ending in R(Theta) via the backup.
         Guarantee type: `sampled`. Failure-rate bound from the i.i.d. samples only (Clopper-Pearson, 95 %).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PLAN = dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
BOX = dict(dx=(-0.015, 0.015), dy=(-0.015, 0.015), yaw_deg=(-8.0, 8.0), mu_scale=(0.6, 1.2), hidden_mass=(0.0, 0.4))  # ASSUMED prior
PL = ["none", "near_handle", "far_centre", "far_left", "far_right"]


def _pin():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"


def worlds(kind):
    """(tag, world) lists.  full = the 62 same worlds as results/pan_fair_box_2026-10-29 (32 vertices + 30 i.i.d. interior);
    overload = 0.8 kg far-centre puck (outside the 0-0.4 kg prior): 16 corners of (dx, dy, yaw, mu) + 10 interior;
    iid = 70 more i.i.d. draws from the prior box (seed 29), for the failure-rate bound together with the 30 interior of `full`."""
    from .vizcoast_faithful.run_baseline import worlds as W
    from .observe_eval import _random_thetas
    import itertools
    if kind == "full":
        return W("full", 0)
    if kind == "iid":
        return [(f"R{i:02d}", dict(placement=t["placement"], hidden_mass=t["hidden_mass"] or None,
                                   perturb={k: t[k] for k in ("dx", "dy", "yaw_deg", "mu_scale")})) for i, t in enumerate(_random_thetas(70, 29))]
    if kind == "overload":
        out = []
        for i, (sx, sy, sw, mu) in enumerate(itertools.product((-1, 1), (-1, 1), (-1, 1), (0.6, 1.2))):
            out.append((f"O{i:02d}", dict(placement="overload", hidden_mass=0.8,
                                          perturb=dict(dx=0.015 * sx, dy=0.015 * sy, yaw_deg=8.0 * sw, mu_scale=mu))))
        rng = np.random.default_rng(31)
        for i in range(10):
            out.append((f"OI{i:02d}", dict(placement="overload", hidden_mass=0.8, perturb=dict(
                dx=float(rng.uniform(-.015, .015)), dy=float(rng.uniform(-.015, .015)), yaw_deg=float(rng.uniform(-8, 8)),
                mu_scale=float(rng.uniform(0.6, 1.2))))))
        return out
    raise ValueError(kind)


def _cert(job):
    _pin()
    from . import segments as G
    tag, w, plan, dense = job
    t0 = time.time()
    try:
        r = G.certify_world(w, plan, observe="zed", obs_seed=G.world_seed(tag), dense=dense, roa=dense, reach=True,
                            branch_dt=G.BRANCH_DT if dense else 0.3)
    except Exception:
        r = dict(world=w, plan=plan, error=traceback.format_exc()[-1500:])
    r.update(tag=tag, wall_s=round(time.time() - t0, 1), dense=dense)
    return r


def _margin(r):
    """falsifier objective: smallest margin over everything the certificate checks (refusal: R margin of the untouched pan)."""
    if r.get("error"):
        return -2.0
    if r.get("refused"):          # the robot commits nothing: not a counterexample to the rule (refusals are counted separately)
        return 1.0
    ms = [s["margin"] for s in r["segments"]]
    if r.get("B0", {}).get("roa") and not r["B0"]["roa"]["ok"]:
        ms.append(-0.5)
    return float(min(ms))


def library(a):
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    plan = dict(PLAN, **json.loads(a.plan)) if a.plan else PLAN
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [(t, w, plan, not a.sparse) for t, w in worlds(a.set) if t not in done]
    print(f"library {a.set}: {len(jobs)} worlds, plan {plan}, dense={not a.sparse}", flush=True)
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_cert, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            segs = [(s["seg"], s["nominal"].get("success"), s["admissible_here"]) for s in r.get("segments", [])]
            print(r["tag"], r.get("refused") or "", segs, "margin", round(_margin(r), 3), r["wall_s"], "s", flush=True)
    (out / f"_DONE_{a.set}").write_text("ok")


# ---------------- falsifier: CMA-ES (Hansen's (mu/mu_w, lambda) update, numpy only) ----------------
class CMA:
    def __init__(self, x0, sigma, lam, seed=0):
        n = self.n = len(x0)
        self.m, self.sigma, self.lam = np.asarray(x0, float), float(sigma), lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.g = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * (z * self.D) @ self.B.T

    def tell(self, X, f):
        X = np.asarray(X)[np.argsort(f)[:self.mu]]
        old = self.m
        self.m = self.w @ X
        y = (self.m - old) / self.sigma
        Cis = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cis @ y
        hs = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.g + 1))) / self.chiN < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        Y = (X - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(D2, 1e-20))
        self.g += 1


DIMS = ("dx", "dy", "yaw_deg", "mu_scale", "hidden_mass")


def _to_world(z, placement):
    """z in [-1, 1]^5 (clipped) -> world in the prior box."""
    z = np.clip(z, -1, 1)
    v = {k: BOX[k][0] + (zi + 1) / 2 * (BOX[k][1] - BOX[k][0]) for k, zi in zip(DIMS, z)}
    return dict(placement=placement, hidden_mass=None if placement == "none" else float(v["hidden_mass"]),
                perturb={k: float(v[k]) for k in DIMS[:4]})


def _from_world(w):
    pr = w["perturb"]
    vals = [pr["dx"], pr["dy"], pr["yaw_deg"], pr["mu_scale"], w.get("hidden_mass") or 0.0]
    return np.array([2 * (x - BOX[k][0]) / (BOX[k][1] - BOX[k][0]) - 1 for k, x in zip(DIMS, vals)])


def falsify(a):
    out = Path(a.out) / f"falsify_{a.placement}"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    plan = dict(PLAN, **json.loads(a.plan)) if a.plan else PLAN
    x0, src = np.zeros(5), "box centre"
    if a.x0_from:                                     # start at the library world of this placement with the smallest margin
        best = None
        for d in a.x0_from:
            for p in Path(d, "raw").glob("*.json"):
                r = json.loads(p.read_text())
                if r["world"]["placement"] == a.placement and not r.get("error"):
                    if best is None or _margin(r) < best[0]:
                        best = (_margin(r), r["world"], r["tag"])
        if best:
            x0, src = _from_world(best[1]), f"library {best[2]} (margin {best[0]:.3f})"
    es = CMA(x0, a.sigma, a.pop, seed=a.seed)
    hist = []
    print(f"falsify {a.placement}: x0 = {src}", flush=True)
    with get_context("spawn").Pool(a.pop, maxtasksperchild=1) as pool:
        for g in range(a.gens):
            X = es.ask()
            jobs = [(f"F_{a.placement}_g{g}_{i}", _to_world(x, a.placement), plan, False) for i, x in enumerate(X)]
            res = pool.map(_cert, jobs)
            f = [_margin(r) for r in res]
            for r, fi in zip(res, f):
                r["objective"] = fi
                (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            es.tell(X, f)
            hist.append(dict(gen=g, f_min=float(min(f)), f_med=float(np.median(f)), sigma=float(es.sigma),
                             best=jobs[int(np.argmin(f))][1]))
            print(json.dumps(hist[-1]), flush=True)
    (out / "history.json").write_text(json.dumps(dict(placement=a.placement, x0=src, pop=a.pop, gens=a.gens, sigma0=a.sigma,
                                                      plan=plan, hist=hist), indent=1))


def recheck(a):
    """dense certificate of every falsifier sample with objective below --thr (the counterexample candidates)."""
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in a.finds:
        for p in sorted(Path(d).glob("falsify_*/raw/*.json")):
            r = json.loads(p.read_text())
            if r.get("objective", 9) < a.thr:
                jobs.append((r["tag"].replace("F_", "D_"), r["world"], r["plan"], True))
    jobs.sort(key=lambda j: j[0])
    jobs = jobs[:a.max]
    print(f"recheck {len(jobs)} candidates", flush=True)
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_cert, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            print(r["tag"], "margin", round(_margin(r), 3), flush=True)


def cp_upper(k, n, conf=0.95):
    """one-sided Clopper-Pearson upper bound on a failure rate after k failures in n i.i.d. trials."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def load(dirs):
    rs = []
    for d in dirs:
        for p in sorted(Path(d, "raw").glob("*.json")):
            rs.append(json.loads(p.read_text()))
    return rs


def seg_table(rs):
    """per segment over the records rs (each = the robot's policy in one true world).  'reached' = the robot got to the start
    boundary with every earlier observable boundary check passing; failure = not successful, aborted by the monitor, or the
    boundary check failed; a failure 'ends in R' when the backup from the segment's end (or abort point) is safe and lands in R."""
    from .segments import SEGMENTS
    rows = []
    for k, seg in enumerate(SEGMENTS):
        reached = [(r, r["segments"][k]) for r in rs if not r.get("error") and not r.get("refused") and len(r["segments"]) > k]
        adm = [s["admissible_here"] for _, s in reached]
        succ = [bool(s["nominal"].get("success")) for _, s in reached]
        fail = [(r, s) for r, s in reached if not (s["nominal"].get("success") and s["boundary_ok"])]
        silent = [(r, s) for r, s in reached if not s["nominal"].get("success") and s["boundary_ok"]]
        endR = [s["backups"][-1]["safe"] and s["backups"][-1]["inR"] for _, s in fail]
        cex = [dict(tag=r["tag"], **(s["counterexample"] or {})) for r, s in reached if s["counterexample"]]
        rows.append(dict(seg=seg, reached=len(reached), admissible_here=sum(adm), success=sum(succ), failures=len(fail),
                         failures_end_in_R=sum(endR), silent_failures=[r["tag"] for r, _ in silent],
                         nominal_unsafe=sum(s["nominal"]["viol"] is not None for _, s in reached),
                         backups_run=sum(s["n_backups"] for _, s in reached),
                         backups_failed=sum(not (b["safe"] and b["inR"]) for _, s in reached for b in s["backups"]),
                         roa_fail=sum(bool(s["roa"] and not s["roa"]["ok"]) for _, s in reached), counterexamples=cex))
    ref = [r for r in rs if r.get("refused")]
    return rows, dict(n=len(rs), errors=[r["tag"] for r in rs if r.get("error")], refused=len(ref),
                      refused_in_R=sum(bool(r["B0"]["inR"]) for r in ref), refused_tags=[(r["tag"], r["refused"][:60]) for r in ref],
                      task_success=sum(bool(len(r.get("segments", [])) == 4 and r["segments"][-1]["nominal"].get("success")) for r in rs))


def summary(a):
    lines = ["# Segment-wise admission, covered pan (`pan/segcert.py`)", "",
             f"Plan {json.dumps(PLAN)}; observe_pose built (zed noise) in the loop; guarantee type: **sampled**.", ""]
    allrows = {}
    for spec in a.dirs:
        d, _, label = spec.partition(":")
        rs = load([d])
        rows, tot = seg_table(rs)
        allrows[label or d] = (rows, tot)
        lines += [f"## {label or d}  (n = {tot['n']} worlds)", "",
                  f"Refused at B0 (no observe fit / no pre-grasp): {tot['refused']} (in R: {tot['refused_in_R']}). "
                  f"Errors: {len(tot['errors'])}. Whole task succeeded: {tot['task_success']}/{tot['n']}.", "",
                  "| segment | reached | admissible here | success | failures | failures ending in R | silent failures | "
                  "nominal entered U | backups run / failed | ROA fail |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['seg']} | {r['reached']} | {r['admissible_here']} | {r['success']} | {r['failures']} | "
                         f"{r['failures_end_in_R']}/{r['failures']} | {len(r['silent_failures'])} | {r['nominal_unsafe']} | "
                         f"{r['backups_run']} / {r['backups_failed']} | {r['roa_fail']} |")
        cx = [c for r in rows for c in r["counterexamples"]]
        if cx:
            lines += ["", "Counterexamples (world, where, time in the run [s], violation):", ""]
            for c in cx[:40]:
                v = c.get("viol") or {}
                lines.append(f"- {c['tag']}: {c.get('where')} t={c.get('t')} {v.get('kind', '')} {v.get('coll') or ''} "
                             f"{('R fails: ' + ','.join(c['R_fail'])) if c.get('R_fail') else ''}")
        lines.append("")
    rs_all = [r for spec in a.dirs for r in load([spec.partition(":")[0]])]
    yaw = dict(camera=[], pinch=[], push=[])
    for r in rs_all:
        if r.get("obs") and r["obs"]["est"].get("fit_ok"):
            yaw["camera"].append(abs(r["obs"]["residual_yaw_deg"]))
        for s in r.get("segments", []):
            o = s["nominal"].get("observed") or {}
            if s["seg"] in ("pinch", "push") and "yaw_arm_deg" in o:
                yaw[s["seg"]].append(abs(o["truth_yaw_rel_claw_deg"]))
    lines += ["## Yaw read from the arm vs the camera (|error| in deg, same worlds; max over n is not a bound)", "",
              "| source | n | p50 | p95 | max |", "|---|---|---|---|---|"]
    for k, lab in (("camera", "camera (observe_pose, zed)"), ("pinch", "arm FK after the pinch"), ("push", "arm FK after the push")):
        v = np.array(yaw[k])
        if len(v):
            lines.append(f"| {lab} | {len(v)} | {np.percentile(v, 50):.2f} | {np.percentile(v, 95):.2f} | {v.max():.2f} |")
    lines.append("")
    if a.iid:
        rs = [r for r in load(a.iid) if r["tag"][:1] in ("I", "R")]       # i.i.d. draws only: full's interior + the iid set
        bad = [r["tag"] for r in rs if r.get("error") or any(s["counterexample"] for s in r.get("segments", []))
               or (r.get("refused") and not r["B0"]["inR"])]
        n = len(rs)
        lines += [f"## Failure-rate bound (i.i.d. draws only, n = {n})", "",
                  f"Worlds with any counterexample: {len(bad)}. One-sided 95 % Clopper-Pearson upper bound on the probability that a "
                  f"world drawn from the prior box has a counterexample: **{cp_upper(len(bad), n):.3f}** (sampled, not a proof).", ""]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("library")
    s.add_argument("--out", required=True)
    s.add_argument("--set", required=True, choices=["full", "overload", "iid"])
    s.add_argument("--procs", type=int, required=True)
    s.add_argument("--sparse", action="store_true")
    s.add_argument("--plan", default=None, help="JSON overrides of PLAN")
    s = sub.add_parser("falsify")
    s.add_argument("--out", required=True)
    s.add_argument("--placement", required=True, choices=PL)
    s.add_argument("--pop", type=int, default=8)
    s.add_argument("--gens", type=int, default=6)
    s.add_argument("--sigma", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--x0-from", nargs="*", default=None)
    s.add_argument("--plan", default=None)
    s = sub.add_parser("recheck")
    s.add_argument("--out", required=True)
    s.add_argument("--finds", nargs="+", required=True)
    s.add_argument("--thr", type=float, default=0.05)
    s.add_argument("--max", type=int, default=24)
    s.add_argument("--procs", type=int, required=True)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    s.add_argument("--iid", nargs="*", default=None, help="dirs whose records are i.i.d. prior draws (for the bound)")
    s.add_argument("dirs", nargs="+", help="DIR[:label]")
    a = ap.parse_args(argv)
    dict(library=library, falsify=falsify, recheck=recheck, summary=summary)[a.cmd](a)


if __name__ == "__main__":
    main()
