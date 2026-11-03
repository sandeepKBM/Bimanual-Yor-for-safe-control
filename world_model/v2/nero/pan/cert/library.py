"""pan/cert/library.py -- offline certificate library over a belief cell (plan 2026-10-29).  SIMULATION ONLY, CPU.

A cell = a box of pose ESTIMATES the robot may hold (dx, dy, yaw) + the contents polytope + the friction interval.  The cell
covers every theta whose true pose = estimate + aim error (aim error inside the residual box, cert/belief AIM_*) and lies in the
prior world box.  Certification samples (seeded, separate from every evaluation set):
    block 1 (structured): every aim-error corner x both friction ends, estimates cycling over the cell corners, contents cycling
                          over the polytope vertices;
    block 2 (random):     estimate, aim error, friction uniform; contents a random convex combination of vertices.
Each sample world is aimed at its estimate (Episode(est=...)); admit.world_trials runs every segment and the backup family from
every branch point; admit.certify picks one backup per segment that passes all samples, or returns counterexamples.

    python -m world_model.v2.nero.pan.cert.library build   --out DIR --n 48 --seed 7 --procs P [--family ...]
    python -m world_model.v2.nero.pan.cert.library certify --out DIR
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import belief as B

FULL_CELL = dict(est_lo=(-B.PRIOR_DXY - B.AIM_POS, -B.PRIOR_DXY - B.AIM_POS, -B.PRIOR_YAW - B.AIM_YAW),
                 est_hi=(B.PRIOR_DXY + B.AIM_POS, B.PRIOR_DXY + B.AIM_POS, B.PRIOR_YAW + B.AIM_YAW))


def samples(cell, n, seed, bel=None):
    """-> list of (tag, world, est, theta).  True pose clipped into the prior box (A1: theta* in the prior)."""
    bel = bel or B.Belief()
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(cell["est_lo"], float), np.asarray(cell["est_hi"], float)
    V = bel.contents_vertices()
    aim = np.array([bel.aim_pos, bel.aim_pos, bel.aim_yaw])
    prior = np.array([bel.dxy, bel.dxy, bel.yaw])
    est_corners = list(itertools.product(*zip(lo, hi)))
    out = []
    struct = list(itertools.product(itertools.product((-1, 1), repeat=3), bel.mu))
    for i in range(n):
        if i < len(struct):
            sgn, mu = struct[i]
            err = np.array(sgn) * aim
            est = np.array(est_corners[i % len(est_corners)])
            pi = V[i % len(V)]
        else:
            est = rng.uniform(lo, hi)
            err = rng.uniform(-aim, aim)
            mu = float(rng.uniform(*bel.mu))
            wts = rng.dirichlet(np.ones(len(V)))
            pi = wts @ V
        true = np.clip(est + err, -prior, prior)
        est = true - err if np.any(np.abs(est + err) > prior) else est      # keep the aim error exact after clipping
        th = dict(dx=float(true[0]), dy=float(true[1]), yaw_deg=float(true[2]), mu_scale=float(mu), **B.contents_world(pi))
        world = B.world_dict(th)
        out.append((f"K{i:03d}", world, dict(dx=float(est[0]), dy=float(est[1]), yaw_deg=float(est[2])),
                    dict(th, aim_err=[float(x) for x in err])))
    return out


def build(a):
    from . import run_trials as RT
    from . import backups as BK
    fam = tuple(float(x) for x in a.family.split(",")) if a.family else BK.FAMILY
    smp = samples(FULL_CELL, a.n, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cell.json").write_text(json.dumps(dict(cell=FULL_CELL, n=a.n, seed=a.seed, family=fam, belief=B.Belief().describe(),
                                                   thetas={t: th for t, _, _, th in smp}), indent=1, default=float))
    kw = dict(family=fam, perturb_lid=a.perturb_lid)
    RT.run([(t, w, e, kw) for t, w, e, _ in smp], out, a.procs)
    (out / "_DONE_build").write_text("ok")


def certify_dir(a):
    from . import admit as AD
    from .. import segments as S
    out = Path(a.out)
    meta = json.loads((out / "cell.json").read_text())
    recs = []
    for p in sorted((out / "raw").glob("*.json")):
        r = json.loads(p.read_text())
        r["theta"] = meta["thetas"].get(r["tag"])
        recs.append(r)
    fam = tuple(meta["family"])
    cert = dict(cell=meta["cell"], n_samples=len(recs), refused_B0=sum(bool(r.get("refused")) for r in recs),
                errors=[r["tag"] for r in recs if r.get("error")], segments={})
    for seg in S.SEGMENTS:
        cert["segments"][seg] = AD.certify(recs, seg, family=fam)
    (out / "certificate.json").write_text(json.dumps(cert, indent=1, default=float))
    for seg, c in cert["segments"].items():
        print(seg, c["ok"], c.get("backup_retreat", c.get("best_retreat")), c["guarantee"], len(c.get("counterexamples", [])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=48)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--procs", type=int, required=True)
    s.add_argument("--family", default=None)
    s.add_argument("--perturb-lid", type=float, default=0.0)
    s = sub.add_parser("certify")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dict(build=build, certify=certify_dir)[a.cmd](a)


if __name__ == "__main__":
    main()
