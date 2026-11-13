"""teleop/fastpath_validate.py -- N-seed RAW + SHIELD validation of MuJoCo solver settings for the teleop carry task (SIMULATION ONLY, CPU only).

noslip_iterations / tolerance CHANGE the contact physics (noslip=0 let the cup slide off and tip in 2 of 3 raw seeds in the smoke study), so
they must be validated against the shipped setting (noslip 4, tol 1e-8) before they are ever enabled by default.  BIMANUAL_TELEOP_FAST is
bit-identical and needs no validation (see fastpath.py); it is forced ON here only for speed.

    cd <repo> ; export PYTHONPATH=$PWD:$PWD/experiments/ab_smooth_vs_bangbang:$PWD/experiments/realtime \\
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    PY=python
    $PY -m world_model.v2.house.teleop.fastpath_validate --seeds 0-8 --modes raw shield --stream open \\
        --cfg 4:1e-8 --cfg 1:1e-8 --cfg 0:1e-8 --cfg 0:1e-6 --pool 56 --out data/molmospaces/teleop/perf/val9

--stream open replays the SAME pre-recorded operator commands in every config (isolates the physics change; closed-loop operators diverge
chaotically after any perturbation).  First --cfg is the reference.  Output: <out>/val_<mode>_n<noslip>_t<tol>_s<seed>.json + summary.json/table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _one(job):
    mode, seed, noslip, tol, stream, max_s, out = job
    os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
    os.environ.setdefault("BIMANUAL_FLMPC_IMPL", "rt")
    import numpy as np
    from . import sim as S
    from . import run as R
    if not hasattr(S.TeleopSim, "_orig_init"):
        S.TeleopSim._orig_init = S.TeleopSim.__init__
    orig = S.TeleopSim._orig_init

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.m.opt.noslip_iterations = noslip
        self.m.opt.tolerance = tol

    S.TeleopSim.__init__ = init
    t0 = time.time()
    summ, rows = R.run_episode(mode == "shield", seed, "carry", "h5", max_s, stream=stream, verbose=False)
    keys = ("sim_s", "task_done", "spills_beads_out", "spilled", "cup_lost_at_s", "max_tray_tilt_deg", "max_cup_tilt_deg",
            "max_cup_slip_on_tray_m", "peak_tray_acc_ms2", "p99_tray_acc_ms2", "physical_house_contacts", "flmpc_infeasible_steps",
            "intervention_fraction_gt2pct", "min_wall_clearance_m", "final_pose")
    r = {k: summ[k] for k in keys}
    r.update(mode=mode, seed=seed, noslip=noslip, tol=tol, wall_s=time.time() - t0,
             rel=np.array([x["rel"] for x in rows]).tolist(), pose=np.array([x["pose"] for x in rows]).tolist())
    with open(os.path.join(out, f"val_{mode}_n{noslip}_t{tol:g}_s{seed}.json"), "w") as f:
        json.dump(r, f, default=str)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-8")
    ap.add_argument("--modes", nargs="+", default=["raw", "shield"], choices=["raw", "shield"])
    ap.add_argument("--cfg", action="append", default=None, help="noslip:tol, first = reference (default 4:1e-8)")
    ap.add_argument("--stream", default="open", choices=["open", "closed"])
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    cfgs = [tuple(c.split(":")) for c in (a.cfg or ["4:1e-8"])]
    cfgs = [(int(n), float(t)) for n, t in cfgs]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(m, s, n, t, a.stream, a.max_s, a.out) for m in a.modes for (n, t) in cfgs for s in seeds]
    if a.pool > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.pool, maxtasksperchild=3) as p:
            res = p.map(_one, jobs, chunksize=1)
    else:
        res = [_one(j) for j in jobs]
    import numpy as np
    ref = cfgs[0]
    by = {(r["mode"], r["noslip"], r["tol"], r["seed"]): r for r in res}
    lines, summary = [], {}
    for mode in a.modes:
        for (n, t) in cfgs:
            rs = [by[(mode, n, t, s)] for s in seeds]
            dev = []
            for s in seeds:
                b, r = by[(mode, ref[0], ref[1], s)], by[(mode, n, t, s)]
                k = min(len(b["rel"]), len(r["rel"]))
                dev.append(float(np.abs(np.array(r["rel"][:k]) - np.array(b["rel"][:k])).max()))
            row = dict(mode=mode, noslip=n, tol=t, n=len(rs),
                       spilled=int(sum(bool(r["spilled"]) for r in rs)), beads_out_total=int(sum(r["spills_beads_out"] for r in rs)),
                       task_done=int(sum(bool(r["task_done"]) for r in rs)),
                       cup_tilt_max_deg_mean=float(np.mean([r["max_cup_tilt_deg"] for r in rs])),
                       cup_tilt_max_deg_worst=float(np.max([r["max_cup_tilt_deg"] for r in rs])),
                       cup_slip_max_m_mean=float(np.mean([r["max_cup_slip_on_tray_m"] for r in rs])),
                       peak_tray_acc_mean=float(np.mean([r["peak_tray_acc_ms2"] for r in rs])),
                       interv_frac_mean=float(np.mean([r["intervention_fraction_gt2pct"] for r in rs])),
                       house_contacts=int(sum(r["physical_house_contacts"] for r in rs)),
                       max_rel_dev_vs_ref_m=float(np.max(dev)))
            summary[f"{mode}_n{n}_t{t:g}"] = row
            lines.append(f"{mode:6s} noslip={n} tol={t:g}  spilled {row['spilled']}/{row['n']} beads {row['beads_out_total']}  done {row['task_done']}/{row['n']}  "
                         f"cup_tilt mean/worst {row['cup_tilt_max_deg_mean']:.2f}/{row['cup_tilt_max_deg_worst']:.2f} deg  slip {row['cup_slip_max_m_mean']:.3f} m  "
                         f"tray_acc {row['peak_tray_acc_mean']:.2f}  interv {row['interv_frac_mean']:.2f}  contacts {row['house_contacts']}  dev_vs_ref {row['max_rel_dev_vs_ref_m']:.4f} m")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
