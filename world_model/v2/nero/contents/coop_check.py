"""nero/contents/coop_check.py -- control-layer A/B on the welded carry (11-02).  SIMULATION ONLY.

Same worlds (seeds fixed per task, cell, index), nominal runs only (no backups), four controller variants:
  none      = LeverCtl (FL-MPC + lever payload feedforward)
  fc        = + carry.FRIC_COMP (joint friction compensation + integral, after FL-MPC)
  coop      = + carry.COOP (internal-error projection of the references)
  fc+coop   = both
Per world and variant: max over the run of measured claw twist (Nm), internal force along the grasp line (N), object tilt (deg),
joint torque ratio, shared object tracking error (mm, deg) and internal error (mm, deg), U events, end in R.

    python -m world_model.v2.nero.contents.coop_check --n 4 --procs 8 --out DIR
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np

from . import baseline_ungated as BU
from . import carry as CR
from . import carry_cert as CC

CELLS = [("tray4", (0.125, 0.25)), ("tray6", (0.0, 0.05)), ("box", (0.5, 1.0)), ("box", (1.5, 2.0)), ("box", (2.5, 3.0))]
STIFF = (0.004, 1.0)       # weld solref [s, dampratio]: 4 ms time constant (>= 2 x the 1 ms physics step)
VARIANTS = {"none": (None, None, None), "fc": (True, None, None), "coop": (None, True, None), "fc+coop": (True, True, None),
            "fc/stiff": (True, None, STIFF), "fc+coop/stiff": (True, True, STIFF), "none/stiff": (None, None, STIFF)}
KEYS = ("twist_max", "F_int_max", "obj_tilt_max", "tau_ratio", "e_obj_mm", "e_obj_deg", "e_int_mm", "e_int_deg", "spill_ratio")


def _job(args):
    task, cell, i, var = args
    fc, coop, ws = VARIANTS[var]
    seed = BU.seed_of(task, cell, i)
    w = CC.sample_world(task, cell, np.random.default_rng(seed))
    t0 = time.time()
    try:
        r = CR.world_trial(w, f"{task}_{cell[0]}_{i}_{var}", branch_backups=False, m_hat=0.5 * sum(cell_contents(task, cell)),
                           fric_comp=fc, coop=coop, weld_solref=ws)
        segs = r["segments"]
        out = dict(task=task, cell=list(cell), i=i, seed=seed, var=var,
                   **{k: max(s["nominal"].get(k, 0.0) for s in segs if s["seg"] != "release") for k in KEYS},
                   **{"rel_" + k: max([s["nominal"].get(k, 0.0) for s in segs if s["seg"] == "release"] or [0.0])
                      for k in ("twist_max", "F_int_max")},
                   viol=next((dict(seg=s["seg"], **s["nominal"]["viol"]) for s in segs if s["nominal"]["viol"]), None),
                   end_in_R=r["end_in_R"][0], wall_s=round(time.time() - t0, 1))
    except Exception as e:
        out = dict(task=task, cell=list(cell), i=i, var=var, error=repr(e))
    return out


def cell_contents(task, cell):
    c = CC.cell_spec(task, cell).contents
    return (c.m_lo, c.m_hi)


def summary(rows):
    g = collections.defaultdict(list)
    for r in rows:
        if "error" not in r:
            g[(r["task"], tuple(r["cell"]), r["var"])].append(r)
    L = [f"{'cell':22s} {'variant':14s} n  twist_Nm  Fint_N  tilt_deg  tau   e_obj_mm/deg  e_int_mm/deg  U  endR | release: twist Fint   (max over n; "
         "first block = held segments lift/carry/place)"]
    for (task, cell, var), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], list(VARIANTS).index(kv[0][2]))):
        mx = lambda k: max(r[k] for r in rs)
        L.append(f"{task + ' ' + str(cell):22s} {var:14s} {len(rs)}  {mx('twist_max'):7.2f}  {mx('F_int_max'):6.1f}  {mx('obj_tilt_max'):7.2f}"
                 f"  {mx('tau_ratio'):.2f}  {mx('e_obj_mm'):5.1f}/{mx('e_obj_deg'):4.2f}   {mx('e_int_mm'):5.1f}/{mx('e_int_deg'):4.2f}"
                 f"   {sum(bool(r['viol']) for r in rs)}  {sum(r['end_in_R'] for r in rs)}"
                 f"  |  {mx('rel_twist_max'):5.2f} {mx('rel_F_int_max'):5.1f}")
    errs = [r for r in rows if "error" in r]
    if errs:
        L.append(f"errors: {len(errs)}: {errs[0]['error'][:200]}")
    return "\n".join(L)


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--cells", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = CELLS if a.cells == "all" else [CELLS[int(k)] for k in a.cells.split(",")]
    jobs = [(t, c, i, v) for t, c in cells for i in range(a.n) for v in a.variants.split(",")]
    print(f"{len(jobs)} runs", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(out / "runs.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    s = summary(rows)
    (out / "SUMMARY.txt").write_text(s + "\n")
    print(s, flush=True)


if __name__ == "__main__":
    main()
