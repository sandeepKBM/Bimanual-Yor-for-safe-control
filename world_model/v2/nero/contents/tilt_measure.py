"""tilt_measure.py -- MEASURE the peak object tilt of the nominal carry per (cell, acceleration level)  (checker step K4).  SIMULATION ONLY.

    python -m world_model.v2.nero.contents.tilt_measure --task box --a 0.25,0.5 --n 20 --procs N --out DIR [--cells 0.5-1.0,1.0-1.5]

For each cell of carry_cert.cells(task) and each a: n episodes, world = carry_cert.sample_world(task, cell, rng(seed)), seed = cell_seed(task, cell, i)
(the same worlds as the certification), nominal carry = hold 0.4 s + all SEGS (world_trial, branch_backups=False), m_hat = the cell's centre, the
default carry physics.  Per episode: peak obj_tilt_deg per segment (track obj_tilt_max, reset per segment) and overall.  Output: episodes.jsonl
(one line per episode, restartable), TABLE.json, README.md.  An episode that crashes is recorded with `error` and excluded from the statistics
(counted in n_errors).  A segment sequence cut short by a U violation (the run stops) is recorded (`stopped_at`) and its peak covers the segments run."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

NOISE = "simulation only: nominal carry (no sensing noise, no injected disturbance); worlds i.i.d. uniform on the cell; peak of the object z-axis tilt from vertical"
LEAVES_OUT = ("weld compliance of the real gripper (sim grasp = stiff rigid weld, config.WELD_STIFF), sensing / estimation error, base motion, "
              "slosh, controller latency, worlds outside the cell")


def u_limit_deg(task):
    """the task's U limit on obj_tilt_deg (stack/tasks.py, pure data); a peak tilt >= it counts as over (conservative for strict limits too)."""
    from world_model.v2.stack import tasks as TK
    return float(dict(TK.task(task).U)["obj_tilt_deg"].hi)


def cell_key(task, cell):
    return f"{task}|{cell[0]}-{cell[1]}"


def _episode(args):
    task, cell, a, i, seed, planner = args
    os.environ["BIMANUAL_PLANNER"] = planner
    import numpy as np
    from . import carry as CR
    from . import carry_cert as CC
    t0 = time.time()
    key = cell_key(task, cell)
    base = dict(key=key, task=task, cell=list(cell), a=a, i=i, seed=seed, planner=planner)
    try:
        w = CC.sample_world(task, cell, np.random.default_rng(seed))
        c = CC.cell_spec(task, cell).contents
        r = CR.world_trial(w, f"{key}_a{a}_{i}", a_plan=a, m_hat=0.5 * (c.m_lo + c.m_hi), branch_backups=False)
        segs = {s["seg"]: s["nominal"]["obj_tilt_max"] for s in r["segments"]}
        viol = next(((s["seg"], s["nominal"]["viol"]) for s in r["segments"] if s["nominal"]["viol"]), None)
        out = dict(base, tilt_seg=segs, tilt_max=max(segs.values()) if segs else None, pre_hold_viol=r.get("pre_hold_viol"),
                   stopped_at=list(viol) if viol else None, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        out = dict(base, error=repr(e), wall_s=round(time.time() - t0, 1))
    return out


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=pathlib.Path(__file__).parent).strip()
    except Exception:
        return "?"


def build_table(rows, tasks, a_levels, n, meta):
    import numpy as np
    cells = {}
    for r in rows:
        cells.setdefault(r["key"], {}).setdefault(repr(float(r["a"])), []).append(r)
    tab = {}
    for key, lv in cells.items():
        task = key.split("|")[0]
        ul = u_limit_deg(task)
        levels = {}
        for a, rs in lv.items():
            ok = sorted((r for r in rs if "error" not in r and r.get("tilt_max") is not None), key=lambda r: r["i"])
            t = np.array([r["tilt_max"] for r in ok])
            n_err, n_stop = len(rs) - len(ok), sum(1 for r in ok if r.get("stopped_at"))
            n_over = int((t >= ul).sum()) if len(t) else 0
            n_fail = n_err + sum(1 for r in ok if r.get("stopped_at") or r["tilt_max"] >= ul)      # union: errored, stopped early, or peak >= U
            levels[a] = dict(n=len(ok), n_total=len(rs), n_errors=n_err, n_stopped=n_stop, n_over=n_over, n_fail=n_fail, u_limit_deg=ul,
                             p50=round(float(np.percentile(t, 50)), 4) if len(t) else None, p95=round(float(np.percentile(t, 95)), 4) if len(t) else None,
                             max=round(float(t.max()), 4) if len(t) else None,
                             per_seg_max={s: round(max(r["tilt_seg"].get(s, 0.0) for r in ok), 4) for s in ("lift", "carry", "place", "release")} if ok else None,
                             seeds=[r["seed"] for r in rs])
        tab[key] = dict(task=task, planner=meta["planner"], u_limit_deg=ul, levels=levels)
    return dict(cells=tab, noise_model=NOISE, leaves_out=LEAVES_OUT, n_requested=n, a_levels=a_levels, tasks=tasks, **meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="box", help="comma list: box, tray2/4/6, pot")
    ap.add_argument("--a", default="0.25,0.5", help="comma list of a_plan levels (m/s^2)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--planner", default=None, help="scripted | closed_chain: sets BIMANUAL_PLANNER in the workers (default: the env / scripted)")
    ap.add_argument("--cells", default="", help="comma list of 'lo-hi' cell filters (default: every cell)")
    a = ap.parse_args(argv)
    import multiprocessing as mp
    from . import carry_cert as CC
    if a.planner:
        os.environ["BIMANUAL_PLANNER"] = a.planner
    from . import carry as CR
    planner = CR.resolve_planner()                          # validated; spawn workers inherit the env
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in a.task.split(",") if t]
    levels = [float(x) for x in a.a.split(",") if x]
    flt = {x for x in a.cells.split(",") if x}
    jl = out / "episodes.jsonl"
    rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()] if jl.exists() else []
    if any(r.get('planner', planner) != planner for r in rows):
        raise SystemExit(f'{jl} holds episodes of another planner than {planner!r}: use a separate --out')
    done = {(r["key"], repr(float(r["a"])), r["i"]) for r in rows}
    jobs = []
    for task in tasks:
        for (cell,) in CC.cells(task):
            if flt and f"{cell[0]}-{cell[1]}" not in flt:
                continue
            for lv in levels:
                for i in range(a.n):
                    if (cell_key(task, cell), repr(lv), i) not in done:
                        jobs.append((task, cell, lv, i, CC.cell_seed(task, cell, i), planner))
    print(f"{len(jobs)} episodes to run ({len(rows)} already done)", flush=True)
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(jl, "a") as f:
        for r in pool.imap_unordered(_episode, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            rows.append(r)
            print(f"{r['key']} a={r['a']} i={r['i']} tilt={r.get('tilt_max')} {r.get('error', '')} {r['wall_s']}s", flush=True)
    meta = dict(planner=planner, git_head=_git_head(), file=str(out / "TABLE.json"), date=time.strftime("%Y-%m-%d"),
                env={k: v for k, v in os.environ.items() if k.startswith("BIMANUAL_")}, carry_physics="carry_cert default (config.carry_physics())")
    tab = build_table(rows, tasks, levels, a.n, meta)
    tmp = out / "TABLE.json.tmp"
    tmp.write_text(json.dumps(tab, indent=1, default=float))
    tmp.replace(out / "TABLE.json")
    (out / "README.md").write_text(
        f"# Peak object tilt of the nominal carry (K4)\n\nMeasured: peak object tilt (deg) over hold + lift/carry/place/release, per cell x a_plan, n={a.n} worlds per row, "
        f"worlds = carry_cert.sample_world with carry_cert.cell_seed (seeds in TABLE.json). Statistics: p50 / p95 / max over n (a max over n, not a bound).\n\n"
        f"Noise model: {NOISE}.\n\nLeaves out: {LEAVES_OUT}.\n\nplanner {planner}; git HEAD {meta['git_head']}; BIMANUAL_* env: {meta['env']}.\n"
        f"Consumer: stack/checks/tilt.py tilt_row (next HIGHER measured level; errored / stopped-early / over-U episodes count as failures; CP bound).\n")
    print("wrote", out / "TABLE.json", flush=True)


if __name__ == "__main__":
    main()
