"""people_track/ps_sweep.py -- parallel sweep of ps_run over seeds x walkers x shields x perception conds.  SIMULATION ONLY, CPU only.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 BIMANUAL_TELEOP_FAST=1 \
    python -m world_model.v2.house.people_track.ps_sweep --seeds 0-8 --walkers yield,oblivious,giveup --shields old,new --conds truth --workers 40 \
        --out /data/local/ss5772/molmospaces/people_shield/sweep9
Writes one JSON per job (resumable: existing JSONs are skipped) and a sweep_summary.json + a printed table.
Depth-only tracker conds need the camera (EGL):  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4  --conds depth
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

OUT = Path(os.environ.get("PEOPLE_SHIELD_OUT", "/data/local/ss5772/molmospaces/people_shield"))


def _job(a):
    seed, walker, shield, cond, out, max_s = a[:6]
    hm = a[6] if len(a) > 6 else None
    tag = f"{walker}_{shield}_{cond.replace(':', '-')}{'_hm' if hm else ''}_s{seed}"
    f = Path(out) / f"{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    from .ps_run import run
    summ, _ = run(seed, walker, shield, cond, max_s, human_mesh=hm)
    f.write_text(json.dumps(summ, indent=1, default=str))
    return summ


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-8")
    ap.add_argument("--walkers", default="yield,oblivious")
    ap.add_argument("--shields", default="old,new")
    ap.add_argument("--conds", default="truth")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--out", default=str(OUT / "sweep"))
    ap.add_argument("--human-mesh", default=None, choices=[None, "articulated"], help="visual Rocketbox humans on the scripted people (default: capsules)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, w, sh, c, str(out), a.max_s, a.human_mesh) for s in parse_seeds(a.seeds) for w in a.walkers.split(",")
            for sh in a.shields.split(",") for c in a.conds.split(",")]
    with mp.Pool(a.workers, maxtasksperchild=3) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    (out / "sweep_summary.json").write_text(json.dumps(res, indent=1, default=str))
    import numpy as np
    keys = {}
    for r in res:
        keys.setdefault((r["walker"], r["shield_variant"], r["cond"]), []).append(r)
    print("walker shield cond | n done time contact(fp) true_contact moving/stat min_true_clear deadlock_s beads")
    for (w, sh, c), rs in sorted(keys.items()):
        n = len(rs)
        print(f"{w:9s} {sh:3s} {c:10s} | n={n} done={sum(bool(r['task_done']) for r in rs)} "
              f"t={np.nanmean([r['task_time_s'] if r['task_time_s'] is not None else np.nan for r in rs]):.1f} "
              f"fp_ct={sum(r['fp_contact_ticks'] for r in rs)} true_ct={sum(r['true_contact_ticks'] for r in rs)} "
              f"(mov {sum(r['true_contact_moving'] for r in rs)} / stat {sum(r['true_contact_stationary'] for r in rs)}) "
              f"min_clr={min((r['min_true_clear_m'] if r['min_true_clear_m'] is not None else 9) for r in rs):.3f} "
              f"dead={np.mean([r['deadlock_s'] for r in rs]):.1f}s beads={sum(r['spills_beads_out'] for r in rs)}")


if __name__ == "__main__":
    main()
