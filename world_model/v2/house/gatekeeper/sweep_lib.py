"""gatekeeper/sweep_lib.py -- multi-seed sweep of the library-augmented gatekeeper (LibGatekeeper) vs the non-library configs.  CPU-only, SIMULATION ONLY.

    python -m world_model.v2.house.gatekeeper.sweep_lib --seeds 0-1 --workers 6 --max-s 45 --out /data/local/ss5772/molmospaces/gatekeeper/lib_smoke [--baselines]
Writes <out>/rows.jsonl (resumable; one line per (config, seed)) and <out>/summary.json.  Structure copied from sweep.py (which is only imported, read-only).
"""
from __future__ import annotations

import argparse, dataclasses, json, os, time
from multiprocessing import get_context
from pathlib import Path

from . import sweep as SW                                            # read-only: CONFIGS / _job / _seeds for the non-library baselines

_TRK = dict(perception=SW.TRK_PERCEPTION, conformal=SW.TRK_CONFORMAL)
# name -> (GKConfig kwargs, lib?)    ; the "_lib" configs use LibGatekeeper (stall_s 1.0 s, grow xy 0.15 / p 0.3 / mu 0.10, debug=True re-verifies every direct recovery)
CONFIGS_LIB = {
    "gk_accel_hold_mon_lib": (dict(person_model="accel", candidates="hold", keep_rule="monitor", cup_chord=True), True),
    "gk_accel_set_mon_lib": (dict(person_model="accel", candidates="set", keep_rule="monitor", cup_chord=True), True),
    "gk_accel_hold_trk_mon_int_lib": (dict(SW.CONFIGS["gk_accel_hold_trk_mon_int"]), True),
    # same-seed non-library counterparts that sweep.CONFIGS does not have
    "gk_accel_set_mon": (dict(person_model="accel", candidates="set", keep_rule="monitor", cup_chord=True), False),
}
LIB_NAMES = tuple(n for n, (_, lib) in CONFIGS_LIB.items() if lib)
BASELINES = ("gk_accel_hold", "gk_accel_hold_mon", "gk_accel_hold_trk_mon_int", "gk_accel_set_mon")      # first three: sweep.CONFIGS (read-only), last: local


def all_names():
    return list(SW.CONFIGS) + [n for n in CONFIGS_LIB if n not in SW.CONFIGS]


def _job(a):
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    name, seed, max_s, task, scene = a
    t0 = time.time()
    if name in SW.CONFIGS:
        r = SW._job(a)
        r["job_s"] = round(time.time() - t0, 1)
        return r
    try:
        from ..teleop import run as RUN
        from .belief import GKConfig
        from .gate import Gatekeeper
        from .gate_lib import LibGatekeeper
        from .run_gk import _brief
        spec, uselib = CONFIGS_LIB[name]
        kw = dict(spec, seed=seed) if spec.get("people_source") == "tracker" else spec
        gk = dataclasses.replace(GKConfig(), **kw)
        g = LibGatekeeper(gk=gk, debug=True) if uselib else Gatekeeper(gk=gk)
        summ, _ = RUN.run_episode(True, seed, task, scene, max_s, with_people=True, verbose=False, filt=g)
        r = _brief(summ)
        r["mode"] = "gatekeeper"
        r["gk"] = g.counters()
        r["error"] = None
    except Exception as e:  # noqa: BLE001
        import traceback
        r = dict(error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-800:])
    r.update(config=name, seed=seed, job_s=round(time.time() - t0, 1))
    return r


def summarize(rows):
    out = {}
    names = []
    for r in rows:
        if r["config"] not in names:
            names.append(r["config"])
    for name in names:
        R = [r for r in rows if r["config"] == name and not r.get("error")]
        if not R:
            continue
        f = lambda k: [r[k] for r in R if r.get(k) is not None]
        G = [r.get("gk") or {} for r in R]
        done = [r for r in R if r.get("task_done")]
        mn = lambda xs: (min(xs) if xs else None)
        d = dict(n=len(R), n_err=sum(1 for r in rows if r["config"] == name and r.get("error")), n_done=len(done), done_rate=len(done) / len(R),
                 mean_time_done_s=(sum(r["task_time_s"] for r in done) / len(done) if done else None),
                 contact_eps=sum(1 for r in R if (r.get("person_contact_ticks") or 0) > 0 or (r.get("wall_collision_ticks") or 0) > 0 or (r.get("physical_house_contacts") or 0) > 0),
                 person_contact_eps=sum(1 for r in R if (r.get("person_contact_ticks") or 0) > 0), wall_collision_eps=sum(1 for r in R if (r.get("wall_collision_ticks") or 0) > 0),
                 spills=sum(1 for r in R if r.get("spilled")), min_person_m=mn(f("min_person_distance_m")),
                 regime2_ticks_mean=sum(g.get("n_regime2_ticks", 0) for g in G) / len(R), falsified_eps=sum(1 for g in G if g.get("n_falsified", 0) > 0),
                 gk_step_ms_mean=sum(g.get("step_ms_mean", 0) for g in G) / len(R), gk_step_ms_max=max([g.get("step_ms_max", 0) for g in G] or [0]),
                 job_s_mean=sum(r.get("job_s", 0) for r in R) / len(R))
        if any("n_rec_calls" in g for g in G):
            ks = ("n_rec_calls", "n_rec_direct", "n_rec_move", "n_rec_look", "n_rec_none", "n_rec_commit", "n_rec_mismatch", "n_rec_move_abort", "n_rec_move_done",
                  "n_look_glances", "n_look_noop", "n_look_deferred", "n_look_nofix", "n_look_infov", "n_look_blocked", "n_look_resolved", "n_regime2_entries")
            d["recoveries"] = {k: sum(g.get(k, 0) for g in G) for k in ks}
            d["lib_size_mean"] = sum(g.get("lib_size", 0) for g in G) / len(R)
            d["lookup_ms_max"] = max(g.get("lookup_ms_max", 0) for g in G)
            d["recover_ms_max"] = max(g.get("recover_ms_max", 0) for g in G)
            d["lib_add_ms_mean"] = sum(g.get("lib_add_ms_mean", 0) for g in G) / len(R)
        out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--configs", default=",".join(LIB_NAMES))
    ap.add_argument("--baselines", action="store_true", help="also run the non-library counterparts (same seeds)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-s", type=float, default=45.0)
    ap.add_argument("--task", default="carry")
    ap.add_argument("--scene", default="h5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fn = out / "rows.jsonl"
    names = a.configs.split(",") + ([b for b in BASELINES if b not in a.configs.split(",")] if a.baselines else [])
    done = {(r["config"], r["seed"]) for r in (json.loads(l) for l in open(fn)) if not r.get("error")} if fn.exists() else set()
    jobs = [(c, s, a.max_s, a.task, a.scene) for s in SW._seeds(a.seeds) for c in names if (c, s) not in done]
    print(f"{len(jobs)} jobs ({len(done)} already done)", flush=True)
    with get_context("fork").Pool(min(a.workers, 6), maxtasksperchild=3) as pool:
        for r in pool.imap_unordered(_job, jobs):
            with open(fn, "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            print(f"{r['config']:30s} seed {r['seed']:3d} done={r.get('task_done')} t={r.get('task_time_s')} err={r.get('error')}", flush=True)
    rows = [json.loads(l) for l in open(fn)]
    S = summarize(rows)
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
