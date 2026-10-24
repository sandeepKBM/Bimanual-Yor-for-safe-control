"""gatekeeper/sweep.py -- multi-seed comparison: shield (brake / escape) vs gatekeeper variants, CPU-only, one episode per process.

    python -m world_model.v2.house.gatekeeper.sweep --seeds 0-19 --workers 24 --max-s 45 --out /data/local/ss5772/molmospaces/gatekeeper/sweep_truth
Writes <out>/rows.jsonl (one line per (config, seed); resumable) and <out>/summary.json (per-config aggregates).  SIMULATION ONLY.
"""
from __future__ import annotations

import argparse, dataclasses, json, os, time
from multiprocessing import get_context
from pathlib import Path

CONFIGS = {"shield_brake": None, "shield_escape": None,
           "gk_speed_hold": dict(person_model="speed", candidates="hold"),
           "gk_accel_hold": dict(person_model="accel", candidates="hold")}
DEFAULT_CONFIGS = tuple(CONFIGS)                         # the truth-people sweep; the default of --configs (the *_trk entries below are opt-in)

# ---- tracker people source (people_track head-camera KF tracks instead of sim.people_state; truth stays the metric + audit) ----
# CPU-ONLY perception: "depth" = PerceptionStack with the DepthDetector on mj_multiRay point clouds (no GL/EGL, no GPU, no neural server).  'seg' and 'param:SIG:LAT:DROP' are
# CPU too; 'depth_perc' / 'neural' / 'fusion' / 'full' need estimated pose and/or a GPU detector server -> not used.
TRK_PERCEPTION = "depth"
_CONF_CANDIDATES = ("/data/local/ss5772/molmospaces/percept/people/calib_out/conformal.json",)       # people_track.calib fit, depth detector, h5 carry, alpha=0.05
TRK_CONFORMAL = next((p for p in _CONF_CANDIDATES if os.path.exists(p)), None)
_TRK = dict(perception=TRK_PERCEPTION, conformal=TRK_CONFORMAL)
CONFIGS["shield_brake_trk"] = dict(_shield="shield_brake", **_TRK)
CONFIGS["shield_escape_trk"] = dict(_shield="shield_escape", **_TRK)
CONFIGS["gk_accel_hold_trk"] = dict(person_model="accel", candidates="hold", people_source="tracker", **_TRK)
# ---- opt-in theory fixes: T1 KEEP-rule monitor (GAP-2), cup chord term (GAP-1), set-membership belief filter ----
CONFIGS["gk_accel_hold_mon"] = dict(person_model="accel", candidates="hold", keep_rule="monitor", cup_chord=True)
CONFIGS["gk_accel_hold_trk_mon"] = dict(person_model="accel", candidates="hold", people_source="tracker", keep_rule="monitor", cup_chord=True, **_TRK)
CONFIGS["gk_accel_hold_trk_mon_int"] = dict(person_model="accel", candidates="hold", people_source="tracker", keep_rule="monitor", cup_chord=True,
                                            belief_filter="intersect", **_TRK)


def _job(a):
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    name, seed, max_s, task, scene = a
    from .run_gk import run_one
    from .belief import GKConfig
    t0 = time.time()
    try:
        spec = CONFIGS[name]
        if spec is None:
            r = run_one(name, seed, max_s, task, scene, True)
        elif "_shield" in spec:
            r = run_one(spec["_shield"], seed, max_s, task, scene, True, people_source="tracker",
                        people_kw=dict(perception=spec["perception"], conformal=spec["conformal"], seed=seed))
        else:
            kw = dict(spec, seed=seed) if spec.get("people_source") == "tracker" else spec
            r = run_one("gatekeeper", seed, max_s, task, scene, True, dataclasses.replace(GKConfig(), **kw))
        r["error"] = None
    except Exception as e:  # noqa: BLE001
        r = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
    r.update(config=name, seed=seed, job_s=round(time.time() - t0, 1))
    return r


def _seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def summarize(rows):
    out = {}
    for name in CONFIGS:
        R = [r for r in rows if r["config"] == name and not r.get("error")]
        if not R:
            continue
        f = lambda k: [r[k] for r in R if r.get(k) is not None]
        done = [r for r in R if r.get("task_done")]
        mn = lambda xs: (min(xs) if xs else None)
        out[name] = dict(n=len(R), n_err=sum(1 for r in rows if r["config"] == name and r.get("error")),
                         done_rate=len(done) / len(R), mean_time_done_s=(sum(r["task_time_s"] for r in done) / len(done) if done else None),
                         spills=sum(1 for r in R if r.get("spilled")), max_cup_slip_m=max(f("max_cup_slip_on_tray_m") or [0]),
                         min_person_m=mn(f("min_person_distance_m")), person_contact_eps=sum(1 for r in R if (r.get("person_contact_ticks") or 0) > 0),
                         wall_collision_eps=sum(1 for r in R if (r.get("wall_collision_ticks") or 0) > 0),
                         house_contact_eps=sum(1 for r in R if (r.get("physical_house_contacts") or 0) > 0),
                         peak_accel_max=max(f("peak_base_accel_ms2") or [0]),
                         falsified_eps=sum(1 for r in R if (r.get("gk") or {}).get("n_falsified", 0) > 0),
                         gk_step_ms_max=max([(r.get("gk") or {}).get("step_ms_max", 0) for r in R] or [0]))
        M = [r["gk"] for r in R if (r.get("gk") or {}).get("mean_filtered_radius") is not None]
        if M:                                            # belief_filter="intersect": set-membership filter statistics
            out[name].update(n_obs_inconsistent=sum(g["n_obs_inconsistent"] for g in M), mean_raw_radius=sum(g["mean_raw_radius"] for g in M) / len(M),
                             mean_filtered_radius=sum(g["mean_filtered_radius"] for g in M) / len(M))
        if any((r.get("gk") or {}).get("falsify_reasons") is not None for r in R):          # keep_rule="monitor": why regime 2 fired
            fr = {}
            for r in R:
                for k, v in ((r.get("gk") or {}).get("falsify_reasons") or {}).items():
                    fr[k] = fr.get(k, 0) + v
            out[name]["falsify_reasons"] = fr
        G = [r["gk"] for r in R if (r.get("gk") or {}).get("n_audit_steps") is not None]
        if G:                                            # tracker-source gatekeeper: assumption audit (metrics only; T1's belief-soundness assumption)
            out[name].update(n_belief_miss=sum(g["n_belief_miss"] for g in G), belief_miss_eps=sum(1 for g in G if g["n_belief_miss"] > 0),
                             n_untracked_near=sum(g["n_untracked_near"] for g in G), untracked_near_eps=sum(1 for g in G if g["n_untracked_near"] > 0),
                             n_audit_person_steps=sum(g["n_audit_person_steps"] for g in G), n_audit_matched=sum(g["n_audit_matched"] for g in G),
                             max_miss_excess_m=max(g["max_miss_excess_m"] for g in G))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-s", type=float, default=45.0)
    ap.add_argument("--task", default="carry")
    ap.add_argument("--scene", default="h5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fn = out / "rows.jsonl"
    done = {(r["config"], r["seed"]) for r in (json.loads(l) for l in open(fn)) if not r.get("error")} if fn.exists() else set()
    jobs = [(c, s, a.max_s, a.task, a.scene) for s in _seeds(a.seeds) for c in a.configs.split(",") if (c, s) not in done]
    print(f"{len(jobs)} jobs ({len(done)} already done)", flush=True)
    with get_context("fork").Pool(a.workers, maxtasksperchild=3) as pool:
        for r in pool.imap_unordered(_job, jobs):
            with open(fn, "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            print(f"{r['config']:14s} seed {r['seed']:3d} done={r.get('task_done')} t={r.get('task_time_s')} err={r.get('error')}", flush=True)
    rows = [json.loads(l) for l in open(fn)]
    S = summarize(rows)
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
