"""scenarios/run_scen.py -- run the scripted scenarios over (param x config x seed), CPU only, resumable.  SIMULATION ONLY.

  python -m world_model.v2.house.scenarios.run_scen --scenario corridor --param 0.4:1.4:0.2 --configs shield_brake,gk_accel_hold --seeds 0-4 --workers 4 \
         --max-s 40 --out /data/local/ss5772/molmospaces/scenarios/corridor
  python -m world_model.v2.house.scenarios.run_scen --scenario occlusion --param 3.5,4.5,5.5 --configs gk_accel_hold,gk_accel_hold_trk --seeds 0-4

--param   corridor: lateral offset d [m];  occlusion: t_appear [s].   "a,b,c" or "lo:hi:step" (inclusive).
--scen-kw JSON of the other scenario fields (corridor: arc_s, side; occlusion: speed, jitter_s, oblivious, hide, corner, cross, exit).
--configs comma list of: shield_brake, shield_escape, gk_speed_hold, gk_accel_hold, gk_accel_hold_trk, or any name given in --gk-extra.
--gk-extra JSON {name: {GKConfig field: value, ...}} -- extra gatekeeper configs (overrides applied to GKConfig()); e.g. '{"gk_a3": {"person_model": "accel", "a_p": 3.0}}'.
Writes <out>/rows.jsonl (one line per (scenario, param, scen_kw, config, seed); resumable) and <out>/summary.json (per (param, config) aggregates).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

OUT = Path("/data/local/ss5772/molmospaces/scenarios")
MAX_WORKERS = 4
_CONF = "/data/local/ss5772/molmospaces/percept/people/calib_out/conformal.json"
_TRK = dict(perception="depth", conformal=_CONF if os.path.exists(_CONF) else None)
CONFIGS = {                                  # name -> None (shield kind == name) | GKConfig overrides
    "shield_brake": None, "shield_escape": None,
    "gk_speed_hold": dict(person_model="speed", candidates="hold"),
    "gk_accel_hold": dict(person_model="accel", candidates="hold"),
    "gk_accel_hold_trk": dict(person_model="accel", candidates="hold", people_source="tracker", **_TRK),
}
SCENARIOS = ("corridor", "occlusion")
GK_SUM = ("n_commit", "n_keep_backup", "n_falsified", "n_regime2_ticks", "n_gk_steps", "n_fallback", "n_veto", "n_deviate", "n_belief_miss", "n_untracked_near")


def parse_params(s):
    if ":" in s:
        lo, hi, st = (float(x) for x in s.split(":"))
        n = int(round((hi - lo) / st)) + 1
        return [round(lo + i * st, 6) for i in range(n)]
    return [float(x) for x in s.split(",")]


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def make_spec(scenario, param, seed, scen_kw):
    """-> (scene key via install(), spec).  Corridor is seed-independent; occlusion timing jitter is seeded by the episode seed."""
    kw = dict(scen_kw or {})
    if scenario == "corridor":
        from . import corridor as C
        spec = C.CorridorSpec(d=float(param), **kw)
        return C.install(spec), spec
    if scenario == "occlusion":
        from . import occlusion as O
        for k in ("hide", "corner", "cross", "exit"):
            if k in kw:
                kw[k] = tuple(kw[k])
        kw.setdefault("seed", int(seed))
        spec = O.OcclusionSpec(t_appear=float(param), **kw)
        return O.install(spec), spec
    raise ValueError(scenario)


def run_job(a):
    """one episode (worker process)."""
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    scenario, param, cfg_name, seed, max_s, scen_kw, configs = a
    t0 = time.time()
    row = dict(scenario=scenario, param=param, scen_kw=scen_kw, config=cfg_name, seed=seed, max_s=max_s)
    try:
        from . import ScenSim, patched
        from ..gatekeeper.belief import GKConfig
        from ..gatekeeper.run_gk import run_one
        key, spec = make_spec(scenario, param, seed, scen_kw)
        row["scene_key"] = key
        spec_c = configs[cfg_name]
        ScenSim.LAST = None
        with patched():
            if spec_c is None:
                r = run_one(cfg_name, seed, max_s, "carry", key, True)
            else:
                kw = dict(spec_c)
                if kw.get("people_source") == "tracker":
                    kw["seed"] = seed
                r = run_one("gatekeeper", seed, max_s, "carry", key, True, dataclasses.replace(GKConfig(), **kw))
        row.update(r)
        if ScenSim.LAST is not None:
            row.update(ScenSim.LAST.extras(r.get("task_time_s")))
        if scenario == "occlusion":
            from . import occlusion as O
            row["t_appear_eff"] = O.effective_t_appear(spec)
        row["error"] = None
    except Exception as e:                                           # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    row["job_s"] = round(time.time() - t0, 1)
    return row


def _key(r):
    return (r["scenario"], float(r["param"]), json.dumps(r.get("scen_kw") or {}, sort_keys=True), r["config"], int(r["seed"]))


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], float(r["param"]), r["config"]), []).append(r)
    out = {}
    for (sc, p, cfg), R in sorted(groups.items()):
        ok = [r for r in R if not r.get("error")]
        s = dict(scenario=sc, param=p, config=cfg, n=len(ok), n_err=len(R) - len(ok))
        if ok:
            f = lambda k: [r[k] for r in ok if r.get(k) is not None]
            mean = lambda xs: (sum(xs) / len(xs) if xs else None)
            done = [r for r in ok if r.get("task_done")]
            s.update(done_rate=len(done) / len(ok), mean_time_done_s=mean([r["task_time_s"] for r in done if r.get("task_time_s") is not None]),
                     mean_stall_s=mean(f("stall_s")), min_person_m=min(f("min_person_distance_m") or [None]) if f("min_person_distance_m") else None,
                     min_wall_m=min(f("min_wall_clearance_m")) if f("min_wall_clearance_m") else None,
                     person_contact_eps=sum(1 for r in ok if (r.get("person_contact_ticks") or 0) > 0),
                     wall_collision_eps=sum(1 for r in ok if (r.get("wall_collision_ticks") or 0) > 0),
                     house_contact_eps=sum(1 for r in ok if (r.get("physical_house_contacts") or 0) > 0),
                     spill_eps=sum(1 for r in ok if r.get("spilled")), mean_final_dist_to_goal_m=mean(f("final_dist_to_goal_m")))
            if any("first_visible_t" in r for r in ok):
                s["mean_first_visible_t"] = mean(f("first_visible_t"))
            G = [r["gk"] for r in ok if r.get("gk")]
            if G:
                s["gk"] = {k: sum(g.get(k, 0) for g in G) for k in GK_SUM if any(k in g for g in G)}
                s["gk"]["step_ms_max"] = max(g.get("step_ms_max", 0.0) for g in G)
        out[f"{sc}|{p:g}|{cfg}"] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--configs", default="shield_brake,gk_accel_hold")
    ap.add_argument("--gk-extra", default=None, help="JSON {name: GKConfig overrides}")
    ap.add_argument("--scen-kw", default=None, help="JSON of extra scenario spec fields")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    configs = dict(CONFIGS)
    for name, ov in (json.loads(a.gk_extra) if a.gk_extra else {}).items():
        configs[name] = dict(ov)
    names = a.configs.split(",")
    bad = [n for n in names if n not in configs]
    if bad:
        raise SystemExit(f"unknown configs {bad}; known: {sorted(configs)}")
    scen_kw = json.loads(a.scen_kw) if a.scen_kw else {}
    out = Path(a.out) if a.out else OUT / a.scenario
    out.mkdir(parents=True, exist_ok=True)
    fn = out / "rows.jsonl"
    rows = [json.loads(l) for l in open(fn)] if fn.exists() else []
    done = {_key(r) for r in rows if not r.get("error")}
    jobs = [(a.scenario, p, c, s, a.max_s, scen_kw, configs) for p in parse_params(a.param) for s in parse_seeds(a.seeds) for c in names
            if (a.scenario, p, json.dumps(scen_kw, sort_keys=True), c, s) not in done]
    workers = max(1, min(a.workers, MAX_WORKERS, len(jobs) or 1))
    print(f"{len(jobs)} jobs ({len(done)} already done), {workers} workers", flush=True)
    os.nice(10)
    if jobs:
        with get_context("fork").Pool(workers, maxtasksperchild=3) as pool:
            for r in pool.imap_unordered(run_job, jobs):
                with open(fn, "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                rows.append(r)
                print(f"{r['scenario']} p={r['param']:g} {r['config']:18s} seed {r['seed']:3d} done={r.get('task_done')} t={r.get('task_time_s')} "
                      f"stall={r.get('stall_s')} minP={r.get('min_person_distance_m')} err={r.get('error')}", flush=True)
    latest = {}
    for r in rows:                                    # a later row (a retry) replaces an earlier error row of the same key
        if _key(r) not in latest or not r.get("error"):
            latest[_key(r)] = r
    (out / "summary.json").write_text(json.dumps(summarize(list(latest.values())), indent=1, default=str))
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
