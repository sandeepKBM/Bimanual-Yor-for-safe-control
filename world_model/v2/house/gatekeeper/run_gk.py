"""gatekeeper/run_gk.py -- one headless episode with the gatekeeper (CPU, no video).  SIMULATION ONLY.

  python -m world_model.v2.house.gatekeeper.run_gk --seed 0 --max-s 30 [--compare] [--out /data/local/ss5772/molmospaces/gatekeeper/smoke_seed0.json]

--compare runs: shield (default fallback=brake), shield (fallback=escape), gatekeeper on the same seed.
--ab runs the 4 gatekeeper configs (person_model x candidates) in parallel processes (CPU) and adds the shield baseline from --baseline (smoke_seed0.json):
  python -m world_model.v2.house.gatekeeper.run_gk --ab --seed 0 --max-s 30 --out /data/local/ss5772/molmospaces/gatekeeper/smoke_AB_seed0.json
--person-model / --candidates select one config for a single run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..teleop import run as RUN
from ..teleop import shield as SH
from .belief import GKConfig
from .gate import Gatekeeper

OUT = Path("/data/local/ss5772/molmospaces/gatekeeper")
KEYS = ["mode", "task_done", "task_time_s", "final_phase", "spilled", "spills_beads_out", "max_cup_slip_on_tray_m", "min_wall_clearance_m",
        "wall_collision_ticks", "min_person_distance_m", "person_contact_ticks", "physical_house_contacts", "shield_qp_fallbacks", "shield_ir_vetoes",
        "final_dist_to_goal_m", "peak_base_speed", "peak_base_accel_ms2", "peak_tray_acc_ms2", "intervention_fraction_gt2pct", "wall_s"]


def _brief(s):
    return {k: s.get(k) for k in KEYS}


def run_one(kind, seed, max_s, task="carry", scene="h5", with_people=True, gk=None, people_source="truth", people_kw=None):
    """people_source "truth" (default, unchanged) | "tracker": the people the FILTER sees come from people_track (sim.people_state stays the metric truth).
    people_kw: dict(perception=<run_pt cond, CPU: depth|seg|param:..>, conformal=<json path|None>, seed=<int>).  Shield kinds: run.run_episode(people_source=,
    people_kw=) -> PerceivedShield (exactly run.py's --people tracker path); gatekeeper: GKConfig(people_source="tracker", perception, conformal, seed)."""
    import dataclasses
    kw = dict(people_kw or {})
    if kind == "gatekeeper":
        if people_source == "tracker":
            gk = dataclasses.replace(gk or GKConfig(), people_source="tracker", **{k: kw[k] for k in ("perception", "conformal", "seed") if k in kw})
        g = Gatekeeper(gk=gk)
        summ, _ = RUN.run_episode(True, seed, task, scene, max_s, with_people=with_people, verbose=False, filt=g)
        summ = _brief(summ); summ["mode"] = "gatekeeper"
        summ["gk"] = g.counters()
        return summ
    cfg = SH.ShieldCfg(fallback="escape") if kind == "shield_escape" else None
    ek = dict(people_source=people_source, people_kw=kw) if people_source != "truth" else {}
    summ, _ = RUN.run_episode(True, seed, task, scene, max_s, cfg=cfg, with_people=with_people, verbose=False, **ek)
    summ = _brief(summ); summ["mode"] = kind
    return summ


AB = [("speed", "hold"), ("accel", "hold"), ("speed", "set"), ("accel", "set")]


def _ab_job(args):
    pm, cd, seed, max_s, task, scene, people, base = args
    import dataclasses
    gk = dataclasses.replace(base, person_model=pm, candidates=cd)
    r = run_one("gatekeeper", seed, max_s, task, scene, people, gk)
    return f"{pm}+{cd}", r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--task", default="carry")
    ap.add_argument("--scene", default="h5")
    ap.add_argument("--no-people", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--v-p-max", type=float, default=GKConfig.v_p_max)
    ap.add_argument("--out", default=str(OUT / "smoke_seed0.json"))
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--baseline", default=str(OUT / "smoke_seed0.json"))
    ap.add_argument("--person-model", default="speed", choices=["speed", "accel"])
    ap.add_argument("--candidates", default="hold", choices=["hold", "set"])
    a = ap.parse_args()
    gk = GKConfig(v_p_max=a.v_p_max, person_model=a.person_model, candidates=a.candidates)
    if a.ab:
        import dataclasses
        from multiprocessing import Pool
        with Pool(len(AB)) as pool:
            out = dict(pool.map(_ab_job, [(pm, cd, a.seed, a.max_s, a.task, a.scene, not a.no_people, gk) for pm, cd in AB]))
        try:
            base = json.loads(Path(a.baseline).read_text())["results"]
            out.update({k: v for k, v in base.items() if k.startswith("shield")})
        except Exception as e:                                                     # noqa: BLE001
            out["baseline_error"] = repr(e)
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(seed=a.seed, max_s=a.max_s, task=a.task, scene=a.scene, people=not a.no_people, gk_cfg=dataclasses.asdict(gk), results=out), indent=1, default=str))
        print(json.dumps({k: (v if k == "baseline_error" else {x: v.get(x) for x in ("task_done", "task_time_s", "spilled")}) for k, v in out.items()}, default=str), flush=True)
        return
    kinds = ["shield_brake", "shield_escape", "gatekeeper"] if a.compare else ["gatekeeper"]
    res = {}
    for k in kinds:
        res[k] = run_one(k, a.seed, a.max_s, a.task, a.scene, not a.no_people, gk)
        print(json.dumps(res[k], indent=1, default=str), flush=True)
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(seed=a.seed, max_s=a.max_s, task=a.task, scene=a.scene, people=not a.no_people, gk_cfg=vars(gk) if not hasattr(gk, "__dataclass_fields__") else gk.__dict__, results=res), indent=1, default=str))


if __name__ == "__main__":
    main()
