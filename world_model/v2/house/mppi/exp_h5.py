"""exp_h5.py -- parallel h5 carry experiments for the mppi planner (CPU sim + GPU planner per worker).  SIMULATION ONLY.

    python -m world_model.v2.house.mppi.exp_h5 --part A --seeds 0 1 2 3 4 5 6 7 8 9 --workers 5 --out $MPPI_OUT/h5_A.json
    python -m world_model.v2.house.mppi.exp_h5 --part B --seeds 0 1 2 --workers 5 --out $MPPI_OUT/h5_B.json
A: operator -> shield  vs  operator -> planner(robust) -> shield, truth mu 0.25.
B: truth mu in {0.18, 0.25, 0.35} x {raw, shield, planner(mu nominal 0.35 | 0.25 | robust CVaR over [0.18,0.35]) x shield on/off}.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

KEEP = ("task_done", "task_time_s", "t_dock_entry_s", "spills_beads_out", "spilled", "cup_lost_at_s", "max_cup_slip_on_tray_m", "max_cup_tilt_deg", "max_tray_tilt_deg",
        "peak_tray_acc_ms2", "p99_tray_acc_ms2", "peak_base_accel_ms2", "min_wall_clearance_m", "wall_collision_ticks", "min_person_distance_m", "person_contact_ticks",
        "physical_house_contacts", "intervention_fraction_gt2pct", "intervention_mean_norm", "intervention_vs_raw_exec_mean_norm", "intervention_vs_raw_exec_fraction_gt5pct",
        "shield_changed_proposal_frac_gt2pct", "shield_changed_proposal_mean", "intervention_labels_ticks", "final_dist_to_goal_m", "mean_speed", "sim_s",
        "planner_warmup_ms", "planner_ms_median", "planner_ms_p95", "mode", "shield_mode", "mu_true", "mu_mode", "seed", "shield_qp_fallbacks")


def _job(spec):
    name, kw = spec
    os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
    from world_model.v2.house.mppi import carry as CA
    t0 = time.time()
    try:
        s, _rows = CA.run_episode(**kw)
        out = {k: s.get(k) for k in KEEP}
        out["wall_s"] = time.time() - t0
    except Exception as e:                                      # keep the sweep alive
        import traceback
        out = dict(error=repr(e), tb=traceback.format_exc()[-800:])
    return name, kw, out


def jobs_A(seeds):
    J = []
    for sd in seeds:
        J.append((f"A|shield|s{sd}", dict(mode="shield", seed=sd)))
        J.append((f"A|planner|s{sd}", dict(mode="planner", seed=sd, mu_mode="robust")))
    return J


def jobs_B(seeds):
    J = []
    for mu in (0.18, 0.25, 0.35):
        for sd in seeds:
            J.append((f"B|raw|mu{mu}|s{sd}", dict(mode="raw", seed=sd, mu_true=mu)))
            J.append((f"B|shield|mu{mu}|s{sd}", dict(mode="shield", seed=sd, mu_true=mu)))
            for lab, kw in (("nom0.35", dict(mu_mode="nominal", mu_nominal=0.35)), ("nom0.25", dict(mu_mode="nominal", mu_nominal=0.25)), ("robust", dict(mu_mode="robust"))):
                for sh in ("on", "off"):
                    J.append((f"B|planner_{lab}|shield_{sh}|mu{mu}|s{sd}", dict(mode="planner", seed=sd, mu_true=mu, shield_mode=sh, **kw)))
    return J


def jobs_C(seeds):
    """tray-tilt jog channel (planner steers c_h to cancel the cup's inertial force) vs the same planner without it."""
    J = []
    for mu in (0.18, 0.25):
        for sd in seeds:
            for sh in ("on", "off"):
                J.append((f"C|planner_robust_tilt|shield_{sh}|mu{mu}|s{sd}", dict(mode="planner", seed=sd, mu_true=mu, shield_mode=sh, mu_mode="robust", use_tilt=True)))
    return J


def jobs_D(seeds):
    """planner ALONE (shield bypassed) with NO dock passthrough: isolates the planner's own cup protection (parts B/C 'shield_off' arms forwarded the raw operator
    command inside the dock radius = raw 4 m/s^2 slews, which contaminates them)."""
    J = []
    for mu in (0.18, 0.25, 0.35):
        for sd in seeds:
            for lab, kw in (("nom0.35", dict(mu_mode="nominal", mu_nominal=0.35)), ("nom0.25", dict(mu_mode="nominal", mu_nominal=0.25)), ("robust", dict(mu_mode="robust")),
                            ("robust_tilt", dict(mu_mode="robust", use_tilt=True))):
                J.append((f"D|planner_{lab}|shield_off_nopass|mu{mu}|s{sd}", dict(mode="planner", seed=sd, mu_true=mu, shield_mode="off", dock_passthrough=False, **kw)))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-s", type=float, default=45.0)
    ap.add_argument("--cfg-kw", default=None, help="json CostCfg overrides for the planner arms")
    ap.add_argument("--resume", action="store_true", help="keep finished entries of --out (no 'error'), rerun the rest")
    ap.add_argument("--planner-only", action="store_true", help="part A: skip the shield-only arm")
    a = ap.parse_args()
    J = {"A": jobs_A, "B": jobs_B, "C": jobs_C, "D": jobs_D}[a.part](a.seeds)
    if a.planner_only:
        J = [(n, kw) for n, kw in J if kw["mode"] == "planner"]
    for _n, kw in J:
        kw["max_s"] = a.max_s
        if a.cfg_kw and kw["mode"] == "planner":
            kw["cfg_kw"] = json.loads(a.cfg_kw)
    res = {}
    if a.resume and Path(a.out).exists():
        old = json.loads(Path(a.out).read_text())
        res = {n: v for n, v in old.items() if "error" not in v["out"]}
        J = [(n, kw) for n, kw in J if n not in res]
        print(f"resume: {len(res)} kept, {len(J)} to run", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=2) as pool:
        for name, kw, out in pool.imap_unordered(_job, J):
            res[name] = dict(kw=kw, out=out)
            print(f"[{len(res)}] {name}: " + (f"done={out.get('task_done')} t={out.get('task_time_s')} beads={out.get('spills_beads_out')} chg={out.get('shield_changed_proposal_frac_gt2pct')}" if "error" not in out else out["error"]), flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
