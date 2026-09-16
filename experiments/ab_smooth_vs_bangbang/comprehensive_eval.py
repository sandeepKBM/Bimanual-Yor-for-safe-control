"""experiments/ab_smooth_vs_bangbang/comprehensive_eval.py

COMPREHENSIVE validation sweep for the bimanual FL-MPC solution
(--controller flmpc) vs the reactive baseline (--controller v4c) on the
NOW-FINAL full-fidelity model (self-collision + arm<->arm + arm<->table +
right-arm<->cup + arm<->cube all real, via scripts/make_cup_collect_safety_demo.py).

This generalizes the ad-hoc validation sweeps in this directory (e.g. the
graceful-slack-QP-fallback check at
/common/home/ss5772/.claude/jobs/f1580deb/tmp/fl_graceful_sweep.py) into one
reusable, parallel, CLI-driven harness that runs the SAME hard geometry
(scenario="crossing", ee_speed_limit=True, cup_jitter_m=0.04, N_CUBES=3,
V_EE_MAX_MS=0.28 fast/hard) across a seed range x controller set, and reports
both a per-seed CSV-ish progress line and a final per-controller aggregate
table -- the "exhaustive check we did" made repeatable.

Usage (owner runs the FULL sweep; see also `# OWNER COMMANDS` in the module
docstring further down for exact copy-paste invocations):

    MUJOCO_GL=egl \\
    /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
    experiments/ab_smooth_vs_bangbang/comprehensive_eval.py \\
        --seeds 0-49 --controllers flmpc,v4c --workers 14 \\
        --out experiments/ab_smooth_vs_bangbang/results_full.json

Each episode is ~5min (full ~15-16s duration_s at DT=0.005s => ~3000-3200
physics steps, both arms' QPs solved every step) so N=50 seeds x 2
controllers = 100 episodes is HOURS even at --workers 14; that's why this is
a CLI harness the owner runs standalone, not something this task itself
executes at full scale (see `--max-steps` for a fast partial-episode smoke).

SMOOTHNESS / JERK METRIC (documented here, not implicit):
  We don't have a torque log for the intruder/ego reference split at the
  Python level cheaply comparable across controllers (both controllers'
  *commanded* torque differs by construction -- flmpc plans accelerations
  and realizes tau=M*a+c, v4c PD-tracks a joint reference through a
  decentralized box-QP -- so a raw torque comparison conflates "different
  control law" with "smoothness"). Instead we use the REALIZED right-EE
  Cartesian path (log["right_ee_pos"], the arm actually doing the fast
  pick/carry/deposit crossing -- the "bang-bang vs smooth" axis this A/B
  directory is about) as the common observable and derive:
    vel  = diff(right_ee_pos) / DT                (finite-diff, 1st deriv)
    acc  = diff(vel)          / DT                (2nd deriv)
    jerk = diff(acc)          / DT                (3rd deriv)
    jerk_rms = sqrt(mean(||jerk||^2))              over the whole episode
  reported in m/s^3. This is a physically legible plant-motion smoothness
  proxy that's directly comparable across controllers (same DT, same EE
  path definition) regardless of internal control-law differences, and it's
  exactly the axis "smooth vs bang-bang" the experiment name refers to.
  Triple finite-differencing amplifies logging noise, so treat jerk_rms as
  a RELATIVE (flmpc vs v4c), not absolute, number.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AB4_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
EVAL_DIR = REPO_ROOT / "eval"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(AB4_DIR), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 0.005  # scripts/make_cup_collect_safety_demo.py / sim_common.DT (scene_yor_canonical.xml timestep)

# ABLATION VARIANTS (workshop-paper knobs isolating the FL-MPC "cooperative"
# and "smooth" claims -- see make_cup_collect_safety_demo.py's
# FL_W_JERK_OVERRIDE / FL_W_POS_INT_OVERRIDE module docstring). Each maps to
# the underlying demo controller string ("flmpc" for all 3 flmpc variants --
# only the module-global overrides differ) plus the two override values to
# set on the demo module BEFORE build_episode (None => leave at FLMPC's own
# default, i.e. identical to plain "flmpc").
#   flmpc            baseline: joint QP, W_JERK/W_POS_INT at FLMPC defaults.
#   flmpc_bangbang   SMOOTHNESS ablation: W_JERK=0.0 (no jerk regularization)
#                    -> right-EE jerk_rms should be >> flmpc's.
#   flmpc_unilateral COOPERATION ablation: W_POS_INT=1e6 (intruder/right arm
#                    rigidly tracks its task, can't deviate to help avoid) ->
#                    only the ego/left arm should show large deflection during
#                    the crossing, unlike flmpc's shared left/right split.
#   v4c              reactive baseline (decentralized box-QP, no FLMPC).
CONTROLLERS = ("flmpc", "flmpc_bangbang", "flmpc_unilateral", "v4c")
VARIANT_OVERRIDES = {
    "flmpc":            dict(demo_controller="flmpc", w_jerk=None, w_pos_int=None),
    "flmpc_bangbang":   dict(demo_controller="flmpc", w_jerk=0.0,  w_pos_int=None),
    "flmpc_unilateral": dict(demo_controller="flmpc", w_jerk=None, w_pos_int=1e6),
    "v4c":              dict(demo_controller="v4c",   w_jerk=None, w_pos_int=None),
}


def jerk_rms(ee_pos: np.ndarray, dt: float = DT) -> float:
    """RMS of ||d^3 ee_pos / dt^3|| over the episode -- see module docstring
    SMOOTHNESS section for why right-EE Cartesian path, not torque."""
    if ee_pos.shape[0] < 4:
        return 0.0
    vel = np.diff(ee_pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))


def deflection_max_m(ee_pos: np.ndarray, idx_settle: int) -> float:
    """max ||ee_pos[k] - ee_pos[idx_settle]|| for k >= idx_settle -- the
    per-arm Cartesian motion DURING the crossing (from the settled hold pose
    at RIGHT_START_T onward), the cooperation-axis observable: see the
    variant docstring in CONTROLLERS/VARIANT_OVERRIDES above."""
    idx_settle = min(idx_settle, ee_pos.shape[0] - 1)
    if idx_settle >= ee_pos.shape[0] - 1:
        return 0.0
    dev = np.linalg.norm(ee_pos[idx_settle:] - ee_pos[idx_settle], axis=1)
    return float(np.max(dev))


def run_one(args: dict) -> dict:
    """Runs in a worker process. Returns a flat metric row (or ok=False + err)."""
    seed = args["seed"]
    variant = args["controller"]
    max_steps = args.get("max_steps")
    t0 = time.perf_counter()
    try:
        import make_cup_collect_safety_demo as demo
        import sim_common as sc
        demo.N_CUBES = 3
        # SPEED-ENVELOPE sweep (#3): the intruder-EE Cartesian speed cap. Default
        # 0.28 (the hard-geometry setting); --v-ee-max overrides it per run so the
        # same benchmark is swept over a range of speeds (0.10/0.20/0.28/0.40) to
        # show FL-MPC's operating envelope vs v4c, not a single point.
        demo.V_EE_MAX_MS = float(args.get("v_ee_max") or 0.28)
        # ABLATION WIRING: set the demo module's FL-MPC override globals
        # BEFORE build_episode (fl = mpc_fl.FLMPC(...) is constructed inside
        # build_episode's setup, reading these at construction time -- see
        # FL_W_JERK_OVERRIDE/FL_W_POS_INT_OVERRIDE in
        # make_cup_collect_safety_demo.py). Explicitly reset both every call
        # (workers are reused across jobs via maxtasksperchild) so a prior
        # job's override can never leak into this one.
        ov = VARIANT_OVERRIDES[variant]
        demo.FL_W_JERK_OVERRIDE = ov["w_jerk"]
        demo.FL_W_POS_INT_OVERRIDE = ov["w_pos_int"]
        controller = ov["demo_controller"]

        info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                                   cup_jitter_m=0.04, n_cubes=3)
        log, _ = demo.run_episode(info, controller=controller, qp_enabled=True, capture=False,
                                   ee_speed_limit=True, table_floor=True, max_steps=max_steps)
        m = demo.summarize(log)

        # spill@lift: same convention noted in the task context -- spill
        # flagged within the first 320 steps (the lift phase), read from the
        # per-step spilled array directly rather than the whole-episode max.
        n_lift = min(320, len(log["spilled"]))
        spill_at_lift = int(np.max(log["spilled"][:n_lift])) if n_lift > 0 else 0

        right_ee = np.asarray(log["right_ee_pos"])
        jerk = jerk_rms(right_ee, DT)
        # SPARC smoothness (#1): spectral arc length of the right-EE SPEED profile
        # -- a robust, established smoothness metric (sim_common.sparc) that does
        # NOT triple-differentiate, so it is far less noise-sensitive than jerk_rms
        # (more negative = LESS smooth; a straight/smooth move is near 0). This is
        # the primary smoothness number for the paper; jerk_rms is kept as a
        # secondary/relative check.
        _rspeed = np.linalg.norm(np.diff(right_ee, axis=0) / DT, axis=1) if right_ee.shape[0] > 1 else np.zeros(1)
        try:
            sparc_right = float(sc.sparc(_rspeed, DT))
        except Exception:
            sparc_right = float("nan")

        # COOPERATION-AXIS metrics: per-arm Cartesian deflection during the
        # crossing, measured from the settled hold pose at RIGHT_START_T
        # (demo.RIGHT_START_T -- same "settle index" convention already used
        # in make_cup_collect_safety_demo.py's own __main__ smoke, see
        # idx_settle = searchsorted(log["t"], RIGHT_START_T) there). Compares
        # how much EACH arm moved to negotiate the crossing: flmpc should
        # split this across both arms; flmpc_unilateral should push nearly
        # all of it onto the left/ego arm (right/intruder rigidly tracks its
        # task and can't help avoid).
        idx_settle = int(np.searchsorted(log["t"], demo.RIGHT_START_T))
        left_ee = np.asarray(log["left_ee_pos"])
        left_deflect_max_m = deflection_max_m(left_ee, idx_settle)
        right_deflect_max_m = deflection_max_m(right_ee, idx_settle)

        return dict(
            seed=seed, controller=variant, ok=True,
            n_steps=int(len(log["t"])),
            collision=bool(m["collision"]),
            spilled_max=int(m["spilled_max"]),
            spill_at_lift=spill_at_lift,
            right_ee_speed_max_ms=float(m["right_ee_speed_max_ms"]),
            h_min=float(m["h_min"]),
            qp_infeasible_frac_left=float(m["qp_infeasible_frac_left"]),
            qp_infeasible_frac_right=float(m["qp_infeasible_frac_right"]),
            qp_infeasible_frac_max=float(max(m["qp_infeasible_frac_left"], m["qp_infeasible_frac_right"])),
            placed_final=int(m["placed_final"]),
            placed_max=int(m["placed_max"]),
            left_qp_deflect_max_nm=float(m["left_qp_deflect_max_nm"]),
            jerk_rms_right_ee=jerk,
            sparc_right_ee=sparc_right,
            left_deflect_max_m=left_deflect_max_m,
            right_deflect_max_m=right_deflect_max_m,
            v_ee_max=float(args.get("v_ee_max") or 0.28),
            wall_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return dict(seed=seed, controller=variant, ok=False,
                     err=str(e)[:200], tb=traceback.format_exc()[-800:],
                     wall_s=time.perf_counter() - t0)


def aggregate(rows: list[dict], controller: str) -> dict:
    g = [r for r in rows if r.get("ok") and r["controller"] == controller]
    n = len(g)
    if n == 0:
        return dict(controller=controller, n=0)
    return dict(
        controller=controller, n=n,
        collision_rate=sum(r["collision"] for r in g) / n,
        spill_rate=sum(r["spilled_max"] > 0 for r in g) / n,
        spill_at_lift_rate=sum(r["spill_at_lift"] > 0 for r in g) / n,
        fling_mean_ms=float(np.mean([r["right_ee_speed_max_ms"] for r in g])),
        fling_worst_ms=float(np.max([r["right_ee_speed_max_ms"] for r in g])),
        h_min_mean=float(np.mean([r["h_min"] for r in g])),
        h_min_worst=float(np.min([r["h_min"] for r in g])),
        qp_infeasible_frac_mean=float(np.mean([r["qp_infeasible_frac_max"] for r in g])),
        placed_final_mean=float(np.mean([r["placed_final"] for r in g])),
        jerk_rms_mean=float(np.mean([r["jerk_rms_right_ee"] for r in g])),
        jerk_rms_worst=float(np.max([r["jerk_rms_right_ee"] for r in g])),
        sparc_mean=float(np.nanmean([r.get("sparc_right_ee", float("nan")) for r in g])),
        left_deflect_mean_m=float(np.mean([r["left_deflect_max_m"] for r in g])),
        right_deflect_mean_m=float(np.mean([r["right_deflect_max_m"] for r in g])),
        v_ee_max=float(g[0].get("v_ee_max", 0.28)),
    )


def print_aggregate_table(rows: list[dict], controllers: list[str]) -> None:
    aggs = [aggregate(rows, c) for c in controllers]
    # jerk_rms_mean = SMOOTH axis (bangbang should blow this up vs flmpc);
    # left/right_defl_mean = COOPERATION axis (unilateral should push nearly
    # all deflection onto left/ego vs flmpc's more even split).
    hdr = (f"{'controller':17s} {'n':>3s} {'coll%':>6s} {'spill%':>7s} {'fling_mean':>10s} {'fling_worst':>11s} "
           f"{'h_min_worst':>11s} {'infeas_mean':>11s} {'placed_mean':>11s} {'jerk_rms_mean':>13s} "
           f"{'sparc':>8s} {'left_defl_m':>11s} {'right_defl_m':>12s}")
    print("\n=== AGGREGATE (comprehensive_eval) ===")
    print(hdr)
    print("-" * len(hdr))
    for a in aggs:
        if a["n"] == 0:
            print(f"{a['controller']:17s}   0  (no successful episodes)")
            continue
        print(f"{a['controller']:17s} {a['n']:3d} "
              f"{100*a['collision_rate']:5.1f}% {100*a['spill_rate']:6.1f}% "
              f"{a['fling_mean_ms']:10.3f} {a['fling_worst_ms']:11.3f} "
              f"{a['h_min_worst']:11.3f} {a['qp_infeasible_frac_mean']:11.3f} "
              f"{a['placed_final_mean']:11.2f} {a['jerk_rms_mean']:13.1f} "
              f"{a['sparc_mean']:8.2f} {a['left_deflect_mean_m']:11.4f} {a['right_deflect_mean_m']:12.4f}")
    print()


def parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, required=True, help="e.g. '0-49' or '0,1,2,5'")
    ap.add_argument("--controllers", type=str, default="flmpc,v4c",
                     help="comma-separated subset of {flmpc,flmpc_bangbang,flmpc_unilateral,v4c} "
                          "(the 3 flmpc_* names are ablation VARIANTS, not separate controllers -- "
                          "see VARIANT_OVERRIDES above)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=str, default="experiments/ab_smooth_vs_bangbang/results.json")
    ap.add_argument("--max-steps", type=int, default=None,
                     help="SMOKE ONLY: truncate each episode (passthrough to run_episode's max_steps). "
                          "Metrics computed on the truncated log are PARTIAL -- most useful for confirming "
                          "the harness end-to-end, not for real collision/spill statistics.")
    ap.add_argument("--maxtasksperchild", type=int, default=1)
    ap.add_argument("--v-ee-max", type=float, default=None,
                     help="Speed-envelope sweep (#3): override the intruder-EE Cartesian speed cap "
                          "(demo.V_EE_MAX_MS; default 0.28). Sweep by launching one run per value "
                          "(e.g. 0.10 0.20 0.28 0.40) to trace FL-MPC's operating envelope vs v4c.")
    args = ap.parse_args()

    seeds = parse_seed_range(args.seeds)
    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    for c in controllers:
        assert c in CONTROLLERS, f"unknown controller {c!r}, expected one of {CONTROLLERS}"

    jobs = [dict(seed=s, controller=c, max_steps=args.max_steps, v_ee_max=args.v_ee_max)
            for c in controllers for s in seeds]
    n_jobs = len(jobs)
    print(f"[comprehensive_eval] {n_jobs} episodes ({len(seeds)} seeds x {len(controllers)} controllers), "
          f"workers={args.workers}, max_steps={args.max_steps or 'FULL'}", flush=True)
    t0 = time.perf_counter()

    rows: list[dict] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=args.maxtasksperchild) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs)):
            rows.append(r)
            if r.get("ok"):
                print(f"[{i+1}/{n_jobs}] seed={r['seed']:3d} ctrl={r['controller']:17s} "
                      f"OK collision={r['collision']} spill_max={r['spilled_max']} "
                      f"fling={r['right_ee_speed_max_ms']:.3f}m/s h_min={r['h_min']:+.3f} "
                      f"jerk_rms={r['jerk_rms_right_ee']:.1f} wall={r['wall_s']:.0f}s", flush=True)
            else:
                print(f"[{i+1}/{n_jobs}] seed={r['seed']:3d} ctrl={r['controller']:17s} "
                      f"FAILED: {r.get('err')}", flush=True)

    wall = time.perf_counter() - t0
    n_ok = sum(r.get("ok") for r in rows)
    n_bad = len(rows) - n_ok
    print(f"\n[comprehensive_eval] done: {n_ok}/{len(rows)} ok, {n_bad} failed, wall={wall/60:.1f}min", flush=True)
    if n_bad:
        print("failures (first 3):")
        for r in [r for r in rows if not r.get("ok")][:3]:
            print(f"  seed={r['seed']} ctrl={r['controller']}: {r.get('err')}")

    print_aggregate_table(rows, controllers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        seeds=seeds, controllers=controllers, max_steps=args.max_steps,
        n_jobs=n_jobs, wall_s=wall,
        rows=rows,
        aggregate={c: aggregate(rows, c) for c in controllers},
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[comprehensive_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
