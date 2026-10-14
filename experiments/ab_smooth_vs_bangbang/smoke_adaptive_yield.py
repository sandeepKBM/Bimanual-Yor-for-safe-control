"""SMOKE TEST (this task only, not the parent's full sweep): ADAPTIVE
COOPERATIVE YIELDING in mpc_fl.FLMPC (see that module's __init__ docstring
for the urgency/cup-share derivation) -- adaptive ON vs OFF on a handful of
HARD randomized seeds (cup_jitter_m=0.04, V_EE_MAX_MS=0.28 [already the
demo's default], 3 cubes -- shorter/harder crossing than the full 6-cube
episode), via scripts/make_cup_collect_safety_demo.py's --controller flmpc
path (build_episode/run_episode/summarize, imported as a library -- nothing
in that script's own CLI/video path is touched here).

Only runs the QP-FILTERED (qp_enabled=True) episode per seed/condition --
NOT the no-QP baseline main() also computes (irrelevant to collision rate)
-- and NEVER renders video (capture=False throughout), so this stays a
quick signal, not the full 50-seed sweep (parent owns that; see this task's
brief).

Run (GPU 5 only):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 \\
    /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
    experiments/ab_smooth_vs_bangbang/smoke_adaptive_yield.py \\
    [--seeds N] [--workers K] [--n-cubes 3] [--cup-jitter-m 0.04]
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_one(job):
    """Runs in its OWN process (spawn context, see main() below) -- thread
    pinning env vars are set here, BEFORE numpy/mujoco/numba get imported
    (spawn re-executes this module fresh in the child, so this is early
    enough), per the task's "pin OMP/etc=1" instruction for parallel runs."""
    seed, adaptive, cup_jitter_m, n_cubes = job
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    from scripts import make_cup_collect_safety_demo as demo

    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                               cup_jitter_m=cup_jitter_m, n_cubes=n_cubes)
    log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                               ee_speed_limit=True, cup_in_barrier=True, table_floor=True,
                               adaptive_yield=adaptive)
    m = demo.summarize(log)
    tilt = log["left_cup_tilt_rad"]
    idx_settle = int(np.searchsorted(log["t"], demo.RIGHT_START_T))
    idx_settle = min(idx_settle, len(tilt) - 1)
    ee_pos = log["left_ee_pos"]
    ee_disp_from_hold = np.linalg.norm(ee_pos[idx_settle:] - ee_pos[idx_settle], axis=1)
    left_ee_speed_max_ms = float(np.max(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1) / demo.sc.DT)) \
        if ee_pos.shape[0] > 1 else 0.0
    return dict(
        seed=seed, adaptive=bool(adaptive), n_cubes=n_cubes, cup_jitter_m=cup_jitter_m,
        collision=bool(m["collision"]),
        qp_infeasible_frac_left=m["qp_infeasible_frac_left"],
        qp_infeasible_frac_right=m["qp_infeasible_frac_right"],
        deposited_max=m["deposited_max"], deposited_final=m["deposited_final"],
        h_min=m["h_min"],
        left_ee_speed_max_ms=left_ee_speed_max_ms,
        right_ee_speed_max_ms=m["right_ee_speed_max_ms"],
        cup_tilt_max_rad_after_hold=float(np.max(tilt[idx_settle:])),
        cup_ee_disp_max_m_after_hold=float(np.max(ee_disp_from_hold)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=9, help="number of hard-seed episodes per condition")
    ap.add_argument("--seed0", type=int, default=1000, help="first seed (avoid seed=0, used by the demo's own easy-case default)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-cubes", type=int, default=3)
    ap.add_argument("--cup-jitter-m", type=float, default=0.04)
    args = ap.parse_args()

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    jobs = [(s, adaptive, args.cup_jitter_m, args.n_cubes)
            for adaptive in (False, True) for s in seeds]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(_run_one, jobs)

    by_cond = {False: [], True: []}
    for r in results:
        by_cond[r["adaptive"]].append(r)

    print(json.dumps(results, indent=2))
    print("\n=== SUMMARY (hard seeds: n_cubes=%d, cup_jitter_m=%.3f, seeds=%s) ===" %
          (args.n_cubes, args.cup_jitter_m, seeds))
    for adaptive in (False, True):
        rs = by_cond[adaptive]
        n = len(rs)
        coll_rate = sum(r["collision"] for r in rs) / n
        infeas_max = max(max(r["qp_infeasible_frac_left"], r["qp_infeasible_frac_right"]) for r in rs)
        infeas_mean = sum(max(r["qp_infeasible_frac_left"], r["qp_infeasible_frac_right"]) for r in rs) / n
        ee_speed_max = max(max(r["left_ee_speed_max_ms"], r["right_ee_speed_max_ms"]) for r in rs)
        tilt_max = max(r["cup_tilt_max_rad_after_hold"] for r in rs)
        disp_max = max(r["cup_ee_disp_max_m_after_hold"] for r in rs)
        print(f"adaptive={adaptive!s:5} n={n:2d}  collision_rate={coll_rate:.3f}  "
              f"qp_infeasible_frac(max/mean)={infeas_max:.3f}/{infeas_mean:.3f}  "
              f"worst_ee_speed_ms={ee_speed_max:.3f}  cup_tilt_max_rad={tilt_max:.4f}  "
              f"cup_ee_disp_max_m={disp_max:.4f}")


if __name__ == "__main__":
    main()
