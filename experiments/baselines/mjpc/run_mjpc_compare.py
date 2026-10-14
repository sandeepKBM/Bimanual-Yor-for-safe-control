"""experiments/baselines/mjpc/run_mjpc_compare.py

Head-to-head: FL-MPC (ours, mpc_fl.FLMPC) vs a tuned REAL MuJoCo MPC (MJPC,
google-deepmind/mujoco_mpc; C++ planner server, see cpp/ + mjpc_planner.py) on
the SAME cup-collect crossing episode, SIMULATION ONLY.

Both controllers run through the UNMODIFIED
scripts/make_cup_collect_safety_demo.py build_episode/run_episode(controller=
"flmpc") loop (same comprehensive_eval setup: N_CUBES=3, cup_jitter 0.04,
V_EE_MAX 0.28, faked grasp, table_floor on). For "mjpc" the demo module's
`mpc_fl` binding is swapped IN MEMORY for a namespace whose FLMPC is
mjpc_planner.MJPCController (no file edited) -- so lift phase, cup re-pin,
weld/release, torque filter/clip and all logging are byte-identical; only the
planner producing the 12 torques differs.

Usage:
  python run_mjpc_compare.py --seeds 0,2,5 --controllers mjpc,flmpc --workers 6 \
      --mjpc-threads 8 --out results.json [--max-steps N] [--weights '{"Clear":1e5}']
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"),
           str(REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"), str(REPO_ROOT / "eval"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 0.005
DEFAULT_TMP = os.environ.get("MJPC_TMP", str(Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "mjpc"))


def _jerk_rms(ee_pos, dt=DT):
    # identical to comprehensive_eval.jerk_rms (copied to avoid importing that
    # module mid-edit by a parallel agent; the definition is 4 lines)
    if ee_pos.shape[0] < 4:
        return 0.0
    vel = np.diff(ee_pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))


def run_one(args: dict) -> dict:
    seed, ctrl = args["seed"], args["controller"]
    t_wall0 = time.perf_counter()
    try:
        import make_cup_collect_safety_demo as demo
        import sim_common as sc
        import mjpc_planner
        demo.N_CUBES = 3
        demo.V_EE_MAX_MS = 0.28
        demo.FL_W_JERK_OVERRIDE = None
        demo.FL_W_POS_INT_OVERRIDE = None
        sc.D_MIN = 0.03
        sc.ALPHA1 = sc.ALPHA2 = 30.0
        sc.COMBINED_RADIUS = sc.R_EGO + sc.R_INT + sc.D_MIN
        cup_jitter = float(args.get("cup_jitter", 0.04))

        info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                                  cup_jitter_m=cup_jitter, n_cubes=3, grasp_mode="faked")
        model, data = info["model"], info["data"]
        cube_init = {c["name"]: data.xpos[c["body_id"]].copy() for c in info["cube_plan"]}

        orig_mpc_fl = demo.mpc_fl
        if ctrl == "mjpc":
            cfg = dict(args.get("mjpc_cfg") or {})
            cfg.setdefault("tmp_dir", DEFAULT_TMP)
            cfg["seed"] = seed
            mjpc_planner.CONTEXT.clear()
            mjpc_planner.CONTEXT.update(info=info, cfg=cfg)
            demo.mpc_fl = types.SimpleNamespace(FLMPC=mjpc_planner.MJPCController)
        try:
            t_ep0 = time.perf_counter()
            log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                                      ee_speed_limit=True, table_floor=True, max_steps=args.get("max_steps"))
            ep_wall = time.perf_counter() - t_ep0
        finally:
            demo.mpc_fl = orig_mpc_fl
            inst = mjpc_planner.CONTEXT.get("instance") if ctrl == "mjpc" else None
            if inst is not None:
                inst.close()
        m = demo.summarize(log)

        n = len(log["t"])
        idx_cross = int(np.searchsorted(log["t"], demo.RIGHT_START_T))
        cross = slice(idx_cross, n)
        solve_ms = 1e3 * np.asarray(log["solve_time"])[cross]
        # moved-and-placed: a SELECTED cube, resting in the keep zone at the end, that
        # actually moved (> 3 cm) from its start (placed_final also counts cube_3,
        # which starts in the zone).
        place_c = np.mean(np.stack([c["place_xyz"][:2] for c in info["cube_plan"]]), axis=0)
        moved_placed = 0
        for c in info["cube_plan"]:
            p = data.xpos[c["body_id"]]
            in_zone = np.hypot(*(p[:2] - place_c)) < demo.KEEP_ZONE_RADIUS_M
            resting = abs(p[2] - (info["table_top_z"] + demo.cct.CUBE_HALF)) < 0.05
            moved = np.linalg.norm(p - cube_init[c["name"]]) > 0.03
            moved_placed += int(in_zone and resting and moved)
        right_ee = np.asarray(log["right_ee_pos"])
        left_ee = np.asarray(log["left_ee_pos"])
        # task-tracking parity: right-arm joint error vs the scripted track_r reference
        tr = info["track_r"]
        q_ref = np.stack([tr.sample(tt) for tt in log["right_track_t"][cross]]) if n > idx_cross else np.zeros((0, 6))
        q_err = np.linalg.norm(np.asarray(log["right_q"])[cross] - q_ref, axis=1) if q_ref.size else np.zeros(1)
        row = dict(
            seed=seed, controller=ctrl, ok=True, n_steps=n,
            collision=bool(m["collision"]), collision_steps=int(np.sum(log["collision"] > 0.5)),
            h_min=float(m["h_min"]), dist_min=float(m["dist_min"]),
            right_cup_clearance_min_m=float(m["right_cup_clearance_min_m"]),
            gripper_tilt_max_deg=float(np.degrees(np.max(log["left_cup_tilt_rad"][cross]))),
            cup_tilt_max_deg=float(np.degrees(np.max(log["true_cup_tilt_rad"][cross]))),
            cup_tilt_rel_max_deg=float(np.degrees(np.max(np.arccos(np.clip(
                np.asarray(log["cup_n_axis"])[cross] @ np.asarray(log["cup_n_axis"])[idx_cross], -1, 1)))))
            if n > idx_cross else 0.0,
            tilt_over_5p73_frac=float(np.mean(log["left_cup_tilt_rad"][cross] > 0.10)),
            jerk_rms_right_ee=_jerk_rms(right_ee), jerk_rms_left_ee=_jerk_rms(left_ee),
            right_ee_speed_max_ms=float(m["right_ee_speed_max_ms"]),
            placed_final=int(m["placed_final"]), moved_and_placed=int(moved_placed),
            right_track_err_rms_rad=float(np.sqrt(np.mean(q_err ** 2))),
            right_track_err_max_rad=float(np.max(q_err)),
            left_ee_excursion_max_m=float(np.max(np.linalg.norm(left_ee[cross] - left_ee[idx_cross], axis=1)))
            if n > idx_cross else 0.0,
            spilled_max=int(m["spilled_max"]),
            table_min_z_margin_m=float(min(m["left_min_arm_z_m"], m["right_min_arm_z_m"]) - info["table_top_z"]),
            solve_ms_mean=float(np.mean(solve_ms)) if solve_ms.size else float("nan"),
            solve_ms_p95=float(np.percentile(solve_ms, 95)) if solve_ms.size else float("nan"),
            solve_ms_max=float(np.max(solve_ms)) if solve_ms.size else float("nan"),
            rtf_controller=float(DT / max(np.mean(solve_ms) / 1e3, 1e-9)) if solve_ms.size else float("nan"),
            episode_wall_s=float(ep_wall), sim_s=float(n * DT),
            rtf_episode=float(n * DT / ep_wall),
            qp_infeasible_frac=float(max(m["qp_infeasible_frac_left"], m["qp_infeasible_frac_right"])),
            cup_jitter=cup_jitter,
        )
        if ctrl == "mjpc":
            inst = mjpc_planner.CONTEXT.get("instance")
            row["mjpc_plan_ms_mean"] = float(np.mean(inst.plan_ms))
            row["mjpc_threads"] = int(cfg.get("threads", 8))
            row["mjpc_cfg"] = {k: v for k, v in cfg.items() if k != "tmp_dir"}
        if args.get("dump"):
            extra = {}
            if ctrl == "mjpc":
                extra = dict(plan_ms=np.asarray(inst.plan_ms), best_cost=np.asarray(inst.best_cost))
            np.savez_compressed(args["dump"], q_ref=q_ref, idx_cross=idx_cross,
                                **{k: np.asarray(v) for k, v in log.items() if isinstance(v, np.ndarray)}, **extra)
        row["wall_s"] = time.perf_counter() - t_wall0
        return row
    except Exception as e:
        return dict(seed=seed, controller=ctrl, ok=False, err=str(e)[:300],
                    tb=traceback.format_exc()[-1500:], wall_s=time.perf_counter() - t_wall0)


COLS = [("collision", "coll", "{!s:>5}"), ("h_min", "h_min", "{:+.3f}"), ("dist_min", "dmin", "{:.3f}"),
        ("right_cup_clearance_min_m", "cupclr", "{:.3f}"), ("gripper_tilt_max_deg", "gtilt", "{:5.2f}"),
        ("cup_tilt_max_deg", "ctilt", "{:5.2f}"), ("cup_tilt_rel_max_deg", "ctRel", "{:5.2f}"), ("jerk_rms_right_ee", "jerkR", "{:7.1f}"),
        ("moved_and_placed", "mvpl", "{:d}"), ("right_track_err_rms_rad", "trkE", "{:.3f}"), ("placed_final", "plf", "{:d}"),
        ("solve_ms_mean", "ms", "{:6.2f}"), ("solve_ms_p95", "ms95", "{:6.2f}"),
        ("rtf_controller", "rtfC", "{:5.2f}"), ("rtf_episode", "rtfE", "{:5.2f}")]


def print_table(rows):
    print(f"{'ctrl':6s} {'seed':>4s} " + " ".join(f"{h:>7s}" for _, h, _ in COLS))
    for r in sorted(rows, key=lambda r: (r["controller"], r["seed"])):
        if not r.get("ok"):
            print(f"{r['controller']:6s} {r['seed']:4d} FAILED {r.get('err')}")
            continue
        print(f"{r['controller']:6s} {r['seed']:4d} " + " ".join(f"{f.format(r[k]):>7s}" for k, _, f in COLS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--controllers", default="mjpc,flmpc")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--cup-jitter", type=float, default=0.04)
    ap.add_argument("--mjpc-threads", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=1)
    ap.add_argument("--act-mode", default="pd", choices=("torque", "pd"))
    ap.add_argument("--plan-every", type=int, default=1)
    ap.add_argument("--weights", default=None, help="JSON dict of residual weight overrides")
    ap.add_argument("--params", default=None, help="JSON dict of residual_* param overrides")
    ap.add_argument("--agent", default=None, help="JSON dict of agent/planner numeric overrides")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-dir", default=None, help="save per-episode log npz here")
    a = ap.parse_args()
    seeds = ([int(x) for x in a.seeds.split(",")] if "-" not in a.seeds
             else list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)))
    cfg = dict(threads=a.mjpc_threads, n_iter=a.n_iter, plan_every=a.plan_every, act_mode=a.act_mode,
               weights=json.loads(a.weights) if a.weights else None,
               params=json.loads(a.params) if a.params else None,
               agent=json.loads(a.agent) if a.agent else None)
    jobs = [dict(seed=s, controller=c, max_steps=a.max_steps, cup_jitter=a.cup_jitter,
                 mjpc_cfg=cfg if c == "mjpc" else None,
                 dump=(str(Path(a.dump_dir) / f"log_{c}_s{s}.npz") if a.dump_dir else None))
            for s in seeds for c in a.controllers.split(",")]
    rows = []
    if a.workers <= 1:
        for j in jobs:
            r = run_one(j)
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "tb"}), flush=True)
            if not r.get("ok"):
                print(r.get("tb"), flush=True)
    else:
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                rows.append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "tb"}), flush=True)
                if not r.get("ok"):
                    print(r.get("tb"), flush=True)
    print_table(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
