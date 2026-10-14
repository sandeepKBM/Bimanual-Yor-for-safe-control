#!/usr/bin/env python
"""collect_ladder.py -- data collector for the WM-requirement spill-predictor
LADDER (experiments/wm_requirement/spill_predictors.py, task 2026-09-10:
"how powerful does the WM need to be?").

Runs the HARD/fast "crossing" episode -- scripts/make_cup_collect_safety_demo.py's
build_episode/run_episode/summarize, imported as a LIBRARY, nothing in that
script's own CLI/video path touched -- for each seed in --seeds A-B, and saves
the per-episode cup kinematics the ladder needs to a single .npz for
score_ladder.py.

HARD geometry (per the task brief; byte-identical to the pattern already
validated by experiments/ab_smooth_vs_bangbang/smoke_adaptive_yield.py, where
this exact call spills ~46/50 seeds -- positive labels are plentiful):
    demo.build_episode(seed, "crossing", ee_speed_limit=True,
                        cup_jitter_m=0.04, n_cubes=3)
    demo.run_episode(info, controller="flmpc", qp_enabled=True,
                      ee_speed_limit=True, table_floor=True)
(demo.V_EE_MAX_MS is already 0.28 by default -- no override needed.)

WHAT'S LOGGED, and how (read scripts/make_cup_collect_safety_demo.py's
run_episode loop before trusting this): run_episode calls mujoco.mj_step
EXACTLY ONCE per logged sample k (confirmed by reading the loop -- no inner
physics-substep loop), so the logging cadence IS the physics cadence and
dt below is exactly demo.sc.DT = 0.005s, not a substep-averaged value.
  - cup_n_axis (T,3): the cup body's own world +z axis
    (data.xmat[cup_body].reshape(3,3) @ [0,0,1]) -- added to run_episode by a
    small ADDITIVE edit (one new log dict key + one line in the existing
    per-step logging block; nothing renamed/removed/changed for any other
    caller of run_episode). log["cup_pos"] (data.xpos[cup_body]) already
    proved that reading data.*[cup_body] mid-loop is valid every step (the
    cup is kept kinematically re-pinned via scene_setup.reattach_cup_rigidly
    earlier in the loop), so data.xmat[cup_body] is equally valid here.
  - a_cup (T,3): WORLD LINEAR ACCELERATION of the cup, via CENTRAL FINITE
    DIFFERENCE of the already-logged log["cup_pos"] at dt=demo.sc.DT, with
    light position smoothing (5-tap moving average) before differencing
    (double-differentiation amplifies numerical/contact-solver noise);
    endpoints clamped to their nearest interior value (no extrapolation).
  - spill_onset_step: first step where log["spilled"] -- the PER-STEP
    scripts.cup_task_utils.count_spilled ball count, already logged every
    step by run_episode, not just the episode max -- rises above its
    initial value (usually 0); None if it never does in the episode.

Saved format (ragged episodes -> per-seed keys, NOT a single 3D array):
  seeds            : (N,) int array of the collected seeds
  dt               : scalar, the logging-cadence timestep [s]
  n_axis_<seed>     : (T,3) cup_n_axis
  a_cup_<seed>      : (T,3) cup world linear acceleration
  onset_<seed>      : scalar int, spill_onset_step or -1 if it never spilled
  T_<seed>          : scalar int, episode length
  spilled_max_<seed>: scalar int, peak ball-out count (0 if never spilled)

Run (GPU not needed -- no video/capture; threads pinned to 1 below):
  MUJOCO_GL=egl /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
    experiments/wm_requirement/collect_ladder.py \\
    --seeds 1000-1005 --out experiments/wm_requirement/traj_smoke.npz
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import make_cup_collect_safety_demo as demo  # noqa: E402  (build_episode/run_episode/summarize; read-only import)

# HARD/fast geometry (see module docstring) -- the parent's N=50 sweep and
# smoke_adaptive_yield.py's own smoke both use exactly this.
N_CUBES_HARD = 3
CUP_JITTER_M_HARD = 0.04


def _smooth(pos: np.ndarray, k: int) -> np.ndarray:
    """k-tap moving average per axis, edge-padded (output length == input
    length, no shrinkage). Identity if the trajectory is shorter than k."""
    if k <= 1 or pos.shape[0] <= k:
        return pos
    pad = k // 2
    kernel = np.ones(k) / k
    out = np.empty_like(pos)
    for d in range(pos.shape[1]):
        padded = np.pad(pos[:, d], (pad, pad), mode="edge")
        out[:, d] = np.convolve(padded, kernel, mode="valid")[: pos.shape[0]]
    return out


def finite_diff_accel(pos: np.ndarray, dt: float, smooth_window: int = 5) -> np.ndarray:
    """Central-difference WORLD linear acceleration (T,3) from a position
    trajectory (T,3): a[t] = (x[t+1] - 2x[t] + x[t-1]) / dt^2, computed on a
    lightly-smoothed copy of the position (smooth_window-tap moving average)
    to tame double-differentiation noise. Endpoints CLAMPED to their nearest
    interior value (a[0]=a[1], a[-1]=a[-2]) rather than extrapolated."""
    T = pos.shape[0]
    a = np.zeros_like(pos)
    if T < 3:
        return a
    pos_s = _smooth(pos, smooth_window)
    a[1:-1] = (pos_s[2:] - 2.0 * pos_s[1:-1] + pos_s[:-2]) / (dt * dt)
    a[0] = a[1]
    a[-1] = a[-2]
    return a


def spill_onset_step(spilled_per_step: np.ndarray, settle_steps: int = 100) -> int | None:
    """First step index where the per-step ball-out count rises above its
    POST-SETTLE baseline; None if it never rises. The baseline is taken AFTER a
    short settle window (settle_steps, ~0.5s at DT=0.005) because count_spilled
    can transiently read balls as "outside" during the initial grasp/settle
    before the pile beds into the cup (this is why an earlier version reported
    'onset never' for an episode that clearly lost 4 balls -- base at step 0 was
    already elevated). We take base = the MIN count over the settle window (the
    settled floor) and search only AFTER the window."""
    T = len(spilled_per_step)
    if T == 0:
        return None
    s = int(min(settle_steps, max(0, T - 1)))
    base = int(np.min(spilled_per_step[: s + 1])) if s > 0 else int(spilled_per_step[0])
    rose = np.where(spilled_per_step[s:] > base)[0]
    return int(rose[0] + s) if len(rose) else None


def collect_one(seed: int) -> dict:
    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                               cup_jitter_m=CUP_JITTER_M_HARD, n_cubes=N_CUBES_HARD)
    log, _frames = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                                     ee_speed_limit=True, table_floor=True)
    dt = float(demo.sc.DT)
    n_axis = log["cup_n_axis"]
    a_cup = finite_diff_accel(log["cup_pos"], dt)
    onset = spill_onset_step(log["spilled"])
    T = int(n_axis.shape[0])
    # Save the RAW per-step spill curve + raw cup position so spill-ONSET
    # labeling and accel smoothing can be re-tuned OFFLINE (episodes are ~5min;
    # never re-simulate just to change a label definition). Also the left-EE
    # cup tilt already logged, handy cross-check.
    return dict(seed=seed, n_axis=n_axis, a_cup=a_cup, onset=onset, T=T, dt=dt,
                spilled_max=int(np.max(log["spilled"])) if T else 0,
                spilled_curve=np.asarray(log["spilled"]).astype(np.int16),
                cup_pos=np.asarray(log["cup_pos"]),
                cup_tilt=np.asarray(log["left_cup_tilt_rad"]))


def _worker(seed: int) -> dict:
    """Pool worker: one full episode. Returns picklable arrays. maxtasksperchild=1
    below gives each episode a FRESH process (MuJoCo model/data + numba state are
    heavy and better not reused across episodes)."""
    t0 = time.perf_counter()
    ep = collect_one(seed)
    ep["wall_s"] = time.perf_counter() - t0
    ep["peak_a"] = float(np.max(np.linalg.norm(ep["a_cup"], axis=1))) if ep["T"] else 0.0
    return ep


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, required=True, help="inclusive range 'A-B' (e.g. 1000-1005)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel episodes (each ~5-9 min; episodes are T~5000 steps).")
    args = ap.parse_args()

    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    nw = max(1, min(args.workers, len(seeds)))

    payload = {}
    dt_ref = None
    n_spilled = 0
    done = 0
    # spawn (not fork): mujoco + numba + egl are not fork-safe under a warm parent.
    ctx = mp.get_context("spawn")
    with ctx.Pool(nw, maxtasksperchild=1) as pool:
        for ep in pool.imap_unordered(_worker, seeds):
            done += 1
            seed = ep["seed"]
            dt_ref = ep["dt"]
            spilled = ep["spilled_max"] > 0
            n_spilled += int(spilled)
            onset_str = str(ep["onset"]) if ep["onset"] is not None else "never"
            print(f"[{done}/{len(seeds)} seed {seed:5d}] T={ep['T']:4d}  spilled={str(spilled):5s} "
                  f"(balls out={ep['spilled_max']})  onset_step={onset_str:>6s}  "
                  f"peak|a_cup|={ep['peak_a']:6.2f} m/s^2  ({ep['wall_s']:.0f}s)", flush=True)
            payload[f"n_axis_{seed}"] = ep["n_axis"]
            payload[f"a_cup_{seed}"] = ep["a_cup"]
            payload[f"onset_{seed}"] = np.array(ep["onset"] if ep["onset"] is not None else -1)
            payload[f"T_{seed}"] = np.array(ep["T"])
            payload[f"spilled_max_{seed}"] = np.array(ep["spilled_max"])
            payload[f"spilled_curve_{seed}"] = ep["spilled_curve"]
            payload[f"cup_pos_{seed}"] = ep["cup_pos"]
            payload[f"cup_tilt_{seed}"] = ep["cup_tilt"]

    payload["seeds"] = np.array(seeds, dtype=int)
    payload["dt"] = np.array(dt_ref)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"\nsaved {len(seeds)} episodes -> {out_path}  ({n_spilled}/{len(seeds)} spilled)", flush=True)


if __name__ == "__main__":
    main()
