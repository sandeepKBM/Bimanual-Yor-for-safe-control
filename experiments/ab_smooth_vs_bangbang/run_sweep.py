"""LEG A distributional sweep: turns the single-geometry A/v4-C anecdote
(run_ab4.py/run_ab7.py) into an aggregate result over N randomized scenes.

Reuses the existing physics/controllers UNCHANGED (sim_common.py's
_patch_real_collision_geoms / all_pair_barriers / min_pair_distance /
HOCBF-QP, run_ab4.run_condition, run_ab4.compute_metrics). This file adds
ONLY: (1) a reproducible scene sampler that perturbs the scene knobs
run_ab4/sim_common already expose as module-level constants (LEFT_GRASP_Q,
RIGHT_HOME_CTRL, RIGHT_INTRUDE_CTRL), (2) a parallel driver that reruns the
existing single-scene machinery once per (scene, controller), (3) aggregation
+ CSV/JSON/markdown/PNG output.

"The scene" = (a) the LEFT (ego) arm's grasp/start joint configuration
(LEFT_GRASP_Q -- also implicitly sets the box's initial position/orientation,
since sim_common.init_grasped_state() teleports the box to the midpoint of
the OPEN gripper pads at whatever LEFT_GRASP_Q is, then closes the fingers
-- so perturbing LEFT_GRASP_Q perturbs the object placement too, for free,
with no separate box-placement code path needed); (b) the RIGHT (intruder)
arm's home pose (RIGHT_HOME_CTRL) and deepest-intrusion target
(RIGHT_INTRUDE_CTRL), i.e. the crossing geometry/depth. All three are plain
np.ndarray module attributes read by bare name at CALL time inside
sim_common.init_grasped_state / sim_common.intruder_ctrl /
run_ab4.intruder_ctrl_nominal -- so monkeypatching sc.LEFT_GRASP_Q /
sc.RIGHT_HOME_CTRL / sc.RIGHT_INTRUDE_CTRL before calling
run_ab4.run_condition(condition) changes the scene with ZERO changes to
sim_common.py/run_ab4.py and zero risk of touching the QP/physics.

Run (measured ~57s/scene for A, ~87s/scene for v4-C single-threaded -- NOT
"seconds/scene", so this driver parallelizes across scenes via a process
pool; 50 scenes x 2 controllers x ~72s serial-equivalent / 24 workers ~= 5-6
wall-clock minutes on this 72-core box):

    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \\
        python run_sweep.py --n-scenes 50 --workers 24

Writes artifacts/leg_a_sweep_results.csv (per scene x controller row),
artifacts/leg_a_sweep_summary.json + .md (aggregate table), and
artifacts/leg_a_sweep.png (summary plot).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Imported in the PARENT process only to read the canonical
# INTRUDER_EXTRA_MARGIN/INTRUDER_MARGIN_SCHEDULE defaults when resolving each
# task's explicit margin below (see _resolve_margin_value). The parent's own
# copy of ab4.INTRUDER_EXTRA_MARGIN is never mutated -- only a worker's own
# freshly re-imported copy is (spawn context, see _run_one) -- so this stays
# the canonical default for the lifetime of the sweep regardless of what
# --coop-margins overrides run in worker subprocesses.
import run_ab4 as ab4

ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

CONTROLLER_CODE = {"A": "A", "B": "B", "v4-C": "C", "C-adapt": "C-adapt",
                    "C-adapt-relax": "C-adapt"}  # display name -> run_ab4.run_condition() arg

# controllers that get run_ab4.run_condition(..., ego_relax_on_infeasible=True)
# -- the bounded ego-QP constraint-relaxation fallback (see run_ab4.py's
# _solve_ego_qp_with_relaxation docstring), orthogonal to which margin
# scheme (static/adaptive) the controller otherwise uses.
RELAX_CONTROLLERS = {"C-adapt-relax"}

# controller display name -> extra-margin override applied to
# ab4.INTRUDER_EXTRA_MARGIN["C"] for that (scene, controller) run only,
# used by the --coop-margins static-margin grid (task 3). Populated in
# main() from --coop-margins; left empty for a bare --controllers run so
# existing A/B/v4-C/C-adapt/C-adapt-relax behavior is untouched.
MARGIN_OVERRIDE: dict[str, float] = {}


def _resolve_margin_value(controller: str, code: str):
    """Resolves, in the PARENT process, the exact INTRUDER_EXTRA_MARGIN
    value a task's worker must apply for controller `code` -- for EVERY
    task, not just --coop-margins-overridden ones. This closes a state-leak
    bug: ProcessPoolExecutor REUSES worker processes across many (scene,
    controller) tasks, so a worker that had previously mutated its own
    ab4.INTRUDER_EXTRA_MARGIN["C"] (e.g. to run a 'v4-C-m0.06' task) left
    that mutation in place for the NEXT task it happened to pick up in the
    same process -- if that next task was a bare 'v4-C' run, it silently
    inherited the previous task's margin instead of the canonical default.
    Making every task carry its own fully-resolved value (used
    unconditionally by the worker, see _run_one) means no task's behavior
    can depend on what a prior task did to that process's module globals.

    Returns None for codes that don't consult INTRUDER_EXTRA_MARGIN at all
    (currently only "C-adapt"/"C-adapt-relax", whose code is "C-adapt" and
    which use INTRUDER_MARGIN_SCHEDULE's per-pair schedule fn instead --
    forcing a value there would be a harmless no-op, so skip the needless
    work). For every other code (A, B, C) returns the --coop-margins
    override for this controller if one was given, else ab4's own
    (unmutated, canonical) default for that code.
    """
    if code in ab4.INTRUDER_MARGIN_SCHEDULE:
        return None
    return MARGIN_OVERRIDE.get(controller, ab4.INTRUDER_EXTRA_MARGIN.get(code, 0.0))

# ---------------------------------------------------------------------------
# Scene-sampling knobs. Perturbation magnitudes chosen SMALL relative to the
# base v4 geometry (probe_v3.py's found LEFT_GRASP_Q / RIGHT_HOME_CTRL /
# RIGHT_INTRUDE_CTRL) so scenes stay "the same qualitative crossing task,
# randomized" rather than drifting into geometry where the arms no longer
# meaningfully approach each other (which would make every controller look
# trivially perfect for the wrong reason -- no real near-collision test).
# ---------------------------------------------------------------------------
D_LEFT_GRASP = 0.12       # [rad] per-joint jitter on the ego grasp/start config (+ box placement)
D_RIGHT_HOME = 0.05       # [rad] per-joint jitter on the intruder's home/ready pose
D_RIGHT_LATERAL = 0.12    # [rad] per-joint jitter on the intrusion TARGET (crossing geometry)
DEPTH_SCALE_RANGE = (0.70, 1.15)   # multiplies the home->intrude vector (crossing depth)
JOINT_CLIP_MARGIN = 0.03  # [rad] stay this far inside the hard joint limits
ACCEPT_H_THRESH = 0.06    # [m] worst-pair h at the intended full-intrusion pose must be below
                           # this (a real near-collision must be geometrically reachable) or resample
MAX_RESAMPLE_TRIES = 40


def _clip_to_ranges(q, ranges, margin=JOINT_CLIP_MARGIN):
    lo = ranges[:, 0] + margin
    hi = ranges[:, 1] - margin
    return np.clip(q, lo, hi)


def _build_probe():
    import sim_common as sc
    import mujoco
    model = sc.load_model()
    data = mujoco.MjData(model)
    _, left_qposadr, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, _ = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    return sc, mujoco, model, data, left_qposadr, right_qposadr


def sample_scenes(n_scenes: int, seed: int) -> list[dict]:
    """Reproducible scene sampler. Each scene gets its own seeded RNG derived
    from (seed, scene_idx) via np.random.SeedSequence, so scenes are
    independent of n_scenes/resample-count and the WHOLE sweep is
    reproducible from a single --seed. Rejection-samples (up to
    MAX_RESAMPLE_TRIES) against a cheap forward-kinematics-only check
    (sc.min_pair_distance on a throwaway probe model/data, NO physics
    stepping) so every accepted scene has a genuinely reachable near-collision
    at its intended full-intrusion pose."""
    sc, mujoco, model, data, left_qposadr, right_qposadr = _build_probe()
    joint_ranges = sc.LEFT_JOINT_RANGES  # identical numeric ranges for both arms (verified via mjcf)

    scenes = []
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_scenes)
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        accepted = None
        for attempt in range(MAX_RESAMPLE_TRIES):
            left_grasp_q = _clip_to_ranges(
                sc.LEFT_GRASP_Q + rng.uniform(-D_LEFT_GRASP, D_LEFT_GRASP, 6), joint_ranges)
            right_home_ctrl = _clip_to_ranges(
                sc.RIGHT_HOME_CTRL + rng.uniform(-D_RIGHT_HOME, D_RIGHT_HOME, 6), joint_ranges)
            depth_scale = float(rng.uniform(*DEPTH_SCALE_RANGE))
            lateral = rng.uniform(-D_RIGHT_LATERAL, D_RIGHT_LATERAL, 6)
            right_intrude_ctrl = _clip_to_ranges(
                right_home_ctrl + depth_scale * (sc.RIGHT_INTRUDE_CTRL - sc.RIGHT_HOME_CTRL) + lateral,
                joint_ranges)

            data.qpos[left_qposadr] = left_grasp_q
            data.qpos[right_qposadr] = right_intrude_ctrl
            mujoco.mj_forward(model, data)
            h_min, dist_min, *_ = sc.min_pair_distance(model, data)

            candidate = dict(
                scene_idx=i, seed=int(cs.generate_state(1)[0]), attempt=attempt,
                left_grasp_q=left_grasp_q.tolist(),
                right_home_ctrl=right_home_ctrl.tolist(),
                right_intrude_ctrl=right_intrude_ctrl.tolist(),
                depth_scale=depth_scale,
                h_min_at_intended_intrusion=float(h_min),
                dist_min_at_intended_intrusion=float(dist_min),
                valid_crossing=bool(h_min < ACCEPT_H_THRESH),
            )
            if h_min < ACCEPT_H_THRESH:
                accepted = candidate
                break
            accepted = candidate  # keep last candidate as fallback
        scenes.append(accepted)
    return scenes


# ---------------------------------------------------------------------------
# Worker: one (scene, controller) run, reusing run_ab4.run_condition /
# compute_metrics verbatim. Runs in a subprocess (spawn context -- safer with
# mujoco's native state than fork) so LEFT_GRASP_Q/RIGHT_HOME_CTRL/
# RIGHT_INTRUDE_CTRL monkeypatches and run_ab4.LOG_DIR redirection never leak
# across concurrent scenes.
# ---------------------------------------------------------------------------

def _run_one(task: dict) -> dict:
    scene = task["scene"]
    controller = task["controller"]
    # `code` is resolved in the PARENT process (run_sweep(), below) and
    # carried in the task dict rather than looked up here via CONTROLLER_CODE
    # -- workers are spawned (mp.get_context("spawn")) as fresh interpreters
    # that re-import this module, so any --coop-margins-driven entries added
    # to CONTROLLER_CODE at runtime in main() would NOT be visible via a
    # module-global lookup inside a worker process.
    code = task["code"]
    out = dict(scene_idx=scene["scene_idx"], seed=scene["seed"], controller=controller,
               margin_override=task.get("margin_override"), ego_relax=bool(task.get("ego_relax", False)),
               error=None)
    try:
        sys.path.insert(0, HERE) if HERE not in sys.path else None
        import numpy as _np
        import sim_common as sc
        import run_ab4 as ab4

        sc.LEFT_GRASP_Q = _np.array(scene["left_grasp_q"])
        sc.RIGHT_HOME_CTRL = _np.array(scene["right_home_ctrl"])
        sc.RIGHT_INTRUDE_CTRL = _np.array(scene["right_intrude_ctrl"])

        # Explicit per-task margin (fixes a state-leak: ProcessPoolExecutor
        # REUSES worker processes across many tasks, so a worker that
        # previously mutated its own ab4.INTRUDER_EXTRA_MARGIN[code] for one
        # task -- e.g. a 'v4-C-m0.06' override -- would leave that mutation
        # in place for the NEXT task it happened to run in the same process,
        # silently contaminating a later bare 'v4-C' run with the wrong
        # margin). `margin_value` is resolved ONCE in the parent
        # (_resolve_margin_value, using the parent's own never-mutated
        # ab4.INTRUDER_EXTRA_MARGIN as the canonical default) for EVERY task
        # -- not just --coop-margins ones -- and applied here
        # UNCONDITIONALLY-per-task (not gated on "was an override
        # requested"), so this worker's INTRUDER_EXTRA_MARGIN[code] is always
        # set fresh from this task's own resolved value right before running
        # it, never left over from whatever the previous task in this same
        # process set it to. None only for codes that don't consult this
        # dict at all (C-adapt/C-adapt-relax; see _resolve_margin_value).
        # This subprocess's ab4.INTRUDER_EXTRA_MARGIN copy is otherwise
        # independent of every other worker's (spawn context, fresh import
        # per process), so this mutation never crosses PROCESS boundaries --
        # only within-process task-to-task reuse was ever a problem.
        margin_value = task.get("margin_value")
        if margin_value is not None:
            ab4.INTRUDER_EXTRA_MARGIN[code] = float(margin_value)
        ego_relax = bool(task.get("ego_relax", False))

        with tempfile.TemporaryDirectory(prefix=f"leg_a_sweep_s{scene['scene_idx']}_{code}_") as tmpdir:
            ab4.LOG_DIR = tmpdir
            t0 = time.time()
            log = ab4.run_condition(code, ego_relax_on_infeasible=ego_relax)
            wall_s = time.time() - t0
        m = ab4.compute_metrics(controller, log)

        out.update(m)
        out["controller"] = controller  # compute_metrics stamps its own arg back in; keep display name
        out["wall_time_s"] = wall_s
        out["step_wall_time_mean_ms"] = wall_s / sc.N_STEPS * 1000.0
        out["valid_crossing"] = scene["valid_crossing"]
        out["h_min_at_intended_intrusion"] = scene["h_min_at_intended_intrusion"]
        out["depth_scale"] = scene["depth_scale"]
    except Exception as e:  # keep sweeping through the other (scene, controller) tasks
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    return out


def run_sweep(n_scenes: int, seed: int, controllers: list[str], workers: int) -> tuple[list[dict], list[dict]]:
    scenes = sample_scenes(n_scenes, seed)
    n_invalid = sum(1 for s in scenes if not s["valid_crossing"])
    print(f"Sampled {n_scenes} scenes (seed={seed}); "
          f"{n_scenes - n_invalid}/{n_scenes} passed the near-collision reachability check "
          f"(h_min < {ACCEPT_H_THRESH} m at intended full intrusion), "
          f"{n_invalid} kept anyway as shallow-crossing fallbacks (flagged valid_crossing=False).")

    tasks = [dict(scene=s, controller=c, code=CONTROLLER_CODE[c], margin_override=MARGIN_OVERRIDE.get(c),
                   margin_value=_resolve_margin_value(c, CONTROLLER_CODE[c]),
                   ego_relax=c in RELAX_CONTROLLERS)
             for s in scenes for c in controllers]
    results = []
    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            status = "ERROR" if r["error"] else ("COLLISION" if r.get("collision") else "ok")
            print(f"  [{done}/{len(tasks)}] scene={r['scene_idx']:3d} controller={r['controller']:5s} "
                  f"{status}  ({time.time() - t0:6.1f}s elapsed)")
    return scenes, results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _pctl(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def add_derived_fields(results: list[dict], success_threshold: float) -> list[dict]:
    """Adds intruder_progress_frac (normalized to THAT SCENE's own A-condition
    achieved intrusion depth, same formula run_ab4.py/run_ab7.py already use)
    and task_success = progress>=threshold AND grasp_retained AND not collision,
    with the 3 sub-conditions kept as separate columns so failures are
    attributable."""
    by_scene_A = {}
    for r in results:
        if r["error"] is None and r["controller"] == "A":
            by_scene_A[r["scene_idx"]] = r

    for r in results:
        r["grasp_retained"] = (not r["box_dropped"]) if r["error"] is None else None
        r["no_collision"] = (not r["collision"]) if r["error"] is None else None
        a_ref = by_scene_A.get(r["scene_idx"])
        if r["error"] is None and a_ref is not None:
            home = a_ref["home_ee_gap"]
            adv_min = a_ref["min_ee_gap"]
            denom = max(home - adv_min, 1e-6)
            prog = (home - r["min_ee_gap"]) / denom
            r["intruder_progress_frac"] = float(prog)
            r["progress_ok"] = bool(prog >= success_threshold)
            r["task_success"] = bool(r["progress_ok"] and r["grasp_retained"] and r["no_collision"])
        else:
            r["intruder_progress_frac"] = None
            r["progress_ok"] = None
            r["task_success"] = None
    return results


def aggregate(results: list[dict], controllers: list[str]) -> dict:
    summary = {}
    for c in controllers:
        rows = [r for r in results if r["controller"] == c]
        ok_rows = [r for r in rows if r["error"] is None]
        n_err = len(rows) - len(ok_rows)
        scored = [r for r in ok_rows if r["task_success"] is not None]

        def dist(field):
            vals = np.array([r[field] for r in ok_rows if r.get(field) is not None
                              and np.isfinite(r[field])], dtype=float)
            if len(vals) == 0:
                return dict(mean=None, std=None, median=None, p10=None, p90=None, n=0)
            return dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        median=float(np.median(vals)), p10=_pctl(vals, 10),
                        p90=_pctl(vals, 90), n=int(len(vals)))

        summary[c] = dict(
            n_scenes_run=len(rows),
            n_errored=n_err,
            collision_rate=float(np.mean([r["collision"] for r in ok_rows])) if ok_rows else None,
            grasp_retention_rate=float(np.mean([r["grasp_retained"] for r in ok_rows])) if ok_rows else None,
            progress_ok_rate=(float(np.mean([r["progress_ok"] for r in scored])) if scored else None),
            task_success_rate=(float(np.mean([r["task_success"] for r in scored])) if scored else None),
            n_scored_for_success=len(scored),
            chatter_hdot_sign_changes_per_s=dist("hdot_sign_changes_per_s"),
            min_ee_gap=dist("min_ee_gap"),
            box_drift_trans_max=dist("box_drift_trans_max"),
            int_track_err_mean=dist("int_track_err_mean"),
            step_wall_time_mean_ms=dist("step_wall_time_mean_ms"),
            qp_infeasible_frac_ego=dist("qp_infeasible_frac_ego"),
        )
    return summary


def write_markdown_table(summary: dict, controllers: list[str], path: str, n_scenes: int, seed: int,
                          success_threshold: float, n_valid_crossing: int):
    lines = []
    lines.append("# LEG A distributional sweep -- summary\n")
    lines.append(f"N scenes = {n_scenes} (seed={seed}), "
                 f"{n_valid_crossing}/{n_scenes} passed the near-collision reachability check.\n")
    lines.append(f"TASK SUCCESS := intruder_progress_frac >= {success_threshold} "
                 f"AND grasp_retained (box_dropped==False, i.e. drift < 0.05m and no table-level drop) "
                 f"AND no_collision (min pair-barrier h never went < 0 / a real contact never occurred).\n")
    header = "| metric | " + " | ".join(controllers) + " |"
    sep = "|---|" + "---|" * len(controllers)
    lines.append(header)
    lines.append(sep)

    def row(label, getter):
        vals = [getter(summary[c]) for c in controllers]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    def fmt(x, pct=False):
        if x is None:
            return "n/a"
        return f"{x*100:.1f}%" if pct else f"{x:.4f}"

    row("n scenes run", lambda s: str(s["n_scenes_run"]))
    row("n errored", lambda s: str(s["n_errored"]))
    row("collision rate", lambda s: fmt(s["collision_rate"], pct=True))
    row("grasp-retention rate", lambda s: fmt(s["grasp_retention_rate"], pct=True))
    row("progress>=thresh rate", lambda s: fmt(s["progress_ok_rate"], pct=True))
    row("TASK SUCCESS rate", lambda s: fmt(s["task_success_rate"], pct=True))
    row("chatter median (hdot sign chg/s)", lambda s: fmt(s["chatter_hdot_sign_changes_per_s"]["median"]))
    row("chatter mean +/- std", lambda s: (f"{s['chatter_hdot_sign_changes_per_s']['mean']:.2f} +/- "
                                            f"{s['chatter_hdot_sign_changes_per_s']['std']:.2f}"
                                            if s["chatter_hdot_sign_changes_per_s"]["mean"] is not None else "n/a"))
    row("min_ee_gap median (m)", lambda s: fmt(s["min_ee_gap"]["median"]))
    row("min_ee_gap p10 (m)", lambda s: fmt(s["min_ee_gap"]["p10"]))
    row("box_drift_trans_max median (m)", lambda s: fmt(s["box_drift_trans_max"]["median"]))
    row("int_track_err_mean median (rad)", lambda s: fmt(s["int_track_err_mean"]["median"]))
    row("step wall time median (ms)", lambda s: fmt(s["step_wall_time_mean_ms"]["median"]))
    row("qp_infeasible_frac_ego mean", lambda s: fmt(s["qp_infeasible_frac_ego"]["mean"]))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_plot(results: list[dict], controllers: list[str], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"A": "#c0392b", "v4-C": "#1f77b4", "MPC-v7": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    def rows_for(c):
        return [r for r in results if r["controller"] == c and r["error"] is None]

    # (0,0) collision rate + task success rate bars
    ax = axes[0, 0]
    x = np.arange(len(controllers))
    coll = [np.mean([r["collision"] for r in rows_for(c)]) if rows_for(c) else np.nan for c in controllers]
    succ = [np.mean([r["task_success"] for r in rows_for(c) if r["task_success"] is not None])
            if any(r["task_success"] is not None for r in rows_for(c)) else np.nan for c in controllers]
    w = 0.35
    ax.bar(x - w / 2, [v * 100 for v in coll], width=w, label="collision rate", color="#c0392b")
    ax.bar(x + w / 2, [v * 100 for v in succ], width=w, label="task success rate", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(controllers)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Collision rate vs. task-success rate")
    ax.legend(fontsize=8)

    # (0,1) grasp retention rate bar
    ax = axes[0, 1]
    grasp = [np.mean([r["grasp_retained"] for r in rows_for(c)]) * 100 if rows_for(c) else np.nan
             for c in controllers]
    ax.bar(x, grasp, color=[colors.get(c, "#888") for c in controllers])
    ax.set_xticks(x); ax.set_xticklabels(controllers)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Grasp-retention rate")

    # (1,0) chatter violin
    ax = axes[1, 0]
    data = [[r["hdot_sign_changes_per_s"] for r in rows_for(c)] for c in controllers]
    data = [d if d else [np.nan] for d in data]
    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(np.arange(1, len(controllers) + 1)); ax.set_xticklabels(controllers)
    ax.set_ylabel("hdot sign changes / s (chatter)")
    ax.set_title("Chatter distribution across scenes")

    # (1,1) min_ee_gap violin
    ax = axes[1, 1]
    data = [[r["min_ee_gap"] for r in rows_for(c)] for c in controllers]
    data = [d if d else [np.nan] for d in data]
    ax.violinplot(data, showmedians=True)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(np.arange(1, len(controllers) + 1)); ax.set_xticklabels(controllers)
    ax.set_ylabel("min end-effector gap (m)")
    ax.set_title("Min EE-EE gap distribution across scenes\n(dashed = 0, not the collision threshold)")

    fig.suptitle("LEG A -- distributional sweep over randomized scenes", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controllers", type=str, default="A,B,v4-C",
                     help="Comma list of display names. Base set: A, B, v4-C (=condition C, "
                          "static 0.06 margin), C-adapt (distance-adaptive margin), "
                          "C-adapt-relax (C-adapt + bounded ego-QP constraint relaxation on "
                          "infeasibility). Extra margin-grid controllers from --coop-margins "
                          "are appended automatically.")
    ap.add_argument("--coop-margins", type=str, default=None,
                     help="Comma list of INTRUDER_EXTRA_MARGIN values, e.g. "
                          "'0.0,0.03,0.06,0.09,0.12'. For each value v, adds a controller "
                          "'v4-C-m<v>' (condition C with ab4.INTRUDER_EXTRA_MARGIN['C']=v for "
                          "that run only) to the sweep, reusing the SAME sampled scenes as "
                          "every other controller in this run (task 3: static-margin grid, "
                          "seed-reproducible, directly comparable).")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--success-threshold", type=float, default=0.8)
    ap.add_argument("--out-dir", type=str, default=ARTIFACTS_DIR)
    args = ap.parse_args()

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    if "A" not in controllers:
        controllers = ["A"] + controllers  # A is the normalization reference for task success

    if args.coop_margins:
        for tok in args.coop_margins.split(","):
            tok = tok.strip()
            if not tok:
                continue
            v = float(tok)
            label = f"v4-C-m{v:g}"
            CONTROLLER_CODE[label] = "C"
            MARGIN_OVERRIDE[label] = v
            if label not in controllers:
                controllers.append(label)

    os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()
    scenes, results = run_sweep(args.n_scenes, args.seed, controllers, args.workers)
    results = add_derived_fields(results, args.success_threshold)
    summary = aggregate(results, controllers)
    n_valid_crossing = sum(1 for s in scenes if s["valid_crossing"])

    # ---- CSV (per scene x controller row) ----
    import csv
    csv_path = os.path.join(args.out_dir, "leg_a_sweep_results.csv")
    all_keys = sorted({k for r in results for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # ---- scenes.json (reproducibility record) ----
    scenes_path = os.path.join(args.out_dir, "leg_a_sweep_scenes.json")
    with open(scenes_path, "w") as f:
        json.dump(dict(n_scenes=args.n_scenes, seed=args.seed, scenes=scenes), f, indent=2)

    # ---- summary json + markdown ----
    summary_json_path = os.path.join(args.out_dir, "leg_a_sweep_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(dict(n_scenes=args.n_scenes, seed=args.seed, controllers=controllers,
                        success_threshold=args.success_threshold,
                        n_valid_crossing=n_valid_crossing, summary=summary,
                        wall_time_s=time.time() - t_start), f, indent=2)

    summary_md_path = os.path.join(args.out_dir, "leg_a_sweep_summary.md")
    write_markdown_table(summary, controllers, summary_md_path, args.n_scenes, args.seed,
                          args.success_threshold, n_valid_crossing)

    # ---- plot ----
    plot_path = os.path.join(args.out_dir, "leg_a_sweep.png")
    make_plot(results, controllers, plot_path)

    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")
    print(f"CSV:     {csv_path}")
    print(f"Scenes:  {scenes_path}")
    print(f"Summary: {summary_json_path}")
    print(f"Table:   {summary_md_path}")
    print(f"Plot:    {plot_path}")
    print("\n" + open(summary_md_path).read())


if __name__ == "__main__":
    main()
