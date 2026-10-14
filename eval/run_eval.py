"""Config-driven sim eval harness: does the (future) world model's semantic
veto help the geometric HJ-reachability safety layer on the bimanual cup
task? Runs N episodes of the cup task (scenes/scene_yor_cup.xml, LEFT arm
grasps the cup+balls and executes a fast lateral transfer/shake of its PD
target -- the validated accel-spill regime, see scene_setup.spill_shake_offset
-- while RIGHT arm just holds home; see eval/scene_setup.py) under one of 4
safety-stack configs and logs per-episode metrics. The failure of interest
is SPILL (a semantic failure invisible to the geometric arm-arm-distance
layer), not collision -- see run_eval.py's CONFIGS-adjacent SCENARIO comment.

Configs (geometric role x semantic role, see eval/safety_module.py):
    none            -- no safety at all (raw task PD).
    geometric_only  -- HOCBF-QP arm-arm collision filter only.
    semantic_only   -- oracle spill-veto only (no collision filter).
    both            -- both layers active.

SCAFFOLD + SMOKE ONLY: placeholder modules (analytic-distance HOCBF-QP for
geometric, ground-truth oracle for semantic) -- see eval/safety_module.py's
module docstring for exactly what a real HJ value / trained WM monitor must
replace.

Run:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=3 conda run -n mujoco_ur5e \\
        python /common/users/ss5772/bimanual/eval/run_eval.py \\
        --configs all --episodes 8 --duration-s 4.5

Writes eval/results/<config>/ep_XXXX.json (per-episode metrics) and
eval/results/all_episodes.csv (one row per episode, all configs).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
_EVAL_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_SIM_COMMON_DIR), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import; do not edit)
from scripts import cup_task_utils as ct  # noqa: E402  (read-only import; do not edit)
import scene_setup  # noqa: E402  (eval/, this task's own new code)
import safety_module as sm  # noqa: E402  (eval/, this task's own new code)

RESULTS_DIR = _EVAL_DIR / "results"

CONFIGS = {
    "none": ("none", "none"),
    "geometric_only": ("hocbf_qp", "none"),
    "semantic_only": ("none", "oracle"),
    "both": ("hocbf_qp", "oracle"),
}

# SCENARIO (fixed from the original collision-only smoke): the LEFT arm
# (grasping the cup) executes a fast lateral transfer/shake of its PD target
# (scene_setup.spill_shake_offset, converted to joint space via
# scene_setup.left_ee_jac_x_direction) at the validated accel-spill
# amplitude/frequency -- this is a SEMANTIC failure (spill) the geometric
# (arm-arm distance) layer is structurally blind to. The RIGHT arm no longer
# runs sc.intruder_ctrl -- it just holds its post-reset ctrl (home pose), so
# collision is NOT the decisive factor in any config; the geometric layer is
# still live (in geometric_only/both) but has nothing to react to, so it
# does not constrain the cup motion -- exactly the point of this eval
# (geometry cannot help against a spill). DEFAULT_DURATION_S covers: ~0.5s
# grasp-settle, ~1.3s shake window, ~2.7s post-shake settle (spilled balls
# finish leaving the cup / landing) so `spilled`/`collision`/`min_h` are all
# measured on a completed motion, not a truncated one.
DEFAULT_DURATION_S = 4.5
CUP_DROP_THRESH_M = 0.05   # same threshold as the box task's box_dropped


def _semantic_precision_recall(spilled_hist: np.ndarray, veto_hist: np.ndarray,
                                horizon_steps: int):
    """Precision/recall of the semantic monitor's veto flag against the
    ground-truth label "a NEW spill happens within `horizon_steps`", evaluated
    only at steps that are not already-spilled (there is nothing "imminent"
    to predict once spill has already occurred)."""
    n = len(spilled_hist)
    tp = fp = fn = tn = 0
    for t in range(n):
        if spilled_hist[t]:
            continue
        label = bool(np.any(spilled_hist[t + 1:min(n, t + 1 + horizon_steps)]))
        pred = bool(veto_hist[t])
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def run_episode(config_name: str, seed: int, duration_s: float) -> dict:
    geometric_kind, semantic_kind = CONFIGS[config_name]
    geometric: sm.GeometricSafety = sm.GEOMETRIC_REGISTRY[geometric_kind]()
    semantic: sm.SemanticSafety = sm.SEMANTIC_REGISTRY[semantic_kind]()
    geometric.reset()
    semantic.reset()

    model = scene_setup.load_model()
    # cup_jitter_m=0.0: episode-to-episode variety comes from the intrusion
    # timing jitter (t_offset) below instead -- jittering the PRE-grasp cup
    # position shifts the delta applied to the balls (see scene_setup.
    # init_cup_grasped_state) relative to their authored cup-local offsets,
    # which can push a ball outside the cup radius before the episode even
    # starts. cup_jitter_m stays a valid knob (scene_setup applies it
    # correctly relative to old_cup_pos) for future use with a fix that also
    # re-centers the balls' target offsets, just not used as the default
    # diversity source here.
    data = scene_setup.init_cup_grasped_state(model, seed=seed, cup_jitter_m=0.0)
    _, left_qposadr, left_dofadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, right_dofadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    q0_left = data.qpos[left_qposadr].copy()
    sc.disable_left_arm_position_servos(model)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    ee_body = model.body(scene_setup.EE_BODY_NAME).id
    cup_body = model.body(ct.CUP_BODY_NAME).id

    # Joint-space pinv for the transfer/tilt angular offset, computed once
    # at the settled grasp pose (see scene_setup.left_ee_jac_rot_pinv
    # docstring for why tilt, not a lateral translation, is used here).
    Jr_pinv = scene_setup.left_ee_jac_rot_pinv(model, data, ee_body, left_dofadr)
    q_transfer_dir = Jr_pinv @ scene_setup.TRANSFER_ROT_AXIS

    # Rigid cup<-EE attachment offset (captured once, post-settle) -- see
    # scene_setup.reattach_cup_rigidly's docstring: init_cup_grasped_state's
    # finger-pad grasp never actually contacts the cup wall in this scene
    # (verified separately), so without this the cup is simply left behind
    # by any EE motion. Re-applied every step below; only the cup BODY is
    # pinned, the 8 balls stay fully free (they are what can spill).
    cup_rel_pos0, cup_rel_quat0 = scene_setup.cup_rel_pose(model, data, ee_body, cup_body)

    n_steps = int(round(duration_s / sc.DT))
    rng = np.random.default_rng(seed + 10_000)
    t_offset = float(rng.uniform(-0.05, 0.05))  # small per-episode transfer-timing jitter

    h_hist = np.zeros(n_steps)
    dist_hist = np.zeros(n_steps)
    ee_pos_hist = np.zeros((n_steps, 3))
    spilled_hist = np.zeros(n_steps, dtype=bool)
    collision_hist = np.zeros(n_steps, dtype=bool)
    veto_hist = np.zeros(n_steps, dtype=bool)
    risk_hist = np.zeros(n_steps)
    solve_time_hist = np.zeros(n_steps)
    feasible_hist = np.ones(n_steps, dtype=bool)
    cup_drift_trans_hist = np.zeros(n_steps)
    cup_drift_rot_hist = np.zeros(n_steps)

    cup_pos_rel0 = None
    cup_quat_rel0 = None

    # The transfer motion's OWN clock -- only advances on steps where the
    # semantic veto is NOT active. This is the actual "slow/hold the risky
    # action" wiring: task_torque's freeze=True already zeroes the
    # position-attraction term for the CURRENT step (pure damping, "hold
    # where you are"), but the transfer schedule itself was still being
    # indexed by wall-clock t -- so a veto that released later would find
    # q_target had kept marching ahead on schedule while q sat frozen,
    # producing a huge position error and a violent catch-up snap (verified
    # empirically: an earlier version of this loop showed rms_ee_jerk
    # jumping ~1300 vs ~17 for the unvetoed baseline, and still spilling 3/8
    # balls, i.e. the veto was actively making things WORSE). Freezing this
    # clock too means a release resumes the transfer from exactly where it
    # paused -- an actual hold, not a pause-then-lurch.
    transfer_t_elapsed = 0.0

    for k in range(n_steps):
        t = k * sc.DT
        # RIGHT arm: no scripted intruder -- it just holds its post-reset
        # ctrl (home pose), so collision is not the decisive factor here (see
        # module-level SCENARIO comment above CONFIGS).
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0

        mujoco.mj_forward(model, data)

        # Kinematically re-pin the cup to the EE every step (see
        # scene_setup.reattach_cup_rigidly docstring) -- only the cup body,
        # never the balls.
        scene_setup.reattach_cup_rigidly(model, data, ee_body, cup_body, cup_rel_pos0, cup_rel_quat0)
        mujoco.mj_forward(model, data)

        verdict = semantic.assess(sm.SemanticObservation(model=model, data=data, t=t))
        if not verdict.veto:
            transfer_t_elapsed += sc.DT
        transfer_off = scene_setup.transfer_tilt_offset(max(0.0, transfer_t_elapsed + t_offset))
        q_target = q0_left + q_transfer_dir * transfer_off
        tau_task, M_ll, bias_l = scene_setup.task_torque(
            model, data, q_target, left_dofadr, left_qposadr, ee_body, cup_body,
            freeze=verdict.veto)

        geo_inputs = sm.GeometricInputs(
            model=model, data=data, left_dofadr=left_dofadr, right_dofadr=right_dofadr,
            tau_task=tau_task, M_ll=M_ll, bias_l=bias_l)
        geo_result = geometric.correct(geo_inputs)
        tau_cmd = np.clip(geo_result.tau, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[left_dofadr] = tau_cmd

        h_hist[k] = geo_result.min_h
        dist_hist[k] = geo_result.min_dist
        ee_pos_hist[k] = data.xpos[ee_body]
        solve_time_hist[k] = geo_result.solve_time_s
        feasible_hist[k] = geo_result.feasible
        veto_hist[k] = verdict.veto
        risk_hist[k] = verdict.risk

        collision_hist[k] = scene_setup.left_right_contact(model, data)
        spilled_hist[k] = ct.count_spilled(model, data) > 0

        R_gb = data.xmat[ee_body].reshape(3, 3)
        rel_pos = R_gb.T @ (data.xpos[cup_body] - data.xpos[ee_body])
        gb_quat = data.xquat[ee_body].copy()
        cup_quat = data.xquat[cup_body].copy()
        rel_quat = scene_setup.quat_mul(scene_setup.quat_conj(gb_quat), cup_quat)
        if cup_pos_rel0 is None:
            cup_pos_rel0 = rel_pos.copy()
            cup_quat_rel0 = rel_quat.copy()
        cup_drift_trans_hist[k] = float(np.linalg.norm(rel_pos - cup_pos_rel0))
        drift_quat = scene_setup.quat_mul(scene_setup.quat_conj(cup_quat_rel0), rel_quat)
        cup_drift_rot_hist[k] = float(scene_setup.quat_angle(drift_quat))

        mujoco.mj_step(model, data)

    rms_ee_jerk, max_ee_jerk = sc.jerk_metrics(ee_pos_hist, sc.DT)
    final_spilled_count = int(ct.count_spilled(model, data))
    ever_spilled = bool(np.any(spilled_hist))
    ever_collision = bool(np.any(collision_hist))
    cup_dropped = bool(np.max(cup_drift_trans_hist) > CUP_DROP_THRESH_M)
    success = (not ever_spilled) and (not ever_collision) and (not cup_dropped)

    horizon_s = getattr(semantic, "horizon_s", 0.15)
    horizon_steps = max(1, int(round(horizon_s / sc.DT)))
    precision, recall = (
        _semantic_precision_recall(spilled_hist, veto_hist, horizon_steps)
        if semantic_kind != "none" else (float("nan"), float("nan")))

    return dict(
        config=config_name, seed=seed, geometric=geometric_kind, semantic=semantic_kind,
        duration_s=duration_s, n_steps=n_steps,
        success=success,
        spilled=ever_spilled, spilled_count=final_spilled_count,
        collision=ever_collision,
        cup_dropped=cup_dropped,
        cup_drift_trans_max_m=float(np.max(cup_drift_trans_hist)),
        cup_drift_rot_max_deg=float(np.degrees(np.max(cup_drift_rot_hist))),
        rms_ee_jerk=rms_ee_jerk, max_ee_jerk=max_ee_jerk,
        min_h=float(np.min(h_hist)), min_dist=float(np.min(dist_hist)),
        mean_solve_time_ms=float(np.mean(solve_time_hist) * 1000.0),
        max_solve_time_ms=float(np.max(solve_time_hist) * 1000.0),
        qp_infeasible_frac=float(np.mean(~feasible_hist)),
        semantic_veto_frac=float(np.mean(veto_hist)),
        semantic_precision=precision, semantic_recall=recall,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="all",
                         help="comma-separated config names, or 'all' (default)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    config_names = list(CONFIGS.keys()) if args.configs == "all" else args.configs.split(",")
    for c in config_names:
        if c not in CONFIGS:
            raise ValueError(f"unknown config {c!r}; choices: {list(CONFIGS.keys())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for config_name in config_names:
        cfg_dir = args.out_dir / config_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.episodes):
            seed = args.seed_start + i
            t0 = time.perf_counter()
            metrics = run_episode(config_name, seed, args.duration_s)
            wall_s = time.perf_counter() - t0
            metrics["wall_time_s"] = wall_s
            ep_path = cfg_dir / f"ep_{i:04d}.json"
            with open(ep_path, "w") as f:
                json.dump(metrics, f, indent=2)
            all_rows.append(metrics)
            print(f"[{config_name} seed={seed}] success={metrics['success']} "
                  f"spilled={metrics['spilled']} collision={metrics['collision']} "
                  f"min_h={metrics['min_h']:.4f} solve_ms={metrics['mean_solve_time_ms']:.2f} "
                  f"({wall_s:.1f}s wall)")

    if all_rows:
        csv_path = args.out_dir / "all_episodes.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} episode rows -> {csv_path}")


if __name__ == "__main__":
    main()
