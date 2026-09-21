#!/usr/bin/env python
"""Bimanual data-collection rollout loop.

Demonstrator: NOT a trained policy. Left arm holds a (seed-perturbed) grasp
pose via position control (box teleported into its fingers and closed with a
seed-perturbed grasp strength -- adapted from
experiments/ab_smooth_vs_bangbang/sim_common.py's `init_grasped_state`, which
this module imports and reuses for model loading / joint bookkeeping /
constants rather than re-deriving them). Right arm runs the SAME deterministic
intrusion sweep already validated in sim_common.py
(`intruder_ctrl`) to bring the right gripper close to the left gripper --
sim_common's own docstring notes this sweep, at sustained amplitude, can
"shake the box loose" and can produce arm-arm contact, which is exactly the
source of spill/drop and collision episodes we want here (no separate
"perturbation policy" needed on top of it).

Per-episode randomization (collection/perturbations.py):
  - "arm_start_shift": small per-joint offsets to both arms' held target pose.
    NOTE (simplification): the demonstrator has no IK, so unlike UR5's
    Cartesian arm_start_shift, offsets here are applied directly in JOINT
    space (joints 1 and 2 of each arm), with the family's *_dx_m/_dy_m
    parameters reused as radian offsets via a fixed 3.0 rad/m scale factor.
    This is a deliberate shortcut to keep this port minimal -- documented
    here and in README.md, not hidden.
  - "object_pose_shift": xy offset applied to the box's teleport pose before
    the grasp is closed (same as UR5's object_position_shift).
  - grasp strength: seeded per episode (not a "family", just an extra scalar
    knob) so that some episodes grip weakly enough to be shaken loose by the
    intrusion sweep or by gravity alone -- this is what produces "drop"
    episodes without needing a real lift phase.

No lift phase: the demonstrator only *holds* the post-grasp pose. This is
enough to generate a mix of success / dropped / collision episodes (see
README.md for why a real lift was cut from this minimal port).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIM_COMMON_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_COMMON_DIR))
import sim_common  # noqa: E402  (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, joint_addrs, intruder_ctrl, DT)

from collection import cameras  # noqa: E402
from collection.episode import (  # noqa: E402
    EpisodeWriter,
    GRIPPER_JOINT_RANGE_M,
    build_episode_metadata,
)
from collection.perturbations import PerturbationSpec, build_perturbation_spec  # noqa: E402

DROP_Z_DELTA_M = 0.03          # box z falling this far below its post-grasp z => "dropped"
ARM_JOINTSPACE_SCALE_RAD_PER_M = 3.0  # see module docstring's arm_start_shift simplification
RIGHT_FINGER_OPEN = 0.035       # right gripper stays open throughout (not grasping anything)
DEFAULT_LEVELS = ("level_0", "level_1", "level_2", "level_3")


def _gripper_state(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    jid = model.joint(f"{side}/joint7").id
    qadr = model.jnt_qposadr[jid]
    return float(np.clip(data.qpos[qadr] / GRIPPER_JOINT_RANGE_M, 0.0, 1.0))


def _left_right_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = model.geom(c.geom1).name
        n2 = model.geom(c.geom2).name
        s1 = n1.split("/", 1)[0] if "/" in n1 else None
        s2 = n2.split("/", 1)[0] if "/" in n2 else None
        if {s1, s2} == {"left", "right"}:
            return True
    return False


def _init_grasped_state(
    model: mujoco.MjModel, spec: PerturbationSpec, grasp_quality: float
) -> mujoco.MjData:
    """Adapted from sim_common.init_grasped_state: same teleport-then-close
    technique, parametrized by (a) an xy box-pose offset (object_pose_shift)
    and (b) a grasp_quality in (0, 1] controlling how far the fingers close
    (1.0 = fully closed target of 0.0 rad; lower = a weaker, partially-open
    hold that gravity / the intrusion sweep can shake loose)."""
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    params = spec.parameters if spec.family == "object_pose_shift" else {}
    box_dx = float(params.get("box_dx_m", 0.0))
    box_dy = float(params.get("box_dy_m", 0.0))

    pad7 = model.geom("left/link7_pad").id
    pad8 = model.geom("left/link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])
    pad_mid = pad_mid + np.array([box_dx, box_dy, 0.0])

    box_jid = model.joint("box_center_free").id
    box_qposadr = model.jnt_qposadr[box_jid]
    box_qveladr = model.jnt_dofadr[box_jid]
    data.qpos[box_qposadr : box_qposadr + 3] = pad_mid
    data.qpos[box_qposadr + 3 : box_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[box_qveladr : box_qveladr + 6] = 0.0
    mujoco.mj_forward(model, data)

    original_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    close_target = (1.0 - float(np.clip(grasp_quality, 0.0, 1.0))) * GRIPPER_JOINT_RANGE_M
    j7 = model.actuator("left/joint7_pos").id
    j8 = model.actuator("left/joint8_pos").id
    data.ctrl[j7] = close_target
    data.ctrl[j8] = -close_target
    for _ in range(500):
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = original_gravity
    for _ in range(200):
        mujoco.mj_step(model, data)
    return data


def _apply_arm_start_shift(q0_left: np.ndarray, q0_right: np.ndarray, spec: PerturbationSpec) -> None:
    if spec.family != "arm_start_shift":
        return
    p = spec.parameters
    q0_left[0] += p["left_dx_m"] * ARM_JOINTSPACE_SCALE_RAD_PER_M
    q0_left[1] += p["left_dy_m"] * ARM_JOINTSPACE_SCALE_RAD_PER_M
    q0_right[0] += p["right_dx_m"] * ARM_JOINTSPACE_SCALE_RAD_PER_M
    q0_right[1] += p["right_dy_m"] * ARM_JOINTSPACE_SCALE_RAD_PER_M
    np.clip(q0_left, sim_common.LEFT_JOINT_RANGES[:, 0], sim_common.LEFT_JOINT_RANGES[:, 1], out=q0_left)
    np.clip(q0_right, sim_common.LEFT_JOINT_RANGES[:, 0], sim_common.LEFT_JOINT_RANGES[:, 1], out=q0_right)


def collect_episode(
    episode_index: int,
    seed: int,
    out_dir: Path,
    *,
    num_steps: int,
    render_size: int,
    render_every: int,
    family: str,
) -> dict:
    rng = np.random.RandomState(seed)
    level = DEFAULT_LEVELS[seed % len(DEFAULT_LEVELS)]
    spec = build_perturbation_spec(family, level, seed)
    grasp_quality = float(rng.uniform(0.55, 1.0))

    model = cameras.build_model(sim_common.SCENE)
    renderer = cameras.DualCameraRenderer(model, height=render_size, width=render_size)

    try:
        data = _init_grasped_state(model, spec, grasp_quality)

        _, left_qposadr, left_dofadr = sim_common.joint_addrs(model, sim_common.LEFT_ARM_JOINTS)
        _, right_qposadr, right_dofadr = sim_common.joint_addrs(model, sim_common.RIGHT_ARM_JOINTS)
        q0_left = data.qpos[left_qposadr].copy()
        q0_right = data.qpos[right_qposadr].copy()
        _apply_arm_start_shift(q0_left, q0_right, spec)

        left_pos_act = [model.actuator(f"left/joint{i}_pos").id for i in range(1, 7)]
        right_pos_act = [model.actuator(f"right/joint{i}_pos").id for i in range(1, 7)]
        right_finger_act = [model.actuator("right/joint7_pos").id, model.actuator("right/joint8_pos").id]
        left_finger_act = [model.actuator("left/joint7_pos").id, model.actuator("left/joint8_pos").id]
        box_body_id = model.body("box_center").id

        box_z0 = float(data.xpos[box_body_id][2])

        writer = EpisodeWriter()
        ever_collision = False
        ever_dropped = False

        for step in range(num_steps):
            t = step * sim_common.DT
            data.ctrl[left_pos_act] = q0_left
            data.ctrl[right_pos_act] = sim_common.intruder_ctrl(t)
            data.ctrl[right_finger_act] = [RIGHT_FINGER_OPEN, -RIGHT_FINGER_OPEN]
            # left finger ctrl stays at whatever _init_grasped_state set (held grasp target)

            mujoco.mj_step(model, data)

            collision = _left_right_contact(model, data)
            box_z = float(data.xpos[box_body_id][2])
            dropped = bool(box_z < (box_z0 - DROP_Z_DELTA_M))
            ever_collision = ever_collision or collision
            ever_dropped = ever_dropped or dropped

            if step % render_every == 0 or step == 0:
                imgs = renderer.render(data)
            # NOTE: when render_every > 1, non-rendered steps reuse the last
            # rendered frame (cheap subsampling) rather than skipping the
            # image keys entirely, so every array in episode.npz stays the
            # same length T = num_steps.
            box_quat = data.xquat[box_body_id].copy()
            box_pose = np.concatenate([data.xpos[box_body_id], box_quat]).astype(np.float32)

            writer.add_step(
                rgb_head=imgs["rgb_head"],
                rgb_wrist_right=imgs["rgb_wrist_right"],
                left_q=data.qpos[left_qposadr].astype(np.float32).copy(),
                left_qd=data.qvel[left_dofadr].astype(np.float32).copy(),
                right_q=data.qpos[right_qposadr].astype(np.float32).copy(),
                right_qd=data.qvel[right_dofadr].astype(np.float32).copy(),
                left_action=data.ctrl[left_pos_act + left_finger_act].astype(np.float32).copy(),
                right_action=data.ctrl[right_pos_act + right_finger_act].astype(np.float32).copy(),
                left_gripper_state=np.float32(_gripper_state(model, data, "left")),
                right_gripper_state=np.float32(_gripper_state(model, data, "right")),
                box_pose=box_pose,
                collision=bool(collision),
                dropped=bool(dropped),
            )

        grasp_success = not ever_dropped
        success = grasp_success and not ever_collision

        metadata = build_episode_metadata(
            episode_index=episode_index,
            seed=seed,
            perturbation=spec,
            policy_name="scripted_hold_plus_intrusion_sweep_v1",
            num_steps=len(writer),
            success=success,
            grasp_success=grasp_success,
            collision=ever_collision,
            spilled_count=0,
        )
        episode_dir = out_dir / metadata["episode_id"]
        writer.save(episode_dir, metadata)
        return metadata
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "collection" / "data" / "smoke")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=150, help="physics steps per episode (DT=%s s)" % sim_common.DT)
    parser.add_argument("--render-size", type=int, default=84)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--family", type=str, default="arm_start_shift", choices=("arm_start_shift", "object_pose_shift"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        meta = collect_episode(
            i,
            seed,
            args.out_dir,
            num_steps=args.num_steps,
            render_size=args.render_size,
            render_every=args.render_every,
            family=args.family,
        )
        results.append(meta)
        print(
            f"[collect] episode={meta['episode_id']} seed={seed} "
            f"success={meta['success']} grasp_success={meta['grasp_success']} "
            f"collision={meta['collision']} steps={meta['num_steps']}"
        )
    dt = time.time() - t0
    n_success = sum(1 for r in results if r["success"])
    print(f"[collect] wrote {len(results)} episodes to {args.out_dir} in {dt:.1f}s "
          f"({n_success}/{len(results)} success)")


if __name__ == "__main__":
    main()
