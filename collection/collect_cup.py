#!/usr/bin/env python
"""Cup-task ("don't-spill-the-cup") data-collection rollout loop -- PROPER YOR
robot (scenes/scene_yor_cup.xml, see scenes/YOR_CANONICAL.md), superseding the
SIMPLIFIED-rig version of this script (see collection/CUP_DATA.md for that
history; collection/CUP_DATA_V2.md documents this rebuild).

Cameras recorded per step (via collection.cameras.build_yor_cup_model() /
YorCupCameraRenderer). `world_model/dataset_adapter.py` is owned by a
separate in-flight task and is NOT edited here -- it unconditionally builds
`image = concat([rgb_head, rgb_wrist_right], axis=-1)`. So, WITHOUT touching
the adapter, the real ZED workspace camera (`zed_cam`, scripts/zed_camera.py,
mounted on Lift_Top -- frames the tabletop + both arms, sees spills) is
stored into BOTH `rgb_head` and `rgb_wrist_right` (the two field names the
adapter already reads), making it THE world model's `image` source (both
halves are the same ZED frame, not two different cameras -- see
collection/CUP_DATA_V2.md). Also recorded: `rgb_zed` (the same ZED frame,
its own key, for direct inspection), and `rgb_wrist_left_ego`/
`rgb_wrist_right_ego` (the two REAL egocentric wrist cams -- auxiliary data
only, not part of `image`). No invented "head_cam" or track-mode debug
camera is used here or fed anywhere downstream.

Demonstrator: NOT a trained policy. The cup is never grasped -- no bimanual
grasp/IK for the cup exists in this repo, so (exactly as the old pipeline
validated in scripts/test_cup_scene.py) the cup is moved by directly
overwriting its freejoint qpos each step ("teleport" open-loop scripting)
while the balls are fully dynamic and pushed via contact. Both arms
otherwise hold their scene's "home" keyframe pose via position control
(LEFT_HOME_CTRL / sim_common.RIGHT_HOME_CTRL -- both verbatim from
scenes/scene_yor_cup.xml's own <keyframe>, which matches
scene_yor_canonical.xml's), except during "collision_intrusion" where the
right arm instead runs `_intrude_toward_left_home` (reuses sim_common's
ramp-in/oscillate/ramp-out TIMING verbatim, robot/pose-agnostic, but with its
own IK-solved intrusion target aimed at the left arm's actual position at
LEFT_HOME_CTRL -- sim_common's own RIGHT_INTRUDE_CTRL targets LEFT_GRASP_Q,
the box task's grasp pose, which sits inside/against this scene's cup rim
and isn't safe to reuse here; see RIGHT_INTRUDE_Q_HOME's docstring below).

Per-episode BEHAVIOR (perturbations.py's "cup_behavior" family,
CUP_BEHAVIOR_LEVELS, unchanged from the old pipeline -- purely a cup-motion
spec, robot-agnostic) selects which demonstrator motion runs:
  - "success_gentle":     small slow upright lateral jitter -> no spill.
  - "spill_shake":        fast upright lateral shake at the accel-spill
                          amplitudes tuned in scripts/test_cup_scene.py's
                          run_upright_shake_test -> spills via acceleration
                          alone (cup stays upright).
  - "spill_tilt":         lift+tilt the cup ~90-115 deg -> spills via tipping.
  - "collision_intrusion": right arm runs `_intrude_toward_left_home` toward
                          the left arm's home region, cup left essentially
                          undisturbed -> exercises arm/arm collision (may
                          incidentally spill if the sweep clips the cup, but
                          is not scripted to).

Ground-truth spill labeling: scripts.cup_task_utils.count_spilled(model,
data) is called every step (same cup geometry/detector as before -- only the
scene's WORLD position changed, and count_spilled works in the cup's own
local frame so it is unaffected). Per-step schema note (episode.py's
STEP_ARRAY_KEYS is shared with the box-pickup task and is NOT modified here):
  - "box_pose"  repurposed to carry the CUP body's (xyz, wxyz-quat) pose each
                step (there is no "box" in this scene). This is the ONE field
                that must NOT leak into the world model's `proprio` --
                world_model/dataset_adapter.py now routes it to a separate
                `object_pose` label field instead (see docs/design/
                state_spaces.md's object-pose-leak fix).
  - "dropped"    repurposed to carry the per-step SPILL flag
                 (count_spilled(t) > 0) instead of "box fell".
  - metadata "spilled_count" is the real final-step count_spilled(...) value.
  - metadata "grasp_success" is repurposed as "cup contents were not
    spilled" (nothing is grasped in this task).
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
import sim_common  # noqa: E402  (LEFT_ARM_JOINTS/RIGHT_ARM_JOINTS/joint_addrs/
                    # RIGHT_HOME_CTRL/RIGHT_INTRUDE_CTRL/intruder_ctrl/DT reused
                    # verbatim -- already targets this exact robot's naming)

from scripts.cup_task_utils import count_spilled  # noqa: E402

from collection import cameras  # noqa: E402
from collection.episode import (  # noqa: E402
    EpisodeWriter,
    GRIPPER_JOINT_RANGE_M,
    build_episode_metadata,
)
from collection.perturbations import (  # noqa: E402
    CUP_BEHAVIOR_LEVELS,
    PerturbationSpec,
    build_perturbation_spec,
)

FINGER_OPEN = 0.035  # both arms stay open throughout (nothing is grasped)

# scenes/scene_yor_cup.xml's own <keyframe name="home"> left-arm ctrl (verbatim;
# identical to scene_yor_canonical.xml's). sim_common.RIGHT_HOME_CTRL is the
# matching right-arm value (same scene family, same keyframe convention).
LEFT_HOME_CTRL = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78])

# collision_intrusion needs its OWN right-arm intrusion target: sim_common's
# RIGHT_INTRUDE_CTRL was IK-solved to reach sim_common.LEFT_GRASP_Q (the box
# task's left-arm GRASP pose, hovering right at the table). Our left arm
# instead holds LEFT_HOME_CTRL throughout (it never approaches the cup --
# reusing LEFT_GRASP_Q here would put the open left gripper's fingers inside/
# against the cup's rim, since the cup's rim (0.63) sits ~4-6cm above the box
# top LEFT_GRASP_Q was tuned for, contaminating the spill signal with
# unscripted arm-cup contact). So RIGHT_INTRUDE_Q_HOME below is a SEPARATE
# damped-least-squares IK solution (same coarse_seed + ik_to_target method as
# scripts/verify_yor_canonical.py, reused via import) that instead aims the
# right arm's grasp site at the LEFT arm's actual `left_arm_ee_grasp` site
# position while the left arm is frozen at LEFT_HOME_CTRL (not LEFT_GRASP_Q).
# Verified (min_pair_distance over the full corrected multi-keypoint proxy,
# see sim_common.py): h = +0.135 at RIGHT_HOME_CTRL -> h = -0.051 at
# RIGHT_INTRUDE_Q_HOME, i.e. a genuine safe->violated swing, comparable in
# magnitude to sim_common's own LEFT_GRASP_Q-targeted pair (+0.129 -> -0.060).
RIGHT_INTRUDE_Q_HOME = np.array(
    [0.973952, 2.696246, -1.794379, 0.753202, -0.284192, 1.916233]
)


def _intrude_toward_left_home(t: float) -> np.ndarray:
    """Right-arm ctrl trajectory toward RIGHT_INTRUDE_Q_HOME (this task's own
    intrusion target -- see constant docstring above), reusing sim_common's
    exact ramp-in/oscillating-hold/ramp-out timeline (T_IN0/T_IN1/T_HOLD_END/
    T_OUT_END/OSC_AMP/OSC_FREQ_HZ are pure timing constants, robot/pose-
    agnostic, so reused verbatim -- only the two endpoint poses differ from
    sim_common.intruder_ctrl)."""
    if t < sim_common.T_IN0:
        s = 0.0
    elif t < sim_common.T_IN1:
        s = sim_common._smoothstep(t, sim_common.T_IN0, sim_common.T_IN1)
    elif t < sim_common.T_HOLD_END:
        osc = sim_common.OSC_AMP * np.sin(2.0 * np.pi * sim_common.OSC_FREQ_HZ * (t - sim_common.T_IN1))
        s = np.clip(1.0 + osc, 0.0, 1.08)
    elif t < sim_common.T_OUT_END:
        s = 1.0 - sim_common._smoothstep(t, sim_common.T_HOLD_END, sim_common.T_OUT_END)
    else:
        s = 0.0
    return sim_common.RIGHT_HOME_CTRL + s * (RIGHT_INTRUDE_Q_HOME - sim_common.RIGHT_HOME_CTRL)


def _smoothstep(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def _gripper_state(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> float:
    """prefix: 'left_arm' or 'right_arm'."""
    jid = model.joint(f"{prefix}_joint7").id
    qadr = model.jnt_qposadr[jid]
    return float(np.clip(data.qpos[qadr] / GRIPPER_JOINT_RANGE_M, 0.0, 1.0))


def _left_right_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Body-name-prefix check (left_arm_* vs right_arm_*) -- geom names on
    this robot don't uniformly carry the side prefix (many are per-mesh
    names like 'link3'), unlike the old rig's 'left/<geom>' convention, so we
    key off the owning BODY's name instead (all bodies in yor_robot_derived
    .mjcf ARE consistently prefixed left_arm_*/right_arm_*)."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        s1 = "left" if b1.startswith("left_arm_") else ("right" if b1.startswith("right_arm_") else None)
        s2 = "left" if b2.startswith("left_arm_") else ("right" if b2.startswith("right_arm_") else None)
        if {s1, s2} == {"left", "right"}:
            return True
    return False


def _num_steps_for(level: str, params: dict, dt: float) -> int:
    """Episode length, derived from the sampled behavior's own timeline (kept
    SHORT for the verification run; the full collection uses the same
    per-behavior lengths, just more episodes)."""
    if level == "success_gentle":
        total_s = params["duration_s"] + 0.3
    elif level == "spill_shake":
        total_s = params["duration_s"] + 0.4
    elif level == "spill_tilt":
        total_s = 0.3 + params["tilt_dur_s"] + 0.4 + 0.5
    elif level == "collision_intrusion":
        total_s = 2.6  # covers intruder_ctrl's ramp-in (0.4-0.8s) + oscillating hold
    else:  # pragma: no cover - guarded by perturbations.build_perturbation_spec
        raise AssertionError(level)
    return int(round(total_s / dt))


def _init_settled_state(model: mujoco.MjModel, settle_s: float = 0.5) -> mujoco.MjData:
    """Reset to 'home' and let the balls settle into the cup under gravity
    before any scripted motion starts."""
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    for _ in range(int(settle_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def collect_episode(
    episode_index: int,
    seed: int,
    out_dir: Path,
    *,
    render_size: int,
    render_every: int,
    level: str | None = None,
) -> dict:
    if level is None:
        level = CUP_BEHAVIOR_LEVELS[seed % len(CUP_BEHAVIOR_LEVELS)]
    spec = build_perturbation_spec("cup_behavior", level, seed)
    params = spec.parameters

    model = cameras.build_yor_cup_model()
    renderer = cameras.YorCupCameraRenderer(model, height=render_size, width=render_size)

    try:
        data = _init_settled_state(model)
        dt = model.opt.timestep
        assert abs(dt - sim_common.DT) < 1e-12

        left_pos_act = [model.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
        right_pos_act = [model.actuator(f"right_arm_joint{i}_pos").id for i in range(1, 7)]
        left_finger_act = [model.actuator("left_arm_joint7_pos").id, model.actuator("left_arm_joint8_pos").id]
        right_finger_act = [model.actuator("right_arm_joint7_pos").id, model.actuator("right_arm_joint8_pos").id]
        _, left_qposadr, left_dofadr = sim_common.joint_addrs(model, sim_common.LEFT_ARM_JOINTS)
        _, right_qposadr, right_dofadr = sim_common.joint_addrs(model, sim_common.RIGHT_ARM_JOINTS)

        cup_bid = model.body("cup").id
        cup_joint = model.joint("cup_free")
        cup_qposadr = int(cup_joint.qposadr[0])
        cup_dofadr = int(cup_joint.dofadr[0])
        base_pos = data.xpos[cup_bid].copy()
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        num_steps = _num_steps_for(level, params, dt)

        writer = EpisodeWriter()
        ever_collision = False
        ever_spilled = False

        for step in range(num_steps):
            t = step * dt
            data.ctrl[left_pos_act] = LEFT_HOME_CTRL
            data.ctrl[left_finger_act] = [FINGER_OPEN, -FINGER_OPEN]
            data.ctrl[right_finger_act] = [FINGER_OPEN, -FINGER_OPEN]

            if level == "collision_intrusion":
                data.ctrl[right_pos_act] = _intrude_toward_left_home(max(0.0, t + params["t_offset_s"]))
                if params["cup_jitter_m"] > 0.0 and step == 0:
                    jitter = np.array([params["cup_jitter_m"], 0.0, 0.0])
                    data.qpos[cup_qposadr:cup_qposadr + 3] = base_pos + jitter
                    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = identity_quat
                    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0
            else:
                data.ctrl[right_pos_act] = sim_common.RIGHT_HOME_CTRL
                if level in ("success_gentle", "spill_shake"):
                    dur = params["duration_s"]
                    x_off = params["amp_m"] * np.sin(2 * np.pi * params["freq_hz"] * t) if t < dur else 0.0
                    data.qpos[cup_qposadr:cup_qposadr + 3] = base_pos + np.array([x_off, 0.0, 0.0])
                    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = identity_quat
                    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0
                elif level == "spill_tilt":
                    lift_dur, hold_dur, settle_dur = 0.3, 0.4, 0.5
                    tilt_dur = params["tilt_dur_s"]
                    tilt_max = np.deg2rad(params["tilt_deg"])
                    lift_h = params["lift_h_m"]
                    t_lift_end = lift_dur
                    t_tilt_end = t_lift_end + tilt_dur
                    t_hold_end = t_tilt_end + hold_dur
                    if t < t_lift_end:
                        lift = lift_h * _smoothstep(t / lift_dur, 0.0, 1.0)
                        angle = 0.0
                    elif t < t_tilt_end:
                        lift = lift_h
                        angle = tilt_max * _smoothstep((t - t_lift_end) / tilt_dur, 0.0, 1.0)
                    else:
                        lift = lift_h
                        angle = tilt_max
                    pos = base_pos + np.array([0.0, 0.0, lift])
                    quat = _axis_angle_quat([1, 0, 0], angle)
                    data.qpos[cup_qposadr:cup_qposadr + 3] = pos
                    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = quat
                    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0

            mujoco.mj_step(model, data)

            collision = _left_right_contact(model, data)
            spilled_count_t = count_spilled(model, data)
            spilled_now = spilled_count_t > 0
            ever_collision = ever_collision or collision
            ever_spilled = ever_spilled or spilled_now

            if step % render_every == 0 or step == 0:
                imgs = renderer.render(data)

            cup_quat = data.xquat[cup_bid].copy()
            cup_pose = np.concatenate([data.xpos[cup_bid], cup_quat]).astype(np.float32)

            writer.add_step(
                rgb_head=imgs["rgb_head"],                  # zed_cam (adapter's image[...,:3]; see cameras.py)
                rgb_wrist_right=imgs["rgb_wrist_right"],    # zed_cam again (adapter's image[...,3:6])
                rgb_zed=imgs["rgb_zed"],                    # zed_cam, dedicated key (debug/inspection)
                rgb_wrist_left_ego=imgs["rgb_wrist_left_ego"],    # left_arm_egocentric -- auxiliary only
                rgb_wrist_right_ego=imgs["rgb_wrist_right_ego"],  # right_arm_egocentric -- auxiliary only
                left_q=data.qpos[left_qposadr].astype(np.float32).copy(),
                left_qd=data.qvel[left_dofadr].astype(np.float32).copy(),
                right_q=data.qpos[right_qposadr].astype(np.float32).copy(),
                right_qd=data.qvel[right_dofadr].astype(np.float32).copy(),
                left_action=data.ctrl[left_pos_act + left_finger_act].astype(np.float32).copy(),
                right_action=data.ctrl[right_pos_act + right_finger_act].astype(np.float32).copy(),
                left_gripper_state=np.float32(_gripper_state(model, data, "left_arm")),
                right_gripper_state=np.float32(_gripper_state(model, data, "right_arm")),
                box_pose=cup_pose,  # repurposed: cup body pose -- LABEL only, see module docstring
                collision=bool(collision),
                dropped=bool(spilled_now),  # repurposed: per-step spill flag
            )

        final_spilled_count = int(count_spilled(model, data))
        grasp_success = not ever_spilled  # repurposed: "cup contents intact"
        success = grasp_success and not ever_collision

        metadata = build_episode_metadata(
            episode_index=episode_index,
            seed=seed,
            perturbation=spec,
            policy_name=f"cup_scripted_{level}_yor_v1",
            num_steps=len(writer),
            success=success,
            grasp_success=grasp_success,
            collision=ever_collision,
            spilled_count=final_spilled_count,
        )
        episode_dir = out_dir / metadata["episode_id"]
        writer.save(episode_dir, metadata)
        return metadata
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "collection" / "data" / "cup_yor_smoke")
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=64,
                         help="MUST be 64 to match world_model/dataset_adapter.py's hardcoded (T,64,64,6) assert")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--level", type=str, default=None, choices=(None, *CUP_BEHAVIOR_LEVELS),
                         help="force one behavior for all episodes; default cycles seed %% 4 for a mix")
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
            render_size=args.render_size,
            render_every=args.render_every,
            level=args.level,
        )
        results.append(meta)
        print(
            f"[collect_cup] episode={meta['episode_id']} seed={seed} level={meta['perturbation_level']} "
            f"success={meta['success']} spilled_count={meta['spilled_count']} "
            f"collision={meta['collision']} steps={meta['num_steps']}"
        )
    dt = time.time() - t0
    n_spilled = sum(1 for r in results if r["spilled_count"] > 0)
    n_collision = sum(1 for r in results if r["collision"])
    n_clean = sum(1 for r in results if r["spilled_count"] == 0 and not r["collision"])
    print(f"[collect_cup] wrote {len(results)} episodes to {args.out_dir} in {dt:.1f}s "
          f"(spilled={n_spilled} collision={n_collision} clean={n_clean})")


if __name__ == "__main__":
    main()
