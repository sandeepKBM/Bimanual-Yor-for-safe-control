#!/usr/bin/env python
"""Run bimanual teleop (scripted, headless by default) and RECORD an episode
via collection/episode.py -- same schema/field-name convention as
collection/collect_cup.py (see collection/CUP_DATA_V2.md), so a converted
episode drops straight into world_model/dataset_adapter.py unchanged:

  - `rgb_head` / `rgb_wrist_right`: the real ZED workspace camera (`zed_cam`,
    scripts/zed_camera.py's `add_zed_camera`, already attached by
    scripts.cup_task_utils.load_yor_cup_model) stored into BOTH fields -- the
    adapter's `image = concat([rgb_head, rgb_wrist_right], axis=-1)` becomes
    the ZED view duplicated across both 3-channel halves (this file does not
    edit the adapter, so it follows the same repurposing collect_cup.py uses).
  - `rgb_zed` (dedicated key, same ZED frame) + `rgb_wrist_left_ego` /
    `rgb_wrist_right_ego` (the two real egocentric wrist cams) recorded as
    auxiliary extras (episode.py's EpisodeWriter persists any extra per-step
    key beyond STEP_ARRAY_KEYS).
  - `left_q`/`left_qd`/`right_q`/`right_qd`/`left_gripper_state`/
    `right_gripper_state`: the 6 components world_model/dataset_adapter.py
    concatenates into the 26-D `proprio` (no separate "proprio" key is stored
    per-episode -- same as collect_cup.py; the adapter builds it downstream).
  - `box_pose`: repurposed to carry the CUP body's pose (label only, routed to
    a separate `object_pose` key by the adapter, never into `proprio`).
  - `collision`: real left_arm_*/right_arm_* body-geom contact this step.
  - `dropped`: repurposed as the per-step SPILL flag
    (scripts.cup_task_utils.count_spilled(model, data) > 0).

Two ways to produce the (side -> targets) trajectory recorded here:
  1. Scripted (`teleop.bimanual_teleop.drive`'s `targets_over_time`) --
     used by `run_smoke()` below, no display required.
  2. Interactive (`teleop.bimanual_teleop.run_interactive`) -- requires a
     display; not wired into this recorder (would need a live per-step
     episode-writer callback wired into that viewer loop instead of `drive`'s
     `on_step`; left as a documented extension, see teleop/README.md).
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

from scripts.cup_task_utils import count_spilled  # noqa: E402
from collection.episode import (  # noqa: E402
    EpisodeWriter,
    build_episode_metadata,
)
from collection.perturbations import build_perturbation_spec  # noqa: E402

from teleop import bimanual_teleop as bt  # noqa: E402

RENDER_SIZE = 64  # matches world_model/dataset_adapter.py's hardcoded (T,64,64,6) assert
ZED_CAM = "zed_cam"
LEFT_EGO = "left_arm_egocentric"
RIGHT_EGO = "right_arm_egocentric"


def _left_right_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Same body-name-prefix contact check as collection/collect_cup.py's
    `_left_right_contact` (reimplemented here -- collect_cup.py itself is not
    in this task's allowed-import list, but the logic is a few lines keyed off
    the same left_arm_*/right_arm_* body-name convention documented in
    scenes/YOR_CANONICAL.md)."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        s1 = "left" if b1.startswith("left_arm_") else ("right" if b1.startswith("right_arm_") else None)
        s2 = "left" if b2.startswith("left_arm_") else ("right" if b2.startswith("right_arm_") else None)
        if {s1, s2} == {"left", "right"}:
            return True
    return False


class TeleopEpisodeRenderer:
    """Renders zed_cam (primary) + both egocentric wrist cams (auxiliary),
    matching collection/cameras.py's YorCupCameraRenderer field-name
    convention exactly (not imported -- that module is outside this task's
    allowed-import list -- but its output keys are reproduced verbatim so a
    recorded episode is schema-identical)."""

    def __init__(self, model: mujoco.MjModel, size: int = RENDER_SIZE):
        self._r = mujoco.Renderer(model, height=size, width=size)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        self._r.update_scene(data, camera=ZED_CAM)
        zed = self._r.render().copy()
        out = {
            "rgb_head": zed,
            "rgb_wrist_right": zed.copy(),
            "rgb_zed": zed.copy(),
        }
        for key, cam in (("rgb_wrist_left_ego", LEFT_EGO), ("rgb_wrist_right_ego", RIGHT_EGO)):
            self._r.update_scene(data, camera=cam)
            out[key] = self._r.render().copy()
        return out

    def close(self) -> None:
        self._r.close()


def record_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets_over_time: dict[str, list["bt.Segment"]],
    *,
    episode_index: int,
    seed: int,
    policy_name: str,
    out_dir: Path,
    render_every: int = 2,
) -> dict:
    """Drives `targets_over_time` via teleop.bimanual_teleop.drive(), appending
    one EpisodeWriter step per physics step, then writes episode.npz +
    metadata.json under out_dir/<episode_id>/."""
    ik = bt.BimanualIK(model)
    renderer = TeleopEpisodeRenderer(model)
    writer = EpisodeWriter()

    left_pos_act = bt.arm_pos_actuators(model, "left")
    right_pos_act = bt.arm_pos_actuators(model, "right")
    left_finger_act = bt.finger_actuators(model, "left")
    right_finger_act = bt.finger_actuators(model, "right")
    left_qposadr = np.array([model.joint(n).qposadr[0] for n in bt.ARM_JOINTS["left"]])
    left_dofadr = np.array([model.joint(n).dofadr[0] for n in bt.ARM_JOINTS["left"]])
    right_qposadr = np.array([model.joint(n).qposadr[0] for n in bt.ARM_JOINTS["right"]])
    right_dofadr = np.array([model.joint(n).dofadr[0] for n in bt.ARM_JOINTS["right"]])

    cup_bid = model.body("cup").id

    ever_collision = False
    ever_spilled = False
    imgs = None

    def on_step(step: int, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal ever_collision, ever_spilled, imgs
        collision = _left_right_contact(m, d)
        spilled_count_t = count_spilled(m, d)
        spilled_now = spilled_count_t > 0
        ever_collision = ever_collision or collision
        ever_spilled = ever_spilled or spilled_now

        if imgs is None or step % render_every == 0:
            imgs = renderer.render(d)

        cup_pose = np.concatenate([d.xpos[cup_bid], d.xquat[cup_bid]]).astype(np.float32)

        writer.add_step(
            rgb_head=imgs["rgb_head"],
            rgb_wrist_right=imgs["rgb_wrist_right"],
            rgb_zed=imgs["rgb_zed"],
            rgb_wrist_left_ego=imgs["rgb_wrist_left_ego"],
            rgb_wrist_right_ego=imgs["rgb_wrist_right_ego"],
            left_q=d.qpos[left_qposadr].astype(np.float32).copy(),
            left_qd=d.qvel[left_dofadr].astype(np.float32).copy(),
            right_q=d.qpos[right_qposadr].astype(np.float32).copy(),
            right_qd=d.qvel[right_dofadr].astype(np.float32).copy(),
            left_action=d.ctrl[left_pos_act + left_finger_act].astype(np.float32).copy(),
            right_action=d.ctrl[right_pos_act + right_finger_act].astype(np.float32).copy(),
            left_gripper_state=np.float32(bt.gripper_state(m, d, "left")),
            right_gripper_state=np.float32(bt.gripper_state(m, d, "right")),
            box_pose=cup_pose,  # repurposed: cup pose, LABEL only (see module docstring)
            collision=bool(collision),
            dropped=bool(spilled_now),  # repurposed: per-step spill flag
        )

    try:
        bt.drive(model, data, targets_over_time, ik=ik, on_step=on_step)
    finally:
        renderer.close()

    final_spilled_count = int(count_spilled(model, data))
    grasp_success = not ever_spilled
    success = grasp_success and not ever_collision

    spec = build_perturbation_spec("cup_behavior", "success_gentle", seed)
    metadata = build_episode_metadata(
        episode_index=episode_index,
        seed=seed,
        perturbation=spec,
        policy_name=policy_name,
        num_steps=len(writer),
        success=success,
        grasp_success=grasp_success,
        collision=ever_collision,
        spilled_count=final_spilled_count,
    )
    episode_dir = out_dir / metadata["episode_id"]
    writer.save(episode_dir, metadata)
    return metadata


def build_smoke_targets(cup_xyz=(0.42, -0.21, 0.55)) -> dict[str, list["bt.Segment"]]:
    """Scripted LEFT-arm sequence: reach the cup -> close gripper -> lift 10cm
    -> move sideways 15cm. Right arm holds its home pose throughout (absent
    from the returned dict -> bt.drive() leaves its ctrl untouched)."""
    x, y, z = cup_xyz
    return {
        "left": [
            bt.Segment(pos=np.array([x, y, z]), gripper=1.0, duration=1.2, hold=0.2),   # reach, open
            bt.Segment(pos=np.array([x, y, z]), gripper=0.0, duration=0.5, hold=0.3),    # close gripper
            bt.Segment(pos=np.array([x, y, z + 0.10]), gripper=0.0, duration=1.0, hold=0.2),  # lift 10cm
            bt.Segment(pos=np.array([x, y + 0.15, z + 0.10]), gripper=0.0, duration=1.0, hold=0.2),  # move sideways
        ]
    }


def run_smoke(out_dir: Path) -> dict:
    model, data = bt.load_scene()
    bt.reset_home(model, data)
    # Let balls settle (mirrors collect_cup.py's _init_settled_state) so the
    # cup starts in a physically valid state before the scripted reach.
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    targets = build_smoke_targets()
    meta = record_episode(
        model, data, targets,
        episode_index=0, seed=0, policy_name="teleop_smoke_left_reach_lift",
        out_dir=out_dir,
    )
    return meta


def save_smoke_png(episode_dir: Path, out_path: Path) -> None:
    """Montage of start / mid / end ZED (primary) + left-wrist-egocentric
    (auxiliary) frames from a recorded episode.npz, for a quick visual smoke
    check -- written to artifacts/teleop_smoke.png."""
    from PIL import Image

    d = np.load(episode_dir / "episode.npz")
    n = d["rgb_zed"].shape[0]
    idx = [0, n // 2, n - 1]
    zed_row = np.concatenate([d["rgb_zed"][i] for i in idx], axis=1)
    ego_row = np.concatenate([d["rgb_wrist_left_ego"][i] for i in idx], axis=1)
    montage = np.concatenate([zed_row, ego_row], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(montage).save(out_path)
    print(f"[record_teleop] wrote {out_path} (top row: zed_cam @ steps {idx}, "
          f"bottom row: left_arm_egocentric @ same steps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "teleop" / "data" / "smoke")
    parser.add_argument("--save-frames", type=Path, default=None,
                         help="Also render a start/mid/end montage PNG to this path.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = run_smoke(args.out_dir)
    dt = time.time() - t0
    print(f"[record_teleop] episode={meta['episode_id']} steps={meta['num_steps']} "
          f"success={meta['success']} collision={meta['collision']} "
          f"spilled_count={meta['spilled_count']} ({dt:.1f}s)")
    if args.save_frames is not None:
        save_smoke_png(args.out_dir / meta["episode_id"], args.save_frames)


if __name__ == "__main__":
    main()
