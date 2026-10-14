#!/usr/bin/env python
"""Bimanual STACKING data-collection rollout loop -- PROPER YOR robot
(scenes/scene_yor_stack.xml, see scenes/YOR_CANONICAL.md / scenes/STACK_TASK.md).

Scene: 8 free-floating 4cm blocks, REACH-PARTITIONED so neither arm can build
the shared central tower alone -- 3 blocks reachable only by the LEFT arm
(scripts/stack_task_utils.BLOCK_SIDE == "left"), 3 only by the RIGHT arm, and
2 "shared" blocks + the tower zone itself in the arms' overlap region (both
per the FK-sweep envelope in scenes/YOR_CANONICAL.md). Demonstrator: NOT a
trained policy -- a scripted, per-behavior-family, multi-phase bimanual
pick-and-place built on teleop.bimanual_teleop's DLS IK (`bt.drive`,
`bt.BimanualIK`, `bt.coarse_seed`, all reused/unmodified) for the arms'
reach/carry ANIMATION.

TRANSPORT-FIDELITY NOTE (read before touching the grasp logic): this shared
robot's arm actuators ship with LOW position gains (kp 5-10, see
scenes/YOR_CANONICAL.md), which the teleop layer's own smoke test already
flags as producing "a few cm of steady-state PD lag" -- and its cup-pickup
smoke test never achieved a real friction grasp at all (box_pose bit-
identical every step, teleop/README.md). Empirically here that lag is a
genuine, non-shrinking (does not improve with more hold time) steady-state
error of several cm, which is intolerant for a real friction/pinch grasp of a
4cm block, AND too imprecise to place one block within a block-width of
another purely by tracking the arm's actual reached site position. Two real
issues were found/handled:
  1. A genuine BUG (fixed in scripts/stack_task_utils.py's
     `_restore_finger_object_collision`): `sim_common._patch_real_collision_geoms`
     (reused for the arm-arm `collision` label) also remaps the finger pad
     geoms' contype/conaffinity (they ship in the same "group=3" class as the
     arm-link collision proxies), which silently made the fingers incapable
     of colliding with ANYTHING but the other arm. Fixed in-memory, in our
     own loader, without touching the shared sim_common.py.
  2. The remaining PD-lag-vs-mm-precision gap is a physical/tuning
     characteristic of the shared, read-only robot MJCF, not something this
     task is scoped to retune. So the ARM's motion (via bt.drive) is real IK,
     used for a plausible reach/carry ANIMATION, while the HELD BLOCK follows
     its own precisely SCRIPTED waypoint path (`attach_scripted`/
     `add_scripted_waypoint`, decoupled from the arm's actual tracking
     error) for the pick-up and transit legs -- but is DETACHED (real
     gravity/contact physics takes back over) a couple cm above its target,
     so whether a placement is clean, misplaced, or topples the tower is a
     genuine physics outcome of the final drop, never scripted. This mirrors
     this repo's own existing precedent (collect_cup.py scripts the cup's
     motion via direct qpos writes for its ENTIRE episode, no contact grasp
     attempted at all) but goes further: only the transit leg is scripted,
     never the task-relevant failure physics (the drop/settle/topple).

Both arms independently fetch their OWN reach-exclusive block first (this is
what makes the task genuinely bimanual: block_l1 is unreachable by the right
arm, block_r1 by the left), then LEFT additionally reaches ACROSS the body
midline to fetch a "shared" block (block_s1, at y=-0.45, on the right-ish side
of the table) to demonstrate cross-reach into the other arm's territory.

Per-episode BEHAVIOR (`STACK_BEHAVIOR_LEVELS`, seeded via seed % 3):
  - "clean_stack":      both arms place their blocks centered, small release
                         gap -> a clean 3-block tower, no topple, no collision.
  - "topple_misplace":  identical for the first two blocks, but the third
                         (cross-reach) block is released offset sideways and
                         from extra height -> real physics topples the tower
                         (count_toppled fires).
  - "collision_bump":   after each arm independently picks its own block, BOTH
                         arms (and their carried blocks) are commanded to the
                         SAME point near the tower center simultaneously --
                         reproducing the exact "both grasp sites sent to one
                         point -> crossed/tangled arms" finding documented in
                         scenes/YOR_CANONICAL.md -- producing a real left/
                         right arm-arm contact (collision=True), which may
                         also knock over/scatter whatever was on the tower.

Ground-truth labels every step: scripts.stack_task_utils.collision (real
left_arm_*/right_arm_* contact), .count_toppled (blocks near the tower zone
that fell over or were knocked off -- the WM-visible TOWER FAILURE), and
.block_dropped (any block knocked off the table/platform onto the floor).
scripts.stack_task_utils.tower_height is also recorded every step (extra,
non-required field) as a dense progress signal.

Per-step schema note (collection/episode.py's STEP_ARRAY_KEYS is shared with
the box-pickup/cup tasks and is NOT modified here, same repurposing
convention collect_cup.py/record_teleop.py use):
  - "box_pose"  repurposed to carry the pose of whichever block is currently
                the demonstrator's "focus" object (the one just grasped/being
                carried), or the fixed tower-zone-marker pose before the first
                grasp. Label only.
  - "dropped"   repurposed to carry the per-step block_dropped() flag.
  - metadata "spilled_count" repurposed as the final count_toppled() value.
  - metadata "grasp_success" repurposed as "tower reached >=3 blocks, none
    toppled" (scripts.stack_task_utils.is_stacked_success); "success"
    additionally requires no collision (collect_cup.py's convention).

Cameras: same convention as teleop/record_teleop.py's TeleopEpisodeRenderer
(reused directly, unmodified) -- the real ZED workspace camera (`zed_cam`,
scripts/zed_camera.py, attached by scripts.stack_task_utils.load_yor_stack_model)
is stored into BOTH `rgb_head` and `rgb_wrist_right` (world_model/
dataset_adapter.py is not touched here either; same ZED-duplication trick as
collect_cup.py), plus `rgb_zed` (dedicated key) and the two real egocentric
wrist cams as auxiliary extras.
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

from scripts.stack_task_utils import (  # noqa: E402
    TABLE_TOP_Z,
    TOWER_XY,
    block_dropped,
    collision as collision_fn,
    count_toppled,
    is_stacked_success,
    load_yor_stack_model,
    tower_height,
)
from collection.episode import (  # noqa: E402
    EpisodeWriter,
    build_episode_metadata,
)
from collection.perturbations import PerturbationSpec  # noqa: E402
from teleop import bimanual_teleop as bt  # noqa: E402
from teleop.record_teleop import TeleopEpisodeRenderer  # noqa: E402

RENDER_SIZE = 64  # matches world_model/dataset_adapter.py's hardcoded (T,64,64,6) assert
STACK_BEHAVIOR_LEVELS = ("clean_stack", "topple_misplace", "collision_bump")
BLOCK_HALF = 0.02
HOVER = 0.15  # pre-grasp/pre-place approach clearance, meters
RELEASE_GAP = 0.015  # release the block this far above its target -- real physics drops it the rest

# Block start positions -- MUST match scenes/scene_yor_stack.xml's <body pos=...>.
BLOCK_POS = {
    "block_l1": (0.34, 0.29, 0.57), "block_l2": (0.42, 0.36, 0.57), "block_l3": (0.50, 0.42, 0.57),
    "block_r1": (0.34, -0.69, 0.57), "block_r2": (0.42, -0.76, 0.57), "block_r3": (0.50, -0.82, 0.57),
    "block_s1": (0.20, -0.45, 0.57), "block_s2": (0.64, -0.02, 0.57),
}
LEFT_PARK = np.array([0.42, 0.35, 0.72])   # above the left platform, clear of the tower
RIGHT_PARK = np.array([0.42, -0.75, 0.72])  # above the right platform, clear of the tower


def _sample_waypoints(start_pos: np.ndarray, waypoints: list, t: float) -> np.ndarray:
    """Piecewise-linear interpolation through `waypoints` (list of
    (pos, duration, hold)), same shape/semantics as
    teleop.bimanual_teleop._SideTrack.sample (position-only, no gripper/quat)
    -- reimplemented here (that class is private to teleop.bimanual_teleop)
    to drive a HELD BLOCK's scripted transit, decoupled from the arm's own
    (imprecise) tracking. Holds at the last waypoint once `t` exceeds the
    total schedule, so extra elapsed time is a stable no-op, not drift."""
    pos = start_pos
    t0 = 0.0
    for wp_pos, duration, hold in waypoints:
        t_end_move = t0 + duration
        t_end_hold = t_end_move + hold
        if t <= t_end_move:
            s = 0.0 if duration <= 0 else float(np.clip((t - t0) / duration, 0.0, 1.0))
            return pos + s * (wp_pos - pos)
        elif t <= t_end_hold:
            return wp_pos
        pos = wp_pos
        t0 = t_end_hold
    return pos


class StackEpisodeRecorder:
    """Drives a multi-phase bimanual stacking trajectory (each phase = one
    teleop.bimanual_teleop.drive() call for one or both arms) and records one
    episode via collection/episode.py's EpisodeWriter, field-name-compatible
    with collection/collect_cup.py / teleop/record_teleop.py (see module
    docstring for the box_pose/dropped repurposing and the scripted-block-
    transport rationale)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.ik = bt.BimanualIK(model)
        self.renderer = TeleopEpisodeRenderer(model, size=RENDER_SIZE)
        self.writer = EpisodeWriter()

        self.left_pos_act = bt.arm_pos_actuators(model, "left")
        self.right_pos_act = bt.arm_pos_actuators(model, "right")
        self.left_finger_act = bt.finger_actuators(model, "left")
        self.right_finger_act = bt.finger_actuators(model, "right")
        self.left_qposadr = np.array([model.joint(n).qposadr[0] for n in bt.ARM_JOINTS["left"]])
        self.left_dofadr = np.array([model.joint(n).dofadr[0] for n in bt.ARM_JOINTS["left"]])
        self.right_qposadr = np.array([model.joint(n).qposadr[0] for n in bt.ARM_JOINTS["right"]])
        self.right_dofadr = np.array([model.joint(n).dofadr[0] for n in bt.ARM_JOINTS["right"]])

        self.ever_collision = False
        self.ever_toppled = False
        self.ever_dropped = False
        self._imgs = None
        self._global_step = 0
        self._focus_block: str | None = None
        self._scripted: dict[str, dict] = {}  # side -> {bid, qadr, dofadr, start_pos, waypoints, elapsed}
        self._tower_zone_pose = np.array(
            [TOWER_XY[0], TOWER_XY[1], TABLE_TOP_Z + 0.001, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def set_focus(self, block_name: str | None) -> None:
        self._focus_block = block_name

    # -- scripted block transport (see module docstring) ---------------------

    def attach_scripted(self, side: str, block_name: str, waypoints: list | None = None) -> None:
        """Start scripting `block_name`'s pose from its CURRENT actual pose
        (queried now, so it's exact) through `waypoints` -- every subsequent
        physics step (until `detach_scripted`) the block's POSITION is forced
        along this precomputed path AND its ORIENTATION is pinned to this
        starting orientation, with zero velocity, fully decoupled from the
        arm's own IK tracking accuracy. Pinning orientation too (not just
        position) matters: the tracking arm's fingers can still incidentally
        overlap a forcibly-positioned block each step (real contype/
        conaffinity collision, see module docstring's fix #1), and a single
        mj_step's contact-resolution torque on an unconstrained orientation
        silently accumulates into a wildly tumbled quat by the time of
        release -- verified empirically (quat drifted from ~identity to
        [0.67, 0.38, 0.61, 0.18] over one transit with position-only
        pinning)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block_name)
        jadr = self.model.body_jntadr[bid]
        self._scripted[side] = {
            "bid": bid,
            "qadr": int(self.model.jnt_qposadr[jadr]),
            "dofadr": int(self.model.jnt_dofadr[jadr]),
            "start_pos": self.data.xpos[bid].copy(),
            "start_quat": self.data.xquat[bid].copy(),
            "waypoints": list(waypoints) if waypoints else [],
            "elapsed": 0,
        }

    def add_scripted_waypoint(self, side: str, pos: np.ndarray, duration: float, hold: float) -> None:
        self._scripted[side]["waypoints"].append((np.asarray(pos, dtype=float), duration, hold))

    def detach_scripted(self, side: str) -> None:
        self._scripted.pop(side, None)

    def _apply_scripted(self, d: mujoco.MjData) -> None:
        if not self._scripted:
            return
        dt = self.model.opt.timestep
        for info in self._scripted.values():
            t = info["elapsed"] * dt
            pos = _sample_waypoints(info["start_pos"], info["waypoints"], t)
            d.qpos[info["qadr"]:info["qadr"] + 3] = pos
            d.qpos[info["qadr"] + 3:info["qadr"] + 7] = info["start_quat"]
            d.qvel[info["dofadr"]:info["dofadr"] + 6] = 0.0
            info["elapsed"] += 1
        mujoco.mj_forward(self.model, d)

    # -- arm animation helpers (real IK via bt.drive; imprecision is fine -- --
    # -- the held block's actual path is scripted, not tied to arm accuracy) -

    def seed_down(self, side: str, target_xyz, seed: int = 0) -> None:
        """Down-biased coarse seed (bt.coarse_seed, reused) so the arm's
        subsequent bt.drive() tracking starts from a "reach down"
        configuration -- purely cosmetic here (a nicer-looking approach),
        not load-bearing for placement accuracy."""
        arm = self.ik.arms[side]
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        seed_q = bt.coarse_seed(self.model, scratch, arm.site_id, arm.gripper_body_id,
                                 np.asarray(target_xyz, dtype=float), arm.qposadr, arm.jnt_range,
                                 n=8000, seed=seed, want_down=True)
        if seed_q is not None:
            arm.last_q = seed_q

    def hold_gripper(self, side: str, gripper: float, duration: float = 0.3, hold: float = 0.3) -> None:
        """Close/open the gripper in place (visual) -- a single-position
        Segment via bt.drive() so it's recorded through the normal path."""
        pos = self.data.site_xpos[self.ik.arms[side].site_id].copy()
        self.run_phase({side: [bt.Segment(pos=pos, gripper=gripper, duration=duration, hold=hold)]})

    def hover_segments(self, xyz, hover: float = HOVER, gripper: float = 1.0):
        x, y, z = xyz
        return [bt.Segment(pos=np.array([x, y, z + hover]), gripper=gripper, duration=1.0, hold=0.1)]

    # -- recording -------------------------------------------------------

    def _on_step(self, step: int, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        self._apply_scripted(d)

        collision = collision_fn(m, d)
        toppled = count_toppled(m, d)
        dropped = block_dropped(m, d)
        self.ever_collision = self.ever_collision or collision
        self.ever_toppled = self.ever_toppled or (toppled > 0)
        self.ever_dropped = self.ever_dropped or dropped

        if self._imgs is None or self._global_step % 2 == 0:
            self._imgs = self.renderer.render(d)
        imgs = self._imgs

        if self._focus_block is not None:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self._focus_block)
            box_pose = np.concatenate([d.xpos[bid], d.xquat[bid]]).astype(np.float32)
        else:
            box_pose = self._tower_zone_pose

        self.writer.add_step(
            rgb_head=imgs["rgb_head"],
            rgb_wrist_right=imgs["rgb_wrist_right"],
            rgb_zed=imgs["rgb_zed"],
            rgb_wrist_left_ego=imgs["rgb_wrist_left_ego"],
            rgb_wrist_right_ego=imgs["rgb_wrist_right_ego"],
            left_q=d.qpos[self.left_qposadr].astype(np.float32).copy(),
            left_qd=d.qvel[self.left_dofadr].astype(np.float32).copy(),
            right_q=d.qpos[self.right_qposadr].astype(np.float32).copy(),
            right_qd=d.qvel[self.right_dofadr].astype(np.float32).copy(),
            left_action=d.ctrl[self.left_pos_act + self.left_finger_act].astype(np.float32).copy(),
            right_action=d.ctrl[self.right_pos_act + self.right_finger_act].astype(np.float32).copy(),
            left_gripper_state=np.float32(bt.gripper_state(m, d, "left")),
            right_gripper_state=np.float32(bt.gripper_state(m, d, "right")),
            box_pose=box_pose,  # repurposed: focus-block pose, LABEL only
            collision=bool(collision),
            dropped=bool(dropped),  # repurposed: per-step block_dropped() flag
            tower_height=np.float32(tower_height(m, d)),  # extra: dense progress label
            count_toppled=np.float32(toppled),            # extra: dense failure label
        )
        self._global_step += 1

    def run_phase(self, targets: dict[str, list["bt.Segment"]], extra_steps: int = 0) -> None:
        bt.drive(self.model, self.data, targets, ik=self.ik, on_step=self._on_step, extra_steps=extra_steps)

    def close(self) -> None:
        self.renderer.close()


def _pick_place_place(rec: StackEpisodeRecorder, side: str, block_name: str, level_idx: int,
                       park: np.ndarray, target_xy=TOWER_XY, xy_offset=(0.0, 0.0),
                       drop_extra: float = 0.0, release_gap: float = RELEASE_GAP,
                       seed: int = 0) -> np.ndarray:
    """One full pick-carry-release cycle for `block_name` via `side`, landing
    at tower level `level_idx` ABOVE `target_xy` (+ optional misplace
    offset). The arm's reach/carry is a real (if imprecise) bt.drive() IK
    animation; the held BLOCK follows its own exact scripted waypoint path
    (see module docstring), released `release_gap` above its target so real
    physics governs the final drop/settle -- clean, misplaced, or toppled is
    a genuine physics outcome. Returns the block's ACTUAL final (x, y) so the
    caller can stack the NEXT block directly on top of where this one really
    landed."""
    rec.set_focus(block_name)
    pick_xyz = np.asarray(BLOCK_POS[block_name], dtype=float)
    pick_hover = pick_xyz + np.array([0.0, 0.0, HOVER])

    z = TABLE_TOP_Z + BLOCK_HALF + level_idx * 2.0 * BLOCK_HALF
    block_target = np.array([target_xy[0] + xy_offset[0], target_xy[1] + xy_offset[1], z + drop_extra])
    release_pos = block_target + np.array([0.0, 0.0, release_gap])
    place_hover = np.array([block_target[0], block_target[1], block_target[2] + HOVER])

    # -- reach + close (arm: real IK animation; block: untouched, still resting) --
    rec.seed_down(side, pick_xyz, seed=seed)
    rec.run_phase({side: rec.hover_segments(pick_xyz, gripper=1.0)
                   + [bt.Segment(pos=pick_xyz, gripper=1.0, duration=0.6, hold=0.1),
                      bt.Segment(pos=pick_xyz, gripper=0.0, duration=0.35, hold=0.1)]})

    # -- carry (block: scripted waypoints; arm: matching real IK animation) --
    rec.attach_scripted(side, block_name, waypoints=[
        (pick_hover, 0.6, 0.1),
        (place_hover, 1.0, 0.1),
        (release_pos, 0.6, 0.3),
    ])
    rec.run_phase({side: [
        bt.Segment(pos=pick_hover, gripper=0.0, duration=0.6, hold=0.1),
        bt.Segment(pos=place_hover, gripper=0.0, duration=1.0, hold=0.1),
        bt.Segment(pos=release_pos, gripper=0.0, duration=0.6, hold=0.3),
    ]})
    rec.detach_scripted(side)  # real physics takes over from HERE -- topple/success is a genuine outcome
    rec.hold_gripper(side, 1.0)  # open (visual)
    rec.run_phase({side: [bt.Segment(pos=place_hover, gripper=1.0, duration=0.6, hold=0.1)]})
    rec.run_phase({side: [bt.Segment(pos=park, gripper=1.0, duration=0.8, hold=0.1)]})
    rec.run_phase({}, extra_steps=100)  # let the drop/settle play out before measuring

    bid = mujoco.mj_name2id(rec.model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    return rec.data.xpos[bid][:2].copy()


def run_clean_stack(rec: StackEpisodeRecorder) -> None:
    xy = _pick_place_place(rec, "left", "block_l1", 0, LEFT_PARK, seed=0)
    xy = _pick_place_place(rec, "right", "block_r1", 1, RIGHT_PARK, target_xy=xy, seed=1)
    _pick_place_place(rec, "left", "block_s1", 2, LEFT_PARK, target_xy=xy, seed=2)  # cross-reach shared block
    rec.set_focus(None)
    rec.run_phase({}, extra_steps=150)  # settle


def run_topple_misplace(rec: StackEpisodeRecorder) -> None:
    xy = _pick_place_place(rec, "left", "block_l1", 0, LEFT_PARK, seed=0)
    xy = _pick_place_place(rec, "right", "block_r1", 1, RIGHT_PARK, target_xy=xy, seed=1)
    # Deliberate misplace: cross-reach block released 4.5cm off-center and
    # from a much larger height -> real physics knocks the stack over.
    _pick_place_place(rec, "left", "block_s1", 2, LEFT_PARK, target_xy=xy, seed=2,
                       xy_offset=(0.045, 0.0), release_gap=0.05)
    rec.set_focus(None)
    rec.run_phase({}, extra_steps=200)  # settle -- let the topple play out


def run_collision_bump(rec: StackEpisodeRecorder) -> None:
    # Each arm independently fetches its own block first (safe, disjoint regions).
    l1_xyz = np.asarray(BLOCK_POS["block_l1"], dtype=float)
    rec.set_focus("block_l1")
    rec.seed_down("left", l1_xyz, seed=0)
    rec.run_phase({"left": rec.hover_segments(l1_xyz, gripper=1.0)
                   + [bt.Segment(pos=l1_xyz, gripper=1.0, duration=0.6, hold=0.1),
                      bt.Segment(pos=l1_xyz, gripper=0.0, duration=0.35, hold=0.1)]})
    rec.attach_scripted("left", "block_l1", waypoints=[(l1_xyz + [0.0, 0.0, HOVER], 0.6, 0.4)])

    r1_xyz = np.asarray(BLOCK_POS["block_r1"], dtype=float)
    rec.set_focus("block_r1")
    rec.seed_down("right", r1_xyz, seed=0)
    rec.run_phase({"right": rec.hover_segments(r1_xyz, gripper=1.0)
                   + [bt.Segment(pos=r1_xyz, gripper=1.0, duration=0.6, hold=0.1),
                      bt.Segment(pos=r1_xyz, gripper=0.0, duration=0.35, hold=0.1)]})
    rec.attach_scripted("right", "block_r1", waypoints=[(r1_xyz + [0.0, 0.0, HOVER], 0.6, 0.4)])

    # The trigger: BOTH arms (and their scripted-carried blocks) commanded to
    # the SAME point near the tower center, simultaneously -- reproduces
    # scenes/YOR_CANONICAL.md's documented finding that sending both grasp
    # sites to one shared point produces visibly crossed/tangled arms, i.e. a
    # real left/right arm-arm contact.
    common = np.array([TOWER_XY[0], TOWER_XY[1], TABLE_TOP_Z + 0.10])
    rec.add_scripted_waypoint("left", common, 1.0, 0.4)
    rec.add_scripted_waypoint("right", common, 1.0, 0.4)
    rec.set_focus("block_l1")
    rec.run_phase({
        "left": [bt.Segment(pos=common, gripper=0.0, duration=1.0, hold=0.4)],
        "right": [bt.Segment(pos=common, gripper=0.0, duration=1.0, hold=0.4)],
    })
    # Release both (real physics settle -- blocks land wherever this bump left them)
    # and retreat clear of the tower zone.
    rec.detach_scripted("left")
    rec.detach_scripted("right")
    rec.set_focus(None)
    rec.run_phase({
        "left": [bt.Segment(pos=LEFT_PARK, gripper=1.0, duration=0.8, hold=0.1)],
        "right": [bt.Segment(pos=RIGHT_PARK, gripper=1.0, duration=0.8, hold=0.1)],
    })
    rec.run_phase({}, extra_steps=150)  # settle


BEHAVIOR_FNS = {
    "clean_stack": run_clean_stack,
    "topple_misplace": run_topple_misplace,
    "collision_bump": run_collision_bump,
}


def _init_settled_state(model: mujoco.MjModel, settle_s: float = 0.3) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    for _ in range(int(settle_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def collect_episode(episode_index: int, seed: int, out_dir: Path, *, level: str | None = None) -> dict:
    if level is None:
        level = STACK_BEHAVIOR_LEVELS[seed % len(STACK_BEHAVIOR_LEVELS)]

    model, _ = load_yor_stack_model()
    data = _init_settled_state(model)

    rec = StackEpisodeRecorder(model, data)
    try:
        BEHAVIOR_FNS[level](rec)

        final_height = tower_height(model, data)
        final_toppled = count_toppled(model, data)
        grasp_success = is_stacked_success(model, data, k=3)  # pure tower-structure condition
        success = grasp_success and not rec.ever_collision

        spec = PerturbationSpec(family="stack_behavior", level=level, seed=seed, parameters={})
        metadata = build_episode_metadata(
            episode_index=episode_index,
            seed=seed,
            perturbation=spec,
            policy_name=f"stack_scripted_{level}_yor_v1",
            num_steps=len(rec.writer),
            success=success,
            grasp_success=grasp_success,
            collision=rec.ever_collision,
            spilled_count=final_toppled,  # repurposed: final count_toppled()
        )
        episode_dir = out_dir / metadata["episode_id"]
        rec.writer.save(episode_dir, metadata)
        metadata["_final_tower_height"] = final_height  # debug/print only, not in metadata.json
        return metadata
    finally:
        rec.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "collection" / "data" / "stack_yor_smoke")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--level", type=str, default=None, choices=(None, *STACK_BEHAVIOR_LEVELS),
                         help="force one behavior for all episodes; default cycles seed %% 3 for a mix")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        meta = collect_episode(i, seed, args.out_dir, level=args.level)
        results.append(meta)
        print(
            f"[collect_stack] episode={meta['episode_id']} seed={seed} level={meta['perturbation_level']} "
            f"success={meta['success']} tower_height={meta['_final_tower_height']} "
            f"toppled={meta['spilled_count']} collision={meta['collision']} steps={meta['num_steps']}"
        )
    dt = time.time() - t0
    n_success = sum(1 for r in results if r["success"])
    n_toppled = sum(1 for r in results if r["spilled_count"] > 0)
    n_collision = sum(1 for r in results if r["collision"])
    print(f"[collect_stack] wrote {len(results)} episodes to {args.out_dir} in {dt:.1f}s "
          f"(success={n_success} toppled={n_toppled} collision={n_collision})")


if __name__ == "__main__":
    main()
