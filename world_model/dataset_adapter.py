"""world_model/dataset_adapter.py

Converts `collection/` episodes (one `episode.npz` + `metadata.json` per
episode dir, schema in `collection/episode.py`) into the flat-npz format
`dreamerv3-torch`'s own `tools.load_episodes` / `tools.save_episodes` /
`tools.sample_episodes` expect (one npz per episode, all per-step arrays
sharing a single leading time axis, written straight into a directory that
`--offline_traindir` points at).

Key mapping (collection -> dreamerv3-torch obs/action dict):

  image    (T, 64, 64, 6) uint8   = concat([rgb_head, rgb_wrist_right], axis=-1)
                                     dreamerv3-torch's WorldModel.preprocess()
                                     hardcodes a single "image" key (divides
                                     it by 255) -- channel-concatenating our
                                     two cams into one 6-channel blob is what
                                     lets us reuse that code with ZERO edits
                                     to the vendored repo. MultiEncoder/
                                     MultiDecoder treat it as one cnn input;
                                     the two cams are recovered downstream by
                                     slicing channels [0:3] / [3:6] if a
                                     future monitor head wants them split.
  proprio  (T, 26) float32        = concat([left_q(6), left_qd(6), right_q(6),
                                     right_qd(6), left_gripper_state(1),
                                     right_gripper_state(1)])
                                     -- OBJECT-POSE-LEAK FIX (see
                                     docs/design/state_spaces.md's "sim2real
                                     line" section): `box_pose` (7-D; the
                                     cup's ground-truth 6-DoF pose for cup
                                     episodes) is NOT in proprio. It is
                                     privileged sim state -- unavailable on
                                     the real robot (no external tracker) --
                                     so feeding it to the WM would let the
                                     model predict spill from the privileged
                                     pose instead of from vision, and it
                                     would not transfer. Down from the
                                     previous 33-D (which folded box_pose
                                     into proprio).
  object_pose (T, 7) float32      = box_pose (xyz + wxyz quat) -- kept as a
                                     separate LABEL field (supervision/eval
                                     only, e.g. a future failure-monitor head
                                     or offline analysis), never fed to the
                                     encoder/dynamics as an observation.
  action   (T, 16) float32        = concat([left_action(8), right_action(8)])
  reward   (T,) float32           = zeros (no task reward defined yet; the
                                     failure/monitor head this task explicitly
                                     defers is where a real signal would come
                                     from -- reward_head still trains, just
                                     on an all-zero target, which is fine for
                                     a world-model-only smoke/setup)
  discount (T,) float32           = ones (no early-termination signal wanted
                                     for WM dynamics; see is_terminal below
                                     for the cont_head's actual target)
  is_first (T,) bool              = True at t=0 only (per-episode boundary,
                                     required by RSSM.observe to reset state)
  is_terminal (T,) bool           = collision[t] | dropped[t] (real per-step
                                     signal from our sim; feeds cont_head only,
                                     does not affect encoder/dynamics/decoder
                                     losses)

Output: one `<episode_id>-<T>.npz` per input episode under `out_dir`,
written via the vendored `tools.save_episodes` so the on-disk format is
byte-for-byte what `tools.load_episodes(out_dir)` will read back in a real
`dreamer.py` run. `object_pose` is an EXTRA key alongside the ones
dreamerv3-torch's WorldModel actually consumes (image/proprio/action/reward/
discount/is_first/is_terminal) -- tools.save_episodes/load_episodes round-trip
arbitrary keys unmodified (plain **kwargs into np.savez_compressed), and
WorldModel only reads the obs_space keys it's constructed with, so the extra
label field is inert unless a future monitor head is explicitly wired to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DREAMERV3_ROOT = "../dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)

import tools as dreamer_tools  # noqa: E402  (vendored dreamerv3-torch tools.py)

# Proprio field order -- fixed so a trained model's mlp encoder input layout
# is reproducible / documented. box_pose is DELIBERATELY excluded (object
# pose is privileged sim state, not a real-robot observation -- see the
# module docstring's "OBJECT-POSE-LEAK FIX" and
# docs/design/state_spaces.md); it is routed to OBJECT_POSE_FIELD as a
# separate label instead.
PROPRIO_FIELDS: tuple[str, ...] = (
    "left_q",
    "left_qd",
    "right_q",
    "right_qd",
    "left_gripper_state",
    "right_gripper_state",
)
ACTION_FIELDS: tuple[str, ...] = ("left_action", "right_action")
OBJECT_POSE_FIELD = "box_pose"  # LABEL only -- see module docstring


def convert_episode_arrays(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """raw: dict loaded from one collection/ episode.npz (keys = episode.STEP_ARRAY_KEYS).
    Returns a dict in dreamerv3-torch's per-episode format."""
    t = raw["rgb_head"].shape[0]

    image = np.concatenate([raw["rgb_head"], raw["rgb_wrist_right"]], axis=-1)
    assert image.shape == (t, 64, 64, 6), image.shape

    proprio_parts = []
    for key in PROPRIO_FIELDS:
        arr = np.asarray(raw[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        proprio_parts.append(arr)
    proprio = np.concatenate(proprio_parts, axis=-1).astype(np.float32)
    assert proprio.shape == (t, 26), proprio.shape

    object_pose = np.asarray(raw[OBJECT_POSE_FIELD], dtype=np.float32)
    assert object_pose.shape == (t, 7), object_pose.shape

    action = np.concatenate(
        [np.asarray(raw[k], dtype=np.float32) for k in ACTION_FIELDS], axis=-1
    ).astype(np.float32)

    is_first = np.zeros(t, dtype=bool)
    is_first[0] = True
    is_terminal = np.asarray(raw["collision"], dtype=bool) | np.asarray(
        raw["dropped"], dtype=bool
    )

    return {
        "image": image.astype(np.uint8),
        "proprio": proprio,
        "object_pose": object_pose,  # LABEL only -- NOT a WM observation, see module docstring
        "action": action,
        "reward": np.zeros(t, dtype=np.float32),
        "discount": np.ones(t, dtype=np.float32),
        "is_first": is_first,
        "is_terminal": is_terminal,
    }


def convert_dir(collection_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Convert every `<collection_dir>/episode_*/episode.npz` into one
    dreamerv3-torch-format npz under out_dir. Returns the written paths."""
    collection_dir = Path(collection_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(collection_dir.glob("episode_*"))
    assert episode_dirs, f"no episode_* dirs found under {collection_dir}"

    written: list[Path] = []
    for ep_dir in episode_dirs:
        npz_path = ep_dir / "episode.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as f:
            raw = {k: f[k] for k in f.keys()}
        converted = convert_episode_arrays(raw)
        # tools.save_episodes writes <out_dir>/<filename>-<length>.npz and
        # returns True; filename here is the episode id (e.g. episode_000000).
        dreamer_tools.save_episodes(out_dir, {ep_dir.name: converted})
        length = len(converted["reward"])
        written.append(out_dir / f"{ep_dir.name}-{length}.npz")
    return written


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--collection-dir",
        default="collection/data/smoke",
    )
    p.add_argument(
        "--out-dir",
        default="world_model/data/dreamer_fmt/smoke",
    )
    args = p.parse_args()
    paths = convert_dir(args.collection_dir, args.out_dir)
    print(f"[dataset_adapter] wrote {len(paths)} episodes to {args.out_dir}")
    for p_ in paths:
        print(f"  {p_}")
