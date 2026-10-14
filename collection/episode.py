"""Bimanual episode schema + writer -- Dreamer/RSSM-friendly layout.

Adapted from VLAism/ur5e_benchmark/src/ur5e_benchmark/benchmark/episode_schema.py
(the `EpisodeMetadata` dataclass + `build_episode_metadata` free function
pattern) and artifact_keys.py (the SSOT-tuple-of-required-keys idea, so a
future writer/reader drift is a loud assertion error, not a silent schema
mismatch). UR5-specific fields (object_type/object_shape/placement_success
tied to the apple-in-box task) were dropped/renamed for the bimanual box-pick
task; grasp_boundary/converter.py's per-episode-file (not per-dataset-row)
layout is the model for storing one .npz + one .json per episode instead of
ur5e_benchmark's multi-file-per-episode (summary/metadata/action_log/...)
split -- simpler and closer to what a Dreamer/RSSM dataloader wants (one
contiguous trajectory blob per episode).

GRIPPER CONVENTION (the one unambiguous choice used everywhere in this
module and in collect.py):
    gripper_state in [0, 1] = qpos(<side>/joint7) / 0.035
        0.0 == fully closed, 1.0 == fully open.
joint7's MJCF range is [0, 0.035] (bimanual_piper_pickup.xml); joint8 mirrors
joint7 with the opposite sign and is NOT separately encoded -- it is not an
independent DOF in practice (both fingers are always commanded together, see
sim_common.py's docstring), so recording joint7 alone is lossless enough for
world-model purposes and avoids an ambiguous "two gripper numbers that should
usually agree" field.

Storage layout per episode, under <out_dir>/<episode_id>/:
    episode.npz   -- all per-step arrays (images + low-dim), see STEP_ARRAY_KEYS
    metadata.json -- scalar/episode-level fields, see METADATA_REQUIRED_KEYS
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collection.perturbations import PerturbationSpec

SCHEMA_COMPAT_VERSION = "bimanual_collection_v1"
GRIPPER_JOINT_RANGE_M = 0.035  # see GRIPPER CONVENTION docstring above

# SSOT: arrays that MUST be present in every episode.npz (mirrors
# artifact_keys.py's REQUIRED_* tuples -- a writer/reader drift becomes an
# assertion, not a silent gap).
STEP_ARRAY_KEYS: tuple[str, ...] = (
    "rgb_head",            # (T, H, W, 3) uint8
    "rgb_wrist_right",      # (T, H, W, 3) uint8
    "left_q",               # (T, 6) float32 -- left arm joint positions
    "left_qd",              # (T, 6) float32 -- left arm joint velocities
    "right_q",              # (T, 6) float32
    "right_qd",              # (T, 6) float32
    "left_action",           # (T, 8) float32 -- ctrl targets: 6 arm + 2 finger
    "right_action",          # (T, 8) float32
    "left_gripper_state",    # (T,) float32 in [0, 1], see GRIPPER CONVENTION
    "right_gripper_state",   # (T,) float32 in [0, 1]
    "box_pose",              # (T, 7) float32 -- xyz + wxyz quat of box_center
    "collision",             # (T,) bool -- left/right arm geom-geom contact this step
    "dropped",               # (T,) bool -- box below drop-height threshold this step
)

METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "episode_id",
    "seed",
    "perturbation_family",
    "perturbation_level",
    "perturbation_parameters",
    "policy_name",
    "num_steps",
    "success",
    "grasp_success",
    "collision",
    "spilled_count",
    "gripper_convention",
    "schema_compat_version",
)


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    seed: int
    perturbation_family: str
    perturbation_level: str
    perturbation_parameters: dict[str, Any]
    policy_name: str
    num_steps: int
    success: bool
    grasp_success: bool
    collision: bool
    spilled_count: int
    gripper_convention: str
    schema_compat_version: str


def build_episode_metadata(
    *,
    episode_index: int,
    seed: int,
    perturbation: PerturbationSpec,
    policy_name: str,
    num_steps: int,
    success: bool,
    grasp_success: bool,
    collision: bool,
    spilled_count: int = 0,
) -> dict[str, Any]:
    """spilled_count is always 0 today (no cup/balls scene exists yet -- see
    perturbations.py's `ball_config` stub and README.md); the field is kept
    in the schema now so the writer/dataloader contract does not change once
    scenes/scene_bimanual_cup.xml lands and a real spill count is computed
    (count of ball_* bodies whose xy leaves the cup's rim radius)."""
    metadata = EpisodeMetadata(
        episode_id=f"episode_{episode_index:06d}",
        seed=int(seed),
        perturbation_family=str(perturbation.family),
        perturbation_level=str(perturbation.level),
        perturbation_parameters=dict(perturbation.parameters),
        policy_name=str(policy_name),
        num_steps=int(num_steps),
        success=bool(success),
        grasp_success=bool(grasp_success),
        collision=bool(collision),
        spilled_count=int(spilled_count),
        gripper_convention=(
            "gripper_state = qpos(<side>/joint7) / 0.035; 0=closed, 1=open"
        ),
        schema_compat_version=SCHEMA_COMPAT_VERSION,
    )
    payload = asdict(metadata)
    missing = [k for k in METADATA_REQUIRED_KEYS if k not in payload]
    assert not missing, f"build_episode_metadata dropped required keys: {missing}"
    return payload


class EpisodeWriter:
    """Accumulates per-step dicts, then writes one .npz + one .json."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    def add_step(self, **fields: Any) -> None:
        missing = [k for k in STEP_ARRAY_KEYS if k not in fields]
        assert not missing, f"EpisodeWriter.add_step missing required keys: {missing}"
        self._steps.append(fields)

    def __len__(self) -> int:
        return len(self._steps)

    def save(self, out_dir: str | Path, metadata: dict[str, Any]) -> Path:
        """Persists every key actually recorded via add_step -- STEP_ARRAY_KEYS
        (required, checked in add_step) PLUS any extra keys a caller chose to
        add (e.g. collect_cup.py's `rgb_zed`, recorded alongside the required
        `rgb_head`/`rgb_wrist_right` as auxiliary data). All steps must carry
        the same key set (asserted below) -- this keeps the box-pickup task's
        collect.py (which only ever passes STEP_ARRAY_KEYS) byte-for-byte
        unaffected while letting other collectors add extra per-step arrays
        without a shared-schema change."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_keys = set(self._steps[0].keys())
        mismatched = [i for i, step in enumerate(self._steps) if set(step.keys()) != all_keys]
        assert not mismatched, f"EpisodeWriter.save: inconsistent per-step keys at steps {mismatched[:5]}"

        arrays: dict[str, np.ndarray] = {}
        for key in all_keys:
            stacked = np.stack([step[key] for step in self._steps], axis=0)
            arrays[key] = stacked

        npz_path = out_dir / "episode.npz"
        np.savez_compressed(npz_path, **arrays)

        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return out_dir


__all__ = [
    "SCHEMA_COMPAT_VERSION",
    "GRIPPER_JOINT_RANGE_M",
    "STEP_ARRAY_KEYS",
    "METADATA_REQUIRED_KEYS",
    "EpisodeMetadata",
    "build_episode_metadata",
    "EpisodeWriter",
]
