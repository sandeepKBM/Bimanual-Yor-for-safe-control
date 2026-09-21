"""Bimanual perturbation families -- adapted from
VLAism/ur5e_benchmark/src/ur5e_benchmark/benchmark/perturbations.py.

Ported: the seeded-`Random`, `PerturbationSpec(family, level, seed,
parameters)` dataclass shape, and the xy-disc-offset sampling helper
(`_sample_xy_offset`) are carried over near-verbatim -- they are
UR5-agnostic. Adapted: family set is rebuilt for the bimanual scene (two
arms instead of one, a graspable "box" instead of the UR5 apple/handle
object) and for a future cup+balls scene that does not exist yet in
scenes/ (`ball_config` is a documented forward-compat stub -- see README).

Families implemented:
  - "arm_start_shift":   per-arm (left AND right) xy/z/yaw offset applied to
                          the pre-grasp teleport target, same idea as UR5's
                          arm_start_shift (radial xy + z range + yaw range).
  - "object_pose_shift":  xy offset applied to the graspable box's teleport
                          pose before grasp-init (UR5's object_position_shift,
                          renamed for the box object).
  - "ball_config":        STUB. The cup+balls scene (scenes/scene_bimanual_
                          cup.xml) initializes a fixed N=8-ball resting pile
                          from its keyframe -- randomizing ball COUNT/initial
                          layout would require regenerating the XML per
                          episode, out of scope here. Still returns count=0,
                          spread_radius_m=0.0, applicable=False.
  - "cup_behavior":       The cup scene now exists (scenes/scene_bimanual_
                          cup.xml); this family seeds the per-episode
                          DEMONSTRATOR BEHAVIOR for collection/collect_cup.py
                          (which cup motion to script + how strongly), so
                          collection produces a genuine MIX of clean/spilled/
                          collided outcomes instead of one scripted motion
                          repeated identically every episode. `level` selects
                          which behavior:
                            "success_gentle"    -- small, slow lateral cup
                                                    jitter (upright, low accel)
                                                    -> should NOT spill.
                            "spill_shake"        -- fast upright lateral shake
                                                    at accel-spill amplitudes
                                                    tuned in scripts/
                                                    test_cup_scene.py's
                                                    run_upright_shake_test
                                                    (0.03-0.045 m @ 6 Hz ->
                                                    5/8 to 8/8 spilled)
                                                    -> should spill via pure
                                                    acceleration, cup stays
                                                    upright.
                            "spill_tilt"         -- cup lift+tilt (~90-110 deg)
                                                    as in run_tilt_test -> all
                                                    balls spill.
                            "collision_intrusion" -- right arm runs sim_common's
                                                    intruder_ctrl sweep toward
                                                    the left arm/cup region
                                                    (cup left gently static or
                                                    lightly jittered) -> should
                                                    produce left/right arm
                                                    contact (collision=True);
                                                    may also incidentally spill
                                                    if the arm clips the cup.
                          Each level's numeric knobs (shake amp/freq, tilt
                          angle/duration, intrusion timing offset, gentle
                          jitter amplitude) are seeded per-episode via
                          `Random(seed)` so same-behavior episodes still vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians, sin
from random import Random
from typing import Any

FAMILIES = ("arm_start_shift", "object_pose_shift", "ball_config", "cup_behavior")
CUP_BEHAVIOR_LEVELS = ("success_gentle", "spill_shake", "spill_tilt", "collision_intrusion")


@dataclass(frozen=True)
class PerturbationSpec:
    family: str
    level: str
    seed: int
    parameters: dict[str, Any]


def _sample_xy_offset(rng: Random, radius: float) -> tuple[float, float]:
    """Verbatim from ur5e_benchmark.benchmark.perturbations._sample_xy_offset."""
    if radius <= 0.0:
        return 0.0, 0.0
    angle = rng.uniform(0.0, 2.0 * pi)
    magnitude = rng.uniform(0.0, radius)
    return magnitude * cos(angle), magnitude * sin(angle)


# Default per-level radii/ranges (meters / degrees). Small enough that most
# seeds still succeed, large enough that a meaningful fraction miss the grasp
# or clip the other arm -- the point is to yield a MIX of success/failure/
# collision episodes, not to sweep a calibrated difficulty curve (that is a
# ur5e_benchmark-style config-driven sweep, out of scope here).
_LEVELS: dict[str, dict[str, Any]] = {
    "level_0": {"xy_radius_m": 0.0, "z_range_m": (0.0, 0.0), "yaw_range_deg": (0.0, 0.0)},
    "level_1": {"xy_radius_m": 0.01, "z_range_m": (-0.005, 0.005), "yaw_range_deg": (-5.0, 5.0)},
    "level_2": {"xy_radius_m": 0.025, "z_range_m": (-0.012, 0.012), "yaw_range_deg": (-12.0, 12.0)},
    "level_3": {"xy_radius_m": 0.045, "z_range_m": (-0.02, 0.02), "yaw_range_deg": (-20.0, 20.0)},
}


def level_names() -> list[str]:
    return sorted(_LEVELS.keys())


def build_perturbation_spec(family: str, level: str, seed: int) -> PerturbationSpec:
    if family not in FAMILIES:
        raise ValueError(f"Unsupported perturbation family '{family}'. Known: {FAMILIES}")

    rng = Random(int(seed))

    if family == "cup_behavior":
        if level not in CUP_BEHAVIOR_LEVELS:
            raise ValueError(f"Unknown cup_behavior level '{level}'. Known: {CUP_BEHAVIOR_LEVELS}")
        if level == "success_gentle":
            parameters = {
                "amp_m": rng.uniform(0.005, 0.012),
                "freq_hz": rng.uniform(1.0, 2.5),
                "duration_s": rng.uniform(1.0, 1.6),
            }
        elif level == "spill_shake":
            parameters = {
                "amp_m": rng.uniform(0.032, 0.045),
                "freq_hz": rng.uniform(5.5, 6.5),
                "duration_s": rng.uniform(1.0, 1.4),
            }
        elif level == "spill_tilt":
            parameters = {
                "tilt_deg": rng.uniform(90.0, 115.0),
                "tilt_dur_s": rng.uniform(0.8, 1.2),
                "lift_h_m": rng.uniform(0.08, 0.12),
            }
        elif level == "collision_intrusion":
            parameters = {
                "t_offset_s": rng.uniform(-0.1, 0.1),
                "cup_jitter_m": rng.uniform(0.0, 0.006),
            }
        else:  # pragma: no cover - guarded above
            raise AssertionError(level)
        return PerturbationSpec(family=family, level=level, seed=int(seed), parameters=parameters)

    if level not in _LEVELS:
        raise ValueError(f"Unknown level '{level}'. Known: {level_names()}")
    cfg = _LEVELS[level]

    if family == "arm_start_shift":
        left_dx, left_dy = _sample_xy_offset(rng, cfg["xy_radius_m"])
        right_dx, right_dy = _sample_xy_offset(rng, cfg["xy_radius_m"])
        z_lo, z_hi = cfg["z_range_m"]
        yaw_lo, yaw_hi = cfg["yaw_range_deg"]
        parameters = {
            "left_dx_m": left_dx,
            "left_dy_m": left_dy,
            "left_dz_m": rng.uniform(z_lo, z_hi),
            "left_dyaw_rad": radians(rng.uniform(yaw_lo, yaw_hi)),
            "right_dx_m": right_dx,
            "right_dy_m": right_dy,
            "right_dz_m": rng.uniform(z_lo, z_hi),
            "right_dyaw_rad": radians(rng.uniform(yaw_lo, yaw_hi)),
        }
    elif family == "object_pose_shift":
        dx, dy = _sample_xy_offset(rng, cfg["xy_radius_m"])
        parameters = {"box_dx_m": dx, "box_dy_m": dy}
    elif family == "ball_config":
        parameters = {
            "applicable": False,
            "reason": "ball count/layout is fixed by scene_bimanual_cup.xml's keyframe; "
                      "use family='cup_behavior' to randomize per-episode cup motion instead",
            "count": 0,
            "spread_radius_m": 0.0,
        }
    else:  # pragma: no cover - guarded above
        raise AssertionError(family)

    return PerturbationSpec(family=family, level=level, seed=int(seed), parameters=parameters)


__all__ = [
    "FAMILIES",
    "CUP_BEHAVIOR_LEVELS",
    "PerturbationSpec",
    "build_perturbation_spec",
    "level_names",
]
