"""assignment rule (pure python, no sim / GPU)."""
import math

from world_model.v2.house.pick import assign as A


def test_spec_constants():
    assert A.GRIPPER_OPEN_M == 0.07                      # YOR/robot/arm/arm.py:12
    assert abs(A.SINGLE_MAX_WIDTH_M - 0.060) < 1e-9
    assert A.SINGLE_MAX_KG == 0.75                       # assumed payload 1.5 kg x 0.5 derate (NOT a repo value)


def test_small_object_single_arm_side():
    p = A.assign((0.056, 0.056, 0.22), (0.6, 0.08, 0.83))          # bottle, left of the centre line (-0.21)
    assert (p.mode, p.arm, p.grasp) == ("single", "left", "pinch")
    p = A.assign((0.055, 0.055, 0.095), (0.52, -0.64, 0.77))       # mug, right
    assert (p.mode, p.arm) == ("single", "right")


def test_wide_or_heavy_is_bimanual():
    p = A.assign((0.30, 0.20, 0.14), (0.5, -0.21, 0.79), yaw=math.pi / 2)    # long axis along robot y
    assert p.mode == "bimanual" and p.arm == "both" and p.grasp == "squeeze"
    assert abs(p.width - 0.30) < 1e-9
    assert abs(abs(math.sin(p.axis)) - 1.0) < 1e-6                           # hands approach along +-y


def test_jaw_limit_boundary():
    w = A.SINGLE_MAX_WIDTH_M
    assert A.assign((w, w, 0.05), (0.5, 0, 0.8)).mode == "single"
    assert A.assign((w + 0.002, w + 0.002, 0.05), (0.5, 0, 0.8)).mode == "infeasible"     # too wide for one hand, too narrow for two


def test_mass_limit_forces_two_arms():
    # narrow but 'heavy' by the VLM's own mass guess: 0.30 m wide so two hands fit
    p = A.assign((0.30, 0.04, 0.2), (0.5, -0.21, 0.8), yaw=math.pi / 2, mass_vlm=2.0)
    assert p.mode in ("bimanual", "infeasible")
    p = A.assign((0.05, 0.05, 0.05), (0.5, 0, 0.8), mass_vlm=A.SINGLE_MAX_KG + 0.1)
    assert p.mode != "single"


def test_infeasible_reasons():
    p = A.assign((0.9, 0.9, 0.4), (0.5, 0, 0.8))
    assert p.mode == "infeasible" and "jaw" in p.why
