"""pick/assign.py -- single-arm vs bimanual assignment RULE (pure python, no sim).  SIMULATION ONLY.

Spec sources
  GRIPPER_OPEN_M = 0.070 m   max jaw opening.  YOR/robot/arm/arm.py:12 `GRIPPER_OPEN = 0.07` (the real driver); the sim model scenes/yor_robot_derived.mjcf
                             joint7/joint8 range 0..0.035 each (lines 341, 464) -> 2 x 0.035 = 0.070 (pad inner faces +-0.035 at full open, measured by FK).
  FINGER_FORCE_N = 20 N      finger actuator forcerange +-20 (scenes/yor_robot_derived.mjcf:86, class "finger").
  TAU_MAX = 100 Nm           arm joint effort (YOR model; experiments/ab_smooth_vs_bangbang/sim_common.py:130).  The planner does not touch it.
  PAYLOAD (per arm)          NO payload number exists in this repo or in the YOR model (searched README, robot/arm/*.py, URDF/MJCF).  ASSUMPTION, clearly NOT a repo/YOR
                             value: ARM_PAYLOAD_RATED_KG = 1.5 kg (the AgileX Piper rated payload recalled from the vendor datasheet; unverified here), derated
                             x PAYLOAD_DERATE = 0.5 for a dynamic lift -> SINGLE_MAX_KG = 0.75 kg.  (Memory note: the real Piper MIT-mode feed-forward torque is capped at
                             +-8 Nm, so the real payload may be lower; the sim's 100 Nm is ~12x more generous.)
  MIN_HAND_SEP_M = 0.25 m    two hands closer than this violate FL-MPC's arm-arm clearance (build.tray_hold_targets docstring: 0.176 m separation is QP-infeasible at hold,
                             0.276 m works) -> a bimanual squeeze needs the squeeze-axis width >= 0.25 m.
  MAX_SPAN_M = 0.45 m        ASSUMPTION: widest object two hands can squeeze (shoulders 0.36 m apart: link1 y = -0.03 / -0.389 measured; arm reach ~0.7 m).
  JAW_MARGIN_M = 0.005       ASSUMPTION: clearance per side for the depth-estimate error (depth sigma ~5 mm) -> usable single-grasp width 0.060 m.
  DENSITY_PRIOR = 100 kg/m3  ASSUMPTION: mass is NOT sensed before the grasp; prior mass = DENSITY_PRIOR x OBB volume (a household-container prior).  A VLM may supply
                             its own mass guess (kept for logging; the veto uses max(prior, VLM guess) only to be conservative toward bimanual).
Rule (all inputs from the DEPTH estimate: footprint OBB extents (a >= b, yaw), height h)
  w = b (narrow footprint side: the top-down pinch closes across it)
  single feasible  iff  w <= GRIPPER_OPEN - 2 JAW_MARGIN   and  m_est <= SINGLE_MAX_KG
  bimanual feasible iff  some footprint axis has width in [MIN_HAND_SEP, MAX_SPAN]  and  m_est <= 2 SINGLE_MAX_KG   (squeeze across that axis)
  both feasible -> single (less coordination); neither -> infeasible (+ reason).
Single arm side: the arm whose shoulder is nearer (object lateral y vs the robot centre line y = -0.21); the planner tries the other arm if IK / the path fails.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

GRIPPER_OPEN_M = 0.070
FINGER_FORCE_N = 20.0
JAW_MARGIN_M = 0.005
SINGLE_MAX_WIDTH_M = GRIPPER_OPEN_M - 2 * JAW_MARGIN_M
ARM_PAYLOAD_RATED_KG = 1.5          # ASSUMPTION (not in repo / YOR model)
PAYLOAD_DERATE = 0.5
SINGLE_MAX_KG = ARM_PAYLOAD_RATED_KG * PAYLOAD_DERATE
MIN_HAND_SEP_M = 0.25
MAX_SPAN_M = 0.45
DENSITY_PRIOR = 100.0               # ASSUMPTION kg / m^3
CENTRE_Y = -0.21                    # robot centre line (robot frame)


@dataclass
class Plan:
    mode: str                       # "single" | "bimanual" | "infeasible"
    arm: str                        # "left" | "right" | "both" | "none"
    grasp: str                      # "pinch" (top-down, jaws across the narrow side) | "squeeze" (both hands, paddles on opposite faces) | "none"
    why: str = ""
    width: float = 0.0              # grasp width the rule used [m]
    mass_est: float = 0.0           # [kg]
    axis: float = 0.0               # bimanual: yaw [rad] of the squeeze axis in the robot frame (hands approach along +-axis)

    def to_json(self):
        return asdict(self)


def est_mass(extent, mass_vlm=None):
    ex = [float(v) for v in extent]
    vol = ex[0] * ex[1] * ex[2]
    m = DENSITY_PRIOR * vol
    return max(m, float(mass_vlm)) if mass_vlm else m


def arm_for(centroid_y):
    """nearest-shoulder arm: +y is robot-left."""
    return "left" if float(centroid_y) > CENTRE_Y else "right"


def single_feasible(width, mass):
    ok = width <= SINGLE_MAX_WIDTH_M and mass <= SINGLE_MAX_KG
    why = []
    if width > SINGLE_MAX_WIDTH_M:
        why.append(f"width {width*100:.1f} cm > usable jaw opening {SINGLE_MAX_WIDTH_M*100:.1f} cm")
    if mass > SINGLE_MAX_KG:
        why.append(f"mass est {mass:.2f} kg > single-arm limit {SINGLE_MAX_KG:.2f} kg (assumed payload)")
    return ok, "; ".join(why)


def bimanual_axis(extent_ab, yaw):
    """squeeze axis: the footprint axis (a at `yaw`, b at yaw+90 deg) whose width fits two hands and whose direction is closest to the robot y axis.
    -> (width, axis_yaw) or (None, None).  extent_ab = (a, b) with a >= b."""
    import math
    cands = []
    for w, ang in ((extent_ab[0], yaw), (extent_ab[1], yaw + math.pi / 2)):
        if MIN_HAND_SEP_M <= w <= MAX_SPAN_M:
            # direction of the hand-to-hand line: for width measured ALONG robot-frame direction `ang`; we want it near +-y (ang ~ +-90 deg)
            d = abs(abs(((ang + math.pi) % math.pi) - math.pi / 2))      # 0 when ang is along y
            cands.append((d, w, ang))
    if not cands:
        return None, None
    cands.sort()
    return cands[0][1], cands[0][2]


def assign(extent, centroid, yaw=0.0, mass_vlm=None):
    """extent = (a, b, h) footprint OBB sizes (a >= b) and height [m], centroid (x, y, z) in the robot frame, yaw = direction of the `a` axis in the robot frame.
    -> Plan."""
    a, b, h = (float(v) for v in extent)
    mass = est_mass((a, b, h), mass_vlm)
    ok1, why1 = single_feasible(b, mass)
    if ok1:
        return Plan("single", arm_for(centroid[1]), "pinch", f"narrow side {b*100:.1f} cm fits the {GRIPPER_OPEN_M*100:.0f} cm jaws; mass est {mass:.2f} kg <= {SINGLE_MAX_KG:.2f}",
                    width=b, mass_est=mass)
    w2, ang = bimanual_axis((a, b), yaw)
    if w2 is not None and mass <= 2 * SINGLE_MAX_KG:
        return Plan("bimanual", "both", "squeeze", f"single infeasible ({why1}); two hands squeeze across {w2*100:.0f} cm", width=w2, mass_est=mass, axis=float(ang))
    if w2 is None:
        why2 = f"no footprint axis in [{MIN_HAND_SEP_M*100:.0f}, {MAX_SPAN_M*100:.0f}] cm for two hands (a={a*100:.0f}, b={b*100:.0f} cm)"
    else:
        why2 = f"mass est {mass:.2f} kg > two-arm limit {2*SINGLE_MAX_KG:.2f} kg"
    return Plan("infeasible", "none", "none", f"{why1}; bimanual: {why2}", width=b, mass_est=mass)
