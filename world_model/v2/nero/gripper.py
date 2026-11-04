"""gripper.py -- NYU (YOR) gripper helpers.  The NYU gripper is the only gripper build (USER 11-03: the AgileX claw variant was removed).

It is driven by ONE nut slide (0 = open .. NUT_TRAVEL_M = closed) that ties both hinge jaws by a polynomial (nyu_gripper.THETA_POLY).
Call sites use an OPENING FRACTION `frac`: 1 = fully open, 0 = closed (nut = (1 - frac) NUT_TRAVEL_M).  Older call sites wrote "AgileX per-finger
metres" (0.05 m = open); those numbers are now fractions (m / 0.05); the result files that still hold the metre fields are history.
"""
from __future__ import annotations

import numpy as np

from . import config as C

OPEN_FRAC, CLOSED_FRAC = 1.0, 0.0


def _nyu():
    from . import nyu_gripper
    return nyu_gripper


def open_frac_to_ctrl(frac: float) -> float:
    """position-actuator ctrl (nut metres) for an opening fraction."""
    return float((1.0 - frac) * _nyu().NUT_TRAVEL_M)


def closed_ctrl() -> float:
    """position-mode ctrl that closes the gripper (NUT_TRAVEL_M)."""
    return float(_nyu().NUT_TRAVEL_M)


def force_ctrl(force: float) -> float:
    """ctrl in force mode (gain 1, bias 0, ctrlrange +-force; pan.handle_grasp._grip_force_mode): closing force (the nut closes towards +)."""
    return float(force)


def set_opening(m, d, side: str, frac: float) -> None:
    """set the gripper qpos to opening fraction frac (nut + both jaws consistent with the joint equality)."""
    ng = _nyu()
    u = (1.0 - frac) * ng.NUT_TRAVEL_M
    th = float(np.polyval(np.asarray(ng.THETA_POLY)[::-1], u))
    d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_gripper_nut").id]] = u
    for k in (1, 2):
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_gripper_joint{k}").id]] = th


def opening_frac(m, d, side: str) -> float:
    """observable (encoder) opening fraction from the nut slide (1 = open, 0 = closed)."""
    u = float(d.qpos[m.jnt_qposadr[m.joint(f"{side}_arm_gripper_nut").id]])
    return 1.0 - u / _nyu().NUT_TRAVEL_M


def gripper_qpos_adr(m, side: str) -> list[int]:
    """qpos addresses of the side's gripper joints (the 2 jaw hinges; the nut is separate)."""
    return [m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith(f"{side}_arm_gripper_joint")]


def grasp_site(side: str) -> str:
    return f"{side}_arm_ee_grasp"      # nyu_gripper.EE_GRASP_Z along the approach axis


def col_geoms(m, side: str, part: str) -> list[int]:
    """collision geom ids of `part` in ('finger1', 'finger2', 'gripper_base'): the pieces `<side>_arm_<part>_col0..N`."""
    pre = f"{side}_arm_{part}_col"
    return [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(pre) and m.geom(g).name[len(pre):].isdigit()]


def col_geom(m, side: str, part: str) -> int:
    """first collision geom of the part (use where a single geom is needed, e.g. a friction lookup)."""
    return col_geoms(m, side, part)[0]


def set_claw_limit(m, acts, claw_N):
    """raise/lower the gripper actuators' force limit to claw_N [N] in memory (the NUT force, config.NUT_FORCE_N = 231 default).
    None = leave the value set at compose (config.claw_default()).  -> the limit in force."""
    if claw_N is None:
        return float(m.actuator_forcerange[acts[0], 1])
    for a in acts:
        m.actuator_forcerange[a] = (-float(claw_N), float(claw_N))
    return float(claw_N)


def default_grasp() -> str:
    """grasp model of the pan segment executor: "weld" (USER 11-03: rigid weld for everything)."""
    return "weld"


def provenance(grasp_mode: str | None = None) -> dict:
    """fields added to result records (11-03): which gripper model and which grasp model produced them."""
    return dict(gripper_variant=C.GRIPPER, grasp_mode=grasp_mode or default_grasp())


# ---- 11-03 USER-approved grasp-point rule: the TCP and the jaw opening for a grasped feature of thickness w --------------------------------
def tcp_offset() -> float:
    """TCP (ee_grasp site) depth from the flange along the approach axis [m]: nyu_gripper.EE_GRASP_Z (TCP_FRAC = 40 % of pivot-to-tip, ASSUMED)."""
    return float(_nyu().EE_GRASP_Z)


def jaw_gap_m(theta: float, z: float | None = None) -> float:
    """inner (pad-to-pad) gap [m] between the two jaws at depth z (default the TCP) along the approach axis, for the hinge angle theta [rad].
    MODEL: each inner edge is a straight line (MEASURED: nyu_gripper.JAW_INNER_*) rotated by theta about its pivot; both jaws symmetric."""
    ng = _nyu()
    z = ng.EE_GRASP_Z if z is None else float(z)
    py, pz = ng.JAW_PIVOT_YZ_M
    k = np.tan(ng.JAW_INNER_ANGLE_RAD)
    p0 = np.array([ng.JAW_INNER_Y160_M, 0.160]) - np.array([py, pz])       # a point on the open inner edge, relative to the pivot
    d0 = np.array([k, 1.0]) / np.hypot(k, 1.0)                              # edge direction (y, z) at theta = 0
    c, s = np.cos(theta), np.sin(theta)
    Rm = np.array([[c, -s], [s, c]])                                        # closing: the tip moves to smaller y and larger z
    p, dd = Rm @ p0, Rm @ d0
    if abs(dd[1]) < 1e-9:
        return float("nan")
    y = py + p[0] + (z - pz - p[1]) * dd[0] / dd[1]
    return float(2.0 * y)


def theta_to_nut(theta: float) -> float:
    """invert nyu_gripper.THETA_POLY (monotone on 0..NUT_TRAVEL_M): hinge angle [rad] -> nut position u [m]."""
    from scipy.optimize import brentq
    ng = _nyu()
    co = np.asarray(ng.THETA_POLY)[::-1]
    return float(brentq(lambda u: np.polyval(co, u) - theta, 0.0, ng.NUT_TRAVEL_M))


def opening_for_width(w: float, clearance: float | None = None) -> dict:
    """jaw opening so that the inner gap at the TCP = w + 2 clearance.  -> dict(frac, nut, theta, gap, feasible, gap_min).
    frac is the `set_opening` fraction (1 = open, 0 = closed).  feasible=False when the target gap is below the gap at fully closed (THETA_MAX_RAD)
    or above the open gap; the opening is then clipped to the nearest limit. """
    ng = _nyu()
    cl = ng.GRASP_CLEARANCE_M if clearance is None else float(clearance)
    target = float(w) + 2.0 * cl
    from scipy.optimize import brentq
    g_open, g_closed = jaw_gap_m(0.0), jaw_gap_m(ng.THETA_MAX_RAD)
    feasible = bool(g_closed <= target <= g_open)
    if target >= g_open:
        th = 0.0
    elif target <= g_closed:
        th = float(ng.THETA_MAX_RAD)
    else:
        th = float(brentq(lambda t: jaw_gap_m(t) - target, 0.0, ng.THETA_MAX_RAD))
    u = theta_to_nut(min(th, float(np.polyval(np.asarray(ng.THETA_POLY)[::-1], ng.NUT_TRAVEL_M))))
    return dict(frac=1.0 - u / ng.NUT_TRAVEL_M, nut=u, theta=th, gap=jaw_gap_m(th), target=target, feasible=feasible, gap_min=g_closed, gap_open=g_open)


APPROACH_THETA_RAD = 0.60   # ASSUMED: nyu approach opening = jaw angle at which the inner edges are parallel to the approach axis (MODEL: = JAW_INNER_ANGLE_RAD 0.589, rounded to 0.60), gap ~38 mm along the pad


def approach_open_frac() -> float:
    """opening fraction used while approaching / releasing a handle: the opening with parallel pads (APPROACH_THETA_RAD) -- the old 0.6-of-travel opening
    (60-72 mm gap) put the lower jaw tip on the stove pad at the pre-grasp."""
    u = theta_to_nut(APPROACH_THETA_RAD)
    return float(1.0 - u / _nyu().NUT_TRAVEL_M)
