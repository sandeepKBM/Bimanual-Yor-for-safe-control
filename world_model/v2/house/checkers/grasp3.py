"""checkers/grasp3.py -- CHECKER 3: exact grasp-load corner check over a richer grasp set.  Pure python/numpy, SIMULATION ONLY.

See GRASP3.md for the derivation.  Summary
  * Object = box (a >= b footprint, h height), frame: x || a, y || b, z up; geometric centre at the origin; CoM = centre + delta, delta in a BOX (com_box).
  * Quasi-static load the grasp must supply at contact centre p_c:   F = m (a_obj + g_up ez),   tau = (CoM - p_c) x F,    g_up = 9.81, a_obj in an acceleration BOX.
  * w = (F, tau) is MULTI-AFFINE in (m, delta_x, delta_y, delta_z, a_x, a_y, a_z);  the grasp wrench set is a convex gauge G(w) <= 1  (ellipsoid + normal-imbalance term),
    G(w) convex => G(w(.)) is convex in each variable separately => its max over the box is attained at a VERTEX.  So the check over the 2*8*8 = 128 corners is EXACT
    (not a sampling); friction capacity is increasing in mu so mu_lo is the worst case.  Enlarging any box can therefore only remove feasible candidates (monotone).
  * Candidates: top-down pinch (narrow footprint side), edge pinch (across thickness), rim pinch (container wall), handle pinch, bimanual squeeze.
  * decide() -> single | bimanual | infeasible (-> Checker 4's probe).
Every number that is not a repo/YOR constant is a named ASSUMPTION below.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from world_model.v2.house.pick import assign as A

# ---------------------------------------------------------------- constants: reused from pick/assign.py (repo / YOR values or assign's own ASSUMPTIONS)
GRIPPER_OPEN_M = A.GRIPPER_OPEN_M            # 0.070  YOR/robot/arm/arm.py:12
JAW_MARGIN_M = A.JAW_MARGIN_M                # 0.005  (assign ASSUMPTION)
SINGLE_MAX_WIDTH_M = A.SINGLE_MAX_WIDTH_M    # 0.060 usable single-grasp width
SINGLE_MAX_KG = A.SINGLE_MAX_KG              # 0.75   ASSUMED payload (assign.py)
MIN_HAND_SEP_M = A.MIN_HAND_SEP_M            # 0.25
MAX_SPAN_M = A.MAX_SPAN_M                    # 0.45
FINGER_FORCE_N = A.FINGER_FORCE_N            # 20 N  finger actuator forcerange +-20 (scenes/yor_robot_derived.mjcf:86); used as the per-jaw squeeze limit F_grip_max
G = 9.81

# ---------------------------------------------------------------- NEW named ASSUMPTIONS (nothing below is a YOR/real-robot spec unless stated)
PAD_FACE_M = 0.03             # pad face 3 cm x 3 cm = sim finger pad geom half-sizes .015 x .015 (scenes/yor_robot_derived.mjcf:346); real pad size unverified
MU_LO_DEFAULT = 0.4           # ASSUMPTION worst-case pad/object friction (sim pads are tuned to mu=2, NOT a real value)
MU_HI_DEFAULT = 0.8           # ASSUMPTION (only used for the optimistic margin report; feasibility uses mu_lo)
COM_FRAC = 0.05               # ASSUMPTION default CoM uncertainty half-width = COM_FRAC * extent per axis (empty, near-uniform household objects; 0.10 sensitivity in GRASP3.md)
HAND_SQUEEZE_FACTOR = 2.0     # ASSUMPTION: a hand squeezing with its closed gripper reacts 2 x F_grip_max (both fingers 20 N each); the arm is assumed able to deliver it
BIMANUAL_PADDLE_M = 2 * PAD_FACE_M   # ASSUMPTION: a closed gripper presents its two 3 cm pads side by side -> 6 cm x 3 cm squeeze paddle
ACCEL_DEFAULT = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.5))   # ASSUMPTION m/s^2 box (x, y, z): base/arm jerk horizontally, lift up to +1.5, brake down to -1.0
WALL_T_M = 0.0075             # ASSUMPTION container wall thickness 0.5-1 cm (jaw check uses WALL_T_MAX_M)
WALL_T_MAX_M = 0.010
RIM_BITE_M = 0.02             # ASSUMPTION: pads overlap the wall 2 cm down from the rim (<= pad face 3 cm)
RIM_CONTACT_LEN_M = 0.01      # ASSUMPTION: contact length along a curved rim (curvature limits the pad patch)
EDGE_INSET_M = 0.025          # ASSUMPTION: pads overlap the object 2.5 cm from the edge (<= pad face 3 cm)
HANDLE_OFFSET_M = 0.03        # ASSUMPTION loop handle: bar centre sits 3 cm outside the body surface
HANDLE_BAR_D_M = 0.015        # ASSUMPTION handle bar diameter / thickness 1.5 cm (must fit the jaws)
HANDLE_LEN_M = 0.06           # ASSUMPTION loop-handle bar length 6 cm
HANDLE_GAP_M = 0.02           # ASSUMPTION gap between handle bar and body for the inner pad (pad thickness 5 mm in the sim)
SHANK_GRIP_FROM_END_M = 0.04  # ASSUMPTION shank handle (pan/pot/ladle): pinch 4 cm from the free end of the bounding box
TORSION_K = 2.0 / 3.0         # soft-finger patch torsion: tau_n <= mu N (2/3) R_eq per contact (uniform pressure disc)
EDGE_MAX_STACK_H_OVER_B = 1.0  # edge pinch only when the object is flat (h <= b)

RIM_CATEGORIES = frozenset({"Cup", "Mug", "Bowl", "Pot", "Pan", "GarbageCan", "Vase", "Plate", "Kettle"})
SHANK_CATEGORIES = frozenset({"Pan", "Pot", "Ladle", "Spatula", "Fork", "Spoon", "Knife", "ButterKnife"})
_PRIORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vlm", "ftdata", "category_priors.json")


def category_hints(category):
    """-> (has_handle, has_rim) from category_priors.json + RIM_CATEGORIES.  Unknown category -> (False, False)."""
    has_handle = False
    try:
        with open(_PRIORS) as f:
            c = json.load(f)["categories"].get(category)
        has_handle = bool(c and c.get("has_handle"))
    except Exception:
        pass
    return has_handle, category in RIM_CATEGORIES


# ---------------------------------------------------------------- data classes
@dataclass(frozen=True)
class Candidate:
    name: str
    arms: int                     # 1 | 2
    width: float                  # jaw width needed (single) / squeeze-axis span (bimanual) [m]
    n: tuple                      # grasp (closing / squeeze) axis, object frame
    t1: tuple
    t2: tuple
    p_c: tuple                    # contact centre, object frame [m]
    patch: tuple                  # (e1, e2) contact patch extents along t1, t2 [m]
    note: str = ""
    access: str = ""              # geometric precondition NOT checked here (documented)

    @property
    def r_nom(self):
        return -np.asarray(self.p_c, float)


@dataclass
class CandidateResult:
    name: str
    arms: int
    ok: bool
    margin: float                 # worst-corner margin = min(1 - G, payload margin, width margin); >= 0 <=> feasible
    width: float
    reason: str
    binding: dict = field(default_factory=dict)
    margin_mu_hi: float = float("nan")
    access: str = ""

    def to_json(self):
        return asdict(self)


@dataclass
class GraspReport:
    candidates: list              # all CandidateResults, ranked: feasible first, by margin desc
    feasible: list
    best: object                  # best feasible CandidateResult or None
    worst_margin: float           # margin of `best` (worst corner), or the least-bad margin if none feasible

    def to_json(self):
        return dict(candidates=[c.to_json() for c in self.candidates], worst_margin=self.worst_margin,
                    best=self.best.to_json() if self.best else None)


# ---------------------------------------------------------------- candidate generation
def _fp(extent):
    e = [float(v) for v in extent]
    a, b = max(e[0], e[1]), min(e[0], e[1])
    return a, b, e[2]


def generate_candidates(extent, category=None, has_handle=None, has_rim=None, handle_kind=None):
    """grasp candidates for a box-like object.  extent = (a, b, h) (footprint order free, a>=b enforced)."""
    a, b, h = _fp(extent)
    hh, hr = category_hints(category) if category else (False, False)
    has_handle = hh if has_handle is None else bool(has_handle)
    has_rim = hr if has_rim is None else bool(has_rim)
    X, Y, Z = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)
    out = []
    # (i) top-down pinch across the narrow footprint side b; pads bite the top of the object
    e2 = min(PAD_FACE_M, h)
    out.append(Candidate("topdown_pinch", 1, b, Y, X, Z, (0.0, 0.0, h / 2 - e2 / 2), (min(PAD_FACE_M, a), e2),
                         note="jaws close along y across b; contact at object top, x centred"))
    # (ii) edge pinch across the thickness of a flat object (jaws vertical, at the midpoint of a long edge)
    if h <= b * EDGE_MAX_STACK_H_OVER_B:
        ins = min(EDGE_INSET_M, b / 2)
        out.append(Candidate("edge_pinch", 1, h, Z, X, Y, (0.0, b / 2 - ins / 2, 0.0), (min(PAD_FACE_M, a), ins),
                             note="jaws close along z across h at the long-edge midpoint; lever = b/2 - inset/2",
                             access="lower jaw must reach under the edge (support-surface clearance / tilt); NOT checked"))
    # (iii) rim pinch on a container wall (inner + outer pad)
    if has_rim:
        bite = min(RIM_BITE_M, h)
        out.append(Candidate("rim_pinch", 1, WALL_T_MAX_M, Y, X, Z, (0.0, b / 2 - WALL_T_M / 2, h / 2 - bite / 2), (RIM_CONTACT_LEN_M, bite),
                             note="radial jaws across the wall at the rim; lever = b/2 (half diameter)",
                             access="inner pad must fit inside the container; NOT checked"))
    # (iv) handle grasp
    if has_handle:
        if handle_kind is None:
            handle_kind = "shank" if category in SHANK_CATEGORIES else "loop"
        if handle_kind == "shank":
            out.append(Candidate("handle_shank_pinch", 1, HANDLE_BAR_D_M, Z, X, Y, (a / 2 - SHANK_GRIP_FROM_END_M, 0.0, 0.0),
                                 (min(PAD_FACE_M, a), min(PAD_FACE_M, b)),
                                 note="vertical jaws across a horizontal shank handle, 4 cm from its free end; bbox-centre CoM (lever a/2-4cm)"))
        else:
            out.append(Candidate("handle_loop_pinch", 1, HANDLE_BAR_D_M, Y, X, Z, (0.0, b / 2 + HANDLE_OFFSET_M, 0.0), (HANDLE_BAR_D_M, min(PAD_FACE_M, HANDLE_LEN_M)),
                                 note="radial jaws across a vertical handle bar, bar centre 3 cm outside the body",
                                 access=f"inner pad needs the {HANDLE_GAP_M*100:.0f} cm handle gap; NOT checked"))
    # (v) bimanual squeeze across a footprint axis in [MIN_HAND_SEP, MAX_SPAN]
    for tag, w, s, t1, t2, f1, f2 in (("a", a, X, Y, Z, b, h), ("b", b, Y, X, Z, a, h)):
        if MIN_HAND_SEP_M <= w <= MAX_SPAN_M:
            out.append(Candidate(f"bimanual_squeeze_{tag}", 2, w, s, t1, t2, (0.0, 0.0, 0.0), (min(BIMANUAL_PADDLE_M, f1), min(PAD_FACE_M, f2)),
                                 note=f"two hands squeeze across {tag}={w*100:.0f} cm; equal load split + equal squeeze (statically indeterminate: assumed)"))
    return out


# ---------------------------------------------------------------- the load model
def _box(v, ext):
    """com_box -> half-extents (3,) in the object frame.  None -> COM_FRAC*extent; scalar -> all axes; 3-seq -> per axis."""
    if v is None:
        return COM_FRAC * np.array([ext[0], ext[1], ext[2]], float)
    v = np.asarray(v, float)
    return np.full(3, float(v)) if v.ndim == 0 else v.astype(float)


def _corners(lo, hi):
    return np.array(list(itertools.product(*[(l, h) for l, h in zip(lo, hi)])), float)


def grip_gauge(cand, m, delta, acc, mu, fgrip=FINGER_FORCE_N):
    """G(w) for arrays: m (K,), delta (K,3), acc (K,3) -> (K,) gauge and a dict of the component shares.  G <= 1 <=> the load is in the grasp wrench set."""
    m = np.atleast_1d(np.asarray(m, float)); delta = np.atleast_2d(delta); acc = np.atleast_2d(acc)
    F = m[:, None] * (acc + np.array([0.0, 0.0, G]))
    r = cand.r_nom[None, :] + delta
    T = np.cross(r, F)
    n, t1, t2 = (np.asarray(v, float) for v in (cand.n, cand.t1, cand.t2))
    fn, f1, f2 = F @ n, F @ t1, F @ t2
    tn, q1, q2 = T @ n, T @ t1, T @ t2
    e1, e2 = cand.patch
    req = math.sqrt(e1 * e2 / math.pi)
    Ft = 2 * mu * fgrip
    Tn = 2 * mu * fgrip * TORSION_K * req
    Tt1 = fgrip * e2                    # torque about t1: pad pressure couple over the t2 extent
    Tt2 = fgrip * e1
    comps = np.stack([np.hypot(f1, f2) / Ft, np.abs(tn) / Tn, np.abs(q1) / Tt1, np.abs(q2) / Tt2], axis=1)
    g1 = np.abs(fn) / (2 * fgrip) + np.linalg.norm(comps, axis=1)
    g2 = np.abs(fn) / fgrip
    g = np.maximum(g1, g2)
    return g, dict(friction_force=comps[:, 0], torsion=comps[:, 1], pitch_t1=comps[:, 2], pitch_t2=comps[:, 3], normal_imbalance=np.abs(fn) / fgrip)


def _payload_margin(cand, m_hi, com_half, fgrip=None):
    """static payload check (assign's SINGLE_MAX_KG already carries the dynamic derate).  Bimanual: equal split + CoM shift along the squeeze axis."""
    if cand.arms == 1:
        return 1.0 - m_hi / SINGLE_MAX_KG
    s = int(np.argmax(np.abs(cand.n)))
    share = m_hi * (0.5 + min(0.5, com_half[s] / max(cand.width, 1e-9)))
    return 1.0 - share / SINGLE_MAX_KG


def _eval(cand, extent3, mass_range, com_half, mu_lo, mu_hi, accel_box, fgrip):
    m_lo, m_hi = float(mass_range[0]), float(mass_range[1])
    if cand.arms == 1:
        wmax = SINGLE_MAX_WIDTH_M
    else:
        wmax = None
    # width gate
    if cand.arms == 1 and cand.width > wmax:
        return CandidateResult(cand.name, 1, False, 1.0 - cand.width / wmax, cand.width,
                               f"jaw width {cand.width*100:.1f} cm > usable {wmax*100:.1f} cm", {"binding": "width"}, float("nan"), cand.access)
    if cand.arms == 2:
        fgrip = fgrip * HAND_SQUEEZE_FACTOR
    M = np.array([m_lo, m_hi])
    D = _corners(-com_half, com_half)
    AC = _corners([x[0] for x in accel_box], [x[1] for x in accel_box])
    K = list(itertools.product(range(len(M)), range(len(D)), range(len(AC))))
    mm = np.array([M[i] for i, _, _ in K]); dd = np.array([D[j] for _, j, _ in K]); aa = np.array([AC[k] for _, _, k in K])
    g, comps = grip_gauge(cand, mm, dd, aa, mu_lo, fgrip)
    i = int(np.argmax(g))
    gmax = float(g[i])
    gm_hi, _ = grip_gauge(cand, mm, dd, aa, mu_hi, fgrip)
    pay = _payload_margin(cand, m_hi, com_half)
    grip_margin = 1.0 - gmax
    margin = min(grip_margin, pay)
    dom = max(comps, key=lambda k: comps[k][i])
    binding = dict(binding="payload" if pay < grip_margin else dom, mass=float(mm[i]), com_delta=[float(x) for x in dd[i]], accel=[float(x) for x in aa[i]],
                   mu=float(mu_lo), gauge=gmax, payload_margin=float(pay))
    reason = "ok" if margin >= 0 else (f"payload: {m_hi:.2f} kg" + (" per hand split" if cand.arms == 2 else "") + f" over {SINGLE_MAX_KG:.2f} kg limit" if pay < 0 else
                                       f"grip load exceeds wrench set at corner (m={mm[i]:.2f} kg, delta={np.round(dd[i],3).tolist()}, a={aa[i].tolist()}); dominant: {dom}")
    return CandidateResult(cand.name, cand.arms, margin >= 0, float(margin), cand.width, reason, binding, float(min(1.0 - np.max(gm_hi), pay)), cand.access)


def check_grasp(extent, mass_range, com_box=None, mu_range=(MU_LO_DEFAULT, MU_HI_DEFAULT), category=None, has_handle=None, accel_box=ACCEL_DEFAULT,
                has_rim=None, handle_kind=None, grip_force=FINGER_FORCE_N):
    """-> GraspReport.  extent=(a,b,h); mass_range=(m_lo,m_hi) kg; com_box = CoM half-extents (scalar | 3-seq in the object frame x||a,y||b,z up | None=COM_FRAC*extent);
    mu_range=(mu_lo, mu_hi) (feasibility uses mu_lo); accel_box=((ax_lo,ax_hi),(ay..),(az..)) m/s^2 INCLUDING lift accel.
    A candidate is feasible iff margin >= 0 at ALL 128 box corners (exact; see module docstring)."""
    a, b, h = _fp(extent)
    com_half = _box(com_box, (a, b, h))
    cands = generate_candidates((a, b, h), category, has_handle, has_rim, handle_kind)
    res = [_eval(c, (a, b, h), mass_range, com_half, float(mu_range[0]), float(mu_range[1]), accel_box, grip_force) for c in cands]
    res.sort(key=lambda r: (not r.ok, -r.margin))
    feas = [r for r in res if r.ok]
    return GraspReport(res, feas, feas[0] if feas else None, (feas[0].margin if feas else (res[0].margin if res else float("-inf"))))


def decide(extent, mass_range, com_box=None, mu_range=(MU_LO_DEFAULT, MU_HI_DEFAULT), category=None, has_handle=None, accel_box=ACCEL_DEFAULT,
           has_rim=None, handle_kind=None, grip_force=FINGER_FORCE_N, centroid_y=None):
    """-> dict(mode in {single, bimanual, infeasible}, arms, arm, grasp, margin, reason, fallback_arms, report).
    single preferred over bimanual (less coordination); infeasible => no candidate passes for ALL corners -> Checker 4's probe takes over."""
    rep = check_grasp(extent, mass_range, com_box, mu_range, category, has_handle, accel_box, has_rim, handle_kind, grip_force)
    s = [r for r in rep.feasible if r.arms == 1]
    bm = [r for r in rep.feasible if r.arms == 2]
    allr = rep.candidates
    least_bad = max(allr, key=lambda r: r.margin) if allr else None
    arm1 = A.arm_for(centroid_y) if centroid_y is not None else "either"
    if s:
        r = s[0]
        return dict(mode="single", arms=1, arm=arm1, grasp=r.name, margin=r.margin, reason=f"{r.name} passes all corners (worst margin {r.margin:.2f})",
                    fallback_arms=1, report=rep)
    if bm:
        r = bm[0]
        why1 = "; ".join(f"{x.name}: {x.reason}" for x in allr if x.arms == 1) or "no single-arm candidate"
        return dict(mode="bimanual", arms=2, arm="both", grasp=r.name, margin=r.margin, reason=f"{r.name} passes all corners (worst margin {r.margin:.2f}); single failed: {why1}",
                    fallback_arms=2, report=rep)
    why = "; ".join(f"{x.name}: {x.reason}" for x in allr) or "no candidate"
    return dict(mode="infeasible", arms=0, arm="none", grasp="none", margin=rep.worst_margin, reason=f"no candidate passes all corners -> probe (Checker 4). {why}",
                fallback_arms=(least_bad.arms if least_bad else 1), report=rep)
