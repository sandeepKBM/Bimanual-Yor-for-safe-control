"""democ/grasp_cert.py -- PLAN-TIME GRASP-HOLD CERTIFICATE for Demo C.  Pure math + (optional) MuJoCo FK.  SIMULATION ONLY.

Question answered: given a set of contacts on the box (two hands / one hand + support / one hand alone) and INTERVAL uncertainty on
payload (mass, COM, optionally planned acceleration), does a static-equilibrium contact-force distribution EXIST for every
parameter value?  A release may be admitted only if certify_hold(...)[0] is True on the contacts that remain.

Model (quasi-static rigid box, box frame, forces N, torques Nm about the box origin):
  * contact i: point p_i, inward unit normal n_i (force the contact exerts ON the box has positive n_i component),
    friction cone |f_t| <= mu_lo * f_n replaced by an INNER m-edge pyramid  f = sum_j lam_j (n + mu_eff d_j), lam >= 0, d_j unit
    tangents.  mu_eff = mu_lo (edges ON the cone => polytope inside the cone => sound; the owner spec's extra cos(pi/m) is
    available via edge_scale='spec' but is unnecessary conservatism).  Torsional friction is ignored (sound).
  * a finger pad = 4 corner contacts of its patch (a rigid pad can realise any centre of pressure in the corner hull).
  * limits: a GROUP of contacts shares one limit on the SUM of normal forces (hand squeeze: see PAD_* / FINGER_* below);
    support-patch contacts are unilateral with no limit.
  * required wrench: contacts must produce  -w,  w = [m(g-a); r x m(g-a) - I alpha]  (g, a, r in the box frame, r = COM offset).
    With a = 0, w is LINEAR in (m, s = m r); the feasible set W is a convex polytope.  SPEC CORRECTION: the image of the (m, r) box
    under (m, m r) is exactly {r_lo m <= s <= r_hi m} -- a polytope -- with the 2 x 2^d vertices (m_lo|m_hi, r corner).  So the vertex
    check over THOSE is EXACT.  (The owner's outer box [m_lo,m_hi] x [min m r, max m r] is sound but needlessly conservative: it
    wrongly rejects e.g. m in [0.5,0.7], |r|<=0.025 on a +-0.03 pad because the corner (m_lo, s_hi) is not a real parameter value.)  Uncertain a: products m*a_k and q_jk = m r_j a_k are treated as
    independent intervals (sound over-approximation); the 3 force + 3 torque extra intervals are independent so they are
    separate vertex variables (up to 10 variables -> 1024 vertices; each degenerate interval is skipped).
  * margin eps* (scaled wrench space: N for force, Nm/L for torque, L = 0.1 m): min over vertices v and the 12 directions +-e_j of
    max{eps : -w_v + eps e_j in W}.  ok <=> eps* >= -TOL.
  * pose (attachment) uncertainty delta (|delta_k| <= d_k): NOT vertex-exact.  SOUND Lipschitz robustification (Kakutani argument in
    the docstring of _pose_slack): the torque error of shifted contacts is bounded per component by rho_k = sum_i |f_i|max *
    sqrt(d_a^2 + d_b^2); we require -w_v + [0; tau] in W for all 8 corners tau in prod[-rho_k, rho_k] (a CUBE, not the spec's 12 axis
    directions: the axis points only span an octahedron, which does not contain the ball/box -> the spec as written is UNSOUND),
    with every contact's normal force additionally capped (cap set Lambda) so |f_i| is bounded.
  * optimistic_internal_force=True when >= 2 hands actively hold the box: the internal force split is set by the controllers,
    so "there EXIST forces" is optimistic there.  One hand + passive support: existence is fine.

Source of the numbers (ALL read from the repo, none invented):
  * finger inner-face patch: mesh nero_gripper_link1/2 as compiled in scenes/yor_nero_democ.mjcf (left_arm_finger{1,2}_col): in the
    gripper_base frame the fingers span z in [0.0615, 0.138] (tip), the inner faces sit at y = +-0.04 (open, q=0.04); the tip is
    tapered: x half-width 0.014 at the tip (0.028 at the root).  PAD_HALF_W = 0.014 is the conservative rectangle inside the
    trapezoid.  The patch along the approach is the finger z-range intersected with the box (the box end face is 2 cm inside
    ee_grasp = kin.INSET).
  * squeeze limit: actuator left/right_arm_gripper_pos forcerange +-10 N (scenes/yor_nero_democ.mjcf line ~389; the AGX gripper URDF
    assets_nero/agx_gripper/urdf/agx_gripper_description.urdf also has effort=10).  The class default ('finger') says +-20 but the
    instantiated actuators override it to 10.  Joint2 is slaved to joint1 by an equality (q2 = -q1), so the actuator force is the
    work-conjugate of the SYMMETRIC closing mode: F_act = N_pad1 + N_pad2 (verified in the cross-check).  Hence the limit is on the
    SUM of the two pad normals of one hand (HAND_NORMAL_LIMIT = 10 N), not 10 N per pad; the antisymmetric part (N1-N2, what carries
    weight through a vertical jaw) is carried by the coupling structure and is unlimited here.  limit_mode='per_pad' gives the
    owner's per-pad reading (10 N each = 20 N total, OPTIMISTIC by 2x for the sum).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linprog

L_SCALE = 0.1                  # m: torque scaling so forces [N] and torque/L [N] are comparable
TOL = 1e-7
EPS_CAP = 1.0e3                # scaled units; eps beyond this is reported as the cap (unbounded direction, e.g. support cone)
N_EDGES = 8
GAP_TOL = 6.0e-6               # m; default unmodelled corner gap: smallest value in the sweep with 0 false-safe on all 216 sim releases (a FIT, not a validation)
RANK_TOL = 5e-2                # creep model: relative singular-value cut for the normal-row rank (null(J_n) = creep DoFs)

# --- gripper pad geometry / limits (see module docstring for the source) ---
FINGER_Z_ROOT, FINGER_Z_TIP = 0.0615, 0.138      # finger span along the gripper approach axis z_g [m]
PAD_HALF_W = 0.014                               # conservative half width (x_g) of the inner-face patch [m]
HAND_NORMAL_LIMIT = 10.0                         # N, sum of the two pad normals (actuator forcerange)
G = 9.81


@dataclass
class Contact:
    p: np.ndarray                  # point in the box frame [m]
    n: np.ndarray                  # inward unit normal (direction of the force on the box)
    mu_lo: float = 0.4
    mu_hi: float = 0.4
    f_max: float | None = None     # limit on the SUM of normal forces of this contact's group (None = unlimited)
    group: str | None = None       # contacts sharing a group share the f_max budget; None = own budget
    hand: str = "support"          # 'left' | 'right' | 'support'
    active: bool = False           # actively commanded (hand pad) vs passive (support)
    pose_unc: bool = True          # subject to attachment-pose uncertainty delta
    kn_scale: float = 1.0          # compliant mode: extra multiplier of this corner's normal stiffness
    finger: int | None = None      # hand pad corners: 1 = the ACTUATED finger (joint1), 2 = the finger slaved through the soft equality (joint2)

    def __post_init__(self):
        self.p = np.asarray(self.p, float)
        n = np.asarray(self.n, float)
        self.n = n / np.linalg.norm(n)


@dataclass
class Params:
    m_lo: float = 0.6
    m_hi: float = 0.6
    com_lo: tuple = (0.0, 0.0, 0.0)          # COM offset box [m], box frame
    com_hi: tuple = (0.0, 0.0, 0.0)
    g: tuple = (0.0, 0.0, -G)                # gravity in the BOX frame
    a_lo: tuple = (0.0, 0.0, 0.0)            # planned linear acceleration of the box, box frame
    a_hi: tuple = (0.0, 0.0, 0.0)
    alpha: tuple = (0.0, 0.0, 0.0)           # angular acceleration, box frame
    inertia: tuple = (0.0, 0.0, 0.0)         # diag inertia about COM, box frame (only used with alpha)
    pose_delta: tuple = (0.0, 0.0, 0.0)      # |delta_k| <= pose_delta[k] attachment shift of pose_unc contacts
    support_cap: float | None = None         # per-contact normal cap for unlimited contacts when pose_delta > 0 (default 2 m_hi |g|)


# ------------------------------------------------------------------------------------------------ geometry builders
def _tangents(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def pad_contacts(hand, R_gb, p_gb, box_half, mu_lo=0.4, mu_hi=None, f_limit=HAND_NORMAL_LIMIT, limit_mode="hand_total",
                 pad_half_w=PAD_HALF_W, z_root=FINGER_Z_ROOT, z_tip=FINGER_Z_TIP, align_tol_deg=15.0, shrink=1.0, finger1_pos=None):
    """The 8 corner contacts (2 pads x 4 corners) of one hand pinching the box with its jaw axis (y_g) along a box axis k and its
    approach (z_g) along a box axis j.  R_gb: columns = gripper axes (x_g,y_g,z_g) expressed in the BOX frame; p_gb: gripper_base
    origin in the box frame.  Box assumed axis-aligned to the jaw within align_tol_deg (small misalignment is ignored).
    finger1_pos: position (box frame) of the actuated finger's body origin (on its pad plane); tags each corner with finger 1 / 2 for the compliant stiffness.
    shrink in (0,1]: HEURISTIC (not a proof) confinement of the centre of pressure to the central `shrink` fraction of the patch; the
    sim cross-check shows equilibria that need the CoP at the patch EDGE are not realised by compliant contacts.
    limit_mode: 'hand_total' (sum of both pad normals <= f_limit, default, see module docstring) | 'per_pad' (each pad <= f_limit)."""
    R_gb = np.asarray(R_gb, float); p_gb = np.asarray(p_gb, float); h = np.asarray(box_half, float)
    xg, yg, zg = R_gb[:, 0], R_gb[:, 1], R_gb[:, 2]
    c = np.cos(np.radians(align_tol_deg))
    k = int(np.argmax(np.abs(yg)))
    j = int(np.argmax(np.where(np.arange(3) == k, -1, np.abs(zg))))
    i = 3 - k - j
    if abs(yg[k]) < c or abs(zg[j]) < c or abs(xg[i]) < c:
        raise ValueError(f"{hand}: gripper axes not aligned with box axes within {align_tol_deg} deg")
    zc = float(-p_gb @ zg); xc = float(-p_gb @ xg); yc = float(-p_gb @ yg)       # box centre in gripper coordinates
    # pad extent along z_g: finger span cut by the box slab; along x_g: pad width cut by the box
    z_lo, z_hi = max(z_root, zc - h[j]), min(z_tip, zc + h[j])
    x_lo, x_hi = max(-pad_half_w, xc - h[i]), min(pad_half_w, xc + h[i])
    if z_hi - z_lo <= 1e-4 or x_hi - x_lo <= 1e-4:
        raise ValueError(f"{hand}: pad patch does not overlap the box (z {z_lo:.3f}..{z_hi:.3f}, x {x_lo:.3f}..{x_hi:.3f})")
    sj, si = np.sign(zg[j]), np.sign(xg[i])
    cj = sorted(((z_lo - zc) * sj, (z_hi - zc) * sj))           # box coordinate range along axis j
    ci = sorted(((x_lo - xc) * si, (x_hi - xc) * si))
    if shrink != 1.0:
        cj = [0.5 * (cj[0] + cj[1]) + shrink * (v - 0.5 * (cj[0] + cj[1])) for v in cj]
        ci = [0.5 * (ci[0] + ci[1]) + shrink * (v - 0.5 * (ci[0] + ci[1])) for v in ci]
    out = []
    mu_hi = mu_lo if mu_hi is None else mu_hi
    for side in (+1.0, -1.0):                                   # +/- box face normal to axis k
        nk = np.zeros(3); nk[k] = -side                         # inward normal of the box face at +side*h_k
        grp = f"pads_{hand}" if limit_mode == "hand_total" else f"pad_{hand}_{'+' if side > 0 else '-'}"
        for a_ in ci:
            for b_ in cj:
                p = np.zeros(3); p[k] = side * h[k]; p[i] = a_; p[j] = b_
                fing = None if finger1_pos is None else (1 if np.sign(finger1_pos[k]) == side else 2)
                out.append(Contact(p, nk, mu_lo, mu_hi, f_max=f_limit, group=grp, hand=hand, active=True, pose_unc=True, finger=fing))
    return out


def support_contacts(patch_corners, normal, mu_lo=0.4, mu_hi=None, name="support"):
    """Unilateral support patch (corners of the box/support overlap, box frame), `normal` = direction the support pushes the box."""
    mu_hi = mu_lo if mu_hi is None else mu_hi
    return [Contact(np.asarray(p, float), normal, mu_lo, mu_hi, f_max=None, group=name, hand="support", active=False, pose_unc=False)
            for p in patch_corners]


def _clip_poly(poly, lo, hi):
    """Sutherland-Hodgman clip of a convex 2-D polygon (list of (x,y)) against the rectangle [lo,hi]."""
    pts = [tuple(p) for p in poly]
    for ax in (0, 1):
        for bound, keep in ((lo[ax], lambda v, b: v >= b), (hi[ax], lambda v, b: v <= b)):
            new = []
            for q in range(len(pts)):
                a, b = pts[q], pts[(q + 1) % len(pts)]
                ina, inb = keep(a[ax], bound), keep(b[ax], bound)
                if ina:
                    new.append(a)
                if ina != inb and abs(b[ax] - a[ax]) > 1e-15:
                    t = (bound - a[ax]) / (b[ax] - a[ax])
                    new.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            pts = new
            if not pts:
                return []
    return pts


def support_patch_from_pedestal(R_box, p_box, box_half, ped_centre, ped_half, mu_lo=0.4, tol=0.003, name="support"):
    """Contacts of the box resting on an axis-aligned pedestal (top plane z = ped_centre_z + ped_half_z, rect = x,y half sizes).
    A box face counts if all 4 corners are within tol of the plane (patch = face clipped to the rectangle); otherwise any single
    box vertex within tol of the plane and inside the rectangle is a point contact.  Fewer contacts than reality => sound."""
    R_box = np.asarray(R_box, float); p_box = np.asarray(p_box, float); h = np.asarray(box_half, float)
    ztop = ped_centre[2] + ped_half[2]
    lo = np.array([ped_centre[0] - ped_half[0], ped_centre[1] - ped_half[1]])
    hi = np.array([ped_centre[0] + ped_half[0], ped_centre[1] + ped_half[1]])
    nrm_box = R_box.T @ np.array([0, 0, 1.0])
    pts, used = [], False
    for k in range(3):
        for s in (+1.0, -1.0):
            idx = [q for q in range(3) if q != k]
            corners = []
            for a_ in (-1, 1):
                for b_ in (-1, 1):
                    p = np.zeros(3); p[k] = s * h[k]; p[idx[0]] = a_ * h[idx[0]]; p[idx[1]] = b_ * h[idx[1]]
                    corners.append(p)
            corners = [corners[0], corners[1], corners[3], corners[2]]               # cyclic order
            zw = [(p_box + R_box @ p)[2] for p in corners]
            if max(zw) - ztop < tol and min(zw) - ztop > -tol:
                xy = [(p_box + R_box @ p)[:2] for p in corners]
                clip = _clip_poly(xy, lo, hi)
                for q in clip:                                                        # back to box coordinates on this face
                    w = np.array([q[0], q[1], ztop]) - p_box
                    pb = R_box.T @ w
                    pb[k] = s * h[k]
                    pts.append(pb)
                used = used or bool(clip)
    if not used:
        for sx, sy, sz in itertools.product((-1, 1), repeat=3):
            p = np.array([sx * h[0], sy * h[1], sz * h[2]]); pw = p_box + R_box @ p
            if abs(pw[2] - ztop) < tol and np.all(pw[:2] >= lo) and np.all(pw[:2] <= hi):
                pts.append(p)
    if not pts:
        return []
    # dedupe
    uniq = []
    for p in pts:
        if all(np.linalg.norm(p - q) > 1e-6 for q in uniq):
            uniq.append(p)
    return support_contacts(uniq, nrm_box, mu_lo, name=name)


# ------------------------------------------------------------------------------------------------ LP machinery
def _edge_dirs(n, m=N_EDGES):
    t1, t2 = _tangents(n)
    ang = 2 * np.pi * np.arange(m) / m
    return np.cos(ang)[:, None] * t1 + np.sin(ang)[:, None] * t2


def mu_effective(mu_lo, m=N_EDGES, edge_scale="vertex"):
    return mu_lo * (np.cos(np.pi / m) if edge_scale == "spec" else 1.0)


def contact_matrix(contacts, m=N_EDGES, edge_scale="vertex"):
    """-> G (6 x nc*m): column = [f; (p x f)/L] for each pyramid edge;  also edge forces F (nc*m x 3)."""
    cols, forces = [], []
    for c in contacts:
        mu = mu_effective(c.mu_lo, m, edge_scale)
        for d in _edge_dirs(c.n, m):
            f = c.n + mu * d
            forces.append(f)
            cols.append(np.concatenate([f, np.cross(c.p, f) / L_SCALE]))
    return np.array(cols).T, np.array(forces)


def _group_rows(contacts, m, extra_cap):
    """inequality rows: per group sum of normals <= f_max; optional per-contact normal cap (for pose robustification)."""
    nc = len(contacts); nl = nc * m
    rows, rhs = [], []
    groups = {}
    for ci, c in enumerate(contacts):
        key = c.group if c.group is not None else f"_own{ci}"
        groups.setdefault(key, []).append(ci)
    for key, idx in groups.items():
        lim = [contacts[i].f_max for i in idx if contacts[i].f_max is not None]
        if not lim:
            continue
        if len(set(lim)) != 1:
            raise ValueError(f"group {key}: inconsistent f_max {lim}")
        r = np.zeros(nl)
        for i in idx:
            r[i * m:(i + 1) * m] = 1.0
        rows.append(r); rhs.append(lim[0])
    caps = np.full(nc, np.inf)
    if extra_cap is not None:
        for ci, c in enumerate(contacts):
            caps[ci] = c.f_max if c.f_max is not None else extra_cap
            if c.f_max is not None and c.group is not None:
                caps[ci] = c.f_max
            r = np.zeros(nl); r[ci * m:(ci + 1) * m] = 1.0
            rows.append(r); rhs.append(caps[ci])
    return (np.array(rows) if rows else np.zeros((0, nl))), np.array(rhs), caps


def _solve_margin(Gm, A_ub, b_ub, w_s, direction):
    """max eps s.t. G lam - eps*direction = -w_s, lam>=0, A_ub lam <= b_ub.  -> eps (or -inf when infeasible)."""
    nl = Gm.shape[1]
    c = np.zeros(nl + 1); c[-1] = -1.0
    A_eq = np.hstack([Gm, -direction[:, None]])
    A = np.hstack([A_ub, np.zeros((A_ub.shape[0], 1))]) if A_ub.shape[0] else None
    res = linprog(c, A_ub=A, b_ub=b_ub if A is not None else None, A_eq=A_eq, b_eq=-w_s,
                  bounds=[(0, None)] * nl + [(-EPS_CAP, EPS_CAP)], method="highs")
    if res.status != 0:
        return -np.inf, None
    return float(res.x[-1]), res.x[:nl]


def _skew_cols():
    return [np.cross(np.eye(3)[j], np.eye(3)[k]) for j in range(3) for k in range(3)]


def wrench_vertices(P: Params):
    """-> (w0 (6,), base [(label, wrench)], cols (6,ne), los, his, names).  w = w0 + base_k + cols @ x, x in prod[los, his].

    base_k: the 2 x 2^d vertices (m in {m_lo,m_hi}) x (COM-offset box corners) of the EXACT gravity wrench
    [m g; (m r) x g].  Correction to the owner spec: the set {(m, s=m r)} = {r_lo m <= s <= r_hi m, m_lo <= m <= m_hi} is ITSELF a
    polytope (linear inequalities in (m, s)), with exactly these vertices; so the vertex check is EXACT in (m, r), not an outer box.
    cols/x: independent-interval over-approximation of the acceleration terms -m a_k (force) and the torque -r x (m a) (q_jk = m r_j a_k
    as independent intervals); empty when a_lo = a_hi = 0.  Unscaled units [N; Nm].  w0 = -I alpha (torque)."""
    g = np.asarray(P.g, float)
    a_lo, a_hi = np.asarray(P.a_lo, float), np.asarray(P.a_hi, float)
    r_lo, r_hi = np.asarray(P.com_lo, float), np.asarray(P.com_hi, float)
    w0 = np.zeros(6); w0[3:] = -np.asarray(P.inertia, float) * np.asarray(P.alpha, float)
    ms = sorted({P.m_lo, P.m_hi})
    rch = [sorted({r_lo[j], r_hi[j]}) for j in range(3)]
    base = []
    for m_ in ms:
        for r in itertools.product(*rch):
            r = np.array(r)
            base.append((dict(m=m_, r=r.tolist()), np.concatenate([m_ * g, np.cross(m_ * r, g)])))

    def prod(lo1, hi1, lo2, hi2):
        c = [lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2]
        return min(c), max(c)
    s = [prod(P.m_lo, P.m_hi, r_lo[j], r_hi[j]) for j in range(3)]
    cols, los, his, names = [], [], [], []
    for k in range(3):
        lo, hi = prod(P.m_lo, P.m_hi, a_lo[k], a_hi[k])
        cols.append(np.concatenate([-np.eye(3)[k], np.zeros(3)])); los.append(lo); his.append(hi); names.append(f"ma{k}")
    q = {(j, k): prod(s[j][0], s[j][1], a_lo[k], a_hi[k]) for j in range(3) for k in range(3)}
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3                                       # T_i extra = q_kj - q_jk
        lo = q[(k, j)][0] - q[(j, k)][1]; hi = q[(k, j)][1] - q[(j, k)][0]
        e = np.zeros(6); e[3 + i] = 1.0
        cols.append(e); los.append(lo); his.append(hi); names.append(f"qT{i}")
    cols = np.array(cols).T; los = np.array(los); his = np.array(his)
    keep = [i for i in range(cols.shape[1]) if his[i] - los[i] > 1e-12]
    for i in range(cols.shape[1]):
        if i not in keep:
            w0 = w0 + cols[:, i] * los[i]
    return w0, base, cols[:, keep], los[keep], his[keep], [names[i] for i in keep]


def _scale(w):
    w = np.array(w, float); w[3:] /= L_SCALE
    return w


def _pose_slack(contacts, P, caps, mu_edge_scale, m):
    """per-component torque error half-width rho_k [Nm] bounding sum_i (delta_i x f_i)_k over |delta| <= pose_delta, |f_i| <= caps_i *
    sqrt(1+mu_eff^2).  Soundness (Kakutani): let Lambda = {lam>=0, group sums <= F, fn_i <= caps_i} (compact convex).  For a fixed
    actual shift delta, T(lam) = {lam' in Lambda : G0 lam' = -w - E(delta) lam}.  If -w + prod[-rho,rho] (torque dims) is inside G0(Lambda)
    (checked at the 8 corners; G0(Lambda) is convex) then T(lam) != {} for every lam in Lambda, T is convex-valued, closed graph, so a
    fixed point exists:  G_actual lam* = G0 lam* + E lam* = -w.  The shifted system therefore balances w."""
    d = np.asarray(P.pose_delta, float)
    rho = np.zeros(3)
    for ci, c in enumerate(contacts):
        if not c.pose_unc:
            continue
        fmax = caps[ci] * np.sqrt(1 + mu_effective(c.mu_lo, m, mu_edge_scale) ** 2)
        for kk in range(3):
            a, b = (kk + 1) % 3, (kk + 2) % 3
            rho[kk] += fmax * np.sqrt(d[a] ** 2 + d[b] ** 2)
    return rho


def _certify_exists(contacts, params: Params, m_edges=N_EDGES, edge_scale="vertex", compute_margin=True, max_vertices=4096, min_margin=0.0):
    """EXISTENCE-ONLY certificate (mode='exists').  -> (ok, eps_star, detail).  See module docstring.  ok is a SOUND claim: the remaining contacts can balance the box for every
    parameter value in the intervals (and every attachment shift, if pose_delta > 0), under the existence-of-forces model."""
    t0 = time.perf_counter()
    contacts = list(contacts)
    hands_active = {c.hand for c in contacts if c.active}
    optimistic = len(hands_active) >= 2
    detail = dict(optimistic_internal_force=optimistic, n_contacts=len(contacts), hands_active=sorted(hands_active), n_lp=0,
                  edge_scale=edge_scale, notes=[])
    if not contacts:
        detail.update(worst_vertex=None, worst_direction=None, elapsed_s=time.perf_counter() - t0)
        return False, -np.inf, detail
    Gm, _ = contact_matrix(contacts, m_edges, edge_scale)
    pose = np.any(np.asarray(params.pose_delta) > 0)
    cap = None
    if pose:
        cap = params.support_cap if params.support_cap is not None else 2.0 * params.m_hi * np.linalg.norm(params.g)
    A_ub, b_ub, caps = _group_rows(contacts, m_edges, cap)
    w0, base, cols, los, his, names = wrench_vertices(params)
    nv = cols.shape[1]
    if len(base) * 2 ** nv > max_vertices:
        raise ValueError(f"{len(base) * 2 ** nv} parameter vertices > max_vertices")
    detail["vertex_vars"] = names
    detail["n_vertices"] = len(base) * 2 ** nv
    rho = np.zeros(3)
    if pose:
        rho = _pose_slack(contacts, params, caps, edge_scale, m_edges)
        detail["pose_rho_Nm"] = rho.tolist()
    dirs = [s * np.eye(6)[j] for j in range(6) for s in (+1.0, -1.0)]
    eps_star, worst = np.inf, (None, None)
    ok = True
    for (blabel, wb), bits in itertools.product(base, itertools.product((0, 1), repeat=nv)):
        x = np.where(np.array(bits, bool), his, los) if nv else np.zeros(0)
        w = w0 + wb + (cols @ x if nv else 0)
        vlabel = dict(blabel, **dict(zip(names, x.tolist())))
        w_s = _scale(w)
        if pose and np.any(rho > 0):                        # corner feasibility of the torque cube (scaled)
            for sg in itertools.product((-1.0, 1.0), repeat=3):
                shift = np.concatenate([np.zeros(3), np.array(sg) * rho / L_SCALE])
                e, _ = _solve_margin(Gm, A_ub, b_ub, w_s - shift, np.zeros(6))
                detail["n_lp"] += 1
                if not np.isfinite(e) or e < -TOL:
                    ok = False
                    eps_star = -np.inf
                    worst = (vlabel, f"pose_corner{sg}")
                    break
            if not ok:
                break
        if not compute_margin:
            e, _ = _solve_margin(Gm, A_ub, b_ub, w_s, np.zeros(6))
            detail["n_lp"] += 1
            if not np.isfinite(e) or e < -TOL:
                ok = False; eps_star = -np.inf
                worst = (vlabel, "feasibility")
                break
            continue
        for di, dv in enumerate(dirs):
            e, _ = _solve_margin(Gm, A_ub, b_ub, w_s, dv)
            detail["n_lp"] += 1
            if e < eps_star:
                eps_star, worst = e, (vlabel, f"{'+' if dv.sum() > 0 else '-'}e{int(np.argmax(np.abs(dv)))}")
        if eps_star < -TOL:
            ok = False
            break
    if not compute_margin and ok:
        eps_star = float("nan")
    if compute_margin:
        ok = ok and eps_star >= min_margin - TOL        # min_margin > 0: require a robust margin (see xcheck: thin-margin equilibria are not realised)
    if eps_star >= EPS_CAP - 1e-6:
        detail["notes"].append("eps* at EPS_CAP: unbounded margin direction (support cone)")
    detail.update(worst_vertex=worst[0], worst_direction=worst[1], elapsed_s=time.perf_counter() - t0)
    return bool(ok), float(eps_star), detail


# ------------------------------------------------------------------------------------------------ COMPLIANT (linear-elastic) mode
@dataclass
class Stiffness:
    """Contact model constants.  The FORCE DISTRIBUTION depends only on the ratios (rho, kappa, rho_s, ct_s/ct); the absolute scale
    (kn, ct) only sets displacement / creep speed.  Defaults = calibrated against the Demo-C MuJoCo scene (grasp_cert_calib.py).
      spring mode (owner spec): every corner is a 3-D linear spring: normal kn_i, tangential kt_i = rho_i kn_i (two directions).
      creep  mode (what MuJoCo does, see calib): normal = spring kn_i, tangential = VISCOUS (force = -ct_i * tangential velocity); the
             steady state has zero normal rate at every contact, so the box moves with a twist v in null(J_n) (creep) plus an elastic u.
      both: a hold must pass under BOTH tangential models."""
    kn: float = 4.7e4              # normal stiffness of one PAD corner [N/m]
    rho: float = 1.0               # pad kt/kn (spring mode)
    kappa: float = 1.0             # support kn / pad kn
    rho_s: float | None = None     # support kt/kn (None = rho)
    kd: float | None = None        # differential jaw-mode stiffness [N/m] (restoring force per unit finger1-minus-finger2 offset a); None = rigid coupling (a = 0)
    tangential: str = "creep"      # 'spring' | 'creep' | 'both'.  creep = what MuJoCo does (measured); spring = owner spec / real rubber pads; both = pass both
    ct: float = 1.2e4              # pad tangential damping per corner [N s/m] (creep mode)
    ct_s: float | None = None      # support damping per corner (None = ct)


def _skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0.0]])


class CompliantModel:
    """Linear statics of the box on N corner contacts (fixed ACTIVE SET: all in contact, all sticking).
       corner i: rel. displacement  r_i = B_i (u - b_h(i)),  B_i = [I, -[p_i]x]  (u = box twist about the box origin, b_h = hand-base twist);
       normal force   fn_i = kn_i (sigma_i s_h + tau_i a_h - g_i - n_i . r_i)    (sigma_i = 1 for the closing pads of hand h, 0 for support;
                      s_h = symmetric jaw closing (force source F_h); a_h = DIFFERENTIAL jaw mode (finger1 +a, finger2 -a, tau_i = +-1) with the
                      restoring spring k_diff = Stiffness.kd (the q2 = -q1 equality is soft in MuJoCo); g_i = unmodelled gap, 0 nominally)
       tangential     ft_i = -kt_i P_t r_i   (spring)   |   -ct_i P_t B_i v   (creep; n_i . B_i v = 0 at every contact)
       equations:  sum_i W_i = -w   (6),   sum_{i in pads(h)} fn_i = F_h   (jaw force source, 1 per hand),
                   sum_{i in pads(h)} tau_i fn_i = kd a_h   (differential mode, 1 per hand with tagged fingers and finite kd),
                   K_h (b_h - b0_h) + sum_{i in h} W_i = 0   (6, only for hands given an arm stiffness; spring mode).
       Everything is affine in (w, F_h, g), so extreme values over a polytope of (w, F) sit at its vertices."""

    def __init__(self, contacts, stiff: Stiffness, tangential="spring", arm=None):
        self.contacts = list(contacts); self.stiff = stiff; self.tangential = tangential
        nc = len(self.contacts); self.nc = nc
        self.hands = sorted({c.hand for c in self.contacts if c.active})
        nh = len(self.hands); self.nh = nh
        arm = arm or {}
        self.arm_hands = [h for h in self.hands if h in arm and tangential == "spring"]
        self.a_hands = [h for h in self.hands if stiff.kd is not None and any(c.hand == h and c.active and c.finger in (1, 2) for c in self.contacts)]
        na = len(self.a_hands); self.na = na
        self.p = np.array([c.p for c in self.contacts]); self.n = np.array([c.n for c in self.contacts])
        self.t1 = np.zeros((nc, 3)); self.t2 = np.zeros((nc, 3))
        for i, c in enumerate(self.contacts):
            self.t1[i], self.t2[i] = _tangents(c.n)
        B = [np.hstack([np.eye(3), -_skew(p)]) for p in self.p]
        self.an = np.array([self.n[i] @ B[i] for i in range(nc)])
        self.at1 = np.array([self.t1[i] @ B[i] for i in range(nc)]); self.at2 = np.array([self.t2[i] @ B[i] for i in range(nc)])
        sup = np.array([c.hand == "support" or not c.active for c in self.contacts])
        self.kn = np.where(sup, stiff.kappa * stiff.kn, stiff.kn) * np.array([c.kn_scale for c in self.contacts])
        rho_s = stiff.rho if stiff.rho_s is None else stiff.rho_s
        self.kt = np.where(sup, rho_s * self.kn, stiff.rho * self.kn)
        ct_s = stiff.ct if stiff.ct_s is None else stiff.ct_s
        self.ct = np.where(sup, ct_s, stiff.ct)
        self.hidx = np.array([self.hands.index(c.hand) if (c.active and c.hand in self.hands) else -1 for c in self.contacts])
        self.tau = np.array([(1.0 if c.finger == 1 else -1.0 if c.finger == 2 else 0.0) if (c.hand in self.a_hands) else 0.0 for c in self.contacts])
        self.aidx = np.array([self.a_hands.index(c.hand) if (c.hand in self.a_hands and c.active) else -1 for c in self.contacts])
        # unknown layout:  [ u/z (nu) | s (nh) | a (na) | creep y | arm b (6 each) ]
        if tangential == "spring":
            self.nu = 6; self.R = np.eye(6); self.Nn = np.zeros((6, 0))
        else:
            U, S_, Vt = np.linalg.svd(self.an, full_matrices=True)
            r = int(np.sum(S_ > RANK_TOL * max(S_[0], 1e-30))) if len(S_) else 0        # tolerance: a 0.02 deg tilt of a support normal must not add DoFs
            self.R = Vt[:r].T; self.Nn = Vt[r:].T; self.nu = r
        self.nv_ = self.Nn.shape[1]
        o = self.nu
        self.iu = slice(0, o); self.is_ = slice(o, o + nh); o += nh
        self.ia = slice(o, o + na); o += na
        self.ic = slice(o, o + self.nv_); o += self.nv_
        self.ib = {h: slice(o + 6 * k, o + 6 * (k + 1)) for k, h in enumerate(self.arm_hands)}
        nx = o + 6 * len(self.arm_hands); self.nx = nx
        self.row_jaw, self.row_a, self.row_arm = 6, 6 + nh, 6 + nh + na
        Fn = np.zeros((nc, nx)); Ft1 = np.zeros((nc, nx)); Ft2 = np.zeros((nc, nx))
        for i in range(nc):
            h = self.hands[self.hidx[i]] if self.hidx[i] >= 0 else None
            rowu = np.zeros(nx); rowu[self.iu] = self.an[i] @ self.R
            rowb = np.zeros(nx)
            if h in self.ib:
                rowb[self.ib[h]] = self.an[i]
            Fn[i] = -self.kn[i] * (rowu - rowb)
            if self.hidx[i] >= 0:
                Fn[i, self.is_.start + self.hidx[i]] += self.kn[i]
            if self.aidx[i] >= 0:
                Fn[i, self.ia.start + self.aidx[i]] += self.kn[i] * self.tau[i]
            for A_, Ft in ((self.at1, Ft1), (self.at2, Ft2)):
                if tangential == "spring":
                    ru = np.zeros(nx); ru[self.iu] = A_[i] @ self.R
                    rb = np.zeros(nx)
                    if h in self.ib:
                        rb[self.ib[h]] = A_[i]
                    Ft[i] = -self.kt[i] * (ru - rb)
                else:
                    ru = np.zeros(nx); ru[self.ic] = A_[i] @ self.Nn
                    Ft[i] = -self.ct[i] * ru
        self.Fn, self.Ft1, self.Ft2 = Fn, Ft1, Ft2
        Wc = np.zeros((nc, 6, nx))
        for i in range(nc):
            for vec, Fm in ((self.n[i], Fn[i]), (self.t1[i], Ft1[i]), (self.t2[i], Ft2[i])):
                Wc[i, :3] += np.outer(vec, Fm); Wc[i, 3:] += np.outer(np.cross(self.p[i], vec), Fm)
        self.Wc = Wc
        rows = [Wc.sum(0)]                                                          # 6 rows : = -w
        for k in range(nh):                                                         # jaw rows : = F_h
            rows.append(sum(Fn[i] for i in range(nc) if self.hidx[i] == k)[None, :] if np.any(self.hidx == k) else np.zeros((1, nx)))
        for k, h in enumerate(self.a_hands):                                        # differential rows : sum tau fn - kd a = 0
            r_ = sum(self.tau[i] * Fn[i] for i in range(nc) if self.aidx[i] == k)
            r_ = np.array(r_, float); r_[self.ia.start + k] -= stiff.kd
            rows.append(r_[None, :])
        for h in self.arm_hands:                                                    # arm rows : K b + sum W_h = K b0
            Kh = np.asarray(arm[h][0], float)
            blk = sum(Wc[i] for i in range(nc) if self.hidx[i] >= 0 and self.hands[self.hidx[i]] == h)
            blk = np.array(blk, float); blk[:, self.ib[h]] += Kh
            rows.append(blk)
        self.A = np.vstack(rows)
        self.arm = {h: (np.asarray(arm[h][0], float), np.asarray(arm[h][1], float) if len(arm[h]) > 1 else np.zeros(6)) for h in self.arm_hands}
        self.cond = float(np.linalg.cond(self.A)) if self.A.shape[0] == self.A.shape[1] else np.inf
        self.singular = (self.A.shape[0] != self.A.shape[1]) or not np.isfinite(self.cond) or self.cond > 1e11

    def _gap_rhs(self):
        nc = self.nc
        rg = np.zeros((self.A.shape[0], nc))
        wn = np.array([np.concatenate([self.n[j], np.cross(self.p[j], self.n[j])]) for j in range(nc)])
        for j in range(nc):
            rg[:6, j] = self.kn[j] * wn[j]
            if self.hidx[j] >= 0:
                rg[self.row_jaw + self.hidx[j], j] = self.kn[j]
                h = self.hands[self.hidx[j]]
                if h in self.arm:
                    k = self.arm_hands.index(h)
                    rg[self.row_arm + 6 * k:self.row_arm + 6 * (k + 1), j] += self.kn[j] * wn[j]
            if self.aidx[j] >= 0:
                rg[self.row_a + self.aidx[j], j] = self.tau[j] * self.kn[j]
        return rg

    def gap_sensitivity(self):
        """Sensitivity of the solution to an unmodelled initial gap g_j (m, + = open) at corner j (normal direction):  fn_j = kn_j (sigma s - g_j - n.r).
        -> dict(dfn (nc,nc), dft (nc,2,nc), du (6,nc), dv (6,nc)).  Exact for the model (linear)."""
        Xg = np.linalg.solve(self.A, self._gap_rhs())
        return dict(dfn=self.Fn @ Xg - np.diag(self.kn), dft=np.stack([self.Ft1 @ Xg, self.Ft2 @ Xg], 1),
                    du=self.R @ Xg[self.iu], dv=(self.Nn @ Xg[self.ic]) if self.nv_ else np.zeros((6, self.nc)))

    def solve(self, W, F, gap=None):
        """W (nv,6) wrenches the contacts must balance (= gravity wrench w; contacts produce -w), F (nv,nh) jaw forces, gap (nc,) optional initial gaps.
        -> dict(fn (nv,nc), ft (nv,nc,2), u (nv,6), v (nv,6) creep twist, x)."""
        W = np.atleast_2d(W); F = np.atleast_2d(F)
        nv = W.shape[0]
        rhs = [-W, F, np.zeros((nv, self.na))]
        for h in self.arm_hands:
            Kh, b0 = self.arm[h]
            rhs.append(np.tile(Kh @ b0, (nv, 1)))
        rhs = np.hstack(rhs).T
        if gap is not None:
            rhs = rhs + (self._gap_rhs() @ np.asarray(gap, float))[:, None]
        x = np.linalg.solve(self.A, rhs).T
        fn = x @ self.Fn.T; ft1 = x @ self.Ft1.T; ft2 = x @ self.Ft2.T
        if gap is not None:
            fn = fn - self.kn * np.asarray(gap, float)
        u = x[:, self.iu] @ self.R.T
        v = x[:, self.ic] @ self.Nn.T if self.nv_ else np.zeros((nv, 6))
        return dict(fn=fn, ft=np.stack([ft1, ft2], -1), u=u, v=v, x=x)


def cartesian_stiffness_from_jacobian(J, Kq, R_box=None, x_hand=None, origin=None):
    """6x6 arm stiffness for CompliantModel(arm=...):  K_x = (J Kq^-1 J^T)^-1  (J 6 x n_dof world [lin; ang] at the hand base, Kq diag/matrix of joint
    stiffness; valid for redundant arms when the joint spring is the only restoring torque).  If R_box, x_hand, origin (box origin, world) are given the result
    is expressed for a twist [v; w] about the BOX ORIGIN in the BOX frame (the convention of CompliantModel):  v_o = v_h + w x (o - x_h)."""
    J = np.asarray(J, float); Kq = np.asarray(Kq, float)
    Kq = np.diag(Kq) if Kq.ndim == 1 else Kq
    C = J @ np.linalg.solve(Kq, J.T)
    if R_box is not None:
        r = np.asarray(origin, float) - np.asarray(x_hand, float)
        T = np.block([[np.eye(3), -_skew(r)], [np.zeros((3, 3)), np.eye(3)]])
        Rb = np.kron(np.eye(2), np.asarray(R_box, float).T)
        C = Rb @ (T @ C @ T.T) @ Rb.T
    return np.linalg.inv(C)


def _grip_intervals(contacts, grip_force, hands):
    out = []
    for h in hands:
        if grip_force is not None and h in grip_force:
            g = grip_force[h]; lo, hi = (g, g) if np.isscalar(g) else (g[0], g[1])
        else:
            lims = [c.f_max for c in contacts if c.hand == h and c.active and c.f_max is not None]
            lo = hi = float(lims[0]) if lims else 0.0
        out.append((float(lo), float(hi)))
    return out


def _certify_compliant(contacts, params: Params, stiffness: Stiffness | None = None, grip_force=None, n_min=0.05, theta_tol_deg=3.0,
                       x_tol=5e-3, v_tol=1e-3, friction="cone", arm=None, gap_tol=GAP_TOL, check_exists=True, max_vertices=4096, **exists_kw):
    """mode='compliant'.  Admission = (1) all corner normals >= n_min, (2) all tangential forces inside the friction cone (or the
    inscribed pyramid) at mu_lo, (3) box rotation <= theta_tol, translation <= x_tol (spring) / creep surface speed <= v_tol (creep),
    (4) the existence LP is feasible -- at EVERY vertex of (m, m r)-polytope x acceleration box x jaw-force box.  Exact for the model
    because the corner forces and u are affine in (w, F) and the constraints are convex (|ft| <= mu fn is a convex function of f).
    gap_tol [m] (default GAP_TOL): every corner has an UNMODELLED initial gap |g_j| <= gap_tol (pad/face misalignment, flatness, IK error; at kn ~ 5e4 N/m
    12 um is already 0.6 N).  The forces are also affine in g, so the worst case over the g-box is bounded in closed form with the L1 row sums of the
    sensitivity matrix (normal: fn_i - gap_tol sum_j|dfn_ij| >= n_min; friction: |ft_i| + gap_tol sum_j||dft_ij|| <= mu (fn_i - gap_tol sum_j|dfn_ij|); same for u, v).
    gap_tol = 0 is the plain owner spec (perfect nominal geometry)."""
    t0 = time.perf_counter()
    stiff = stiffness or Stiffness()
    contacts = list(contacts)
    hands_active = sorted({c.hand for c in contacts if c.active})
    detail = dict(mode="compliant", optimistic_internal_force=len(hands_active) >= 2, n_contacts=len(contacts), hands_active=hands_active, notes=[],
                  stiffness=dict(stiff.__dict__), gap_tol=gap_tol, n_min=n_min, theta_tol_deg=theta_tol_deg, x_tol=x_tol, v_tol=v_tol, friction=friction)
    if not contacts:
        detail.update(worst_vertex=None, worst_direction=None, elapsed_s=time.perf_counter() - t0, margins={})
        return False, -np.inf, detail
    w0, base, cols, los, his, names = wrench_vertices(params)
    nvv = cols.shape[1]
    ints = _grip_intervals(contacts, grip_force, hands_active)
    fvars = [k for k, (lo, hi) in enumerate(ints) if hi - lo > 1e-12]
    if len(base) * 2 ** (nvv + len(fvars)) > max_vertices:
        raise ValueError("too many parameter vertices")
    Wv, Fv, labels = [], [], []
    for (bl, wb), bits in itertools.product(base, itertools.product((0, 1), repeat=nvv)):
        x = np.where(np.array(bits, bool), his, los) if nvv else np.zeros(0)
        w = w0 + wb + (cols @ x if nvv else 0)
        for fb in itertools.product((0, 1), repeat=len(fvars)):
            F = np.array([lo for lo, hi in ints], float)
            for q, kk in enumerate(fvars):
                F[kk] = ints[kk][1] if fb[q] else ints[kk][0]
            Wv.append(w); Fv.append(F); labels.append(dict(bl, **dict(zip(names, x.tolist())), F_g=F.tolist()))
    Wv = np.array(Wv); Fv = np.array(Fv).reshape(len(Wv), len(hands_active))
    detail["n_vertices"] = len(Wv)
    models = ("spring", "creep") if stiff.tangential == "both" else (stiff.tangential,)
    mu = np.array([c.mu_lo for c in contacts])
    fscale = max(1e-9, float(np.max([lo for lo, hi in ints] + [params.m_hi * np.linalg.norm(params.g)])))
    R_c = float(np.max(np.linalg.norm([c.p for c in contacts], axis=1)))
    ok = True
    rel_margin = np.inf; worst = (None, None)
    allm = {}
    for tm in models:
        M = CompliantModel(contacts, stiff, tangential=tm, arm=arm)
        if M.singular:
            detail["notes"].append(f"{tm}: stiffness matrix singular (cond {M.cond:.2e}) -> reject (active set not determined)")
            allm[tm] = dict(singular=True, cond=M.cond)
            ok = False; rel_margin = -np.inf
            continue
        S_ = M.solve(Wv, Fv)
        fn, ft = S_["fn"], S_["ft"]
        if gap_tol > 0:
            G = M.gap_sensitivity()
            d_fn = gap_tol * np.abs(G["dfn"]).sum(1)                            # (nc,) worst-case normal-force loss
            d_ft = gap_tol * np.linalg.norm(G["dft"], axis=1).sum(1)            # (nc,) worst-case tangential-force gain
            d_ut = gap_tol * np.linalg.norm(G["du"][:3], axis=0).sum(); d_ur = gap_tol * np.linalg.norm(G["du"][3:], axis=0).sum()
            d_sp = gap_tol * (np.linalg.norm(G["dv"][:3], axis=0) + R_c * np.linalg.norm(G["dv"][3:], axis=0)).sum()
        else:
            d_fn = d_ft = np.zeros(M.nc); d_ut = d_ur = d_sp = 0.0
        fn_w = fn - d_fn                                                        # worst-case normal force per corner
        nrm_m = fn_w.min(1) - n_min
        ftn = (np.linalg.norm(ft, axis=-1) if friction == "cone" else np.max(np.abs(ft), -1) * np.sqrt(2.0)) + d_ft
        fr_m = np.where(fn_w > 1e-9, 1.0 - ftn / np.maximum(mu * fn_w, 1e-12), -np.inf).min(1)
        trans = np.linalg.norm(S_["u"][:, :3], axis=1) + d_ut; rot = np.degrees(np.linalg.norm(S_["u"][:, 3:], axis=1) + d_ur)
        if tm == "spring":
            dsp_m = np.minimum(1 - rot / theta_tol_deg, 1 - trans / x_tol)
        else:
            sp = np.linalg.norm(S_["v"][:, :3], axis=1) + R_c * np.linalg.norm(S_["v"][:, 3:], axis=1) + d_sp
            dsp_m = np.minimum(1 - sp / v_tol, np.minimum(1 - rot / theta_tol_deg, 1 - trans / x_tol))
        comp = np.minimum(np.minimum(nrm_m / fscale, fr_m), dsp_m)
        k = int(np.argmin(comp))
        allm[tm] = dict(normal_margin_N=float(nrm_m.min()), friction_margin=float(fr_m.min()), disp_margin=float(dsp_m.min()),
                        min_fn_N=float(fn.min()), min_fn_worst_gap_N=float(fn_w.min()), max_ft_over_mu_fn=float(1 - fr_m.min()), max_rot_deg=float(rot.max()), max_trans_mm=float(1e3 * trans.max()),
                        max_creep_surface_speed_mm_s=(float(1e3 * sp.max()) if tm == "creep" else None), cond=M.cond,
                        worst_vertex=labels[k], worst_condition=["normal", "friction", "displacement"][int(np.argmin([nrm_m[k] / fscale, fr_m[k], dsp_m[k]]))])
        if comp.min() < rel_margin:
            rel_margin = float(comp.min()); worst = (labels[k], f"{tm}:{allm[tm]['worst_condition']}")
        if comp.min() < 0:
            ok = False
    detail["compliant"] = allm
    if check_exists:
        kw = dict(exists_kw); kw.setdefault("compute_margin", False)
        ok_e, eps_e, det_e = _certify_exists(contacts, params, **kw)
        detail["exists_ok"] = bool(ok_e); detail["exists_eps"] = float(eps_e)
        if not ok_e:
            ok = False; rel_margin = min(rel_margin, -1.0); worst = (det_e.get("worst_vertex"), "exists")
    detail.update(worst_vertex=worst[0], worst_direction=worst[1], elapsed_s=time.perf_counter() - t0)
    return bool(ok), float(rel_margin), detail


def certify_hold(contacts, params: Params, mode="compliant", **kw):
    """-> (ok, eps, detail).  mode='compliant' (default): existence AND linear-elastic consistency (see _certify_compliant; eps = worst
    DIMENSIONLESS margin of normal-force / friction / displacement, ok <=> >= 0).  mode='exists': force-existence LP only
    (eps = wrench-space margin, N); keywords of that mode: m_edges, edge_scale, compute_margin, max_vertices, min_margin."""
    if mode == "exists":
        return _certify_exists(contacts, params, **kw)
    if mode != "compliant":
        raise ValueError(mode)
    return _certify_compliant(contacts, params, **kw)


# ------------------------------------------------------------------------------------------------ scene glue
@dataclass
class SceneContacts:
    contacts: list
    g_box: np.ndarray
    R_box: np.ndarray
    p_box: np.ndarray
    box_half: np.ndarray


def contacts_from_scene(model, data, which=("left", "right", "support"), mu_lo=0.4, mu_hi=None, f_limit=HAND_NORMAL_LIMIT,
                        limit_mode="hand_total", box_geom="democ_box_geom", box_body="democ_box", shrink=1.0):
    """Contacts for the CURRENT hand/box poses (FK only; does not read MuJoCo contact forces).  Hands are assumed closed on the box
    (jaw faces touch the two box faces).  Support = pedestal top (democ_table_pedestal) overlap patch.  -> SceneContacts
    (box-frame contacts, gravity in the box frame).  Requires mj_forward to be current."""
    import mujoco
    bid = model.body(box_body).id
    R_box = data.xmat[bid].reshape(3, 3).copy(); p_box = data.xpos[bid].copy()
    h = model.geom_size[model.geom(box_geom).id].copy()
    out = []
    for hand in which:
        if hand == "support":
            gid = model.geom("democ_table_pedestal").id
            out += support_patch_from_pedestal(R_box, p_box, h, data.geom_xpos[gid], model.geom_size[gid], mu_lo)
            continue
        gb = model.body(f"{hand}_arm_gripper_base").id
        Rg = data.xmat[gb].reshape(3, 3)
        f1 = R_box.T @ (data.xpos[model.body(f"{hand}_arm_finger1").id] - p_box)
        out += pad_contacts(hand, R_box.T @ Rg, R_box.T @ (data.xpos[gb] - p_box), h, mu_lo, mu_hi, f_limit, limit_mode, shrink=shrink, finger1_pos=f1)
    return SceneContacts(out, R_box.T @ np.array([0, 0, -G]), R_box, p_box, h)


def release_ok(scene_state: SceneContacts, hand: str, params: Params, **kw):
    """Is the box held by the contacts that remain after `hand` lets go?  -> (ok, eps, detail).  Gravity is taken from the scene
    unless params.g was changed from the default."""
    rest = [c for c in scene_state.contacts if c.hand != hand]
    P = params
    if tuple(P.g) == (0.0, 0.0, -G):
        P = replace(P, g=tuple(scene_state.g_box))
    return certify_hold(rest, P, **kw)
