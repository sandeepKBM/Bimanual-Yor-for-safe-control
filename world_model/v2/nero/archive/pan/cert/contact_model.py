"""pan/cert/contact_model.py -- conservative quasi-static contact model for the two-claw lift hold (USER 2026-10-30).  SIMULATION
ONLY, pure geometry + LP (no MuJoCo stepping, no contact forces read).

Question: with both claws closed, can the pan + lid + contents be held off the stove for every theta in the belief, every pan
acceleration in the box and every tilt up to TILT_MAX, by contact forces the conservative model allows?  Admission rests on this
model; MuJoCo (and Drake) only try to break it (cert/contact_falsify.py).

Model (labels per AGENTS.md "Bounds and assumptions"):
  * Bodies: PAN (pan + hidden puck, the puck's weight acting anywhere in the contents polytope) and LID (separate; resting on the
    24 wall tops, clamped at one edge by the right claw's upper finger).  Quasi-static: loads m * g_eff, g_eff = g - a.  MODEL
  * Contacts are POINTS with Coulomb friction and ZERO torsional friction; the cone is replaced by an inner N_EDGES pyramid
    (edges on the cone, so the pyramid lies inside it: sound).  MODEL
  * Friction is NOT a belief state (USER 2026-10-30): a fixed interval, worst end per check.  Holding uses
    mu = max(geom pair) * MU_SCALE_LO.  MU_SCALE_LO = 0.6 = the prior's lowest friction scale.  ASSUMED (prior)
  * Where a finger touches: the finger's convex hull (the shape MuJoCo collides) against the target surface plane; contact
    points = the part of the hull's extreme face within FLUSH_H of the surface, clipped to the surface region.  A flush pad
    gives its corners (any centre of pressure in their hull); a tilted pad gives an edge or a corner (no moment from that
    finger).  FLUSH_H: ASSUMED.
  * Squeeze: the gripper actuator is a position servo clipped at forcerange 10 N, finger 2 slaved to finger 1 by an equality,
    so the SUM of the two pads' normal forces is <= F_HAND (as democ/grasp_cert, verified there).  Joint-limit forces that add
    squeeze in the sim are ignored (conservative).  MODEL (MJCF)
  * Lid tabs (lateral stops inside the wall) are ignored: lid retention is friction only (conservative).
  * Two claws hold one body: "there exist contact forces" is OPTIMISTIC about the internal force split (it is set by the arm
    controllers, not chosen).  ASSUMED A_int; the falsifier tests it.
  * The puck is a free body in the sim; the model treats its weight as acting at a fixed point in the contents polytope
    (sliding inside the disc stays inside the polytope; impact loads are not modelled).  MODEL (quasi-static)

Exactness: for fixed geometry (one pose sample), the set of loads with load factor k* >= 1 is convex, and the loads are
bilinear in (contents vertex pi, g_eff corner).  So checking the vertex pairs is EXACT over the contents polytope x the g_eff
box (proof under the MODEL).  The pose residual delta changes the geometry (the LP matrix), so it is SAMPLED on a grid (27
points); the guarantee is "proof over contents/acceleration/tilt, sampled over the pose residual".
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from world_model.v2.nero.pan.cert import belief as B
from world_model.v2.nero.archive.democ import grasp_cert as GC

G = 9.81
MU_SCALE_LO = B.PRIOR_MU[0]   # ASSUMED (prior): worst friction scale for holding; friction is a fixed interval, not narrowed
N_EDGES = 8                   # design: friction pyramid edges (inner)
FLUSH_H = 1.0e-4              # ASSUMED: hull points within 0.1 mm of the surface count as touching (sets the pad patch)
A_BOX = 0.5                   # ASSUMED: |a_k| <= A_BOX per axis [m/s^2] (= model_check.A_MAX; planned lift peak 6*0.05/1.5^2 = 0.13)
TILT_MAX_DEG = 3.0            # ASSUMED: pan tilt during the lift (= skills_check POST lift tilt); gravity anywhere in that cone
TILT_WIDE_DEG = 11.0          # ASSUMED (10-30, USER "do both"): widened bound = max pan tilt seen in the 8 smoke lifts (n = 8, not a bound)
CLAW_REL_MM = 2.0             # ASSUMED (USER 11-01): the two claws keep their relative pose during the lift within this [mm] ...
CLAW_REL_DEG = 2.0            # ASSUMED (USER 11-01): ... and this [deg]; the hold model treats both claws as one rigid fixture
PUSH_SLIP_POS = 0.003        # ASSUMED: jaw point slip along the handle during push + clamp [m] (seen 2.7 / 1.8 mm, I03 / V10, n = 2)
PUSH_SLIP_YAW = 3.0           # ASSUMED: pan yaw change during push + clamp, vs the estimate [deg] (seen 0.75 / 1.7, n = 2)
# The pan does NOT turn with the left claw: the claw twists 3-6 deg about its jaw axis during the push (trace 10-30).  So the
# robot's pan estimate at the lift start = the zed estimate moved by the jaw point's FK displacement, heading kept; the pose
# residual delta is a rotation about the JAW POINT plus a shift.
K_CAP = 50.0                  # design: load-factor cap in the LP (reported as >= K_CAP)
SURF = dict(left=("handle_top", "handle_bottom"), right=("lid_top", "bottom_under"))   # (upper finger, lower finger) targets


def assumptions():
    return dict(MU_SCALE_LO=(MU_SCALE_LO, "ASSUMED prior min; friction fixed interval, worst end"), FLUSH_H_m=(FLUSH_H, "ASSUMED"),
                A_BOX=(A_BOX, "ASSUMED"), TILT_MAX_DEG=(TILT_MAX_DEG, "ASSUMED"), TILT_WIDE_DEG=(TILT_WIDE_DEG, "ASSUMED, max seen n=8"), PUSH_SLIP=((PUSH_SLIP_POS, PUSH_SLIP_YAW), "ASSUMED"),
                CLAW_REL=((CLAW_REL_MM, CLAW_REL_DEG), "ASSUMED (USER 11-01): claw-to-claw pose drift during the lift, mm / deg"),
                AIM=((B.AIM_POS, B.AIM_YAW), "ASSUMED from MEASURED max over n=200"), A_int="ASSUMED: internal force split",
                tabs="ignored (conservative)", torsion="zero (conservative)",
                squeeze="sum of pad normals = |actuator force|; [0, F] when a finger joint is at its limit (MODEL)",
                split="Demo C compliant model (creep, kd rigid) -- STIFF/GAP_TOL ASSUMED (Demo C fit, not refit)",
                touch=("GAP_MAX 0.1 mm / PEN_MAX 2 mm", "ASSUMED"), compliant_tols=dict(N_MIN=0.05, THETA_TOL=3.0, X_TOL=5e-3, V_TOL=1e-3))


# ------------------------------------------------------------------------------------------------------------ pan model (MODEL)
class PanModel:
    """pan + lid geometry and mass properties in the PAN BODY frame, read from the compiled MJCF (MODEL)."""

    def __init__(self, ep):
        m, p = ep.m, ep.p
        self.m_pan, self.m_lid = float(m.body_mass[ep.pan]), float(m.body_mass[ep.lid])
        self.c_pan = m.body_ipos[ep.pan].copy()
        lid_R = np.eye(3)                                            # seated lid: same orientation as the pan
        self.lid_o = np.asarray(ep.lid_seat, float)
        self.c_lid = self.lid_o + lid_R @ m.body_ipos[ep.lid]
        self.centre = m.site_pos[m.site("pan_centre").id].copy()
        self.u = ep.R0.T @ p.handle_axis()                            # handle axis in the pan frame
        self.w = np.cross([0, 0, 1.0], self.u)
        from world_model.v2.nero.pan import covered as CV
        self.z_puck = p.pan_bottom_t + CV.ASSUMED_COVERED["puck_h"] / 2
        self.base_mu = {}                                            # sliding friction per geom BEFORE the world's mu_scale
        sc = float((ep.world.get("perturb") or {}).get("mu_scale") or 1.0)
        for n in ("pan_handle", "pan_bottom", "lid_disk", "pan_wall0", "left_arm_finger1_col"):
            self.base_mu[n] = float(m.geom_friction[m.geom(n).id, 0]) / sc
        hg = m.geom("pan_handle").id
        Rh = _quat_mat(m.geom_quat[hg])
        self.handle = dict(pos=m.geom_pos[hg].copy(), R=Rh, half=m.geom_size[hg].copy())
        bg = m.geom("pan_bottom").id
        self.r_bottom = float(m.geom_size[bg, 0])
        self.z_bottom_under = float(m.geom_pos[bg, 2] - m.geom_size[bg, 1])
        lg = m.geom("lid_disk").id
        self.r_lid = float(m.geom_size[lg, 0])
        self.z_lid_top = float(self.lid_o[2] + m.geom_pos[lg, 2] + m.geom_size[lg, 1])
        self.rim = []                                                # wall-top contact points (2 per wall, mid-thickness)
        for g in range(m.ngeom):
            if (m.geom(g).name or "").startswith("pan_wall"):
                Rw, pw, h = _quat_mat(m.geom_quat[g]), m.geom_pos[g], m.geom_size[g]
                for sgn in (-1, 1):
                    self.rim.append(pw + Rw @ np.array([0.0, sgn * h[1], h[2]]))
        self.rim = np.array(self.rim)

    def mu(self, a, b, scale):
        return max(self.base_mu[a], self.base_mu[b]) * scale

    def surfaces(self):
        """target surfaces in the pan frame: point on plane, outward normal (towards the finger), 2-D region test + polygon."""
        h = self.handle
        out = {}
        for nm, sgn in (("handle_top", 1), ("handle_bottom", -1)):
            k = int(np.argmin(h["half"]))                         # thin axis of the handle box
            n = sgn * h["R"][:, k]
            if n[2] * sgn < 0:
                n = -n
            ax = [i for i in range(3) if i != k]
            corners = [h["pos"] + n * h["half"][k] + a * h["R"][:, ax[0]] * h["half"][ax[0]] + b * h["R"][:, ax[1]] * h["half"][ax[1]]
                       for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            out[nm] = dict(p=h["pos"] + n * h["half"][k], n=n, poly=np.array(corners))
        disc = lambda r, z, n: dict(p=np.array([0.0, 0.0, z]), n=np.array([0, 0, n], float),
                                    poly=np.array([[r * np.cos(t), r * np.sin(t), z] for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)]))
        out["bottom_under"] = disc(self.r_bottom, self.z_bottom_under, -1.0)   # inscribed 64-gon: a subset of the disc (conservative)
        out["lid_top"] = disc(self.r_lid, self.z_lid_top, 1.0)
        return out


def _quat_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


# ------------------------------------------------------------------------------------------------------------ fingers (robot FK)
def finger_hulls(m, d, side):
    """[(name, world hull vertices)] for the two finger collision meshes of `side` (robot FK only)."""
    out = []
    for f in (1, 2):
        g = m.geom(f"{side}_arm_finger{f}_col").id
        did = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[did]:m.mesh_vertadr[did] + m.mesh_vertnum[did]].astype(float)
        V = V[ConvexHull(V).vertices]
        Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        out.append((f"{side}_arm_finger{f}_col", pg + V @ Rg.T))
    return out


def _clip(poly, a, b):
    """keep the part of the 2-D convex polygon poly with a . x <= b."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a @ p - b, a @ q - b
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            out.append(p + (q - p) * fp / (fp - fq))
    return out


def _hull2(P):
    if len(P) < 3:
        return P
    try:
        return P[ConvexHull(P).vertices]
    except Exception:                  # collinear points: keep the two extremes
        d = P - P.mean(axis=0)
        u = np.linalg.svd(d)[2][0]
        t = d @ u
        return P[[int(np.argmin(t)), int(np.argmax(t))]]


GRID = 0.0005                 # design: sampling step of the finger's height map over the surface region [m]


def contact_points(Vw, surf, flush_h=FLUSH_H, grid=GRID):
    """finger hull vertices (world) vs a surface (world: p, n outward towards the finger, poly = region corners on the plane).
    Height map: for each grid point q of (region ∩ hull shadow), t(q) = where the ray q + t n enters the hull (t < 0: the hull
    reaches below the plane there).  t is convex over q (lower boundary of a convex body), so {t <= min t + flush_h} is convex
    and the hull of its grid points lies inside it (an inner set: sound for the centre of pressure).
    -> (contact points on the plane (k,3), min t [m], n grid points in the overlap).  Empty if the shadow misses the region."""
    n = surf["n"] / np.linalg.norm(surf["n"])
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    hull = ConvexHull(Vw)
    A, b = hull.equations[:, :3], hull.equations[:, 3]
    S2 = np.c_[(Vw - surf["p"]) @ e1, (Vw - surf["p"]) @ e2]
    R2 = np.c_[(surf["poly"] - surf["p"]) @ e1, (surf["poly"] - surf["p"]) @ e2]
    R2 = R2[ConvexHull(R2).vertices]
    lo, hi = np.maximum(S2.min(0), R2.min(0)), np.minimum(S2.max(0), R2.max(0))
    if np.any(hi < lo):
        return np.zeros((0, 3)), np.inf, 0
    xs, ys = np.arange(lo[0], hi[0] + grid, grid), np.arange(lo[1], hi[1] + grid, grid)
    Q = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    E = np.roll(R2, -1, axis=0) - R2
    inside = np.all(E[None, :, 0] * (Q[:, None, 1] - R2[None, :, 1]) - E[None, :, 1] * (Q[:, None, 0] - R2[None, :, 0]) >= -1e-12,
                    axis=1)                                          # CCW region: left of every edge
    Q = Q[inside]
    if len(Q) == 0:
        return np.zeros((0, 3)), np.inf, 0
    X = surf["p"] + Q[:, :1] * e1 + Q[:, 1:] * e2
    an = A @ n
    c = X @ A.T + b                                                  # (k, facets): a.q + b
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = -c / an
    t_in = np.where(an < -1e-12, tt, -np.inf).max(axis=1)
    t_out = np.where(an > 1e-12, tt, np.inf).min(axis=1)
    par_ok = np.all(np.where(np.abs(an) <= 1e-12, c <= 1e-12, True), axis=1)
    hit = (t_in <= t_out) & par_ok
    if not hit.any():
        return np.zeros((0, 3)), np.inf, 0
    t_min = float(t_in[hit].min())
    sel = hit & (t_in <= t_min + flush_h)
    P2 = _hull2(Q[sel])
    pts = np.array([surf["p"] + x[0] * e1 + x[1] * e2 for x in P2])
    return pts, t_min, int(hit.sum())


# ------------------------------------------------------------------------------------------------------------ the hold LP
def _edges(n, mu, k=N_EDGES):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return [n + mu * (np.cos(2 * np.pi * j / k) * t1 + np.sin(2 * np.pi * j / k) * t2) for j in range(k)]


def g_eff_corners(a_box=A_BOX, tilt_deg=TILT_MAX_DEG):
    """box containing g_dir * G - a over |tilt| <= tilt_deg and |a_k| <= a_box (8 corners, world frame, pointing down)."""
    s, c = np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))
    xy = G * s + a_box
    return [np.array([x, y, z]) for x in (-xy, xy) for y in (-xy, xy) for z in (-(G + a_box), -(G * c - a_box))]


class Hold:
    """one geometry sample: contacts (world) for the pan and the lid + the LP matrices.  load_factor(loads) -> k*."""

    def __init__(self, contacts, rim_pts, mu_rim, o, f_hand):
        """contacts: list of dict(body 'pan'|'lid', hand, p (k,3), n (into the body), mu).  rim_pts: wall-top points (world),
        force ON the lid is +z_pan (n_rim), reaction on the pan.  o: moment reference point."""
        self.o = o
        cols, bodies, hands = [], [], []
        for c in contacts:
            for p in c["p"]:
                for e in _edges(c["n"], c["mu"]):
                    cols.append(np.r_[e, np.cross(p - o, e)])
                    bodies.append(c["body"])
                    hands.append(c["hand"])
        self.n_robot = len(cols)
        self.rim_n = rim_pts["n"]
        for p in rim_pts["p"]:
            for e in _edges(self.rim_n, mu_rim):
                cols.append(np.r_[e, np.cross(p - o, e)])
                bodies.append("rim")
                hands.append(None)
        Gc = np.array(cols).T if cols else np.zeros((6, 0))
        nv = Gc.shape[1]
        self.A_eq = np.zeros((12, nv + 1))
        for j, b in enumerate(bodies):
            if b == "pan":
                self.A_eq[:6, j] = Gc[:, j]
            elif b == "lid":
                self.A_eq[6:, j] = Gc[:, j]
            else:                                                   # rim: +f on the lid, -f on the pan
                self.A_eq[6:, j] = Gc[:, j]
                self.A_eq[:6, j] = -Gc[:, j]
        rows, b_ub = [], []
        for hnd in ("left", "right"):
            r = np.zeros(nv + 1)
            idx = [j for j, h in enumerate(hands) if h == hnd]
            if idx:
                r[idx] = 1.0                                        # n . (n + mu t) = 1: sum lambda = sum of normal forces
                rows.append(r)
                fh = f_hand[hnd] if isinstance(f_hand, dict) else f_hand
                b_ub.append(float(fh[1]) if isinstance(fh, (tuple, list)) else float(fh))
        self.A_ub, self.b_ub = (np.array(rows), np.array(b_ub)) if rows else (None, None)
        self.nv = nv

    def load_factor(self, w_pan, w_lid):
        """max k s.t. contact forces balance k * (loads); w_* = 6-vectors [force; moment about o] of the loads on each body."""
        A = self.A_eq.copy()
        A[:6, -1], A[6:, -1] = w_pan, w_lid
        c = np.zeros(self.nv + 1)
        c[-1] = -1.0
        r = linprog(c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=A, b_eq=np.zeros(12),
                    bounds=[(0, None)] * self.nv + [(0, K_CAP)], method="highs")
        return float(r.x[-1]) if r.status == 0 else 0.0


def _wrench(F, x, o):
    return np.r_[F, np.cross(x - o, F)]


# ------------------------------------------------------------------------------------------------------------ admission
GAP_MAX = 1.0e-4              # ASSUMED: a finger counts as touching only within 0.1 mm (V11: fingers 0.8 mm off the wall carry nothing)
PEN_MAX = 0.002               # ASSUMED: a pose sample that puts a finger > 2 mm inside the pan contradicts the gap reading
SQUEEZE_COS = np.cos(np.radians(45.0))   # design: a contact is pressed by the jaw force only if its normal is within 45 deg of
                                          # the finger's closing direction; otherwise it is a passive contact on the (rigid) claw
SCAN_MARGIN = 0.003           # design: collision margin used to find near contacts (only dist <= GAP_MAX is used)


def _scratch(ep, pm):
    """a private copy of the model (finger margins raised) + data, used ONLY as a geometry engine: mj_fwdPosition, never stepped."""
    if getattr(pm, "_scratch", None) is None:
        import copy
        mc = copy.copy(ep.m)
        fg = [mc.geom(f"{s}_arm_finger{f}_col").id for s in ("left", "right") for f in (1, 2)]
        mc.geom_margin[fg] = SCAN_MARGIN
        mc.geom_gap[fg] = 0.0
        tgt = {ep.pan: "pan", ep.lid: "lid"}
        pm._scratch = (mc, mujoco.MjData(mc), {g: s for g, s in zip(fg, [s for s in ("left", "right") for _ in (1, 2)])}, tgt)
    return pm._scratch


def _set_free(m, d, body, R, p):
    a = m.jnt_qposadr[m.body_jntadr[body]]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = p, q


def geometry(ep, pm, T_pan, mu_scale, flush_h=FLUSH_H, lid_rel=None):
    """robot contacts for the pan at world pose T_pan = (R, p), the lid at lid_rel (pan frame; None = seated), robot q = the
    CURRENT encoders.  Contact points/normals/distances from MuJoCo's collider on a scratch copy (geometry only, no forces).
    Per finger: the contacts within flush_h of its deepest one (rigid: only the extreme points touch).
    -> (contacts, rim, info, consistent)."""
    mc, dc, fing, tgt = _scratch(ep, pm)
    dc.qpos[:] = ep.d.qpos
    Rp, pp = T_pan
    _set_free(mc, dc, ep.pan, Rp, pp)
    Rl, pl = (np.eye(3), pm.lid_o) if lid_rel is None else lid_rel
    _set_free(mc, dc, ep.lid, Rp @ Rl, pp + Rp @ pl)
    if ep.hid is not None:
        _set_free(mc, dc, ep.hid, np.eye(3), np.array([0.0, 0.0, -10.0]))      # the puck's load is in the LP, its geometry is not
    mujoco.mj_fwdPosition(mc, dc)
    per = {}
    for j in range(dc.ncon):
        c = dc.contact[j]
        for gf, go, sgn in ((c.geom1, c.geom2, 1.0), (c.geom2, c.geom1, -1.0)):
            if gf in fing and mc.geom_bodyid[go] in tgt:
                n = sgn * c.frame[:3]                                       # frame normal points geom1 -> geom2: force ON the object
                per.setdefault(gf, []).append(dict(d=float(c.dist), p=c.pos.copy(), n=n / np.linalg.norm(n),
                                                   body=tgt[mc.geom_bodyid[go]], geom=mc.geom(go).name))
    contacts, info, consistent = [], {}, True
    jaw = {}
    for side in ("left", "right"):                                       # closing direction of each finger (robot FK)
        cen = [ep.d.xpos[ep.m.body(f"{side}_arm_finger{f}").id] for f in (1, 2)]
        for f, j in zip((1, 2), ep.fjoint[side]):
            ax = ep.d.xaxis[j] / np.linalg.norm(ep.d.xaxis[j])
            jaw[(side, f)] = ax if ax @ (cen[2 - f] - cen[f - 1]) > 0 else -ax
    for gf, side in fing.items():
        cs = [c for c in per.get(gf, []) if not (c["body"] == "lid" and c["d"] < -PEN_MAX)]   # deep in a seated lid: the lid was pushed aside
        name = mc.geom(gf).name.replace("_arm_", "_").replace("_col", "")
        if not cs:
            info[name] = dict(n_pts=0, d_min_mm=None)
            consistent = False
            continue
        dmin = min(c["d"] for c in cs)
        pen_pan = min((c["d"] for c in cs if c["body"] == "pan"), default=0.0)
        if dmin > GAP_MAX or pen_pan < -PEN_MAX:
            consistent = False
        use = [c for c in cs if c["d"] <= dmin + flush_h] if dmin <= GAP_MAX else []
        for c in use:
            fnum = 1 if "finger1" in mc.geom(gf).name else 2
            contacts.append(dict(body=c["body"], hand=side, p=c["p"][None, :], n=c["n"], finger=fnum,
                                 squeezed=bool(c["n"] @ jaw[(side, fnum)] >= SQUEEZE_COS),
                                 mu=pm.mu("left_arm_finger1_col", "lid_disk" if c["body"] == "lid" else "pan_bottom", mu_scale)))
        P = np.array([c["p"] for c in use])
        info[name] = dict(n_pts=len(use), d_min_mm=round(1e3 * dmin, 3), on=sorted({c["geom"] for c in use}),
                          spread_mm=round(1e3 * float(np.ptp(P, axis=0).max()) if len(P) > 1 else 0.0, 2))
    rim = dict(p=np.array([pp + Rp @ x for x in pm.rim]), n=Rp @ np.array([0, 0, 1.0]))
    return contacts, rim, info, consistent


def loads(pm, T_pan, pi, g, o):
    """(w_pan, w_lid) for contents vertex pi = (m, s_u, s_w) and g_eff g (world)."""
    Rp, pp = T_pan
    W = lambda x: pp + Rp @ x
    m, su, sw = (float(v) for v in pi)
    S = m * W(pm.centre + np.array([0, 0, pm.z_puck])) + Rp @ (su * pm.u + sw * pm.w)   # first moment of the puck (world)
    w_pan = _wrench(pm.m_pan * g, W(pm.c_pan), o) + np.r_[m * g, np.cross(S - m * o, g)]
    w_lid = _wrench(pm.m_lid * g, W(pm.c_lid), o)
    return w_pan, w_lid


def delta_grid(lever, pos=B.AIM_POS + PUSH_SLIP_POS, yaw=B.AIM_YAW + PUSH_SLIP_YAW, n_pos=5, n_yaw=3):
    """pose-residual samples about the jaw anchor.  The zed error (shift AIM_POS, yaw AIM_YAW about the pan centre) seen from a
    jaw point `lever` m from the centre is a rotation AIM_YAW about the jaw plus a shift up to AIM_POS + lever * sin(AIM_YAW);
    then the push slip.  Samples off the handle / rim are dropped by the contact-consistency rule in certify_hold."""
    P = pos + lever * np.sin(np.radians(B.AIM_YAW))
    vp, vy = np.linspace(-P, P, n_pos), np.linspace(-yaw, yaw, n_yaw)
    return [(dx, dy, dyaw) for dx in vp for dy in vp for dyaw in vy]


def apply_delta(T, centre_w, delta):
    """true pose = rotate the estimate by dyaw about the anchor point centre_w (world), then shift by (dx, dy)."""
    R, p = T
    dx, dy, dyaw = delta
    a = np.radians(dyaw)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return Rz @ R, Rz @ (p - centre_w) + centre_w + np.array([dx, dy, 0.0])


JLIM_TOL = 1.0e-4             # design: a finger joint within 0.1 mm of (or past) its range end counts as "at the limit"


def squeeze(ep):
    """per-claw jaw force interval (lo, hi) = the sum of the two pads' normal forces.  MODEL (MJCF): position servo clipped at
    forcerange, finger 2 slaved by an equality, so the sum equals |actuator force| -- unless a finger joint sits at its range end,
    where the joint limit takes an unknown part: then [0, |actuator force|] (the forces are affine in it: both ends are checked)."""
    m, d = ep.m, ep.d
    out = {}
    for s in ("left", "right"):
        a = m.actuator(f"{s}_arm_gripper_pos").id
        F = float(min(abs(d.actuator_force[a]), m.actuator_forcerange[a, 1]))
        at_lim = False
        for j in ep.fjoint[s]:
            q, (lo, hi) = float(d.qpos[m.jnt_qposadr[j]]), m.jnt_range[j]
            at_lim |= q <= lo + JLIM_TOL or q >= hi - JLIM_TOL
        out[s] = (0.0, F) if at_lim else (F, F)
    return out


# ------------------------------------------------------------------------------------------------------------ compliant split
# The existence LP is optimistic with two claws (it may pick an internal force the arm controllers never apply: V10, 10-30).
# The Demo C compliant model (democ/grasp_cert.CompliantModel, USER choice 10-30) fixes the split: every contact point is a normal
# spring, tangential = viscous creep (what MuJoCo does), jaw force source per claw, claws held rigidly (FL-MPC tracking).  The
# forces are then determined and affine in (load, jaw force), so its vertex check is exact for that model.
STIFF = GC.Stiffness()        # ASSUMED: Demo C calibration (creep, kd = rigid coupling), NOT refit for the pan scene
GAP_TOL_C = GC.GAP_TOL        # ASSUMED: unmodelled initial gap per contact (Demo C fit, 6 um), not refit
N_MIN = 0.05                  # = Demo C: every contact keeps >= 0.05 N
THETA_TOL, X_TOL, V_TOL = 3.0, 5e-3, 1e-3   # = Demo C: pan rotation/translation in the claws, creep surface speed [m/s]


def compliant(contacts, o, Wv, Fv, stiff=STIFF, gap_tol=GAP_TOL_C):
    """Demo C margins for the pan + lid as ONE rigid body.  Wv (nv,6) loads (moment about o), Fv (nv, n_hands) jaw forces in sorted
    hand order.  -> dict(margin (min over vertices; ok <=> >= 0), worst index, forces fn (nv,nc), ft (nv,nc,2), model)."""
    gcs = [GC.Contact(p=c["p"][0] - o, n=c["n"], mu_lo=c["mu"], mu_hi=c["mu"], hand=c["hand"], active=True, finger=c["finger"])
           if c.get("squeezed", True) else GC.Contact(p=c["p"][0] - o, n=c["n"], mu_lo=c["mu"], mu_hi=c["mu"], hand="support", active=False)
           for c in contacts]
    if not gcs:
        return dict(margin=-np.inf, why="no contacts")
    M = GC.CompliantModel(gcs, stiff, tangential=stiff.tangential)
    if M.singular:
        return dict(margin=-np.inf, why=f"singular (cond {M.cond:.1e})")
    hands_all = sorted({c["hand"] for c in contacts})
    hidx = [hands_all.index(h) for h in M.hands]
    S_ = M.solve(Wv, Fv[:, hidx] if hidx else np.zeros((len(Wv), 0)))
    fn, ft = S_["fn"], S_["ft"]
    G_ = M.gap_sensitivity()
    d_fn = gap_tol * np.abs(G_["dfn"]).sum(1)
    d_ft = gap_tol * np.linalg.norm(G_["dft"], axis=1).sum(1)
    d_ut = gap_tol * np.linalg.norm(G_["du"][:3], axis=0).sum()
    d_ur = gap_tol * np.linalg.norm(G_["du"][3:], axis=0).sum()
    R_c = float(np.max(np.linalg.norm(M.p, axis=1)))
    d_sp = gap_tol * (np.linalg.norm(G_["dv"][:3], axis=0) + R_c * np.linalg.norm(G_["dv"][3:], axis=0)).sum()
    mu = np.array([c.mu_lo for c in gcs])
    fn_w = fn - d_fn
    fscale = max(1e-9, float(np.max(Fv)) if Fv.size else 0.0, float(np.max(np.linalg.norm(Wv[:, :3], axis=1))))
    nrm_m = fn_w.min(1) - N_MIN
    ftn = np.linalg.norm(ft, axis=-1) + d_ft
    fr_m = np.where(fn_w > 1e-9, 1.0 - ftn / np.maximum(mu * fn_w, 1e-12), -np.inf).min(1)
    trans = np.linalg.norm(S_["u"][:, :3], axis=1) + d_ut
    rot = np.degrees(np.linalg.norm(S_["u"][:, 3:], axis=1) + d_ur)
    sp = np.linalg.norm(S_["v"][:, :3], axis=1) + R_c * np.linalg.norm(S_["v"][:, 3:], axis=1) + d_sp
    dsp_m = np.minimum(np.minimum(1 - sp / V_TOL, 1 - rot / THETA_TOL), 1 - trans / X_TOL)
    comp = np.minimum(np.minimum(nrm_m / fscale, fr_m), dsp_m)
    k = int(np.argmin(comp))
    why = ["normal", "friction", "displacement"][int(np.argmin([nrm_m[k] / fscale, fr_m[k], dsp_m[k]]))]
    return dict(margin=float(comp[k]), worst=k, why=why, fn=fn, ft=ft, M=M, min_fn=float(fn_w.min()),
                max_creep_mm_s=float(1e3 * sp.max()), max_rot_deg=float(rot.max()))


def lid_factor(rim, mu_rim, o, w_lid, f_on_lid):
    """lid on the rim (friction only, tabs ignored): max t s.t. rim forces balance t * (lid gravity-inertia load) + the given finger
    wrench on the lid.  Passive supports only, so existence is sound here.  lid holds <=> t >= 1."""
    cols = [np.r_[e, np.cross(p - o, e)] for p in rim["p"] for e in _edges(rim["n"], mu_rim)]
    A = np.c_[np.array(cols).T, w_lid]
    c = np.zeros(A.shape[1])
    c[-1] = -1.0
    r = linprog(c, A_eq=A, b_eq=-f_on_lid, bounds=[(0, None)] * (A.shape[1] - 1) + [(0, K_CAP)], method="highs")
    return float(r.x[-1]) if r.status == 0 else 0.0


def certify_hold(ep, pm, T_hat, contents_V, deltas, mu_scale=MU_SCALE_LO, gs=None, f_hand=None, require_contact=True, anchor=None,
                 lid_rel=None):
    """min load factor k* over deltas x contents vertices x g_eff corners.  ok <=> k* >= 1.
    A delta for which some finger has no contact point is INCONSISTENT with the robot's closed-claw gap readings and is skipped
    (conditioning the aim error on an observation, rule 4); the count is reported."""
    gs = g_eff_corners() if gs is None else gs
    f_hand = squeeze(ep) if f_hand is None else f_hand
    worst, n_lp, skipped, per_delta = (np.inf, None), 0, 0, []
    worst_c, lid_min = (np.inf, None), np.inf
    centre_w = T_hat[1] + T_hat[0] @ pm.centre if anchor is None else np.asarray(anchor, float)
    for dl in deltas:
        T = apply_delta(T_hat, centre_w, dl)
        contacts, rim, info, consistent = geometry(ep, pm, T, mu_scale, lid_rel=lid_rel)
        if require_contact and not consistent:
            skipped += 1
            per_delta.append(dict(delta=dl, skipped=True, info=info))
            continue
        o = T[1] + T[0] @ pm.centre
        mu_rim = pm.mu("pan_wall0", "lid_disk", mu_scale)
        H = Hold(contacts, rim, mu_rim, o, f_hand)
        kd = (np.inf, None)
        Wv, WL, lab = [], [], []
        hands = sorted({c["hand"] for c in contacts})
        f_vert = list(itertools.product(*[sorted(set(f_hand[h])) for h in hands])) if hands else [()]
        for pi in contents_V:
            for g in gs:
                wp, wl = loads(pm, T, pi, g, o)
                k = H.load_factor(wp, wl)
                n_lp += 1
                if k < kd[0]:
                    kd = (k, dict(pi=[float(x) for x in pi], g=[float(x) for x in g]))
                for fv in f_vert:
                    Wv.append(wp + wl)
                    WL.append(wl)
                    lab.append((pi, g, fv))
        Wv, WL = np.array(Wv), np.array(WL)
        Fv = np.array([lb[2] for lb in lab], float).reshape(len(lab), len(hands))
        C = compliant(contacts, o, Wv, Fv) if hands else dict(margin=-np.inf, why="no contacts")
        lid_t = np.inf
        if "fn" in C:
            M = C["M"]
            lid_i = [i for i, c in enumerate(contacts) if c["body"] == "lid"]
            for v in range(len(Wv)):
                fw = np.zeros(6)
                for i in lid_i:
                    f = C["fn"][v, i] * M.n[i] + C["ft"][v, i, 0] * M.t1[i] + C["ft"][v, i, 1] * M.t2[i]
                    fw += np.r_[f, np.cross(contacts[i]["p"][0] - o, f)]
                if lid_i or v % max(1, len(f_vert)) == 0:
                    lid_t = min(lid_t, lid_factor(rim, mu_rim, o, WL[v], fw))
        cm = C["margin"]
        per_delta.append(dict(delta=dl, k=kd[0], worst=kd[1], info=info, consistent=consistent, c_margin=cm, c_why=C.get("why"),
                              lid_t=lid_t, c_min_fn=C.get("min_fn"), c_creep_mm_s=C.get("max_creep_mm_s"),
                              c_worst=None if "worst" not in C else dict(pi=[float(x) for x in lab[C["worst"]][0]],
                                                                         g=[float(x) for x in lab[C["worst"]][1]], F=list(lab[C["worst"]][2]))))
        if kd[0] < worst[0]:
            worst = (kd[0], dict(delta=dl, **kd[1]))
        if cm < worst_c[0]:
            worst_c = (cm, dict(delta=dl, why=C.get("why"), **(per_delta[-1]["c_worst"] or {})))
        lid_min = min(lid_min, lid_t)
    k = float(worst[0]) if np.isfinite(worst[0]) else None
    cmin = float(worst_c[0]) if worst_c[1] is not None else None          # -inf = no contacts / singular in some sample
    ok = bool(k is not None and k >= 1.0 and cmin is not None and cmin >= 0.0 and lid_min >= 1.0)
    return dict(ok=ok, k_star=k, worst=worst[1], c_margin=cmin, c_worst=worst_c[1], lid_t=float(lid_min) if np.isfinite(lid_min) else None,
                n_lp=n_lp, deltas_skipped=skipped, n_deltas=len(deltas), per_delta=per_delta,
                guarantee=f"proof under MODEL (contents, accel, tilt, jaw force) / sampled (pose residual, {len(deltas)})",
                assumptions=assumptions())
