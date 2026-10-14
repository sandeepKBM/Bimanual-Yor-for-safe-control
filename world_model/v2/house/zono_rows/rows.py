"""rows.py -- zonotope HOCBF rows for FLMPC.step(table_rows=...).  SIMULATION ONLY (pure math on a MjModel/MjData).

Generalises experiments/realtime/fast_barriers.table_plane_barrier_rows (and sim_common's original) from the plane n = e_z to the ACTIVE FACET
of each nearby zonotope.  For one keypoint k (world point p, radius r, Jacobian J over the arm's dofs) and one zonotope with active facet
(n, d) -- the facet attaining  sd_lb(p) = max_f(n_f.p - d_f)  (see zono_hrep) -- :

    h      = n.p - d - r - margin                       (>= 0  <=>  the sphere (p, r) is `margin` outside the facet's half-space)
    hdot   = n.(J qd [+ v_extra])                       n is constant (static obstacle)  -> ndot = 0
    psi1   = hdot + alpha1 h
    hddot  = n.(Jd qd) + (n J) M^-1 (tau - bias)        [+ n.a_extra, not modelled]
    psi2   = hddot + alpha1 hdot + alpha2 psi1  >= 0
  <=>  -(nJ M^-1) tau <= -(nJ M^-1) bias + n.(Jd qd) + alpha1 hdot + alpha2 psi1
i.e. exactly the row  (a, b) = (-A_h, const_term)  that table_plane_barrier_rows returns, with n instead of e_z and the facet offset d
instead of z_floor (and `margin` TIGHTENING instead of relaxing: the table function uses z_thresh = z_floor - margin).  The rows are
in the TORQUE convention  a @ tau <= b  for ONE arm's dofs; FLMPC.step turns them into acceleration-space rows with its own M, c
(a @ (M a0 + c) <= b), so pass bias = c = qfrc_bias - qfrc_passive and M_inv = inv(arm mass block) of the same state (ZonoCtl does).

v_extra (K,3) [m/s]: world velocity of each keypoint NOT produced by the arm's own joints (the kinematic mocap base moving).  J qd only
sees the arm joints, so without it a moving base is invisible to hdot.  Base acceleration is not modelled.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

from .zono_hrep import HRep, hrep_of
from . import zono_fast as ZF

REPO = Path(__file__).resolve().parents[4]
for _p in (REPO / "experiments" / "ab_smooth_vs_bangbang", REPO / "experiments" / "realtime", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sim_common as sc  # noqa: E402

try:                                              # numba Jacobian kernel (bit-identical to mj_jac / mj_jacDot); optional
    import fast_barriers as _fb                   # noqa: E402
except Exception:                                 # pragma: no cover
    _fb = None

ALPHA1, ALPHA2 = sc.ALPHA1, sc.ALPHA2
FAST = os.environ.get("ZONO_ROWS_REF", "0") != "1"       # numba candidate search / row selection (identical rows); ZONO_ROWS_REF=1 = the original dense numpy path


# --------------------------------------------------------------------------------------------------------------- keypoints
@dataclass
class KeypointSet:
    """(body_id, local offset, radius) triples packed for the Jacobian kernel.  `names` is only for reporting."""
    bodies: np.ndarray                 # (K,) int64
    offs: np.ndarray                   # (K,3,1)
    radii: np.ndarray                  # (K,)
    names: list = field(default_factory=list)

    @staticmethod
    def from_list(kps, names=None):
        bodies = np.array([int(b) for b, _o, _r in kps], dtype=np.int64)
        offs = np.empty((len(kps), 3, 1))
        for k, (_b, o, _r) in enumerate(kps):
            offs[k, :, 0] = o
        radii = np.array([r for _b, _o, r in kps], dtype=np.float64)
        return KeypointSet(bodies, offs, radii, list(names) if names else [""] * len(kps))

    def __len__(self):
        return int(self.bodies.shape[0])

    def __add__(self, other):
        return KeypointSet(np.concatenate([self.bodies, other.bodies]), np.concatenate([self.offs, other.offs]),
                           np.concatenate([self.radii, other.radii]), self.names + other.names)


def as_kpset(kps) -> KeypointSet:
    return kps if isinstance(kps, KeypointSet) else KeypointSet.from_list(kps)


def arm_keypoints(model, side: str) -> KeypointSet:
    """the arm-link collision-proxy spheres FL-MPC's pair barriers use (sim_common.barrier_keypoints), side in {'left','right'}."""
    kps = sc.barrier_keypoints(model, f"{side}_arm_")
    names = [model.body(b).name for b, _o, _r in kps]
    return KeypointSet.from_list(kps, names)


def tray_weld(model, side: str):
    """(gripper_base body id, rel (3,), R (3,3)) of the weld `hold_<side>`: the tray origin sits at `rel` in the gripper_base frame and
    R maps tray-frame vectors to gripper_base-frame vectors (b1q * relquat == b2q)."""
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"hold_{side}")
    if eq < 0:
        raise KeyError(f"no weld 'hold_{side}' in the model (compose with with_tray=True)")
    gb = model.body(f"{side}_arm_gripper_base").id
    rel = model.eq_data[eq, 3:6].copy()
    q = model.eq_data[eq, 6:10].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return gb, rel, R.reshape(3, 3)


def tray_local_points(hx=0.15, hy=0.10, t_half=0.003, spacing=0.05, ball_r=0.03, ball_xs=(-0.10, 0.0, 0.10),
                      include_rim=True, include_ball=True):
    """[(point in the TRAY frame, radius, label)]: top and bottom rim every <= `spacing` m, the top centre, and the ball spheres
    (radius ball_r at (x, 0, ball_r), x in ball_xs).  The tray body origin is the top surface, the box spans z in [-2 t_half, 0]."""
    def rim(z):
        pts = []
        for (ax, ay, bx, by) in ((-hx, -hy, hx, -hy), (hx, -hy, hx, hy), (hx, hy, -hx, hy), (-hx, hy, -hx, -hy)):
            n = max(int(math.ceil(math.hypot(bx - ax, by - ay) / spacing)), 1)
            for i in range(n):
                s = i / n
                pts.append((ax + s * (bx - ax), ay + s * (by - ay), z))
        return pts

    pts = []
    if include_rim:
        pts += [(p, 0.0, "tray_top_rim") for p in rim(0.0)] + [(p, 0.0, "tray_bot_rim") for p in rim(-2 * t_half)]
        pts.append(((0.0, 0.0, 0.0), 0.0, "tray_centre"))
    if include_ball:
        pts += [((x, 0.0, ball_r), ball_r, "ball") for x in ball_xs]
    return pts


def tray_keypoints(model, side: str, hx=0.15, hy=0.10, t_half=0.003, spacing=0.05, ball_r=0.03,
                   ball_xs=(-0.10, 0.0, 0.10), include_rim=True, include_ball=True) -> KeypointSet:
    """tray rim / ball points as FIXED offsets in `side`'s gripper_base frame, from the weld relpose (eq 'hold_<side>').
    The tray is a free body held by two welds, so MuJoCo's Jacobian of a point on it w.r.t. the arm joints is zero; a point rigidly attached
    to the gripper_base body (through the weld) has the correct arm Jacobian.  Points: see tray_local_points.
    include_rim / include_ball select the subsets (defaults = the original behaviour: both)."""
    gb, rel, R = tray_weld(model, side)
    pts = tray_local_points(hx, hy, t_half, spacing, ball_r, ball_xs, include_rim, include_ball)
    kps = [(gb, rel + R @ np.asarray(p), r) for p, r, _n in pts]
    return KeypointSet.from_list(kps, [f"{side}:{n}" for _p, _r, n in pts])


# --------------------------------------------------------------------------------------------------------------- tracked keypoints
@dataclass
class TrackedSet:
    """TRACKED object keypoints = (free body, local offset in that body's frame, radius).  The world point comes from the body's CURRENT pose in
    MjData -- in simulation this is a stand-in for a perception tracker (pose + twist of the object); nothing here is known to the real robot
    beyond what a tracker would provide.  See tracked_states for how J / Jd / velocity are formed."""
    bodies: np.ndarray                 # (K,) int64 -- the tracked (free) bodies
    offs: np.ndarray                   # (K,3) offsets in the tracked body's frame
    radii: np.ndarray                  # (K,)
    names: list = field(default_factory=list)

    def __len__(self):
        return int(self.bodies.shape[0])


def tracked_keypoints(model, specs, hx=0.15, hy=0.10, t_half=0.003, spacing=0.05) -> TrackedSet:
    """specs: iterable of body names (or (name, offset(3,), radius) tuples).  'tray' expands to the rim + centre points (radius 0, in the tray
    frame; same points as tray_keypoints without the ball); any other body name -> one point at its origin with the radius of its geoms
    (sphere: geom_size[0]; otherwise the largest geom_rbound).  The ball radius therefore comes from the model's geom size."""
    bodies, offs, radii, names = [], [], [], []
    for sp in specs:
        if isinstance(sp, str):
            b = model.body(sp).id
            if sp == "tray":
                for p, r, n in tray_local_points(hx, hy, t_half, spacing, include_rim=True, include_ball=False):
                    bodies.append(b); offs.append(p); radii.append(r); names.append(f"tracked:tray:{n}")
            else:
                gs = np.flatnonzero(model.geom_bodyid == b)
                r = 0.0
                for g in gs:
                    r = max(r, float(model.geom_size[g, 0]) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else float(model.geom_rbound[g]))
                bodies.append(b); offs.append((0.0, 0.0, 0.0)); radii.append(r); names.append(f"tracked:{sp}")
        else:
            nm, off, r = sp
            bodies.append(model.body(nm).id); offs.append(tuple(off)); radii.append(float(r)); names.append(f"tracked:{nm}")
    return TrackedSet(np.array(bodies, np.int64), np.array(offs, float).reshape(-1, 3), np.array(radii, float), names)


# --------------------------------------------------------------------------------------------------------------- zonotope set
class ZonoSet:
    """HReps of a list of zonotopes + stacked AABBs for the activation prefilter."""

    def __init__(self, zonos, names=None):
        self.h = [hrep_of(z) for z in zonos]
        self.names = list(names) if names is not None else [f"Z{i}" for i in range(len(self.h))]
        self.lo = np.array([h.lo for h in self.h]).reshape(-1, 3)
        self.hi = np.array([h.hi for h in self.h]).reshape(-1, 3)

    def __len__(self):
        return len(self.h)

    def packed(self) -> "ZF.Packed":
        """flat facet arrays + AABBs + group ids (built once, on first use; the H-reps themselves are built in __init__ = once per perception update)."""
        pk = getattr(self, "_packed", None)
        if pk is None:
            ZF.warmup()
            pk = self._packed = ZF.Packed(self.h)
        return pk


def as_zonoset(z) -> ZonoSet:
    return z if isinstance(z, ZonoSet) else ZonoSet(z)


# --------------------------------------------------------------------------------------------------------------- kinematics
_TAB = {}


def _fast_kp(model, data, bodies, offs, dofadr, key):
    """fast_barriers.keypoint_states with the chain tables cached under `key` (the body list + dof columns are static; keypoint_states re-hashes
    them on every call).  Bit-identical result.  Returns the same (p, J, Jd, v, qd, ok) tuple."""
    ent = _TAB.get(key)
    if ent is None or ent[0] is not model:
        ent = _TAB[key] = (model, _fb._chain_tables(model, bodies, dofadr))
    roots, chain_dof, chain_col, chain_len, ok = ent[1]
    p = ZF.world_points(data, bodies, np.ascontiguousarray(offs, dtype=np.float64))
    n, K = len(dofadr), bodies.shape[0]
    JT = np.zeros((K, n, 3)); JdT = np.zeros((K, n, 3))
    _fb._jac_jacdot_batch(p, bodies, roots, chain_dof, chain_col, chain_len, data.cdof, data.cdof_dot, data.cvel, data.subtree_com, n, JT, JdT)
    J = JT.transpose(0, 2, 1)
    qd = data.qvel[dofadr]
    return p, J, JdT.transpose(0, 2, 1), J @ qd, qd, ok


def kp_states(model, data, ks: KeypointSet, dofadr):
    """(p (K,3), J (K,3,n), Jd (K,3,n), v (K,3), qd (n,)) at the current MjData state (mj_forward must have been called)."""
    dofadr = np.asarray(dofadr)
    if _fb is not None:
        if FAST and hasattr(_fb, "_jac_jacdot_batch"):
            key = (id(model), ks.bodies.tobytes(), dofadr.tobytes())
            p, J, Jd, v, qd, ok = _fast_kp(model, data, ks.bodies, ks.offs, dofadr, key)
        else:
            p, J, Jd, v, qd, ok = _fb.keypoint_states(model, data, ks.bodies, ks.offs, dofadr)
        if ok:
            return p, J, Jd, v, qd
    K, n = len(ks), len(dofadr)
    p = np.empty((K, 3)); J = np.empty((K, 3, n)); Jd = np.empty((K, 3, n))
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    for k in range(K):
        b = int(ks.bodies[k])
        p[k] = data.xpos[b] + data.xmat[b].reshape(3, 3) @ ks.offs[k, :, 0]
        mujoco.mj_jac(model, data, jp, jr, p[k], b); J[k] = jp[:, dofadr]
        mujoco.mj_jacDot(model, data, jp, jr, p[k], b); Jd[k] = jp[:, dofadr]
    qd = data.qvel[dofadr]
    return p, J, Jd, J @ qd, qd


def tracked_measured(model, data, ts: TrackedSet):
    """(p (K,3), v (K,3)): world point of every tracked keypoint from its body's pose, and its MEASURED world velocity (full-qvel Jacobian of the tracked
    body, times qvel -- evaluated from MuJoCo's body velocity cvel, which is exactly that product).  Arm-independent: compute once per step and pass
    to tracked_states(meas=...) for both arms.  mj_forward must have been called."""
    if FAST:
        return ZF.tracked_pv(model, data, ts.bodies, np.ascontiguousarray(ts.offs, dtype=np.float64))
    K = len(ts)
    b = ts.bodies
    xm = data.xmat[b].reshape(K, 3, 3)
    p = data.xpos[b] + np.einsum("kij,kj->ki", xm, ts.offs)
    groups = getattr(ts, "_groups", None)
    if groups is None:
        groups = ts._groups = [(int(bi), np.flatnonzero(b == bi)) for bi in np.unique(b)]
        ts._jp = np.zeros((3, model.nv)); ts._jr = np.zeros((3, model.nv))
    jp, jr = ts._jp, ts._jr
    v = np.empty((K, 3))
    qv = data.qvel
    for bi, ix in groups:
        x0 = data.xpos[bi]
        mujoco.mj_jac(model, data, jp, jr, x0, bi)
        vo, om = jp @ qv, jr @ qv
        r = p[ix] - x0
        v[ix, 0] = vo[0] + om[1] * r[:, 2] - om[2] * r[:, 1]
        v[ix, 1] = vo[1] + om[2] * r[:, 0] - om[0] * r[:, 2]
        v[ix, 2] = vo[2] + om[0] * r[:, 1] - om[1] * r[:, 0]
    return p, v


def tracked_states(model, data, ts: TrackedSet, dofadr, gb, meas=None):
    """(p (K,3), J (K,3,n), Jd (K,3,n), v (K,3), qd (n,)) of TRACKED keypoints for ONE arm (dofs `dofadr`, gripper_base body `gb`).
      p   world point from the tracked body's pose in MjData  (xpos + xmat @ offset)            -- tracker stand-in
      v   MEASURED world velocity of that point: full-qvel Jacobian of the tracked (free) body, times qvel.  Already contains whatever the
          kinematic base / the tray weld is doing, so NO v_extra must be added for these points.
      J, Jd  the point is treated as RIGIDLY ATTACHED to `gb` at its current location: mj_jac / mj_jacDot of gripper_base at p, columns
          `dofadr`.  The object's own (rolling / sliding) acceleration is an unmodelled disturbance.
    meas = (p, v) from tracked_measured (shared by both arms), computed here if None.  mj_forward must have been called."""
    K = len(ts)
    p, v = tracked_measured(model, data, ts) if meas is None else meas
    Rg = data.xmat[gb].reshape(3, 3)
    off_gb = (p - data.xpos[gb]) @ Rg                          # = (R_gb^T (p - x_gb)) per point
    if FAST and _fb is not None and hasattr(_fb, "_jac_jacdot_batch"):
        dofadr = np.asarray(dofadr)
        _p, J, Jd, _v, qd, ok = _fast_kp(model, data, np.full(K, gb, dtype=np.int64), off_gb[:, :, None].copy(), dofadr, (id(model), "trk", int(gb), K, dofadr.tobytes()))
        if ok:
            return p, J, Jd, v, qd
    ksg = KeypointSet(np.full(K, gb, dtype=np.int64), off_gb[:, :, None].copy(), ts.radii.copy(), list(ts.names))
    _p, J, Jd, _v, qd = kp_states(model, data, ksg, dofadr)
    return p, J, Jd, v, qd


def tracked_arm_velocity(model, data, ts: TrackedSet, dofadr, gb, p=None):
    """(K,3) = J qd of the tracked points rigidly attached to `gb` (the velocity the ARM's joints give them); skips the measured-velocity Jacobians
    that tracked_states computes.  p (K,3) = tracked_measured's points if already known."""
    K = len(ts)
    if p is None:
        xm = data.xmat[ts.bodies].reshape(K, 3, 3)
        p = data.xpos[ts.bodies] + np.einsum("kij,kj->ki", xm, ts.offs)
    off_gb = (p - data.xpos[gb]) @ data.xmat[gb].reshape(3, 3)
    if FAST and _fb is not None and hasattr(_fb, "_jac_jacdot_batch"):
        dofadr = np.asarray(dofadr)
        bodies = np.full(K, gb, dtype=np.int64)
        v = _fast_kp(model, data, bodies, off_gb[:, :, None].copy(), dofadr, (id(model), "trk", int(gb), K, dofadr.tobytes()))
        if v[5]:
            return v[3]
    ksg = KeypointSet(np.full(K, gb, dtype=np.int64), off_gb[:, :, None].copy(), ts.radii.copy(), list(ts.names))
    return kp_states(model, data, ksg, dofadr)[3]


# --------------------------------------------------------------------------------------------------------------- the rows
def _hocbf_assemble(nrm, h, Jk, w, vk, M_inv, bias, alpha1, alpha2):
    """shared HOCBF row algebra.  nrm (R,3) unit normals (constant), h (R,), Jk (R,3,n) point Jacobians, w (R,3) = Jd qd per row, vk (R,3) the
    point's world velocity.  Returns (-A_h, const) with a @ tau <= const."""
    hdot = np.einsum("ri,ri->r", nrm, vk)
    hddot_rest = np.einsum("ri,ri->r", nrm, w)
    psi1 = hdot + alpha1 * h
    nJ = np.einsum("ri,rij->rj", nrm, Jk)                                    # n @ J
    A_h = nJ @ np.asarray(M_inv)                                             # (n J) M^-1
    const = -(A_h @ np.asarray(bias)) + hddot_rest + alpha1 * hdot + alpha2 * psi1
    return -A_h, const, hdot


def zono_rows_from_states(p, J, Jd, v, qd, radii, names, zonos, M_inv, bias, margin=0.0, alpha1=ALPHA1, alpha2=ALPHA2, d_act=0.25,
                          kp_radius=True, v_extra=None, v_extra_n=None, max_rows=None, per_facet=2, info=None, fast=None):
    """zono_barrier_rows' body on precomputed keypoint states (p (K,3), J/Jd (K,3,n), v (K,3), qd (n,), radii (K,), names).
    v_extra_n: if given, v_extra (array or callable p->(K',3)) applies to the FIRST v_extra_n keypoints only (the arm's own); the rest
    (tracked points, whose measured velocity already includes the base motion) get zero.
    fast (default FAST): numba candidate search (zono_fast; grid-culled) + numba row selection -- the SAME rows as the dense numpy path
    (zono_rows_from_states_ref) to ~1e-15."""
    if fast is None:
        fast = FAST
    if not fast:
        return zono_rows_from_states_ref(p, J, Jd, v, qd, radii, names, zonos, M_inv, bias, margin, alpha1, alpha2, d_act, kp_radius, v_extra, v_extra_n,
                                         max_rows, per_facet, info)
    zs = as_zonoset(zonos)
    n = J.shape[2]
    r = radii if kp_radius else np.zeros(len(radii))
    pk = zs.packed()
    kk, zii, fidx, nrm, h = ZF.facet_pairs(p, r, pk, d_act, margin)
    if not len(h):
        if info is not None:
            info.update(n_pairs=0, h_min=float("inf"), dropped=0)
        return np.zeros((0, n)), np.zeros(0)
    if callable(v_extra):
        v_extra = v_extra(p if v_extra_n is None else p[:v_extra_n])
    if v_extra is not None and v_extra_n is not None and len(v_extra) < len(p):
        v_extra = np.vstack([np.asarray(v_extra), np.zeros((len(p) - len(v_extra), 3))])
    K = len(p)
    vx = np.zeros((K, 3)) if v_extra is None else np.asarray(v_extra, float)
    keep, A, const, hdot = ZF.assemble_rows(kk, nrm, h, pk.gid3[fidx], J, Jd, v, vx, qd, M_inv, bias, alpha1, alpha2, max_rows, per_facet)
    dropped = len(h) - len(keep)
    if info is not None:
        kk, nrm, h, zii = kk[keep], nrm[keep], h[keep], zii[keep]
        j = int(np.argmin(h))
        info.update(n_pairs=int(len(h)), h_min=float(h.min()), dropped=int(dropped), ids=list(map(tuple, np.stack([kk, zii], 1).tolist())), argmin_kp=int(kk[j]), argmin_zono=int(zii[j]),
                    argmin_kp_name=names[int(kk[j])], argmin_zono_name=zs.names[int(zii[j])], argmin_hdot=float(hdot[j]), argmin_n=nrm[j].tolist(),
                    argmin_p=p[kk[j]].tolist())
    return A, const


def zono_rows_from_states_ref(p, J, Jd, v, qd, radii, names, zonos, M_inv, bias, margin=0.0, alpha1=ALPHA1, alpha2=ALPHA2, d_act=0.25,
                              kp_radius=True, v_extra=None, v_extra_n=None, max_rows=None, per_facet=2, info=None):
    """zono_barrier_rows' body on precomputed keypoint states (p (K,3), J/Jd (K,3,n), v (K,3), qd (n,), radii (K,), names).
    v_extra_n: if given, v_extra (array or callable p->(K',3)) applies to the FIRST v_extra_n keypoints only (the arm's own); the rest
    (tracked points, whose measured velocity already includes the base motion) get zero."""
    zs = as_zonoset(zonos)
    n = J.shape[2]
    r = radii if kp_radius else np.zeros(len(radii))
    # --- activation prefilter: distance to the AABB (lower bound of the distance to the zonotope)
    gap = np.maximum(np.maximum(zs.lo[None] - p[:, None], p[:, None] - zs.hi[None]), 0.0)         # (K,Z,3)
    near = np.linalg.norm(gap, axis=2) - r[:, None] < d_act
    kk_l, nn_l, h_l, zi_l = [], [], [], []
    for zi in np.flatnonzero(near.any(0)):
        H = zs.h[zi]
        ks_idx = np.flatnonzero(near[:, zi])
        S = p[ks_idx] @ H.N.T - H.d                                          # (k, F)
        a = S.argmax(1)
        sd = S[np.arange(len(ks_idx)), a]
        act = sd - r[ks_idx] < d_act
        if act.any():
            kk_l.append(ks_idx[act]); nn_l.append(H.N[a[act]]); h_l.append(sd[act] - r[ks_idx[act]] - margin)
            zi_l.append(np.full(int(act.sum()), zi))
    if not kk_l:
        if info is not None:
            info.update(n_pairs=0, h_min=float("inf"), dropped=0)
        return np.zeros((0, n)), np.zeros(0)
    if callable(v_extra):
        v_extra = v_extra(p if v_extra_n is None else p[:v_extra_n])
    if v_extra is not None and v_extra_n is not None and len(v_extra) < len(p):
        v_extra = np.vstack([np.asarray(v_extra), np.zeros((len(p) - len(v_extra), 3))])
    kk = np.concatenate(kk_l); nrm = np.concatenate(nn_l); h = np.concatenate(h_l); zii = np.concatenate(zi_l)
    dropped = 0
    # row ranking for the max_rows cap: urgency = psi2 at ZERO acceleration (FL: tau = bias), i.e. how violated the row is if the arm does nothing:
    #   psi2_free = n.(Jd qd) + alpha1 hdot + alpha2 (hdot + alpha1 h).   (NOT h alone: a far, fast-approaching point is more urgent than a near, still one.)
    # Rows of the same zonotope facet from several keypoints are near-parallel -> keep only the `per_facet` most urgent per (zonotope, facet).
    if max_rows is not None and len(h) > max_rows:
        vk0 = v[kk] if v_extra is None else v[kk] + np.asarray(v_extra)[kk]
        hd0 = np.einsum("ri,ri->r", nrm, vk0)
        free = np.einsum("ri,ri->r", nrm, np.einsum("kij,j->ki", Jd, qd)[kk]) + alpha1 * hd0 + alpha2 * (hd0 + alpha1 * h)
        order = np.argsort(free, kind="stable")
        seen, keep = {}, []
        for i in order:
            key = (int(zii[i]), tuple(np.round(nrm[i], 3)))
            if seen.get(key, 0) >= per_facet:
                continue
            seen[key] = seen.get(key, 0) + 1
            keep.append(i)
            if len(keep) >= max_rows:
                break
        keep = np.array(keep)
        dropped = len(h) - len(keep)
        kk, nrm, h, zii = kk[keep], nrm[keep], h[keep], zii[keep]
    vk = v[kk] if v_extra is None else v[kk] + np.asarray(v_extra)[kk]
    w = np.einsum("kij,j->ki", Jd, qd)[kk]                                   # Jd_k @ qd
    A, const, hdot = _hocbf_assemble(nrm, h, J[kk], w, vk, M_inv, bias, alpha1, alpha2)
    if info is not None:
        j = int(np.argmin(h))
        info.update(n_pairs=int(len(h)), h_min=float(h.min()), dropped=int(dropped), ids=[(int(a_), int(b_)) for a_, b_ in zip(kk, zii)], argmin_kp=int(kk[j]), argmin_zono=int(zii[j]),
                    argmin_kp_name=names[int(kk[j])], argmin_zono_name=zs.names[int(zii[j])], argmin_hdot=float(hdot[j]), argmin_n=nrm[j].tolist(),
                    argmin_p=p[kk[j]].tolist())
    return A, const


def zono_barrier_rows(model, data, kps, dofadr, M_inv, bias, zonos, margin=0.0, alpha1=ALPHA1, alpha2=ALPHA2, d_act=0.25,
                      kp_radius=True, v_extra=None, max_rows=None, per_facet=2, info=None, tracked=None, tracked_gb=None, tracked_meas=None):
    """HOCBF rows (a_rows (R, n), b_rows (R,)), a @ tau <= b, ONE per (keypoint, zonotope) pair with  sd_lb - r < d_act, active facet.

    kps      KeypointSet or list of (body_id, offset, radius) of ONE arm;  dofadr / M_inv / bias must be the SAME arm's.
    zonos    ZonoSet, or a list of HRep / (c, G) / objects with .c .G (converted per call -> pass a ZonoSet in loops).
    margin   [m] extra clearance, POSITIVE = tighter (h = sd_lb - r - margin).  NOTE the opposite sign convention to
             table_plane_barrier_rows, whose positive margin relaxes z_floor.
    d_act    [m] activation distance of the sphere surface to the zonotope (conservative sd_lb).
    kp_radius  include each keypoint's sphere radius (the table function's default is False).
    v_extra  (K,3) world velocity of the ARM keypoints not due to the arm joints (kinematic base), or a callable p(K,3)->(K,3), or None.
             Never applied to `tracked` points (their measured velocity already includes the base).
    max_rows keep only the max_rows MOST URGENT rows (lowest psi2 at zero acceleration; at most `per_facet` per (zonotope, facet)); None = all.
             The FL-MPC QP is a dual-coordinate-ascent solver: many near-parallel active rows leave residual violation, so keep this small (4-8).
    tracked  optional TrackedSet (object keypoints from the measured body pose; see tracked_states), needs `tracked_gb` = this arm's
             gripper_base body id.  They go through exactly the same facet / d_act / max_rows / margin logic as the arm keypoints.
    info     optional dict, filled with n_pairs, h_min, n_zonos_near, dropped.
    Returns arrays with R possibly 0 (shape (0, n), (0,))."""
    ks = as_kpset(kps)
    p, J, Jd, v, qd = kp_states(model, data, ks, dofadr)
    radii, names, n_arm = ks.radii, ks.names, len(ks)
    if tracked is not None and len(tracked):
        pt, Jt, Jdt, vt, _ = tracked_states(model, data, tracked, dofadr, tracked_gb, meas=tracked_meas)
        p, J, Jd, v = np.concatenate([p, pt]), np.concatenate([J, Jt]), np.concatenate([Jd, Jdt]), np.concatenate([v, vt])
        radii, names = np.concatenate([radii, tracked.radii]), list(names) + list(tracked.names)
    return zono_rows_from_states(p, J, Jd, v, qd, radii, names, zonos, M_inv, bias, margin=margin, alpha1=alpha1, alpha2=alpha2, d_act=d_act,
                                 kp_radius=kp_radius, v_extra=v_extra, v_extra_n=(n_arm if len(p) > n_arm else None), max_rows=max_rows,
                                 per_facet=per_facet, info=info)


# --------------------------------------------------------------------------------------------------------------- keep-the-tray-level rows
def common_mount_body(model) -> int:
    """lowest common ancestor body of the two gripper_base bodies (= the body BOTH arms are mounted on; 'Lift_Top' in the YOR model)."""
    def chain(b):
        out = []
        while b > 0:
            out.append(b); b = int(model.body_parentid[b])
        return out
    cl, cr = chain(model.body("left_arm_gripper_base").id), set(chain(model.body("right_arm_gripper_base").id))
    return next(b for b in cl if b in cr)


def tray_tilt_row(model, data, side, dofadr, M_inv, bias, theta_max, alpha1=ALPHA1, alpha2=ALPHA2, info=None):
    """HARD tilt row for the tray as implied by `side`'s gripper through its weld:  u = R_gb . (tray +z in the gripper_base frame) (world up-axis
    of the tray), h = u_z - cos(theta_max) >= 0.  omega = Jr qd, alpha = Jr qdd + Jrd qd (rotational Jacobians of gripper_base);
      hdot = (omega x u)_z,   hddot = (alpha x u + omega x (omega x u))_z,   qdd = M^-1 (tau - bias)
    and the HOCBF condition hddot + alpha1 hdot + alpha2 (hdot + alpha1 h) >= 0, written (a, b) with a @ tau <= b.  (alpha x u)_z = (u x e_z).alpha.
    Returns (a (1,n), b (1,)).  Base rotation about z cannot change u_z, so no base term is needed; base accel is not modelled."""
    gb, _rel, Rt = tray_weld(model, side)
    u = data.xmat[gb].reshape(3, 3) @ Rt[:, 2]
    dofadr = np.asarray(dofadr)
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    pt = data.xpos[gb]
    mujoco.mj_jac(model, data, jp, jr, pt, gb); Jr = jr[:, dofadr].copy()
    mujoco.mj_jacDot(model, data, jp, jr, pt, gb); Jrd = jr[:, dofadr].copy()
    qd = data.qvel[dofadr]
    om = Jr @ qd
    udot = np.cross(om, u)
    h = float(u[2] - math.cos(theta_max))
    hdot = float(udot[2])
    rest = float(np.cross(om, udot)[2] + np.cross(Jrd @ qd, u)[2])
    wv = np.array([u[1], -u[0], 0.0])
    A_h = (wv @ Jr) @ np.asarray(M_inv)
    psi1 = hdot + alpha1 * h
    const = -float(A_h @ np.asarray(bias)) + rest + alpha1 * hdot + alpha2 * psi1
    if info is not None:
        info.update(tilt_h=h, tilt_hdot=hdot, tilt_deg=float(math.degrees(math.acos(np.clip(u[2], -1, 1)))))
    return -A_h[None, :], np.array([const])


def tray_origin_height(model, data, side, mount):
    """(p (3,), z_rel): the tray-origin point implied by `side`'s gripper through its weld, and its height above the `mount` body origin."""
    gb, rel, _R = tray_weld(model, side)
    p = data.xpos[gb] + data.xmat[gb].reshape(3, 3) @ rel
    return p, float(p[2] - data.xpos[mount][2])


def grip_height_rows(model, data, side, dofadr, M_inv, bias, mount, z_ref, delta=0.015, alpha1=ALPHA1, alpha2=ALPHA2, info=None):
    """HARD band rows z_ref - delta <= z_rel <= z_ref + delta (two one-sided HOCBF rows, normals -/+ e_z) for the tray-origin point implied
    by `side`'s gripper, z_rel = its height above the `mount` body.  The arm-only Jacobian is used, so a base / lift translation (which moves
    point and mount together) leaves z_rel untouched; the mount's own ACCELERATION (which does act on the arm dynamics) is not modelled.
    Returns (a (2,n), b (2,))."""
    gb, rel, _R = tray_weld(model, side)
    dofadr = np.asarray(dofadr)
    p, z = tray_origin_height(model, data, side, mount)
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, p, gb); J = jp[:, dofadr].copy()
    mujoco.mj_jacDot(model, data, jp, jr, p, gb); Jd = jp[:, dofadr].copy()
    qd = data.qvel[dofadr]
    ez = np.array([0.0, 0.0, 1.0])
    nrm = np.stack([ez, -ez])                                        # row 0: z >= z_ref - delta ; row 1: z <= z_ref + delta
    h = np.array([z - (z_ref - delta), (z_ref + delta) - z])
    v = np.tile(J @ qd, (2, 1))
    w = np.tile(Jd @ qd, (2, 1))
    A, const, hdot = _hocbf_assemble(nrm, h, np.stack([J, J]), w, v, M_inv, bias, alpha1, alpha2)
    if info is not None:
        info.update(band_z=z, band_h=h.tolist(), band_hdot=hdot.tolist())
    return A, const


def merge_rows(*row_sets):
    """stack several (a_rows, b_rows) pairs (e.g. table-plane rows + zonotope rows) -> one pair."""
    rs = [(np.asarray(a), np.asarray(b)) for a, b in row_sets if a is not None and len(b)]
    if not rs:
        return None
    return np.vstack([a for a, _ in rs]), np.concatenate([b for _, b in rs])
