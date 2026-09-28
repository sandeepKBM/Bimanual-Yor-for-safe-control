"""Fast, BIT-IDENTICAL drop-in replacements for the per-step barrier helpers in
experiments/ab_smooth_vs_bangbang/sim_common.py (+ two demo logging helpers).

NEW file -- nothing existing is edited. SIMULATION ONLY (pure math on a MjData).

Why the originals are slow: sim_common.all_pair_barriers does ~2 mj_jac/mj_jacDot
pybind calls + ~6 small numpy ops per keypoint (71 keypoints) and then a pure-
Python double loop over every (ego, int) keypoint pair (45 x 26 = 1170 pairs in
the cup-collect demo) building one dict per pair; table_plane_barrier_rows and
min_pair_distance repeat the same per-keypoint / per-pair Python work.

How the fast path stays bit-identical (verified by gate_equivalence.py, 0 ulp):
  * mj_jac / mj_jacDot are re-implemented in a tiny numba kernel that reproduces
    MuJoCo's own arithmetic operation-for-operation (engine_support.c):
        jacp[:,i]    = cdof[i,3:] + cross(cdof[i,:3], p - subtree_com[root])
        jacdot[:,i]  = cdof_dot[i,3:] + cross(cdof_dot[i,:3], off) + cross(cdof[i,:3], pvel)
        pvel         = cvel[b,3:] - cross(off, cvel[b,:3])
    for the hinge-only arm chains (asserted at build time; free/ball joints are
    rejected -> falls back to the original function). Only the requested dof
    columns are produced.
  * Every BLAS-backed op of the original (xmat@offset, J@qvel, np.linalg.norm,
    n@v, n@J@M_inv, ...) is done with a BATCHED np.matmul, which numpy dispatches
    item-by-item to the very same cblas_ddot/cblas_dgemv call the unbatched
    original makes (same strides/lengths) -> identical rounding, including
    OpenBLAS's FMA summation order. Elementwise ops are IEEE-exact either way.
  * Ordering (ego-major pairs, first-occurrence argmin, stable top-K sort) is
    preserved exactly.

Public API (signatures identical to the originals):
  all_pair_barriers(model, data, ego_dofadr, int_dofadr, ego_prefix=..., int_prefix=..., extra_ego_keypoints=None)
      -> PairList (read-only Sequence; materializes the SAME list of dicts lazily
         on first element access; FastFLMPC reads its .arrays without building dicts)
  table_plane_barrier_rows(model, data, kps, dofadr, M_inv, bias, z_floor, margin=0.0, alpha1=..., alpha2=...)
  min_pair_distance(model, data, ego_prefix=..., int_prefix=...)
  right_arm_to_cup_clearance(model, data, cup_kps, right_kps, left_ee_body)   (demo logging helper)
  FastFLMPC  (subclass of mpc_fl.FLMPC; step() identical except it gathers the
              top-K pairs from PairList arrays instead of sorting 1170 dicts)
  install(demo_module) / uninstall()  -- monkeypatch for experiments.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

import _demo_env  # noqa: F401  (sys.path shim only)
import sim_common as sc
import mpc_fl

_HINGE = 3
_SLIDE = 2


# ---------------------------------------------------------------------------
# numba kernel: exact MuJoCo mj_jac / mj_jacDot for many points, selected cols
# ---------------------------------------------------------------------------
@njit(cache=True)
def _jac_jacdot_batch(p, bodies, roots, chain_dof, chain_col, chain_len,
                      cdof, cdof_dot, cvel, subtree_com, ncol, JT, JdT):
    """p: (K,3) world points; bodies/roots: (K,) body id and its root body id;
    chain_dof/chain_col: (K, maxc) dof index along the (dof_parentid) chain and
    the output column it maps to (-1 = not requested); chain_len: (K,).
    Writes JT, JdT: (K, ncol, 3) (pre-zeroed) -- i.e. the TRANSPOSED storage,
    so the caller's J = JT.transpose(0, 2, 1) has per-item strides (8, 24):
    exactly the F-contiguous layout `jac[:, dofadr]` fancy indexing produces
    in the original (which matters: numpy dispatches F- vs C-ordered operands
    to different cblas_dgemv variants with different rounding)."""
    K = p.shape[0]
    for k in range(K):
        r = roots[k]
        o0 = p[k, 0] - subtree_com[r, 0]
        o1 = p[k, 1] - subtree_com[r, 1]
        o2 = p[k, 2] - subtree_com[r, 2]
        b = bodies[k]
        w0 = cvel[b, 0]
        w1 = cvel[b, 1]
        w2 = cvel[b, 2]
        # mju_transformSpatial(pvel, cvel[b], 0, point, subtree_com[root]) (motion):
        # pvel[3:] = cvel[3:] - cross(dif, cvel[:3]), dif = point - com
        c0 = o1 * w2 - o2 * w1
        c1 = o2 * w0 - o0 * w2
        c2 = o0 * w1 - o1 * w0
        pv0 = cvel[b, 3] - c0
        pv1 = cvel[b, 4] - c1
        pv2 = cvel[b, 5] - c2
        for j in range(chain_len[k]):
            col = chain_col[k, j]
            if col < 0:
                continue
            i = chain_dof[k, j]
            a0 = cdof[i, 0]
            a1 = cdof[i, 1]
            a2 = cdof[i, 2]
            # jacp = cdof[3:] + cross(cdof[:3], offset)
            JT[k, col, 0] = cdof[i, 3] + (a1 * o2 - a2 * o1)
            JT[k, col, 1] = cdof[i, 4] + (a2 * o0 - a0 * o2)
            JT[k, col, 2] = cdof[i, 5] + (a0 * o1 - a1 * o0)
            d0 = cdof_dot[i, 0]
            d1 = cdof_dot[i, 1]
            d2 = cdof_dot[i, 2]
            t10 = d1 * o2 - d2 * o1
            t11 = d2 * o0 - d0 * o2
            t12 = d0 * o1 - d1 * o0
            t20 = a1 * pv2 - a2 * pv1
            t21 = a2 * pv0 - a0 * pv2
            t22 = a0 * pv1 - a1 * pv0
            JdT[k, col, 0] = cdof_dot[i, 3] + t10 + t20
            JdT[k, col, 1] = cdof_dot[i, 4] + t11 + t21
            JdT[k, col, 2] = cdof_dot[i, 5] + t12 + t22


# ---------------------------------------------------------------------------
# per-model static chain tables (cached)
# ---------------------------------------------------------------------------
_CHAIN_CACHE: dict = {}


def _model_chains(model):
    key = id(model)
    ent = _CHAIN_CACHE.get(key)
    if ent is not None and ent[0] is model:
        return ent[1]
    chains = {}
    for b in range(model.nbody):
        bb = b
        while bb and not model.body_dofnum[bb]:
            bb = int(model.body_parentid[bb])
        out = []
        if bb:
            i = int(model.body_dofadr[bb] + model.body_dofnum[bb] - 1)
            while i >= 0:
                out.append(i)
                i = int(model.dof_parentid[i])
        chains[b] = out
    _CHAIN_CACHE[key] = (model, chains)
    return chains


_TABLE_CACHE: dict = {}


def _chain_tables(model, bodies: np.ndarray, dofadr) -> tuple:
    """(roots, chain_dof, chain_col, chain_len, ok) for a body list + requested cols."""
    dofadr_t = tuple(int(x) for x in np.asarray(dofadr).reshape(-1))
    key = (id(model), bodies.tobytes(), dofadr_t)
    ent = _TABLE_CACHE.get(key)
    if ent is not None and ent[0] is model:
        return ent[1]
    chains = _model_chains(model)
    col_of = {}
    for c, dof in enumerate(dofadr_t):
        col_of.setdefault(dof, c)   # (duplicate dofadr would break exactness: flagged below)
    ok = len(col_of) == len(dofadr_t)
    K = bodies.shape[0]
    maxc = max([len(chains[int(b)]) for b in bodies] + [1])
    chain_dof = np.zeros((K, maxc), np.int64)
    chain_col = -np.ones((K, maxc), np.int64)
    chain_len = np.zeros(K, np.int64)
    roots = np.zeros(K, np.int64)
    for k, b in enumerate(bodies):
        b = int(b)
        roots[k] = int(model.body_rootid[b])
        ch = chains[b]
        chain_len[k] = len(ch)
        for j, i in enumerate(ch):
            jt = int(model.jnt_type[model.dof_jntid[i]])
            if jt not in (_HINGE, _SLIDE):
                ok = False     # ball/free: MuJoCo recomputes cdof_dot -> not replicated
            chain_dof[k, j] = i
            chain_col[k, j] = col_of.get(i, -1)
    res = (roots, chain_dof, chain_col, chain_len, ok)
    _TABLE_CACHE[key] = (model, res)
    return res


# ---------------------------------------------------------------------------
# keypoint-set packing (cached for the static barrier_keypoints lists)
# ---------------------------------------------------------------------------
_KPS_CACHE: dict = {}


def _pack_kps(kps):
    """list of (body_id, offset(3,), radius) -> (bodies int64 (K,), offs (K,3,1), radii list).
    Static lists (sim_common.barrier_keypoints results, cached forever there) are
    cached by identity; any other list is packed fresh (cheap)."""
    key = id(kps)
    ent = _KPS_CACHE.get(key)
    if ent is not None and ent[0] is kps and ent[1] == len(kps):
        return ent[2]
    bodies = np.array([int(b) for b, _o, _r in kps], dtype=np.int64)
    offs = np.empty((len(kps), 3, 1))
    for k, (_b, o, _r) in enumerate(kps):
        offs[k, :, 0] = o
    radii = np.array([r for _b, _o, r in kps], dtype=np.float64)
    res = (bodies, offs, radii)
    if any(kps is v for v in sc._KEYPOINT_CACHE.values()):
        _KPS_CACHE[key] = (kps, len(kps), res)
    return res


def _world_points(data, bodies, offs):
    """== np.stack([data.xpos[b] + data.xmat[b].reshape(3,3) @ off ...]) bit-for-bit
    (batched matmul -> the same per-item cblas_dgemv)."""
    R = data.xmat[bodies].reshape(-1, 3, 3)
    return data.xpos[bodies] + (R @ offs)[:, :, 0]


def keypoint_states(model, data, bodies, offs, dofadr):
    """(p (K,3), J (K,3,n), Jd (K,3,n), v (K,3), ok) -- == per-keypoint
    _kp_world / point_jac_at(..)[:, dofadr] / point_jacdot_at(..)[:, dofadr] / J @ qvel[dofadr]."""
    roots, chain_dof, chain_col, chain_len, ok = _chain_tables(model, bodies, dofadr)
    p = _world_points(data, bodies, offs)
    n = len(dofadr)
    K = bodies.shape[0]
    JT = np.zeros((K, n, 3))
    JdT = np.zeros((K, n, 3))
    _jac_jacdot_batch(p, bodies, roots, chain_dof, chain_col, chain_len,
                      data.cdof, data.cdof_dot, data.cvel, data.subtree_com, n, JT, JdT)
    J = JT.transpose(0, 2, 1)       # (K,3,n), each J[k] F-contiguous like jac[:, dofadr]
    Jd = JdT.transpose(0, 2, 1)
    qd = data.qvel[dofadr]
    v = J @ qd                      # (K,3): per-item dgemv == J_k @ qd
    return p, J, Jd, v, qd, ok


# ---------------------------------------------------------------------------
# all_pair_barriers
# ---------------------------------------------------------------------------
_PAIR_KEYS = ("h", "dist", "eid", "iid", "n", "Je", "Ji", "JeDot", "JiDot", "v_ego", "v_int", "hdot")


class PairList(Sequence):
    """Read-only Sequence of the SAME dicts sim_common.all_pair_barriers returns
    (same order, same values), materialized lazily on first element access /
    iteration. NOT a list subclass on purpose: C-level fast paths
    (PySequence_Fast, str.join, ...) would read an unbuilt list subclass as
    empty; a Sequence either works through __iter__/__getitem__ or fails loudly.
    `.arrays` exposes the vectorized fields (ego-major flat order) so
    FastFLMPC / other array-aware consumers never build the 1000+ dicts.
    Use `.as_list()` if a real list is required."""

    __slots__ = ("arrays", "_list")

    def __init__(self, arrays):
        self.arrays = arrays
        self._list = None

    def as_list(self):
        if self._list is None:
            a = self.arrays
            E, I = a["E"], a["I"]
            h = a["h"].tolist()
            dist = a["dist"].tolist()
            hdot = a["hdot"].tolist()
            eids = a["eids"].tolist()
            iids = a["iids"].tolist()
            n = a["n"]
            Je_l = list(a["Je"]); JeD_l = list(a["JeDot"]); ve_l = list(a["v_ego"])
            Ji_l = list(a["Ji"]); JiD_l = list(a["JiDot"]); vi_l = list(a["v_int"])
            out = []
            k = 0
            for e in range(E):
                eid, Jek, JeDk, vek = eids[e], Je_l[e], JeD_l[e], ve_l[e]
                for i in range(I):
                    out.append(dict(h=h[k], dist=dist[k], eid=eid, iid=iids[i], n=n[k],
                                    Je=Jek, Ji=Ji_l[i], JeDot=JeDk, JiDot=JiD_l[i],
                                    v_ego=vek, v_int=vi_l[i], hdot=hdot[k]))
                    k += 1
            self._list = out
        return self._list

    def __len__(self):
        return self.arrays["E"] * self.arrays["I"]

    def __getitem__(self, idx):
        return self.as_list()[idx]

    def __iter__(self):
        return iter(self.as_list())

    def __add__(self, other):
        return self.as_list() + list(other)

    def __radd__(self, other):
        return list(other) + self.as_list()

    def h_min(self, n_first=None):
        """== float(min(p["h"] for p in pairs[:n_first])), without building dicts
        (min is exact, so bit-identical). n_first=None -> all pairs."""
        h = self.arrays["h"]
        return float((h if n_first is None else h[:n_first]).min())


def all_pair_barriers(model, data, ego_dofadr, int_dofadr,
                      ego_prefix=sc.EGO_PREFIX, int_prefix=sc.INT_PREFIX,
                      extra_ego_keypoints=None):
    ego_kps = sc.barrier_keypoints(model, ego_prefix)
    int_kps = sc.barrier_keypoints(model, int_prefix)
    eb, eo, er = _pack_kps(ego_kps)
    if extra_ego_keypoints:
        xb, xo, xr = _pack_kps(list(extra_ego_keypoints))
        eb = np.concatenate([eb, xb]); eo = np.concatenate([eo, xo]); er = np.concatenate([er, xr])
    ib, io, ir = _pack_kps(int_kps)
    pe, Je, JeD, ve, _, ok_e = keypoint_states(model, data, eb, eo, ego_dofadr)
    pi, Ji, JiD, vi, _, ok_i = keypoint_states(model, data, ib, io, int_dofadr)
    if not (ok_e and ok_i):
        return _ORIG["all_pair_barriers"](model, data, ego_dofadr, int_dofadr, ego_prefix, int_prefix,
                                           extra_ego_keypoints)
    E, I = eb.shape[0], ib.shape[0]
    d = (pe[:, None, :] - pi[None, :, :]).reshape(E * I, 3)
    sq = (d[:, None, :] @ d[:, :, None])[:, 0, 0]          # == d.dot(d) (cblas_ddot) per pair
    dist = np.sqrt(sq)                                       # == np.linalg.norm(d)
    n = d / np.maximum(dist, 1e-9)[:, None]
    rel = (ve[:, None, :] - vi[None, :, :]).reshape(E * I, 3)
    hdot = (n[:, None, :] @ rel[:, :, None])[:, 0, 0]       # == float(n @ (v_ego - v_int))
    h = dist - ((er[:, None] + ir[None, :]).reshape(-1) + sc.D_MIN)
    arrays = dict(E=E, I=I, h=h, dist=dist, n=n, hdot=hdot, eids=eb, iids=ib,
                  e_idx=np.repeat(np.arange(E), I), i_idx=np.tile(np.arange(I), E),
                  Je=Je, Ji=Ji, JeDot=JeD, JiDot=JiD, v_ego=ve, v_int=vi, p_ego=pe, p_int=pi)
    return PairList(arrays)


# ---------------------------------------------------------------------------
# table_plane_barrier_rows
# ---------------------------------------------------------------------------
_EZ = np.array([0.0, 0.0, 1.0])


def table_plane_barrier_rows(model, data, kps, dofadr, M_inv, bias, z_floor,
                             margin=0.0, alpha1=sc.ALPHA1, alpha2=sc.ALPHA2, kp_radius=False):
    b, o, _r = _pack_kps(kps)
    p, J, Jd, v, qd, ok = keypoint_states(model, data, b, o, dofadr)
    if not ok:
        if kp_radius:
            return _ORIG["table_plane_barrier_rows"](model, data, kps, dofadr, M_inv, bias, z_floor,
                                                      margin, alpha1, alpha2, kp_radius=True)
        return _ORIG["table_plane_barrier_rows"](model, data, kps, dofadr, M_inv, bias, z_floor,
                                                  margin, alpha1, alpha2)
    K = b.shape[0]
    z_thresh = z_floor - margin
    h = p[:, 2] - z_thresh if not kp_radius else p[:, 2] - _r - z_thresh   # kp_radius: --ctrlfixes F2
    hdot = v[:, 2]
    w = Jd @ qd                                              # (K,3) == Jd_k @ qvel
    nK = np.broadcast_to(_EZ, (K, 1, 3))
    hddot_rest = (nK @ w[:, :, None])[:, 0, 0]               # == float(n @ (Jd @ qd))
    psi1 = hdot + alpha1 * h
    nJ = nK @ J                                              # (K,1,n) == n @ J
    A_h = nJ @ np.asarray(M_inv)                             # (K,1,n) == (n @ J) @ M_inv
    Ab = (A_h @ np.asarray(bias)[:, None])[:, 0, 0]          # == A_h @ bias
    const_term = -Ab + hddot_rest + alpha1 * hdot + alpha2 * psi1
    return -A_h[:, 0, :], const_term


# ---------------------------------------------------------------------------
# min_pair_distance (ground-truth logging, same result tuple)
# ---------------------------------------------------------------------------
def min_pair_distance(model, data, ego_prefix=sc.EGO_PREFIX, int_prefix=sc.INT_PREFIX):
    ego_kps = sc.barrier_keypoints(model, ego_prefix)
    int_kps = sc.barrier_keypoints(model, int_prefix)
    eb, eo, er = _pack_kps(ego_kps)
    ib, io, ir = _pack_kps(int_kps)
    pe = _world_points(data, eb, eo)
    pi = _world_points(data, ib, io)
    E, I = eb.shape[0], ib.shape[0]
    d = (pe[:, None, :] - pi[None, :, :]).reshape(E * I, 3)
    dist = np.sqrt((d[:, None, :] @ d[:, :, None])[:, 0, 0])
    h = dist - ((er[:, None] + ir[None, :]).reshape(-1) + sc.D_MIN)
    k = int(np.argmin(h))            # first occurrence == the original's strict `<` scan
    e, i = divmod(k, I)
    dk = float(dist[k])
    n = d[k] / max(dk, 1e-9)
    return (float(h[k]), dk, ego_kps[e][0], int_kps[i][0], n, ego_kps[e][1], int_kps[i][1])


# ---------------------------------------------------------------------------
# demo logging helper: right_arm_to_cup_clearance (value only)
# ---------------------------------------------------------------------------
def right_arm_to_cup_clearance(model, data, cup_kps, right_kps, left_ee_body):
    R_ee = data.xmat[left_ee_body].reshape(3, 3)
    p_ee = data.xpos[left_ee_body]
    _cb, co, cup_r = _pack_kps(list(cup_kps))
    cup_pts = p_ee + (R_ee @ co)[:, :, 0]                    # == [p_ee + R_ee @ off ...]
    rb, ro, rr = _pack_kps(right_kps)
    pr = _world_points(data, rb, ro)
    diff = cup_pts[None, :, :] - pr[:, None, :]              # (R, C, 3)
    dd = np.linalg.norm(diff, axis=-1)                       # same add.reduce path as axis=1 per row
    d = dd - cup_r[None, :] - rr[:, None]
    return float(d.min())


# ---------------------------------------------------------------------------
# FastFLMPC: identical step(), cheaper top-K gather from a PairList
# ---------------------------------------------------------------------------
def topk_pairs(pairs_all, K):
    if isinstance(pairs_all, PairList):
        a = pairs_all.arrays
        idx = np.argsort(a["h"], kind="stable")[:K]          # == Python's stable sorted()[:K]
        out = []
        for k in idx.tolist():
            e, i = divmod(k, a["I"])
            out.append(dict(h=float(a["h"][k]), dist=float(a["dist"][k]), eid=int(a["eids"][e]),
                            iid=int(a["iids"][i]), n=a["n"][k], Je=a["Je"][e], Ji=a["Ji"][i],
                            JeDot=a["JeDot"][e], JiDot=a["JiDot"][i], v_ego=a["v_ego"][e],
                            v_int=a["v_int"][i], hdot=float(a["hdot"][k])))
        return out
    return sorted(pairs_all, key=lambda p: p["h"])[:K]


class FastFLMPC(mpc_fl.FLMPC):
    """mpc_fl.FLMPC with step() unchanged EXCEPT the pair gather: passes the
    already-top-K list (bit-identical content/order) into the parent step, whose
    own `sorted(...)[:K_PAIRS]` on a K-element list is then a no-op reorder."""

    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs_all, *args, **kwargs):
        pairs_k = topk_pairs(pairs_all, self.K_PAIRS)
        return super().step(model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                            q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs_k, *args, **kwargs)


# ---------------------------------------------------------------------------
# install / uninstall (experiments only -- the owner integrates for real)
# ---------------------------------------------------------------------------
_ORIG: dict = {
    "all_pair_barriers": sc.all_pair_barriers,
    "table_plane_barrier_rows": sc.table_plane_barrier_rows,
    "min_pair_distance": sc.min_pair_distance,
}
_INSTALLED: list = []


def install(demo=None, *, logging_helpers=True, flmpc=True):
    """Monkeypatch the fast path into sim_common (+ the demo module's logging
    helper and FLMPC class reference). Returns nothing; see uninstall()."""
    def _set(obj, attr, val):
        _INSTALLED.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)
    _set(sc, "all_pair_barriers", all_pair_barriers)
    _set(sc, "table_plane_barrier_rows", table_plane_barrier_rows)
    if logging_helpers:
        _set(sc, "min_pair_distance", min_pair_distance)
        if demo is not None:
            _set(demo, "right_arm_to_cup_clearance", right_arm_to_cup_clearance)
    if flmpc:
        _set(mpc_fl, "FLMPC", FastFLMPC)


def uninstall():
    while _INSTALLED:
        obj, attr, val = _INSTALLED.pop()
        setattr(obj, attr, val)
