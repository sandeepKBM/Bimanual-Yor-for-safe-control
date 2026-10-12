"""zono_fast.py -- numba kernels for the zonotope HOCBF rows and the base-filter pairs.  SIMULATION ONLY (pure numpy + numba).

What is fast here, and what is NOT approximated:

  * The H-rep of every zonotope is built ONCE (zono_hrep.zono_hrep, at ZonoSet construction = once per perception update) and packed into flat
    arrays (all facet normals / offsets, per-zonotope offsets, AABBs, facet group ids) lazily on the first use (ZonoSet.packed()).
  * A uniform-grid hash (cell -> ids of the zonotopes whose AABB, inflated by `reach`, overlaps the cell) is built once per (zonotope set, reach)
    and used to cull candidate zonotopes per point before the exact test.  The cull is a SUPERSET filter (a point p can only be within
    `d_act + r` of a zonotope's AABB if p lies in the inflated AABB, hence in one of the cells that were registered), so it never drops a pair that
    the dense test would keep.
  * The exact test per surviving (point, zonotope) pair is the SAME arithmetic as rows.zono_rows_from_states / base_filter.facet_pairs:
        near  :  ||gap(p, AABB)|| - r < d_act ,   sd = max_f (n_f.p - d_f) ,   act : sd - r < d_act ,   h = sd - r - margin ,   normal = n_active.
    Pairs are returned sorted by (zonotope, keypoint) -- the order of the dense implementation -- so rows are identical to ~1e-16 (the only
    difference is the summation order of the 3-term dot products, BLAS vs scalar loop).
  * The max_rows / per_facet urgency selection (a python loop with dict + rounded-normal tuples before) is a numba loop over precomputed
    per-facet group ids that encode the very same rounded-normal keys.
"""
from __future__ import annotations

import math

import numpy as np
import numba as nb


# ------------------------------------------------------------------------------------------------------------------------- packing
class Packed:
    """flat arrays of a ZonoSet.  Nall (F,3), dall (F,), off (Z+1,), lo/hi (Z,3), gid3 (F,) rounded-normal group ids (rows.py ranking key: zonotope +
    round(n, 3)), gid1000 (F,) the same for base_filter (zonotope + round(1000 n))."""

    def __init__(self, hreps):
        self.Z = len(hreps)
        self.Nall = np.ascontiguousarray(np.vstack([h.N for h in hreps]) if self.Z else np.zeros((0, 3)), dtype=np.float64)
        self.dall = np.ascontiguousarray(np.concatenate([h.d for h in hreps]) if self.Z else np.zeros(0), dtype=np.float64)
        self.Nx, self.Ny, self.Nz = (np.ascontiguousarray(self.Nall[:, i]) for i in range(3))          # structure-of-arrays copy for the vectorised facet scan
        self.fmax = int(max((h.n_facets for h in hreps), default=1))
        self.sym = all(h.n_facets % 2 == 0 and np.array_equal(h.N[h.n_facets // 2:], -h.N[:h.n_facets // 2]) for h in hreps)      # antipodal-halves layout of zono_hrep
        self.off = np.concatenate([[0], np.cumsum([h.n_facets for h in hreps])]).astype(np.int64) if self.Z else np.zeros(1, np.int64)
        self.lo = np.ascontiguousarray(np.array([h.lo for h in hreps]).reshape(-1, 3), dtype=np.float64)
        self.hi = np.ascontiguousarray(np.array([h.hi for h in hreps]).reshape(-1, 3), dtype=np.float64)
        zid = np.repeat(np.arange(self.Z), np.diff(self.off)) if self.Z else np.zeros(0, int)
        self.zid = zid
        self.gid3 = self._gid(np.concatenate([zid[:, None], np.round(self.Nall, 3) + 0.0], 1))
        self.gid1000 = self._gid(np.concatenate([zid[:, None].astype(float), np.round(self.Nall * 1000) + 0.0], 1))
        self._grids = {}

    @staticmethod
    def _gid(key):
        if not len(key):
            return np.zeros(0, np.int64)
        _u, inv = np.unique(key, axis=0, return_inverse=True)
        return np.asarray(inv).reshape(-1).astype(np.int64)

    def grid(self, reach, cell=0.5):
        k = (round(float(reach), 9), float(cell))
        g = self._grids.get(k)
        if g is None:
            g = self._grids[k] = Grid(self.lo, self.hi, reach, cell)
        return g


class Grid:
    """uniform grid hash.  Zonotope z is registered in every cell that its AABB inflated by `reach` overlaps; query(p) = ids registered in p's cell."""

    def __init__(self, lo, hi, reach, cell=0.5):
        Z = len(lo)
        self.cell = float(cell)
        if Z == 0:
            self.x0 = np.zeros(3); self.dim = np.ones(3, np.int64)
            self.keys = np.zeros(0, np.int64); self.zids = np.zeros(0, np.int64)
            return
        a, b = lo - reach, hi + reach
        self.x0 = a.min(0)
        ia = np.floor((a - self.x0) / self.cell).astype(np.int64)
        ib = np.floor((b - self.x0) / self.cell).astype(np.int64)
        self.dim = ib.max(0) + 1
        ks, zs = [], []
        for z in range(Z):
            gx, gy, gz = np.meshgrid(np.arange(ia[z, 0], ib[z, 0] + 1), np.arange(ia[z, 1], ib[z, 1] + 1), np.arange(ia[z, 2], ib[z, 2] + 1), indexing="ij")
            kk = (gx.ravel() * self.dim[1] + gy.ravel()) * self.dim[2] + gz.ravel()
            ks.append(kk); zs.append(np.full(kk.size, z, np.int64))
        keys, zids = np.concatenate(ks), np.concatenate(zs)
        o = np.lexsort((zids, keys))                      # by cell, zonotope ids ascending inside a cell
        self.keys, self.zids = keys[o], zids[o]


# ------------------------------------------------------------------------------------------------------------------------- kernels
@nb.njit(cache=True, inline="always")
def _scan(px, py, pz, r, d_act, f0, f1, Nx, Ny, Nz, dall, buf):
    """(best signed distance, facet) of the point over the facets f0..f1-1 of one zonotope, ties -> the first facet (the arg-max of n_f.p - d_f).
    The facets come in antipodal halves [n_1..n_h, -n_1..-n_h] (zono_hrep layout, checked at pack time): the dot product is shared, so one pass over the
    first half yields both candidates; the two running maxima are merged at the end (plus-half wins ties = the lower index).  Early exit as soon as one
    facet already puts the point d_act + r outside (sd >= that value: the pair is inactive whatever the remaining facets say -- the caller rejects it)."""
    h = (f1 - f0) // 2
    bp = -1e300
    bm = -1e300
    ip = f0
    im = f0 + h
    for f in range(f0, f0 + h):
        a = Nx[f] * px + Ny[f] * py + Nz[f] * pz
        sp = a - dall[f]
        sm = -a - dall[f + h]
        if sp > bp:
            bp = sp
            ip = f
        if sm > bm:
            bm = sm
            im = f + h
        if sp - r >= d_act or sm - r >= d_act:
            break
    if bp >= bm:
        return bp, ip
    return bm, im


@nb.njit(cache=True, inline="always")
def _scan_plain(px, py, pz, r, d_act, f0, f1, Nx, Ny, Nz, dall, buf):
    best = -1e300
    bf = f0
    for f in range(f0, f1):
        s = Nx[f] * px + Ny[f] * py + Nz[f] * pz - dall[f]
        if s > best:
            best = s
            bf = f
            if best - r >= d_act:
                break
    return best, bf


@nb.njit(cache=True)
def _pairs_dense(p, r, lo, hi, off, Nx, Ny, Nz, dall, d_act, buf, sym, ok, oz, of, osd):
    K, Z = p.shape[0], lo.shape[0]
    n = 0
    for z in range(Z):
        for k in range(K):
            g2 = 0.0
            for c in range(3):
                g = max(max(lo[z, c] - p[k, c], p[k, c] - hi[z, c]), 0.0)
                g2 += g * g
            lim = d_act + r[k]
            if lim > 0.0 and g2 < lim * lim:                       # == sqrt(g2) - r < d_act without the sqrt
                if sym:
                    best, bf = _scan(p[k, 0], p[k, 1], p[k, 2], r[k], d_act, off[z], off[z + 1], Nx, Ny, Nz, dall, buf)
                else:
                    best, bf = _scan_plain(p[k, 0], p[k, 1], p[k, 2], r[k], d_act, off[z], off[z + 1], Nx, Ny, Nz, dall, buf)
                if best - r[k] < d_act:
                    ok[n] = k; oz[n] = z; of[n] = bf; osd[n] = best
                    n += 1
    return n


@nb.njit(cache=True)
def _pairs_grid(p, r, lo, hi, off, Nx, Ny, Nz, dall, d_act, buf, sym, gx0, gcell, gdim, gkeys, gzids, ok, oz, of, osd):
    K = p.shape[0]
    n = 0
    for k in range(K):
        ix = int(math.floor((p[k, 0] - gx0[0]) / gcell))
        iy = int(math.floor((p[k, 1] - gx0[1]) / gcell))
        iz = int(math.floor((p[k, 2] - gx0[2]) / gcell))
        if ix < 0 or iy < 0 or iz < 0 or ix >= gdim[0] or iy >= gdim[1] or iz >= gdim[2]:
            continue
        key = (ix * gdim[1] + iy) * gdim[2] + iz
        i0 = np.searchsorted(gkeys, key, side="left")
        i1 = np.searchsorted(gkeys, key, side="right")
        for i in range(i0, i1):
            z = gzids[i]
            g2 = 0.0
            for c in range(3):
                g = max(max(lo[z, c] - p[k, c], p[k, c] - hi[z, c]), 0.0)
                g2 += g * g
            lim = d_act + r[k]
            if lim > 0.0 and g2 < lim * lim:                       # == sqrt(g2) - r < d_act without the sqrt
                if sym:
                    best, bf = _scan(p[k, 0], p[k, 1], p[k, 2], r[k], d_act, off[z], off[z + 1], Nx, Ny, Nz, dall, buf)
                else:
                    best, bf = _scan_plain(p[k, 0], p[k, 1], p[k, 2], r[k], d_act, off[z], off[z + 1], Nx, Ny, Nz, dall, buf)
                if best - r[k] < d_act:
                    ok[n] = k; oz[n] = z; of[n] = bf; osd[n] = best
                    n += 1
    return n


@nb.njit(cache=True)
def _select(order, gid, zero_based_groups, per_facet, max_rows):
    counts = np.zeros(zero_based_groups, np.int64)
    keep = np.empty(min(max_rows, order.shape[0]), np.int64)
    n = 0
    for i in order:
        g = gid[i]
        if counts[g] >= per_facet:
            continue
        counts[g] += 1
        keep[n] = i
        n += 1
        if n >= keep.shape[0]:
            break
    return keep[:n]


# ------------------------------------------------------------------------------------------------------------------------- API
GRID_MIN_ZONOS = 48           # below this the dense double loop (AABB test ~8 us for 250 pts x 30 zonotopes) is faster than the hash
GRID_CELL = 0.5


def facet_pairs(p, r, pk: Packed, d_act, margin, use_grid=None, reach_extra=0.0):
    """(kk, zii, fidx, nrm, h) for every (point, zonotope) pair with  sd_lb - r < d_act  (active facet), sorted by (zonotope, point) -- the dense order.
    p (K,3), r (K,).  h = sd - r - margin."""
    K = p.shape[0]
    if K == 0 or pk.Z == 0:
        e = np.zeros(0, np.int64)
        return e, e, e, np.zeros((0, 3)), np.zeros(0)
    p = np.ascontiguousarray(p, dtype=np.float64)
    r = np.ascontiguousarray(r, dtype=np.float64)
    cap = K * pk.Z
    ok = np.empty(cap, np.int64); oz = np.empty(cap, np.int64); of = np.empty(cap, np.int64); osd = np.empty(cap)
    if use_grid is None:
        use_grid = pk.Z >= GRID_MIN_ZONOS
    if use_grid:
        g = pk.grid(d_act + float(r.max()) + reach_extra, GRID_CELL)
        buf = np.empty(pk.fmax)
        n = _pairs_grid(p, r, pk.lo, pk.hi, pk.off, pk.Nx, pk.Ny, pk.Nz, pk.dall, float(d_act), buf, pk.sym, g.x0, g.cell, g.dim, g.keys, g.zids, ok, oz, of, osd)
        o = np.argsort(oz[:n] * K + ok[:n], kind="stable")
    else:
        buf = np.empty(pk.fmax)
        n = _pairs_dense(p, r, pk.lo, pk.hi, pk.off, pk.Nx, pk.Ny, pk.Nz, pk.dall, float(d_act), buf, pk.sym, ok, oz, of, osd)
        o = np.arange(n)
    kk, zz, ff, sd = ok[:n][o], oz[:n][o], of[:n][o], osd[:n][o]
    return kk, zz, ff, pk.Nall[ff], sd - r[kk] - margin


def select_urgent(urgency_order, gid, per_facet, max_rows):
    """indices (into the candidate list) kept: walk `urgency_order`, take a candidate unless its group already has `per_facet`, stop at max_rows."""
    ng = int(gid.max()) + 1 if len(gid) else 1
    return _select(np.ascontiguousarray(urgency_order, dtype=np.int64), np.ascontiguousarray(gid, dtype=np.int64), ng, int(per_facet), int(max_rows))


# ------------------------------------------------------------------------------------------------------------------------- fused ranking + assembly
@nb.njit(cache=True)
def _rows_kernel(kk, nrm, h, gid, J, Jd, v, vx, qd, Minv, bias, a1, a2, max_rows, per_facet, ng):
    R = kk.shape[0]
    n = qd.shape[0]
    w = np.empty((R, 3))
    vk = np.empty((R, 3))
    for r in range(R):
        k = kk[r]
        for i in range(3):
            s = 0.0
            for c in range(n):
                s += Jd[k, i, c] * qd[c]
            w[r, i] = s
            vk[r, i] = v[k, i] + vx[k, i]
    if max_rows >= 0 and R > max_rows:
        free = np.empty(R)
        for r in range(R):
            hd0 = nrm[r, 0] * vk[r, 0] + nrm[r, 1] * vk[r, 1] + nrm[r, 2] * vk[r, 2]
            free[r] = (nrm[r, 0] * w[r, 0] + nrm[r, 1] * w[r, 1] + nrm[r, 2] * w[r, 2]) + a1 * hd0 + a2 * (hd0 + a1 * h[r])
        order = np.argsort(free, kind="mergesort")
        counts = np.zeros(ng, np.int64)
        keep = np.empty(max_rows, np.int64)
        m = 0
        for ii in range(R):
            r = order[ii]
            g = gid[r]
            if counts[g] >= per_facet:
                continue
            counts[g] += 1
            keep[m] = r
            m += 1
            if m >= max_rows:
                break
        keep = keep[:m]
    else:
        keep = np.arange(R)
    m = keep.shape[0]
    A = np.empty((m, n))
    const = np.empty(m)
    hdot = np.empty(m)
    nJ = np.empty(n)
    Ah = np.empty(n)
    for j in range(m):
        r = keep[j]
        k = kk[r]
        for c in range(n):
            nJ[c] = nrm[r, 0] * J[k, 0, c] + nrm[r, 1] * J[k, 1, c] + nrm[r, 2] * J[k, 2, c]
        for c2 in range(n):
            s = 0.0
            for c in range(n):
                s += nJ[c] * Minv[c, c2]
            Ah[c2] = s
        ah_b = 0.0
        for c in range(n):
            ah_b += Ah[c] * bias[c]
        hd = nrm[r, 0] * vk[r, 0] + nrm[r, 1] * vk[r, 1] + nrm[r, 2] * vk[r, 2]
        rest = nrm[r, 0] * w[r, 0] + nrm[r, 1] * w[r, 1] + nrm[r, 2] * w[r, 2]
        psi1 = hd + a1 * h[r]
        const[j] = -ah_b + rest + a1 * hd + a2 * psi1
        hdot[j] = hd
        for c in range(n):
            A[j, c] = -Ah[c]
    return keep, A, const, hdot


def assemble_rows(kk, nrm, h, gid, J, Jd, v, vx, qd, Minv, bias, a1, a2, max_rows, per_facet):
    """ranking (urgency = psi2 at zero acceleration, <= per_facet per group, <= max_rows; max_rows None = all) + HOCBF row assembly in one numba call.
    Returns (keep (m,), A (m,n), const (m,), hdot (m,)) -- keep indexes the candidate list (kk, nrm, h)."""
    c = np.ascontiguousarray
    ng = int(gid.max()) + 1 if len(gid) else 1
    return _rows_kernel(c(kk, dtype=np.int64), c(nrm), c(h), c(gid, dtype=np.int64), c(J), c(Jd), c(v), c(vx), c(qd, dtype=np.float64), c(Minv), c(bias, dtype=np.float64),
                        float(a1), float(a2), -1 if max_rows is None else int(max_rows), int(per_facet), ng)


@nb.njit(cache=True)
def _tracked_pv(bodies, offs, xpos, xmat, cvel, rootid, subtree_com):
    K = bodies.shape[0]
    p = np.empty((K, 3))
    v = np.empty((K, 3))
    for k in range(K):
        b = bodies[k]
        for i in range(3):
            p[k, i] = xpos[b, i] + xmat[b, 3 * i] * offs[k, 0] + xmat[b, 3 * i + 1] * offs[k, 1] + xmat[b, 3 * i + 2] * offs[k, 2]
        # cvel = (angular, linear) of the body, the linear part referred to subtree_com[root]:  v(p) = lin + ang x (p - com)
        r0 = p[k, 0] - subtree_com[rootid[b], 0]
        r1 = p[k, 1] - subtree_com[rootid[b], 1]
        r2 = p[k, 2] - subtree_com[rootid[b], 2]
        v[k, 0] = cvel[b, 3] + cvel[b, 1] * r2 - cvel[b, 2] * r1
        v[k, 1] = cvel[b, 4] + cvel[b, 2] * r0 - cvel[b, 0] * r2
        v[k, 2] = cvel[b, 5] + cvel[b, 0] * r1 - cvel[b, 1] * r0
    return p, v


def tracked_pv(model, data, bodies, offs):
    """world point and world velocity of body-fixed points (bodies (K,), offs (K,3)) from the forwarded MjData (xpos / xmat / cvel; no Jacobians)."""
    return _tracked_pv(bodies, offs, data.xpos, data.xmat, data.cvel, model.body_rootid, data.subtree_com)


# ------------------------------------------------------------------------------------------------------------------------- base-filter rows
@nb.njit(cache=True)
def _bf_rows_kernel(kk, nrm, h, p, varm, ox, oy, gamma, a_brk, retreat_cap, u0, gid, max_rows, per_facet, ng):
    R = kk.shape[0]
    A = np.empty((R, 3))
    b = np.empty(R)
    for r in range(R):
        k = kk[r]
        nx, ny, nz = nrm[r, 0], nrm[r, 1], nrm[r, 2]
        A[r, 0] = -nx
        A[r, 1] = -ny
        A[r, 2] = -(-nx * (p[k, 1] - oy) + ny * (p[k, 0] - ox))
        hh = h[r]
        if hh >= 0.0:
            beta = min(gamma * hh, math.sqrt(2.0 * a_brk * hh))
        else:
            beta = max(gamma * hh, -retreat_cap)
        b[r] = nx * varm[k, 0] + ny * varm[k, 1] + nz * varm[k, 2] + beta
    if R > max_rows:
        urg = np.empty(R)
        for r in range(R):
            urg[r] = -(A[r, 0] * u0[0] + A[r, 1] * u0[1] + A[r, 2] * u0[2] - b[r])
        order = np.argsort(urg, kind="mergesort")
        counts = np.zeros(ng, np.int64)
        keep = np.empty(max_rows, np.int64)
        m = 0
        for ii in range(R):
            r = order[ii]
            g = gid[r]
            if counts[g] >= per_facet:
                continue
            counts[g] += 1
            keep[m] = r
            m += 1
            if m >= max_rows:
                break
        keep = keep[:m]
    else:
        keep = np.arange(R)
    return keep, A[keep], b[keep]


def bf_rows(kk, nrm, h, p, varm, origin, gamma, a_brk, retreat_cap, u0, gid, max_rows, per_facet):
    """base-filter CBF rows  A u <= b  (twist u = (vx, vy, wz)) for the candidate pairs, capped at max_rows (most violated at u0 first, <= per_facet per group)."""
    c = np.ascontiguousarray
    ng = int(gid.max()) + 1 if len(gid) else 1
    return _bf_rows_kernel(c(kk, dtype=np.int64), c(nrm), c(h), c(p), c(varm), float(origin[0]), float(origin[1]), float(gamma), float(a_brk), float(retreat_cap),
                           c(u0, dtype=np.float64), c(gid, dtype=np.int64), int(max_rows), int(per_facet), ng)


@nb.njit(cache=True)
def _world_pts(xpos, xmat, bodies, offs):
    K = bodies.shape[0]
    p = np.empty((K, 3))
    for k in range(K):
        b = bodies[k]
        for i in range(3):
            p[k, i] = xpos[b, i] + xmat[b, 3 * i] * offs[k, 0, 0] + xmat[b, 3 * i + 1] * offs[k, 1, 0] + xmat[b, 3 * i + 2] * offs[k, 2, 0]
    return p


def world_points(data, bodies, offs):
    """== fast_barriers._world_points (xpos[b] + xmat[b].reshape(3,3) @ off) to rounding; offs (K,3,1)."""
    return _world_pts(data.xpos, data.xmat, bodies, offs)


# ------------------------------------------------------------------------------------------------------------------------- JIT warm-up
_WARM = False


def warmup():
    """compile / load every kernel once (dummy data with the real dtypes), so the first control step does not pay the numba JIT (seconds).  Idempotent;
    called when a ZonoSet is packed (= at perception time, not in the control loop)."""
    global _WARM
    if _WARM:
        return
    _WARM = True
    from .zono_hrep import zono_hrep
    pk = Packed([zono_hrep(np.zeros(3), np.eye(3))])
    p = np.array([[0.5, 0.0, 0.0], [3.0, 3.0, 3.0]])
    r = np.zeros(2)
    for g in (False, True):
        kk, zz, ff, nrm, h = facet_pairs(p, r, pk, 1.0, 0.0, use_grid=g)
    gid = pk.gid3[ff]
    J = np.zeros((2, 3, 6)); Jd = np.zeros((2, 3, 6)); v = np.zeros((2, 3))
    for mr in (None, 1):
        assemble_rows(kk, nrm, h, gid, J, Jd, v, v, np.zeros(6), np.eye(6), np.zeros(6), 1.0, 1.0, mr, 2)
    bf_rows(kk, nrm, h, p, v, np.zeros(2), 1.0, 0.4, 0.15, np.zeros(3), pk.gid1000[ff], 1, 3)
    bf_rows(kk, nrm, h, p, v, np.zeros(2), 1.0, 0.4, 0.15, np.zeros(3), pk.gid1000[ff], 48, 3)
    select_urgent(np.arange(len(h)), gid, 2, 4)
    world_points(type("D", (), dict(xpos=np.zeros((2, 3)), xmat=np.zeros((2, 9))))(), np.zeros(1, np.int64), np.zeros((1, 3, 1)))
    _tracked_pv(np.zeros(1, np.int64), np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((2, 9)), np.zeros((2, 6)), np.zeros(2, np.int32), np.zeros((2, 3)))
