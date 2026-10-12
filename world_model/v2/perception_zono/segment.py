"""segment.py -- cloud pre-processing and segmentation. SIMULATION ONLY. numpy / scipy only (no sklearn in this env: DBSCAN is implemented here).

  preprocess     voxel centroid filter (labels = majority) then a kNN-count outlier filter (removes the gross 5-20 cm outliers and dropout debris)
  gt_segments    ground-truth segments: the hit geom's object id (isolates the geometry step)
  unsup_segments tray plane (RANSAC, normal within 25 deg of gravity -- the robot knows gravity) removal, then DBSCAN on xyz (+ k * normal)
  seg_metrics    object-level and point-level errors of a clustering vs the GT labels
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

VOX = 0.003
OUT_R, OUT_K = 0.012, 4


def preprocess(cl, vox=VOX, out_r=OUT_R, out_k=OUT_K):
    """-> dict(pts, label, cam) after voxel centroid + radius-outlier filtering."""
    P, L, C = cl.pts, cl.label, cl.cam
    if len(P) == 0:
        return dict(pts=P, label=L, cam=C)
    key = np.floor(P / vox).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    kid = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uk, inv = np.unique(kid, return_inverse=True)
    cnt = np.bincount(inv)
    Pm = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    nl = int(L.max()) + 1
    h = np.bincount(inv * nl + L, minlength=len(uk) * nl).reshape(len(uk), nl)
    lab = h.argmax(1).astype(np.int32)
    first = np.zeros(len(uk), np.int64)
    first[inv[::-1]] = np.arange(len(P))[::-1]
    cam = C[first]
    tree = cKDTree(Pm)
    nb = tree.query_ball_point(Pm, out_r, return_length=True)
    keep = nb >= out_k
    return dict(pts=Pm[keep], label=lab[keep], cam=cam[keep])


def gt_segments(pp):
    """{object id -> point indices}; the tray (0) is returned too."""
    return {int(k): np.flatnonzero(pp["label"] == k) for k in np.unique(pp["label"])}


# ------------------------------------------------------------------------------------------------ normals
def estimate_normals(pts, origins, k=12):
    """kNN-PCA normals, oriented toward the camera origin of each point. origins (n,3)."""
    n = len(pts)
    if n < k + 1:
        return np.tile([0.0, 0.0, 1.0], (n, 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    Q = pts[idx] - pts[idx].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    w, V = np.linalg.eigh(C)
    N = V[:, :, 0]
    flip = np.einsum("ni,ni->n", N, origins - pts) < 0
    N[flip] *= -1
    return N


# ------------------------------------------------------------------------------------------------ plane
def ransac_plane(pts, rng, n_it=300, tol=0.006, max_tilt_deg=25.0):
    """dominant near-horizontal plane. Returns (normal, d, sigma) with n.x = d (n up), or None."""
    if len(pts) < 40:
        return None
    best, best_in = None, 0
    cmin = np.cos(np.deg2rad(max_tilt_deg))
    for _ in range(n_it):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        if nrm[2] < 0:
            nrm = -nrm
        if nrm[2] < cmin:
            continue
        r = np.abs(pts @ nrm - s[0] @ nrm)
        c = int((r < tol).sum())
        if c > best_in:
            best_in, best = c, (nrm, float(s[0] @ nrm))
    if best is None or best_in < 40:
        return None
    nrm, d = best
    inl = np.abs(pts @ nrm - d) < tol
    c = pts[inl].mean(0)
    U, S, Vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    nrm = Vt[2] if Vt[2][2] > 0 else -Vt[2]
    d = float(c @ nrm)
    res = pts @ nrm - d
    near = np.abs(res) < 0.015
    sig = 1.4826 * np.median(np.abs(res[near] - np.median(res[near])))
    return nrm, d, float(sig)


# ------------------------------------------------------------------------------------------------ DBSCAN
def dbscan(X, eps, min_samples):
    """plain DBSCAN (Euclidean). Returns labels (-1 = noise)."""
    n = len(X)
    if n == 0:
        return np.zeros(0, np.int64)
    tree = cKDTree(X)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    deg = np.bincount(pairs.ravel(), minlength=n) + 1
    core = deg >= min_samples
    m = core[pairs[:, 0]] & core[pairs[:, 1]]
    pc = pairs[m]
    g = coo_matrix((np.ones(len(pc)), (pc[:, 0], pc[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    lab = np.full(n, -1, np.int64)
    lab[core] = comp[core]
    # border points: nearest core neighbour
    border = np.flatnonzero(~core)
    if len(border) and core.any():
        ci = np.flatnonzero(core)
        t2 = cKDTree(X[ci])
        d, j = t2.query(X[border], distance_upper_bound=eps)
        ok = np.isfinite(d)
        lab[border[ok]] = comp[ci[j[ok]]]
    # relabel to 0..K-1
    u, inv = np.unique(lab[lab >= 0], return_inverse=True)
    out = np.full(n, -1, np.int64)
    out[lab >= 0] = inv
    return out


def unsup_segments(pp, views, rng, use_normals=False, eps=0.009, min_samples=6, min_cluster=8, lam=0.008, plane_k=3.0, plane=None, plane_thr=0.008):
    """returns (labels, info). labels: -2 = tray plane, -1 = noise / dropped, >= 0 object cluster id."""
    P = pp["pts"]
    n = len(P)
    lab = np.full(n, -1, np.int64)
    info = dict(plane=False)
    pl = ransac_plane(P, rng) if plane is None else plane          # plane = (normal, d, sigma) from kinematics (the robot holds the tray)
    obj = np.ones(n, bool)
    if pl is not None:
        nrm, d, sig = pl
        res = P @ nrm - d
        thr = max(plane_k * sig, 0.003) if plane is None else plane_thr
        tray = np.abs(res) < thr
        lab[tray] = -2
        obj = res >= thr
        info.update(plane=True, sigma=sig, thr=thr, n_tray=int(tray.sum()))
    idx = np.flatnonzero(obj)
    if len(idx) == 0:
        return lab, info
    X = P[idx]
    if use_normals:
        origins = np.array([v.origin for v in views])[pp["cam"][idx]]
        N = estimate_normals(X, origins)
        X = np.concatenate([X, lam * N], 1)
    cl = dbscan(X, eps, min_samples)
    # drop tiny clusters
    sz = np.bincount(cl[cl >= 0]) if (cl >= 0).any() else np.zeros(0, int)
    small = np.flatnonzero(sz < min_cluster)
    cl[np.isin(cl, small)] = -1
    u, inv = np.unique(cl[cl >= 0], return_inverse=True)
    out = np.full(len(idx), -1, np.int64)
    out[cl >= 0] = inv
    lab[idx] = out
    return lab, info


def seg_metrics(gt, lab, tray_id=0):
    """gt (n,) GT object ids (0 = tray), lab (n,) clustering (-2 tray plane, -1 noise, >=0 clusters).
    object-level: for each GT object with >= 5 points the best cluster by IoU over ALL points; recovered if IoU >= 0.5.
    point-level: fraction of the object points that sit in their matched cluster; fraction of cluster points that are wrong (contamination)."""
    objs = [k for k in np.unique(gt) if k != tray_id and (gt == k).sum() >= 5]
    ncl = int(lab.max() + 1) if (lab >= 0).any() else 0
    ious, rec, cont, ass = [], 0, [], []
    for k in objs:
        m = gt == k
        if ncl == 0:
            ious.append(0.0)
            continue
        ov = np.bincount(lab[m & (lab >= 0)], minlength=ncl)
        csz = np.bincount(lab[lab >= 0], minlength=ncl)
        iou = ov / (m.sum() + csz - ov)
        c = int(iou.argmax())
        ious.append(float(iou[c]))
        rec += iou[c] >= 0.5
        ass.append(float(ov[c] / m.sum()))
        cont.append(float(1 - ov[c] / max(csz[c], 1)))
    tray_leak = float(((lab >= 0) & (gt == tray_id)).sum() / max((gt == tray_id).sum(), 1))
    return dict(n_gt=len(objs), n_clusters=ncl, recovered=int(rec), mean_iou=float(np.mean(ious)) if ious else float("nan"),
                recall_pts=float(np.mean(ass)) if ass else float("nan"), contamination=float(np.mean(cont)) if cont else float("nan"),
                tray_leak=tray_leak)


def match_clusters(gt, lab, objs):
    """object id -> point indices of its best-IoU cluster (None if no cluster)."""
    out = {}
    ncl = int(lab.max() + 1) if (lab >= 0).any() else 0
    csz = np.bincount(lab[lab >= 0], minlength=ncl) if ncl else None
    for k in objs:
        m = gt == k
        if ncl == 0 or m.sum() == 0:
            out[k] = None
            continue
        ov = np.bincount(lab[m & (lab >= 0)], minlength=ncl)
        iou = ov / (m.sum() + csz - ov)
        c = int(iou.argmax())
        out[k] = np.flatnonzero(lab == c) if iou[c] > 0 else None
    return out
