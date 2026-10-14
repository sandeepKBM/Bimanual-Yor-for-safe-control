"""perceive.py -- stereo cloud -> robot self-filter -> plane removal -> DBSCAN -> zonotope per object.  SIMULATION ONLY.

Inputs the pipeline may use: left/right images, the camera pose (from the robot's own kinematics: base odometry + the assumed head mount), the
robot's own FK (self-filter), gravity (the support planes are near-horizontal).  Ground truth (MuJoCo depth / segmentation) is used ONLY for evaluation
and for printing the name next to a cluster; it is never an input to the segmentation, the zonotopes or the planner.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import cv2
import mujoco
from scipy.spatial import cKDTree, ConvexHull

from world_model.v2.perception_zono import segment as SEG
from world_model.v2.perception_zono.fit import fit_hull_zono, fit_box
from world_model.v2.perception_zono.zonoutil import ZUnion
from world_model.v2.tray_rep.sets.zono import Zonotope
from . import stereo as ST

UNSEEN_DEPTH = 0.08        # m  ASSUMED depth of an object behind its visible front surface (occlusion extrusion, along the viewing direction)
INFL_EXTRA = 0.005         # m  on top of the measured p95 stereo error


# ------------------------------------------------------------------------------------------------ ground-truth keys (EVAL / LABELS ONLY)
def gt_keys(m):
    """geom id -> object key string (eval only).  THOR bodies 'desk_1__Desk_204_1...' -> 'desk_1'; world geoms 'bottle_a_neck_vis' -> 'bottle_a'."""
    keys = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        # climb to the top-level body below the world (or below yor_root)
        top = b
        while top > 0 and int(m.body_parentid[top]) != 0:
            top = int(m.body_parentid[top])
        nm = m.body(top).name or ""
        if top == 0 or not nm:
            gn = m.geom(g).name or f"geom{g}"
            for suf in ("_vis", "_col", "_arrow"):
                gn = gn.replace(suf, "")
            if gn.endswith("_neck"):
                gn = gn[:-5]
            if gn.startswith("wall_") or gn.startswith("lintel"):
                gn = "wall"
            elif gn.startswith(("floor", "office_floor", "threshold", "marker")):
                gn = "floor"
            keys.append(gn)
        elif "__" in nm:
            keys.append(nm.split("__")[0])
        else:
            keys.append(nm)
    return np.array(keys)


def geom_pts(m, d, g):
    """world-frame boundary-ish sample points of a geom: box corners + face centres, cylinder/capsule rings, sphere samples, mesh vertices."""
    typ = int(m.geom_type[g])
    R = d.geom_xmat[g].reshape(3, 3)
    p = d.geom_xpos[g]
    s = m.geom_size[g]
    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        a0, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        return m.mesh_vert[a0:a0 + n] @ R.T + p
    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        g1 = np.array([-1.0, 0.0, 1.0])
        c = np.array([[i, j, k] for i in g1 for j in g1 for k in g1]) * s
        return c @ R.T + p
    if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
        u = np.random.default_rng(0).normal(size=(120, 3))
        return p + s[0] * u / np.linalg.norm(u, axis=1, keepdims=True)
    if typ in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([s[0] * np.cos(a), s[0] * np.sin(a), np.zeros(24)], 1)
        c = np.concatenate([ring + [0, 0, s[1]], ring - [0, 0, s[1]], ring])
        return c @ R.T + p
    return np.zeros((0, 3))


# ------------------------------------------------------------------------------------------------ cloud
def robot_mask(seg, cls_of_geom, dilate_px=9):
    """pixels covered by the robot's OWN body (rendered from its kinematic model) + a dilation for stereo edge fattening.  seg: (H,W,2) geom/type ids."""
    gid = seg[..., 0]
    ok = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
    c = np.full(gid.shape, "H", dtype="<U1")
    c[ok] = cls_of_geom[gid[ok]]
    mk = (c != "H").astype(np.uint8)
    mk = cv2.dilate(mk, np.ones((dilate_px, dilate_px), np.uint8))
    return mk.astype(bool)


def voxel_reduce(P, pix, vox):
    key = np.floor(P / vox).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    kid = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uk, first, inv = np.unique(kid, return_index=True, return_inverse=True)
    cnt = np.bincount(inv)
    Pm = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    return Pm, pix[first]


def radius_filter(P, r=0.015, k=6):
    nb = cKDTree(P).query_ball_point(P, r, return_length=True)
    return nb >= k


def _vertical_plane(P, rng, n_it=250, tol=0.02, min_extent=0.8, min_pts=1500):
    """dominant vertical plane (normal horizontal) with a large horizontal extent (a wall).  Returns (n, d, inlier mask) or None."""
    if len(P) < min_pts:
        return None
    best, bc = None, 0
    for _ in range(n_it):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) > 0.12:
            continue
        c = int((np.abs(P @ n - s[0] @ n) < tol).sum())
        if c > bc:
            bc, best = c, (n, float(s[0] @ n))
    if best is None or bc < min_pts:
        return None
    n, d = best
    inl = np.abs(P @ n - d) < tol
    t = np.array([-n[1], n[0], 0.0])
    span = np.ptp(P[inl] @ t)
    return (n, d, inl) if span >= min_extent else None


def peel_planes(P, rng, max_h=2):
    """remove large horizontal planes (floor, desk top, ...) and large vertical planes (walls).  Returns (keep mask, list of plane dicts)."""
    keep = np.ones(len(P), bool)
    planes = []
    for _ in range(max_h):
        idx = np.flatnonzero(keep)
        pl = SEG.ransac_plane(P[idx], rng, n_it=250, tol=0.006, max_tilt_deg=6.0)
        if pl is None:
            break
        n, d, sig = pl
        inl = idx[np.abs(P[idx] @ n - d) < max(3 * sig, 0.006)]
        u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
        area = np.ptp(P[inl] @ u) * np.ptp(P[inl] @ v)
        if len(inl) < 6000 or area < 0.35:
            break
        keep[inl] = False
        inl = inl[np.abs(P[inl, 2] - np.median(P[inl, 2])) < 0.02]      # a support surface is flat: drop any inlier far from its median height
        planes.append(dict(kind="horizontal", n=n, d=float(d), z=float(np.median(P[inl, 2])), idx=inl, area=float(area), sigma=float(sig)))
    for _ in range(2):
        idx = np.flatnonzero(keep)
        vp = _vertical_plane(P[idx], rng)
        if vp is None:
            break
        n, d, inl = vp
        keep[idx[inl]] = False
        planes.append(dict(kind="vertical", n=n, d=float(d), idx=idx[inl]))
    return keep, planes


# ------------------------------------------------------------------------------------------------ zonotope helpers
def zono_vertices(Z: Zonotope, max_gen=14):
    G = Z.G
    if G.shape[1] > max_gen:
        Z = Z.reduce(max_gen)
        G = Z.G
    m = G.shape[1]
    S = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1) * 2.0 - 1.0
    return Z.c + S @ G.T


def zono_edges(Z: Zonotope, ang_deg=1.5):
    """(vertices V (k,3), edges [(i,j)]) of the convex hull of the zonotope, coplanar triangle edges merged away."""
    V = zono_vertices(Z)
    try:
        h = ConvexHull(V)
    except Exception:
        return V, []
    eq = h.equations[:, :3]
    cmin = np.cos(np.deg2rad(ang_deg))
    edges = set()
    for i, (simp, nb) in enumerate(zip(h.simplices, h.neighbors)):
        for k in range(3):
            j = nb[k]
            if float(eq[i] @ eq[j]) < cmin:
                a, b = sorted([int(simp[(k + 1) % 3]), int(simp[(k + 2) % 3])])
                edges.add((a, b))
    return V, sorted(edges)


def extrude_unseen(Z: Zonotope, view_dir_xy, L=UNSEEN_DEPTH):
    """conservative occlusion extrusion: the surface behind the visible hull is unseen, so extend the set by L along the horizontal viewing direction."""
    u = np.array([view_dir_xy[0], view_dir_xy[1], 0.0])
    u /= np.linalg.norm(u) + 1e-12
    return Zonotope(Z.c + 0.5 * L * u, np.concatenate([Z.G, 0.5 * L * u[:, None]], 1))


def drop_to_support(Z: Zonotope, z_support):
    """shadow volume: the part below the lowest visible point down to the support plane is unseen (occluded by the object itself), so extend the
    set downwards to the plane (Minkowski sum with a vertical segment)."""
    zmin = float(Z.c[2] - np.abs(Z.G[2]).sum())
    h = zmin - z_support
    if h <= 0.0:
        return Z
    return Zonotope(Z.c + np.array([0, 0, -0.5 * h]), np.concatenate([Z.G, np.array([[0.0], [0.0], [0.5 * h]])], 1))


def inflate_world(Z: Zonotope, r):
    return Zonotope(Z.c, np.concatenate([Z.G, np.eye(3) * r], 1))


# ------------------------------------------------------------------------------------------------ the pipeline
def run(m, d, rig, o, cls, keys, base_xy, base_yaw, order=3, seed=0, roi=((0.05, 1.47), (-1.30, 0.80)), zmax=1.55, sgbm=None, eval_pack=None,
        stereo_ev=None):
    """full perception on one stereo capture `o` (rig.capture output incl. gt_*).  Returns a result dict.
    `eval_pack`: optional dict(depth_ev) (for the inflation law).  Timings in res['timing']."""
    T = {}
    rng = np.random.default_rng(seed)
    K = rig.K
    t0 = time.perf_counter()
    disp, t_sgbm = ST.compute_disparity(o["left"], o["right"], sgbm)
    T["sgbm_s"] = t_sgbm
    depth = ST.depth_from_disparity(disp, K)
    origin, R = rig.left_pose(d)
    ev = ST.evaluate(depth, o["gt_depth"]) if stereo_ev is None else stereo_ev
    # ---- cloud, self-filter
    rmask = robot_mask(o["gt_seg"], cls)
    gt_eval = o["gt_depth"].copy()                      # scene-only evaluation: robot body (untextured, self-filtered anyway) and the SGBM margin excluded
    gt_eval[rmask] = np.nan
    gt_eval[:, :int(sgbm.getNumDisparities()) if sgbm is not None else 192] = np.nan
    ev_scene = ST.evaluate(depth, gt_eval)
    P, pix = ST.depth_to_cloud(depth, K, origin, R, mask=~rmask)
    n_raw = len(P)
    # ---- ROI (robot frame) + voxel + outliers
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    Rb = np.array([[c, -s], [s, c]])
    Pr = (P[:, :2] - base_xy) @ Rb                    # robot-frame xy
    sel = (Pr[:, 0] > roi[0][0]) & (Pr[:, 0] < roi[0][1]) & (Pr[:, 1] > roi[1][0]) & (Pr[:, 1] < roi[1][1]) & (P[:, 2] < zmax) & (P[:, 2] > -0.05)
    P, pix = P[sel], pix[sel]
    T["n_roi_raw"] = len(P)
    Pv, pixv = voxel_reduce(P, pix, 0.004)
    ok = radius_filter(Pv, 0.016, 6)
    Pv, pixv = Pv[ok], pixv[ok]
    T["n_after_filters"] = len(Pv)
    # ---- planes
    keep, planes = peel_planes(Pv, rng)
    lab_pl = np.full(len(Pv), -1)
    for i, pl in enumerate(planes):
        lab_pl[pl["idx"]] = i
    Po = Pv[keep]
    lab = SEG.dbscan(Po, 0.018, 8)
    sz = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(0, int)
    small = np.flatnonzero(sz < 60)
    lab[np.isin(lab, small)] = -1
    u_, inv = np.unique(lab[lab >= 0], return_inverse=True)
    lab2 = np.full(len(Po), -1, np.int64)
    lab2[lab >= 0] = inv
    lab = lab2
    ncl = int(lab.max() + 1) if (lab >= 0).any() else 0
    # floor residue (points the floor plane tolerance missed, e.g. around the wheels) is not an object: a cluster must rise >= 10 cm above the floor
    floor_z = next((pl["z"] for pl in planes if pl["kind"] == "horizontal" and pl["z"] < 0.15), 0.0)
    for k in range(ncl):
        if Po[lab == k, 2].max() < floor_z + 0.10:
            lab[lab == k] = -1
    u2, inv2 = np.unique(lab[lab >= 0], return_inverse=True)
    lab3 = np.full(len(Po), -1, np.int64)
    lab3[lab >= 0] = inv2
    lab = lab3
    ncl = int(lab.max() + 1) if (lab >= 0).any() else 0
    T["perceive_s"] = time.perf_counter() - t0
    # ---- GT labels per voxel point (EVAL ONLY)
    gseg = o["gt_seg"][..., 0].ravel()
    gk_pix = np.where(gseg[pixv] >= 0, keys[np.maximum(gseg[pixv], 0)], "none")
    gk_o = gk_pix[keep]
    # ---- zonotopes
    zs = []
    desk_planes = [pl for pl in planes if pl["kind"] == "horizontal" and pl["z"] > 0.4]
    err_fn = lambda z: ST.error_at_range(ev_scene if ev_scene["bins"] else ev, min(z, 1.49), "p95")
    z_desk = max((pl["z"] for pl in desk_planes), key=lambda z: sum(len(p["idx"]) for p in desk_planes if p["z"] == z)) if desk_planes else None
    items = []
    for k in range(ncl):
        Pk = Po[lab == k]
        items.append(("obj", k, Pk))
    for pl in desk_planes:                          # the support surface (desk top) as a thin slab zonotope
        items.append(("support", len([i for i in items if i[0] == "support"]), Pv[pl["idx"]]))
    for kind, k, Pk in items:
        t1 = time.perf_counter()
        Z = fit_hull_zono(Pk, order=order) if kind == "obj" else fit_box(Pk, "up")     # a plane is a thin oriented box (Girard reduction fattens thin sheets)
        rng_z = float(np.linalg.norm(Pk.mean(0) - origin))
        delta = err_fn(rng_z) + INFL_EXTRA
        vdir = Pk.mean(0)[:2] - origin[:2]
        Zx = extrude_unseen(Z, vdir) if kind == "obj" else Z
        if kind == "obj" and z_desk is not None and z_desk - 0.03 < Pk[:, 2].min() < z_desk + 0.30 and Pk[:, 2].mean() > z_desk:
            Zx = drop_to_support(Zx, z_desk)
        Zi = inflate_world(Zx, delta)
        if kind == "obj":
            lk = gk_o[lab == k]
            vals, cnt = np.unique(lk, return_counts=True)
            gk = str(vals[cnt.argmax()])
            purity = float(cnt.max() / cnt.sum())
        else:
            gk, purity = "desk_top", float("nan")
        zs.append(dict(kind=kind, k=int(k), n_pts=int(len(Pk)), Z=Z, Zx=Zx, Zi=Zi, delta=float(delta), range_m=rng_z, gt_key=gk, purity=purity,
                       centre=Z.c.copy(), order=order, fit_s=time.perf_counter() - t1,
                       contains_pts=float(ZUnion([Z]).contains(Pk).mean() if len(Pk) else 1.0),
                       aabb=Zi.interval_hull()))
    T["total_s"] = time.perf_counter() - t0
    return dict(pix_voxel=pixv, z_desk=z_desk, disp=disp, depth=depth, ev=ev, ev_scene=ev_scene, rmask=rmask, origin=origin, R=R, P_voxel=Pv, planes=planes, keep=keep,
                lab_obj=lab, P_obj=Po, plane_lab=lab_pl, zonos=zs, timing=T, gt_pix_keys=gk_pix, n_raw_cloud=n_raw, K=K)


def gt_recovery(res, min_pts=150):
    """object-level recovery vs the GT labels of the cloud points (eval only): for each GT key with >= min_pts voxel points among the NON-plane points,
    best-IoU cluster, recovered if IoU >= 0.5."""
    lab = res["lab_obj"]
    gk = res["gt_pix_keys"][res["keep"]]
    out = {}
    ncl = int(lab.max() + 1) if (lab >= 0).any() else 0
    csz = np.bincount(lab[lab >= 0], minlength=ncl) if ncl else np.zeros(0, int)
    for key in np.unique(gk):
        if key in ("none", "floor", "wall"):
            continue
        mk = gk == key
        if mk.sum() < min_pts:
            continue
        if ncl == 0:
            out[str(key)] = dict(n=int(mk.sum()), iou=0.0, cluster=None, recovered=False)
            continue
        ov = np.bincount(lab[mk & (lab >= 0)], minlength=ncl)
        iou = ov / (mk.sum() + csz - ov)
        c = int(iou.argmax())
        out[str(key)] = dict(n=int(mk.sum()), iou=float(iou[c]), cluster=c, recovered=bool(iou[c] >= 0.5))
    return out


def gt_containment(m, d, keys, zonos, objs, zmax=None):
    """for each recovered object key: fraction of its GT collision-geom sample points (corners / rings / hull vertices) inside the INFLATED zonotope
    union of the matched cluster(s) and the worst outside distance (>= 0).  Eval only."""
    out = {}
    for key in objs:
        gs = [g for g in range(m.ngeom) if keys[g] == key and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_group[g] in (3, 4)]
        if not gs:
            continue
        X = np.concatenate([geom_pts(m, d, g) for g in gs])
        if zmax is not None:
            X = X[X[:, 2] <= zmax]
        zz = [z for z in zonos if z["kind"] == "obj" and z["gt_key"] == key]
        if not zz:
            out[key] = dict(contained=None)
            continue
        U = ZUnion([z["Zi"] for z in zz])
        inside = U.contains(X)
        out[key] = dict(contained=float(inside.mean()), n_pts=int(len(X)), worst_out_m=float(U.outside_depth(X).max()),
                        raw_contained=float(ZUnion([z["Zx"] for z in zz]).contains(X).mean()))
    return out
