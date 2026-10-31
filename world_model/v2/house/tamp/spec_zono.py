"""tamp/spec_zono.py -- STEP 2 of the spec certifier: static obstacles as PERCEIVED ZONOTOPES instead of the occupancy grid.  SIMULATION ONLY (CPU ray casting, no GL / GPU).

    python -m world_model.v2.house.tamp.spec_zono perceive [--est-pose] [--out DIR]     # head-ZED ray-cast mapping pass -> cloud -> zonotopes -> DIR/h5_zonos.npz (+ meta json)
    python -m world_model.v2.house.tamp.spec_zono compare [--zonos NPZ]                   # zonotope clearance vs the TRUE occupancy grid on route + random poses

PIPELINE (all existing components; the split rule is the only new logic)
  sensing   loc.mapper.HeadDepth (mj_multiRay on the static house, the depth_sim.CamSpec noise law: sigma_z(z), grazing mask, dropout, outliers, quantisation; z <= 4 m) along
            loc.mapper.exploration_path (the scripted mapping pass: drive the out_route there and back, spin 360 deg every metre and at every waypoint, 2 frames/s).  Points are
            placed with the TRUE pose (default; isolates perception + fit error) or the ESTIMATED pose of loc.estimator.PoseEstimator at map_loc_scale 0.5 (--est-pose; adds drift).
  filter    hits with height in [0.15, 1.15] m (the mapper's obstacle band), 2 cm voxels, voxels seen >= MIN_HITS times (isolated stereo outliers die here).
  cluster   8-connected components of the 4 cm xy projection, clusters < MIN_CLUSTER voxels dropped, only within BAND m of the route.
  split     recursive PCA bisection of every cluster in xy until a piece is straight (perpendicular half-extent <= MAX_DEV) and short (<= MAX_LEN): a house wall is a long thin
            slab; one hull zonotope of a whole L / U shaped wall system would fill the room.
  fit       perception_zono.fit.fit_hull_zono (outer zonotope of conv(points), order 2, gravity-up frame) per piece (fallback: oriented box if the LP returns a non-containing /
            oversized set; < 8 points: their bounding box), inflated by half a voxel (points are voxel centres).  CONTAINS every filtered voxel cell; unseen surface is NOT covered.
COMPARISON (compare_with_grid): for a pose list, the footprint rectangle's clearance to the zonotopes (EXACT signed distance, ir.sd_exact; the footprint is the prism FP x [0.15, 1.15])
against the TRUE planner grid (teleop/scene.load_occupancy: nav.build_occupancy, 1 cm, EDT to the nearest occupied cell centre; the min over a 1 cm sampling of the footprint boundary).
  false-safe   zonotopes clear by >= d_wall  AND  true grid clearance < d_wall (within d_wall or colliding)       <- the headline (dangerous direction)
  false-unsafe zonotopes < d_wall  AND  true grid >= d_wall (needless refusal)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.zono import Zonotope

from ..teleop import geom as GM
from . import spec_cert as SC

OUT = Path("/data/local/ss5772/molmospaces/spec_cert")
VOX = 0.02
MIN_HITS = 3
XY_CELL = 0.04
MIN_CLUSTER = 12
BAND = 3.0
MAX_LEN = 1.2
MAX_DEV = 0.06
MIN_PIECE = 25                       # voxels


# ====================================================================================== perception
def perceive_cloud(scene_name="h5", est_pose=False, seed=0, frame_hz=2.0, verbose=True):
    """-> (voxel centres (n,3), hit counts (n,), meta).  See the module docstring."""
    import mujoco
    from world_model.v2.house import build as B
    from ..loc import mapper as MP
    from ..loc.estimator import LocCfg, PoseEstimator, scaled
    from ..teleop import scene as SCN
    sc = SCN.get_scene(scene_name)
    cfg = MP.MapCfg(seed=seed, frame_hz=frame_hz)
    spec = sc.spec_loader() if sc.spec_loader else B.load_house_spec(sc.house, True)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = MP.HeadDepth(m, d, cfg, seed=seed)
    route = np.asarray(sc.out_route, float)
    poses = MP.exploration_path(route, cfg)
    est = PoseEstimator(scaled(LocCfg(), cfg.map_loc_scale), seed=cfg.mapping_loc_seed, true_pose0=poses[0]) if est_pose else None
    every = int(round(1.0 / (cfg.frame_hz * 0.02)))
    t0 = time.time()
    keys = np.zeros((0, 3), np.int32)
    cnt = np.zeros(0, np.int32)
    buf = []
    nf = 0
    for k, p in enumerate(poses[1:], 1):
        if est is not None:
            est.update(k * 0.02, p, 0.02)
        if k % every:
            continue
        ok, pts, nofree, _ = cam.capture(p)
        o, ex, ey, ez, hit, fx, fy, fz = cam.place(ok, pts, nofree, est.pose if est is not None else p)
        sel = hit & (ez >= cfg.obs_zlo) & (ez <= cfg.obs_zhi)
        buf.append(np.floor(np.stack([ex[sel], ey[sel], ez[sel]], 1) / VOX).astype(np.int32))
        nf += 1
        if len(buf) >= 25:
            keys, cnt = _merge(keys, cnt, buf)
            buf = []
    keys, cnt = _merge(keys, cnt, buf)
    meta = dict(frames=nf, wall_s=time.time() - t0, est_pose=bool(est_pose), seed=seed, frame_hz=frame_hz, route=route.tolist(), path_poses=len(poses),
                noise="loc.mapper.HeadDepth / depth_sim.CamSpec law", band=[cfg.obs_zlo, cfg.obs_zhi], vox=VOX, min_hits=MIN_HITS,
                map_ate_m=(est.errors()["ate_rmse_m"] if est is not None else 0.0))
    keep = cnt >= MIN_HITS
    P = (keys[keep].astype(float) + 0.5) * VOX
    if verbose:
        print(f"[perceive] {nf} frames, {len(keys)} voxels, {int(keep.sum())} with >= {MIN_HITS} hits, {meta['wall_s']:.0f} s", flush=True)
    return P, cnt[keep], meta


def _merge(keys, cnt, buf):
    if not buf:
        return keys, cnt
    K = np.concatenate([keys] + buf)
    C = np.concatenate([cnt] + [np.ones(len(b), np.int32) for b in buf])
    u, inv = np.unique(K, axis=0, return_inverse=True)
    c = np.zeros(len(u), np.int64)
    np.add.at(c, inv.ravel(), C)
    return u, c.astype(np.int32)


# ====================================================================================== cluster / split / fit
def _near_route(P, route, band):
    from scipy.spatial import cKDTree
    dense = np.concatenate([route[i] + (route[i + 1] - route[i]) * np.linspace(0, 1, max(int(np.hypot(*(route[i + 1] - route[i])) / 0.1), 2))[:, None]
                            for i in range(len(route) - 1)])
    return cKDTree(dense).query(P[:, :2])[0] <= band


def clusters_xy(P):
    """8-connected components of the xy projection -> list of index arrays."""
    ij = np.floor((P[:, :2] - P[:, :2].min(0)) / XY_CELL).astype(int)
    img = np.zeros(ij.max(0)[::-1] + 1, bool)
    img[ij[:, 1], ij[:, 0]] = True
    lab, n = ndimage.label(img, structure=np.ones((3, 3), int))
    L = lab[ij[:, 1], ij[:, 0]]
    return [np.nonzero(L == k)[0] for k in range(1, n + 1)]


def split_pieces(P2, idx, max_len=MAX_LEN, max_dev=MAX_DEV, min_piece=MIN_PIECE):
    """recursive PCA bisection in xy of the points idx -> list of index arrays (straight and short pieces)."""
    out, stack = [], [idx]
    while stack:
        S = stack.pop()
        X = P2[S]
        mu = X.mean(0)
        if len(S) == 1:
            out.append(S)
            continue
        w, V = np.linalg.eigh(np.cov((X - mu).T))
        u = V[:, 1]
        t = (X - mu) @ u
        dev = np.abs((X - mu) @ V[:, 0]).max()
        ext = t.max() - t.min()
        if (ext <= max_len and (dev <= max_dev or len(S) < 2 * min_piece)):      # small sets: only the length matters (a sparse spread-out set is split, not boxed)
            out.append(S)
            continue
        o = np.argsort(t, kind="stable")                                  # split by RANK (voxel columns share an xy: ties would leave one side empty)
        stack += [S[o[:len(S) // 2]], S[o[len(S) // 2:]]]
    return out


def fit_obstacles(P, route, band=BAND, verbose=True):
    """voxel centres P (n,3) -> list of Zonotope (world frame) + info."""
    from world_model.v2.perception_zono import fit as FIT
    t0 = time.time()
    near = _near_route(P, np.asarray(route, float), band)
    P = P[near]
    cl = [c for c in clusters_xy(P) if len(c) >= MIN_CLUSTER]
    pieces = []
    for c in cl:
        pieces += split_pieces(P[:, :2], c)
    zs, n_fb = [], 0
    for p in pieces:
        Q = P[p]
        if len(Q) < 8:                                                       # too few points for a hull: their axis-aligned box (fit_box would return a 1 mm cube)
            Z = Zonotope.from_box(Q.min(0), Q.max(0))
        else:
            Z = FIT.fit_hull_zono(Q, order=2, frame="up")
            if not _sane(Z, Q):                                              # the hull LP / order reduction occasionally returns a huge or non-containing set
                Z, n_fb = FIT.fit_box(Q, frame="up"), n_fb + 1               # oriented min-area box: always contains the piece
        zs.append(FIT.inflate(Z, 0.5 * VOX))                                 # the points are voxel CENTRES: cover the voxel cells (half a voxel)
    info = dict(n_box_fallbacks=n_fb, n_points=int(len(P)), n_clusters=len(cl), n_pieces=len(pieces), fit_s=time.time() - t0, band=band, max_len=MAX_LEN, max_dev=MAX_DEV)
    if verbose:
        print(f"[fit] {len(P)} pts in band, {len(cl)} clusters, {len(zs)} zonotopes, {info['fit_s']:.1f} s", flush=True)
    return zs, info, P, pieces


def _sane(Z, Q, ratio=4.0, tol=1e-6):
    """Z contains all points Q (to tol) and its xy bounding box is at most `ratio` times the points' (plus 2 cm)."""
    from world_model.v2.perception_zono.zonoutil import halfspaces
    lo, hi = Z.interval_hull()
    if np.prod(hi[:2] - lo[:2] + 0.02) > ratio * np.prod(Q[:, :2].max(0) - Q[:, :2].min(0) + 0.02):
        return False
    A, b = halfspaces(Z)
    return bool((b[None] - Q @ A.T).min() >= -tol)


def containment(zs, P, pieces, tol=1e-6):
    """fraction of the filtered points inside their own zonotope (should be 1.0: the hull LP is an outer approximation)."""
    from world_model.v2.perception_zono.zonoutil import halfspaces
    tot = ins = 0
    for Z, p in zip(zs, pieces):
        A, b = halfspaces(Z)
        s = b[None] - P[p] @ A.T
        ins += int((s.min(1) >= -tol).sum())
        tot += len(p)
    return ins / max(tot, 1)


def save_zonos(path, zs, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {}
    for i, Z in enumerate(zs):
        arr[f"c{i}"], arr[f"G{i}"] = Z.c, Z.G
    np.savez(path, n=len(zs), **arr)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))


def load_zonos(path):
    z = np.load(path)
    return [Zonotope(z[f"c{i}"], z[f"G{i}"]) for i in range(int(z["n"]))]


# ====================================================================================== comparison with the true grid
def grid_clearance(grid, poses, fp=None, step=0.01):
    """TRUE planner grid: min over a `step` sampling of the footprint boundary of the EDT [m] at each pose (x, y, yaw) -> (n,)."""
    fp = GM.FP if fp is None else fp
    B = GM.boundary_pts(fp, step=step)
    P = np.asarray(poses, float)
    c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
    X = P[:, 0:1] + c * B[None, :, 0] - s * B[None, :, 1]
    Y = P[:, 1:2] + s * B[None, :, 0] + c * B[None, :, 1]
    return grid.lookup(X, Y).min(1)


def zono_clearance(zonos, poses, fp=None, far=1.0):
    """signed distance of the footprint prism FP x [Z_LO, Z_HI] at each pose to the nearest zonotope (exact, ir.sd_exact; > `far` is a lower bound) -> (n,)."""
    fp = GM.FP if fp is None else fp
    half = np.array([0.5 * (fp["x"][1] - fp["x"][0]), 0.5 * (fp["y"][1] - fp["y"][0]), SC.HZ])
    cen = np.array([0.5 * (fp["x"][1] + fp["x"][0]), 0.5 * (fp["y"][1] + fp["y"][0]), SC.PZ])
    P = np.asarray(poses, float)
    R = SC._rz(P[:, 2])
    pos = np.stack([P[:, 0], P[:, 1], np.zeros(len(P))], 1)
    ZA = Zonotope(cen, np.diag(half)).linear(R, pos)
    best = np.full(len(P), np.inf)
    for Z in zonos:
        best = np.minimum(best, ir.sd_exact_batch(ZA.minkowski(Z.neg()), far))
    return best


def compare_with_grid(zonos, grid, poses, d_wall=0.10, fp=None, name=""):
    """-> dict(n, false_safe, false_unsafe, worst_false_safe, ...) -- see the module docstring."""
    cg = grid_clearance(grid, poses, fp)
    cz = zono_clearance(zonos, poses, fp)
    fs = (cz >= d_wall) & (cg < d_wall)
    fu = (cz < d_wall) & (cg >= d_wall)
    near = (cg < 0.6) | (cz < 0.6)
    e = (cz - cg)[near & (cg > 0.0)]
    out = dict(name=name, n=len(cg), false_safe=int(fs.sum()), false_unsafe=int(fu.sum()), both_safe=int(((cz >= d_wall) & (cg >= d_wall)).sum()),
               both_unsafe=int(((cz < d_wall) & (cg < d_wall)).sum()), colliding_true=int((cg <= 0.0).sum()),
               worst_false_safe_m=float((cg - d_wall)[fs].min()) if fs.any() else None,
               worst_false_safe_zono_minus_grid_m=float((cz - cg)[fs].max()) if fs.any() else None,
               err_zono_minus_grid_near_m=dict(n=int(len(e)), mean=float(e.mean()) if len(e) else None, p05=float(np.percentile(e, 5)) if len(e) else None,
                                              p95=float(np.percentile(e, 95)) if len(e) else None, min=float(e.min()) if len(e) else None, max=float(e.max()) if len(e) else None))
    out["false_safe_idx"] = np.nonzero(fs)[0].tolist()
    return out, cg, cz


def route_poses(route, spacing=0.2, yaw_end=None):
    """poses every `spacing` m along a polyline with the heading of the path tangent (the last segment's heading at the end)."""
    R = np.asarray(route, float)
    out = []
    for a, b in zip(R[:-1], R[1:]):
        L = float(np.hypot(*(b - a)))
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in np.arange(0.0, L - 1e-9, spacing):
            q = a + (b - a) * s / L
            out.append((q[0], q[1], h))
    out.append((R[-1][0], R[-1][1], math.atan2(*(R[-1] - R[-2])[::-1]) if yaw_end is None else yaw_end))
    return np.array(out)


def random_corridor_poses(n=200, seed=0):
    """n random poses in the h5 corridor / doorway region: x in [1.8, 2.5], y in [2.0, 5.3], half with yaw near -pi/2 (+-0.5), half uniform."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(1.8, 2.5, n), rng.uniform(2.0, 5.3, n)
    yaw = np.where(rng.random(n) < 0.5, -math.pi / 2 + rng.uniform(-0.5, 0.5, n), rng.uniform(-math.pi, math.pi, n))
    return np.stack([x, y, yaw], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["perceive", "refit", "compare"])
    ap.add_argument("--est-pose", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--zonos", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    tag = "_est" if a.est_pose else ""
    from ..teleop import scene as SCN
    sc = SCN.get_scene("h5")
    if a.cmd in ("perceive", "refit"):
        if a.cmd == "perceive":
            P, cnt, meta = perceive_cloud("h5", a.est_pose, a.seed)
        else:                                                                # re-run cluster / split / fit on the saved cloud (float32 voxel centres)
            P, cnt = (lambda z: (z["P"].astype(float), z["cnt"]))(np.load(out / f"h5_cloud{tag}.npz"))
            meta = json.loads((out / f"h5_zonos{tag}.json").read_text())
        zs, info, Pb, pieces = fit_obstacles(P, sc.out_route)
        meta.update(info)
        meta["containment"] = containment(zs, Pb, pieces)
        meta["gen_per_zono_mean"] = float(np.mean([Z.n_gen for Z in zs]))
        save_zonos(out / f"h5_zonos{tag}.npz", zs, meta)
        if a.cmd == "perceive":
            np.savez_compressed(out / f"h5_cloud{tag}.npz", P=P.astype(np.float32), cnt=cnt)
        print(json.dumps({k: v for k, v in meta.items() if k != "route"}, default=float), flush=True)
        return
    zs = load_zonos(a.zonos or out / f"h5_zonos{tag}.npz")
    grid = SCN.load_occupancy(sc)
    from ..teleop.scene import OccGrid
    sets = {"out_route_0.2m": route_poses(sc.out_route, 0.2),
            "seed1_standoff_route": route_poses([[2.14, 3.56], [2.135, 2.0]], 0.2, yaw_end=-math.pi / 2),
            "random_corridor_200": random_corridor_poses(200, a.seed)}
    res = {}
    for k, P in sets.items():
        r, cg, cz = compare_with_grid(zs, grid, P, name=k)
        res[k] = r
        print(json.dumps(r, default=float), flush=True)
    (out / f"h5_zonos_compare{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
