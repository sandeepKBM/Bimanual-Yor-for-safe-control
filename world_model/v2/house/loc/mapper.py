"""loc/mapper.py -- perceived occupancy grid from a head-ZED mapping pass.  SIMULATION ONLY (CPU ray casting, no GL, no ZED SDK, no network).

What the real robot has: YOR/robot/slam_node_.py builds a map from the ZED.  We do not know its parameters; this is a stand-in with documented,
configurable assumptions (MapCfg):
  * a SCRIPTED EXPLORATION drives the nominal route (lab -> doors -> cafe counter) and spins 360 deg in place every `spin_every_m` metres and at every
    waypoint (the forward-looking ZED only sees what it faces), 4 frames/s;
  * the head ZED (torso mount (0.05, -0.20955, 1.15), pitched down; the PROPOSED mount of zed_demo/rig.py) is a depth_sim.CamSpec: stereo noise law
    sigma_z = sqrt(floor^2 + (z^2/(f_native*baseline)*sigma_disp)^2), grazing mask, dropout, outliers, quantisation -- the same model as
    perception_bridge / depth_sim (z_min 0.3 m, z_max shortened to `z_map_max` for mapping);
  * each frame's points are placed with the ESTIMATED base pose at the frame time (loc.estimator.PoseEstimator driven along the same route) =>
    the map inherits the localization drift (the map is in the estimated-pose frame of the mapping run, aligned with the world at the start pose);
  * hits with height in [obs_zlo, obs_zhi] (above the noisy floor, below the robot top) mark the cell occupied; every ray clears the cells it
    crosses while its height is inside the band; log-odds update (l_occ / l_free, clamped); unknown cells (never updated) -> free (`unknown`).
Truth enters only as the geometry the rays hit and as the evaluation reference (map_error).  The obstacle band / z-threshold mirrors nav.build_occupancy
(Z_BASE 0.04 .. ROBOT_TOP 1.15) except the lower edge is raised to `obs_zlo` because depth noise at several metres exceeds 4 cm (a real pipeline needs
a ground-plane removal margin) -> low obstacles (thresholds, rugs, < obs_zlo) are invisible to the perceived map: reported as false-free.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import mujoco
from numba import njit
from scipy import ndimage

from .estimator import LocCfg, PoseEstimator, compose, wrap

CACHE = Path("data/molmospaces/percept/loc/maps")


@dataclass
class MapCfg:
    res: float = 0.02
    # camera (robot frame, lift 0) -- PROPOSED torso mount (zed_demo/rig.py HEAD_POS), confirm on hardware
    mount: tuple = (0.05, -0.20955, 1.15)
    pitch_deg: float = 30.0
    fovy: float = 70.0
    W: int = 256
    H: int = 144
    z_map_max: float = 4.0
    # obstacle band
    obs_zlo: float = 0.15
    obs_zhi: float = 1.15
    # log-odds
    dz: float = 0.05                       # height bin of the 3-D log-odds grid (projected to 2-D: a column is occupied if ANY bin is)
    l_occ: float = 0.60
    l_free: float = -0.40
    l_clamp: float = 4.0
    occ_thresh: float = 0.5
    trunc_k: float = 2.0                   # free-space trace stops k * sigma_z(z) before the hit (stereo noise would otherwise erase thin obstacles)
    despeckle: bool = True                 # drop occupied cells with no occupied 8-neighbour (stereo speckle)
    close_iters: int = 1                   # 3x3 binary closing of the occupied set (bridges the ray gaps between wall hits)
    unknown: str = "free"                  # never-observed cells: "free" | "occ"
    # exploration script
    frame_hz: float = 2.0
    speed: float = 0.4
    spin_rate: float = 0.8
    spin_every_m: float = 1.0
    mapping_loc_seed: int = 777
    map_loc_scale: float = 0.5             # the mapping pass is localized better than online driving (scan-matching / loop closure of a SLAM backend, offline):
                                           # LocCfg error magnitudes x this.  ASSUMPTION; 1.0 = raw online estimator (blurred map: see report), 0 = perfect poses
    seed: int = 0

    def asdict(self):
        return asdict(self)


@dataclass
class PerceivedMap:
    res: float
    x0: float
    y0: float
    logodds: np.ndarray
    cfg: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def occ_mask(self, cfg: MapCfg | None = None):
        cfg = cfg or MapCfg(**{k: v for k, v in self.cfg.items() if k in MapCfg.__dataclass_fields__})
        occ = (self.logodds > cfg.occ_thresh).any(axis=0)                       # max-projection over the height bins
        if cfg.unknown == "occ":
            occ = occ | (self.logodds == 0.0).all(axis=0)
        if cfg.despeckle:
            nb = ndimage.convolve(occ.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - occ
            occ = occ & (nb > 0)
        if cfg.close_iters:
            occ = ndimage.binary_closing(occ, structure=np.ones((3, 3), bool), iterations=cfg.close_iters)
        return occ

    def to_occgrid(self, cfg: MapCfg | None = None):
        from ..teleop.scene import OccGrid
        occ = self.occ_mask(cfg)
        return OccGrid(self.res, self.x0, self.y0, occ, ndimage.distance_transform_edt(~occ) * self.res)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, res=self.res, x0=self.x0, y0=self.y0, logodds=self.logodds.astype(np.float32),
                            cfg=np.array(repr(self.cfg)), meta=np.array(repr(self.meta)))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        import ast
        return PerceivedMap(float(z["res"]), float(z["x0"]), float(z["y0"]), z["logodds"], ast.literal_eval(str(z["cfg"])),
                            ast.literal_eval(str(z["meta"])))


# ------------------------------------------------------------------------------------------------ log-odds integration (numba)
@njit(cache=True)
def _integrate(L, res, x0, y0, ox, oy, oz, ex, ey, ez, hit, zlo, zhi, l_occ, l_free, clamp, fx=None, fy=None, fz=None, dz=0.05):
    nz, ny, nx = L.shape
    for i in range(ex.shape[0]):
        # free-space trace ends at (fx, fy, fz) (the hit point pulled back by k sigma_z, voxblox-style truncation) -- default: the endpoint itself
        tx = ex[i] if fx is None else fx[i]
        ty = ey[i] if fy is None else fy[i]
        tz = ez[i] if fz is None else fz[i]
        ddx, ddy, ddz = tx - ox, ty - oy, tz - oz
        d2 = math.hypot(ddx, ddy)
        n = int(d2 / (0.5 * res)) + 1
        last_ix, last_iy, last_iz = -1, -1, -1
        for k in range(n):
            f = (k + 0.5) / n
            z = oz + f * ddz
            if z < zlo or z > zhi:
                continue
            ix = int((ox + f * ddx - x0) / res)
            iy = int((oy + f * ddy - y0) / res)
            if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                continue
            iz = int((z - zlo) / dz)
            if iz >= nz:
                iz = nz - 1
            if ix == last_ix and iy == last_iy and iz == last_iz:
                continue
            last_ix, last_iy, last_iz = ix, iy, iz
            if fx is None and hit[i]:                       # no truncation: do not clear the endpoint cell of a hit (it is handled below)
                eix = int((ex[i] - x0) / res)
                eiy = int((ey[i] - y0) / res)
                if ix == eix and iy == eiy:
                    continue
            v = L[iz, iy, ix] + l_free
            L[iz, iy, ix] = v if v > -clamp else -clamp
        if hit[i] and ez[i] >= zlo and ez[i] <= zhi:
            ix = int((ex[i] - x0) / res)
            iy = int((ey[i] - y0) / res)
            iz = int((ez[i] - zlo) / dz)
            if iz >= nz:
                iz = nz - 1
            if 0 <= ix < nx and 0 <= iy < ny:
                v = L[iz, iy, ix] + l_occ
                L[iz, iy, ix] = v if v < clamp else clamp


# ------------------------------------------------------------------------------------------------ sensor
class HeadDepth:
    """ray-cast ZED-like depth on a (static) MuJoCo model; noise = depth_sim.CamSpec law.  Returns endpoints in the camera/base frame so the caller can
    place them with ANY pose (the estimate)."""

    def __init__(self, m, d, cfg: MapCfg, seed=0):
        from world_model.v2.tray_rep.perception import depth_sim as D
        self.m, self.d, self.cfg = m, d, cfg
        self.spec = D.CamSpec(name="zed_map", mount="world", pos=(0, 0, 0), R=tuple(np.eye(3).ravel()), fovy=cfg.fovy, W=cfg.W, H=cfg.H, z_max=cfg.z_map_max)
        W, H = cfg.W, cfg.H
        f = (H / 2.0) / math.tan(math.radians(cfg.fovy) / 2.0)
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        self.xn, self.yn = ((u - W / 2) / f).ravel(), ((v - H / 2) / f).ravel()          # image right / down, per unit z
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4242]))
        self.mask = np.array([0, 0, 0, 1, 1, 0], np.uint8)        # house collision hulls + floor/walls (same groups as nav.build_occupancy)
        self.geom = np.full(W * H, -1, np.int32)
        self.dist = np.full(W * H, -1.0)
        self.nrm = np.zeros(3 * W * H)

    def axes(self, pose, mount=None):
        """camera origin + (forward, left, up) world unit axes for a base pose (x, y, yaw)."""
        c = self.cfg
        mx, my, mz = c.mount if mount is None else mount
        cy, sy = math.cos(pose[2]), math.sin(pose[2])
        o = np.array([pose[0] + cy * mx - sy * my, pose[1] + sy * mx + cy * my, mz])
        p = math.radians(c.pitch_deg)
        fwd = np.array([math.cos(p) * cy, math.cos(p) * sy, -math.sin(p)])
        left = np.array([-sy, cy, 0.0])
        up = np.array([math.sin(p) * cy, math.sin(p) * sy, math.cos(p)])
        return o, fwd, left, up

    def capture(self, true_pose):
        """-> (valid (N,), pts_cam (N,3) = (fwd z-depth, left, up) coordinates of the NOISY points in the camera axes, no_return_free (N,) bool for
        rays that returned nothing within range (free along the ray to z_map_max*0.9)).  Rays are cast from the TRUE camera pose."""
        s, c = self.spec, self.cfg
        o, fwd, left, up = self.axes(true_pose)
        # image right = -left ; image down = -up
        dirs = fwd[None] + self.xn[:, None] * (-left)[None] + self.yn[:, None] * (-up)[None]
        nz = np.linalg.norm(dirs, axis=1)
        dirs_u = np.ascontiguousarray(dirs / nz[:, None])
        n = len(dirs_u)
        self.geom[:] = -1; self.dist[:] = -1.0
        mujoco.mj_multiRay(self.m, self.d, o, dirs_u.ravel(), self.mask, True, -1, self.geom, self.dist, self.nrm, n, s.z_max * 1.2)
        hit = (self.geom >= 0) & (self.dist > 0)
        z = self.dist * (1.0 / nz)                            # z-depth along the optical axis (dirs have unit forward component)
        rng = self.rng
        e = rng.normal(size=n)
        zn = z + s.sigma_z(z) * e
        u_out, u_drop = rng.random(n), rng.random(n)
        zn = np.where(u_out < s.p_out, z + rng.uniform(0.05, 0.20, n) * np.where(rng.random(n) < 0.5, -1, 1), zn)
        zn = np.round(zn / s.quant_m) * s.quant_m
        cosinc = np.abs(np.einsum("ij,ij->i", self.nrm.reshape(-1, 3), dirs_u))
        ok = hit & (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        nofree = (~hit) & (u_drop >= s.p_drop)
        # camera-axes coordinates (fwd, left, up) of the noisy point
        pts = np.stack([zn, -self.xn * zn, -self.yn * zn], 1)
        return ok, pts, nofree, (o, fwd, left, up)

    def place(self, ok, pts, nofree, est_pose):
        """world endpoints of the returns using the ESTIMATED base pose -> (origin, ex, ey, ez, hit) arrays for _integrate."""
        o, fwd, left, up = self.axes(est_pose)
        P = o[None] + pts[:, :1] * fwd[None] + pts[:, 1:2] * left[None] + pts[:, 2:3] * up[None]
        zmax = self.spec.z_max * 0.9
        sel = ok | nofree
        # a no-return ray: free along the unit ray direction to zmax (reconstruct its direction from the pixel)
        Pn = o[None] + zmax * (fwd[None] + self.xn[:, None] * (-left)[None] + self.yn[:, None] * (-up)[None]) / np.linalg.norm(
            fwd[None] + self.xn[:, None] * (-left)[None] + self.yn[:, None] * (-up)[None], axis=1)[:, None]
        E = np.where(ok[:, None], P, Pn)
        # free-trace end: pull the hit back along the ray by k sigma_z (no-return rays: full range)
        rng_ = np.linalg.norm(E - o[None], axis=1) + 1e-9
        pull = np.where(ok, self.cfg.trunc_k * self.spec.sigma_z(np.maximum(pts[:, 0], 0.0)), 0.0)
        F = o[None] + (E - o[None]) * (np.maximum(rng_ - pull, 0.0) / rng_)[:, None]
        return o, E[sel, 0].copy(), E[sel, 1].copy(), E[sel, 2].copy(), ok[sel].copy(), F[sel, 0].copy(), F[sel, 1].copy(), F[sel, 2].copy()


# ------------------------------------------------------------------------------------------------ exploration script
def exploration_path(route_xy, cfg: MapCfg, tick=0.02):
    """true base poses (N,3) at 1/tick Hz: drive the polyline forward-facing at cfg.speed, spin 360 deg at the start, every spin_every_m metres and at
    every waypoint, then drive it back (reversed, facing the way of travel)."""
    pts = np.asarray(route_xy, float)
    poses = []

    def drive(poly):
        nonlocal poses
        yaw = None
        s_since = 0.0
        p = None
        for a, b in zip(poly[:-1], poly[1:]):
            L = float(np.hypot(*(b - a)))
            if L < 1e-6:
                continue
            h = math.atan2(*(b - a)[::-1])
            if yaw is None or abs(wrap(h - yaw)) > 1e-3:           # turn in place to the new heading
                y0 = h if yaw is None else yaw
                n = int(abs(wrap(h - y0)) / (cfg.spin_rate * tick)) + 1
                for k in range(1, n + 1):
                    poses.append((a[0], a[1], wrap(y0 + wrap(h - y0) * k / n)))
                yaw = h
            n = max(int(L / (cfg.speed * tick)), 1)
            for k in range(1, n + 1):
                q = a + (b - a) * k / n
                poses.append((q[0], q[1], yaw))
                s_since += L / n
                if s_since >= cfg.spin_every_m:
                    s_since = 0.0
                    spin(q[0], q[1], yaw)
            spin(b[0], b[1], yaw)
            s_since = 0.0

    def spin(x, y, yaw):
        n = int(2 * math.pi / (cfg.spin_rate * tick))
        for k in range(1, n + 1):
            poses.append((x, y, wrap(yaw + 2 * math.pi * k / n)))

    spin(pts[0, 0], pts[0, 1], math.atan2(*(pts[1] - pts[0])[::-1]))
    drive(pts)
    drive(pts[::-1])
    return np.array(poses)


def build_map(scene, cfg: MapCfg | None = None, loc_cfg: LocCfg | None = None, verbose=True) -> PerceivedMap:
    """run the scripted mapping pass for `scene` (teleop.scene.SceneSpec) and return the PerceivedMap."""
    from world_model.v2.house import build as B
    from ..teleop import scene as SC
    cfg = cfg or MapCfg()
    t0 = time.time()
    spec = scene.spec_loader() if scene.spec_loader else B.load_house_spec(scene.house, True)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    true = SC.load_occupancy(scene)
    ny, nx = int(round(true.occ.shape[0] * true.res / cfg.res)), int(round(true.occ.shape[1] * true.res / cfg.res))
    L = np.zeros((int(math.ceil((cfg.obs_zhi - cfg.obs_zlo) / cfg.dz)), ny, nx), np.float32)
    cam = HeadDepth(m, d, cfg, seed=cfg.seed)
    route = np.asarray(scene.out_route, float)
    start = scene.waypoints["lab_start"]
    poses = exploration_path(route, cfg)
    from .estimator import scaled
    est = PoseEstimator(scaled(loc_cfg or LocCfg(), cfg.map_loc_scale), seed=cfg.mapping_loc_seed, true_pose0=poses[0])
    every = int(round(1.0 / (cfg.frame_hz * 0.02)))
    nf = 0
    for k, p in enumerate(poses[1:], 1):
        est.update(k * 0.02, p, 0.02)
        if k % every:
            continue
        ok, pts, nofree, _ = cam.capture(p)
        o, ex, ey, ez, hit, fx, fy, fz = cam.place(ok, pts, nofree, est.pose)
        _integrate(L, cfg.res, true.x0, true.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi,
                   cfg.l_occ, cfg.l_free, cfg.l_clamp, fx, fy, fz, cfg.dz)
        nf += 1
    meta = dict(frames=nf, duration_s=len(poses) * 0.02, path_m=float(np.hypot(*np.diff(poses[:, :2], axis=0).T).sum()), wall_s=time.time() - t0,
                map_ate=est.errors(), loc_cfg=(loc_cfg or LocCfg()).asdict(), route=route.tolist())
    if verbose:
        print(f"[mapper] {nf} frames, {meta['duration_s']:.0f} s sim, {meta['wall_s']:.0f} s wall, pose ATE rms {meta['map_ate']['ate_rmse_m']*100:.1f} cm", flush=True)
    return PerceivedMap(cfg.res, true.x0, true.y0, L, cfg.asdict(), meta)


def true_grid_zband(scene, cfg: MapCfg | None = None, use_cache=True):
    """TRUE obstacle grid at the mapper's resolution restricted to what the perceiver is meant to see: a cell is occupied if some house collision geom
    whose TOP is above cfg.obs_zlo is hit below cfg.obs_zhi (nav.build_occupancy's rule with Z_BASE raised from 0.04 to obs_zlo).  Cells that are
    obstacles only for the standard grid (door sills, rugs, 4..15 cm) are NOT in it -> they are the 'low obstacles' the mapper cannot see by design."""
    from world_model.v2.house import build as B, nav as N
    from ..teleop import scene as SC
    from ..teleop.scene import OccGrid
    cfg = cfg or MapCfg()
    f = CACHE / f"truth_zband_{scene.name}_{cfg.res:.3f}_{cfg.obs_zlo:.2f}_{cfg.obs_zhi:.2f}.npz"
    ref = SC.load_occupancy(scene)
    if use_cache and f.exists():
        occ = np.load(f)["occ"]
    else:
        spec = scene.spec_loader() if scene.spec_loader else B.load_house_spec(scene.house, True)
        m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        top = N._geom_top(m, d)
        ny, nx = int(round(ref.occ.shape[0] * ref.res / cfg.res)), int(round(ref.occ.shape[1] * ref.res / cfg.res))
        occ = np.zeros((ny, nx), bool)
        mask = np.array([0, 0, 0, 1, 1, 0], np.uint8)
        up = np.array([0.0, 0.0, 1.0]); gid = np.zeros(1, np.int32)
        for iy in range(ny):
            y = ref.y0 + (iy + 0.5) * cfg.res
            for ix in range(nx):
                x = ref.x0 + (ix + 0.5) * cfg.res
                z0 = -0.3
                for _ in range(6):
                    dist = mujoco.mj_ray(m, d, np.array([x, y, z0]), up, mask, 1, -1, gid)
                    if dist < 0:
                        break
                    zh = z0 + dist
                    if zh >= cfg.obs_zhi:
                        break
                    if top[int(gid[0])] > cfg.obs_zlo:
                        occ[iy, ix] = True
                        break
                    z0 = zh + 0.002
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, occ=occ)
    return OccGrid(cfg.res, ref.x0, ref.y0, occ, ndimage.distance_transform_edt(~occ) * cfg.res)


def map_path(scene, cfg: MapCfg | None = None, tag=""):
    return CACHE / f"perceived_{scene.name}{tag}.npz"


def resolve_occgrid(scene, occ_map=None):
    """OccGrid for the shield from: None (cache/build default), npz path, PerceivedMap, or OccGrid."""
    from ..teleop.scene import OccGrid
    if isinstance(occ_map, OccGrid):
        return occ_map
    if isinstance(occ_map, PerceivedMap):
        return occ_map.to_occgrid()
    if occ_map is None:
        f = map_path(scene)
        if not f.exists():
            build_map(scene).save(f)
        occ_map = f
    return PerceivedMap.load(occ_map).to_occgrid()


# ------------------------------------------------------------------------------------------------ evaluation vs truth
def map_error(perc, true, route_xy, band=1.0, tol=(0.03, 0.05), n_fp=None):
    """perceived OccGrid vs true OccGrid, restricted to cells within `band` metres of the route (the part of the map the robot can use).
       recall_tol / false_free      : true-occupied cells whose nearest PERCEIVED occupied cell is <= tol / > tol (false-free = missed obstacle, DANGEROUS)
       spurious                     : perceived-occupied cells with no true obstacle within tol (false-occupied = needless detour/deadlock)
       edt_err_*                    : signed (perceived EDT - true EDT) over the FOOTPRINT-boundary samples along the route (positive = perceived
                                      clearance larger than the real one = the dangerous direction); at points with true EDT < 0.5 m.
       unknown_frac_band            : never-observed cells within the band (treated as free by the shield)"""
    from ..teleop import geom as GM
    route = np.asarray(route_xy, float)
    # band mask on the TRUE grid (1 cm)
    yy, xx = np.mgrid[0:true.occ.shape[0], 0:true.occ.shape[1]]
    cx, cy = true.x0 + (xx + 0.5) * true.res, true.y0 + (yy + 0.5) * true.res
    dense = np.concatenate([route[i] + (route[i + 1] - route[i]) * np.linspace(0, 1, max(int(np.hypot(*(route[i + 1] - route[i])) / 0.05), 2))[:, None]
                            for i in range(len(route) - 1)])
    from scipy.spatial import cKDTree
    dband = cKDTree(dense).query(np.stack([cx.ravel(), cy.ravel()], 1))[0].reshape(cx.shape)
    inb = dband <= band
    tocc = true.occ & inb
    out = dict(band_m=band, n_true_occ_cells_band=int(tocc.sum()))
    pe = perc.lookup(cx[tocc], cy[tocc])              # perceived EDT at each true-occupied cell
    for t in tol:
        out[f"recall_{int(t*1000)}mm"] = float((pe <= t).mean()) if tocc.any() else float("nan")
        out[f"false_free_{int(t*1000)}mm"] = float((pe > t).mean()) if tocc.any() else float("nan")
    # perceived occupied cells in band vs true edt
    iy, ix = np.nonzero(perc.occ)
    px, py = perc.x0 + (ix + 0.5) * perc.res, perc.y0 + (iy + 0.5) * perc.res
    pb = cKDTree(dense).query(np.stack([px, py], 1))[0] <= band
    te = true.lookup(px[pb], py[pb])
    for t in tol:
        out[f"spurious_{int(t*1000)}mm"] = float((te > t).mean()) if pb.any() else float("nan")
    out["n_perc_occ_cells_band"] = int(pb.sum())
    # signed EDT error at footprint-boundary points along the route (poses: position along the route, headings = travel direction and +-90 deg)
    pts = GM.boundary_pts(step=0.04)
    errs, tr = [], []
    for k in range(0, len(dense), 2):
        for dyaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            if k + 1 < len(dense):
                h = math.atan2(*(dense[k + 1] - dense[k])[::-1])
            else:
                h = 0.0
            R = GM.rot(h + dyaw)
            w = pts @ R.T + dense[k]
            et, ep = true.lookup(w[:, 0], w[:, 1]), perc.lookup(w[:, 0], w[:, 1])
            sel = et < 0.5
            errs.append((ep - et)[sel]); tr.append(et[sel])
    e = np.concatenate(errs) if errs else np.zeros(0)
    out.update(edt_err_n=int(len(e)), edt_err_mean_m=float(e.mean()) if len(e) else float("nan"),
               edt_err_p95_m=float(np.percentile(e, 95)) if len(e) else float("nan"), edt_err_p99_m=float(np.percentile(e, 99)) if len(e) else float("nan"),
               edt_err_max_m=float(e.max()) if len(e) else float("nan"), edt_err_p5_m=float(np.percentile(e, 5)) if len(e) else float("nan"),
               edt_err_frac_gt_3cm=float((e > 0.03).mean()) if len(e) else float("nan"),
               edt_err_frac_gt_5cm=float((e > 0.05).mean()) if len(e) else float("nan"))
    return out
