"""gpusense/core.py -- GPU depth sensor core (SIMULATION ONLY; pure MuJoCo EGL + torch, no robot code, no network).

Replaces the per-pixel CPU ray cast (tray_rep.perception.depth_sim via mj_multiRay) by MuJoCo's EGL depth buffer:

  * ONE render pass per camera gives z-depth (reverse-Z depth buffer, decoded in float64) AND the geom id of every pixel (segmentation
    colour pass, `mjRND_SEGMENT|mjRND_IDCOLOR`).  The id image is the privileged baseline (person index) and the self filter
    (robot / tray / cup / bead geoms) -- exactly what the ray-cast filter did by `geom_id`.
  * The ray-cast's group filter (mj_multiRay `geomgroup` mask) is reproduced with `mjvOption.geomgroup`; the head camera's trick
    ("person geoms are moved to group 5 for the cast only") is reproduced by a temporary `geom_group` override applied around
    `mjv_updateScene`; the wrist cameras' `bodyexclude` (the camera's own body is transparent to rays) by hiding that body's geoms
    (override to a group outside the mask).  Renderer model copy is private to this module (the sim's model is never touched).
  * The depth_sim.CamSpec noise model is applied in torch on the GPU: stereo-law sigma_z, outliers, quantisation, dropout, range
    gates, grazing-incidence invalid (normals from the depth image instead of mj_multiRay's analytic normal), latency is the caller's
    (queue).  RNG: torch Philox generator seeded by (seed, tick, crc32(cam name)) -> a snapshot restore replays the same frame (the
    numbers differ from the numpy stream of the ray-cast: statistically equivalent, see equiv.py).

Persistent per camera: one mujoco.Renderer (EGL context + framebuffer) + precomputed ray tables on the GPU.
`warmup()` creates the contexts, renders twice, runs the noise kernels once and reports the time.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import math
import time
import zlib
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

CUTOFF_MULT = 1.2
HIDE_ORDER = (1, 2, 5, 0, 3, 4)


@dataclass
class GpuCamCfg:
    name: str
    spec: object                                   # depth_sim.CamSpec (W, H, fovy, noise parameters, z range)
    groups: tuple = (3, 4, 5)                      # mj_multiRay geomgroup mask of the ray-cast
    override: tuple = ()                           # ((geom_ids ndarray, group), ...) applied around update_scene (head: persons -> 5)
    hide_geoms: tuple = ()                         # geom ids the cast ignores (bodyexclude) -> hidden from the render
    cutoff_mult: float = CUTOFF_MULT               # ray length cutoff = z_max * cutoff_mult (the ray-cast's cutoff)
    self_geom: np.ndarray | None = None            # (ngeom,) bool: own geoms, dropped from the cloud (still occlude)
    person_of: np.ndarray | None = None            # (ngeom,) int: person index of a geom (-1 none)
    near: float = 0.004
    kind: str = "cloud"                              # cloud (people / wrist) | map | seg: what warmup() exercises
    cup_geom: np.ndarray | None = None             # (ngeom,) bool: the cup (seg_cup RGB stand-in)
    cutoff: float | None = None                    # absolute ray-length cutoff [m]; default spec.z_max * cutoff_mult
    cull: bool = True                              # reproduce mj_multiRay's `cutoff`: a GEOM-culling radius (|geom_xpos - origin| - rbound > cutoff
                                                   # => the geom is never tested; planes have rbound 0, so a floor whose centre is far is invisible)

    def ray_cutoff(self):
        return self.spec.z_max * self.cutoff_mult if self.cutoff is None else self.cutoff


def snapshot_model_bytes(m, path):
    """write the model as a .mjb for the renderer (own copy); framebuffer large enough for 640x360."""
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)
    mujoco.mj_saveModel(m, path, None)
    return path


class _Cam:
    """one persistent EGL renderer + GPU ray tables + (optional) CUDA graph of the dense noise stage."""

    def __init__(self, m, cfg: GpuCamCfg, device, use_graph=True):
        self.cfg, self.m = cfg, m
        s = cfg.spec
        self.W, self.H = int(s.W), int(s.H)
        self.r = mujoco.Renderer(m, self.H, self.W)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[list(cfg.groups)] = 1
        self.opt.sitegroup[:] = 0
        self.opt.jointgroup[:] = 0
        self.opt.tendongroup[:] = 0
        if hasattr(self.opt, "flexgroup"):
            self.opt.flexgroup[:] = 0
        self.hide_group = next(g for g in HIDE_ORDER if g not in cfg.groups)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        far = 100.0                                   # rays are not length-limited in mj_multiRay (cutoff only culls geoms)
        self.near, self.far = float(cfg.near), float(far)
        self.rgb = np.zeros((self.H, self.W, 3), np.uint8)
        self.dep = np.zeros((self.H, self.W), np.float32)
        self.hide = np.asarray(cfg.hide_geoms, int)
        self._lut = {}
        # GPU tables (z-depth image <-> ray)
        W, H, f = self.W, self.H, s.f_px
        u = (torch.arange(W, dtype=torch.float64, device=device) + 0.5 - W / 2.0) / f
        v = -(torch.arange(H, dtype=torch.float64, device=device) + 0.5 - H / 2.0) / f
        Y, X = torch.meshgrid(v, u, indexing="ij")                     # (H, W)
        self.X, self.Y = X, Y
        self.n_cam = torch.sqrt(X * X + Y * Y + 1.0)
        self.cth = 1.0 / self.n_cam
        self.dirs = torch.stack([X, Y, -torch.ones_like(X)], -1) / self.n_cam[..., None]     # (H, W, 3) camera frame, unit
        ng = m.ngeom
        self.self_t = torch.as_tensor(cfg.self_geom if cfg.self_geom is not None else np.zeros(ng, bool), device=device)
        self.person_t = torch.as_tensor(cfg.person_of if cfg.person_of is not None else np.full(ng, -1), dtype=torch.int64, device=device)
        # reverse-Z decode constants (float32 arithmetic as in mujoco.Renderer)
        zf, zn = np.float32(self.far), np.float32(self.near)
        c = -(zf + zn) / (zf - zn)
        dd = -(np.float32(2) * zf * zn) / (zf - zn)
        self.c_coef = float(np.float32(-0.5) * c - np.float32(0.5))
        self.d_coef = float(np.float32(-0.5) * dd)
        self.device = device
        self.use_graph = use_graph
        self.graph = None

    # ------------------------------------------------------------------ render
    def _geom_lut(self, sc, hide, ov_key, eff_group):
        """segmentation value (= scene geom index + 1) -> model geom id (-1: background / non-geom).
        mjv_addGeoms emits the visible model geoms in model order, so the table is computed directly from the effective group array
        (no per-geom Python loop) and VALIDATED on a sample of scene entries every call; any mismatch falls back to the full loop."""
        n = int(sc.ngeom)
        vis = np.flatnonzero(self.opt.geomgroup[eff_group].astype(bool) & (self.m.geom_rgba[:, 3] > 0))
        geoms = sc.geoms
        if len(vis) == n:
            idx = np.linspace(0, n - 1, min(n, 24)).astype(int) if n else np.zeros(0, int)
            if all(geoms[int(i)].objid == vis[i] and geoms[int(i)].segid == i for i in idx):
                lut = np.full(n + 2, -1, np.int32)
                lut[1:n + 1] = vis
                return lut
        self.n_lut_fallback = getattr(self, "n_lut_fallback", 0) + 1
        key = (n, hash(hide.tobytes()), ov_key)
        lut = self._lut.get(key)
        if lut is None:
            lut = np.full(n + 2, -1, np.int32)
            og = int(mujoco.mjtObj.mjOBJ_GEOM)
            for i in range(n):
                g = geoms[i]
                if g.segid >= 0 and int(g.objtype) == og:
                    lut[g.segid + 1] = g.objid
            if len(self._lut) > 8:
                self._lut.clear()
            self._lut[key] = lut
        return lut

    def raw(self, d, origin, R):
        """-> (raw depth float32 (H,W) top-down, geom id image int32 (H,W) top-down)."""
        r, m, cfg = self.r, self.m, self.cfg
        hide, ov, saved = self.hide, cfg.override, []
        for g, grp in ov:
            saved.append((g, m.geom_group[g].copy()))
            m.geom_group[g] = grp
        if cfg.cull:
            dc = np.linalg.norm(d.geom_xpos - origin, axis=1) - m.geom_rbound
            far = np.flatnonzero(dc > cfg.ray_cutoff())
            hide = np.union1d(hide, far).astype(int) if len(hide) else far
        if len(hide):
            saved.append((hide, m.geom_group[hide].copy()))
            m.geom_group[hide] = self.hide_group
        try:
            r.update_scene(d, camera=self.cam, scene_option=self.opt)
            eff = m.geom_group.copy()
        finally:
            for g, old in reversed(saved):
                m.geom_group[g] = old
        sc = r._scene
        s = cfg.spec
        top = self.near * math.tan(math.radians(s.fovy) / 2.0)
        fwd, up = -R[:, 2], R[:, 1]
        for i in range(2):
            c = sc.camera[i]
            c.pos[:] = origin
            c.forward[:] = fwd
            c.up[:] = up
            c.frustum_near, c.frustum_far = self.near, self.far
            c.frustum_top, c.frustum_bottom = top, -top
            c.frustum_center = 0.0
        fl = sc.flags
        for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX,
                  mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_HAZE):
            fl[f] = 0
        fl[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        fl[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        r._gl_context.make_current()
        mujoco.mjr_render(r._rect, sc, r._mjr_context)
        mujoco.mjr_readPixels(self.rgb, self.dep, r._rect, r._mjr_context)
        rgb = self.rgb[::-1]
        seg = rgb[..., 0].astype(np.int32) + (rgb[..., 1].astype(np.int32) << 8) + (rgb[..., 2].astype(np.int32) << 16)
        lut = self._geom_lut(sc, hide, len(ov), eff)
        return self.dep[::-1], lut[np.minimum(seg, len(lut) - 1)]

    def decode(self, raw, geom):
        """GPU: raw reverse-Z depth (float32) + geom ids -> (z (H,W) float64, 0 where nothing; geom int32, -1 none).  Graph-safe."""
        rawd = raw.double()
        z = self.d_coef / (rawd + self.c_coef)
        none = (rawd <= 0) | (z >= self.far * 0.999) | ~torch.isfinite(z) | (geom < 0)
        return torch.where(none, torch.zeros_like(z), z), torch.where(none, torch.full_like(geom, -1), geom)

    def depth_geom(self, d, origin, R):
        """-> (z (H,W) float64 GPU, 0 where nothing is hit; geom (H,W) int32 GPU, -1 none)."""
        dep, geom = self.raw(d, origin, R)
        dev = self.device
        return self.decode(torch.from_numpy(np.ascontiguousarray(dep)).to(dev), torch.from_numpy(np.ascontiguousarray(geom)).to(dev))

    # ------------------------------------------------------------------ cloud stage (people / wrist): dense, graph-capturable
    def _alloc(self):
        H, W, dev = self.H, self.W, self.device
        f64 = dict(dtype=torch.float64, device=dev)
        self.b_raw = torch.zeros((H, W), dtype=torch.float32, device=dev)
        self.b_geom = torch.full((H, W), -1, dtype=torch.int32, device=dev)
        self.b_n = torch.zeros((H, W), **f64)
        self.b_u = torch.zeros((4, H, W), **f64)
        self.b_roi = torch.ones((H, W), dtype=torch.bool, device=dev)
        self.b_pose = torch.zeros(12, **f64)

    def _dense(self):
        """static buffers -> (ok (HW,), pack (HW, 8) [x y z zn geom person row col], n_self, z_true image).  No host sync."""
        s = self.cfg.spec
        H, W = self.H, self.W
        z, geom = self.decode(self.b_raw, self.b_geom)
        hit = (geom >= 0) & (z > 0) & self.b_roi
        cosinc = depth_cos_incidence(self, z, hit)
        e, u_drop, u_out = self.b_n, self.b_u[0], self.b_u[1]
        e_out = (0.05 + 0.15 * self.b_u[2]) * torch.where(self.b_u[3] < 0.5, -1.0, 1.0)
        zn = z + sigma_z(s, z) * e
        zn = torch.where(u_out < s.p_out, z + e_out, zn)
        if s.quant_m > 0:
            zn = torch.round(zn / s.quant_m) * s.quant_m
        ok = hit & (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        gl = geom.clamp(min=0).long()
        slf = self.self_t[gl] & (geom >= 0)
        n_self = (ok & slf).sum()
        ok = ok & ~slf
        o, Rt = self.b_pose[:3], self.b_pose[3:].reshape(3, 3)
        dw = self.dirs.reshape(-1, 3) @ Rt.T
        zf = zn.reshape(-1)
        pts = o + (zf / self.cth.reshape(-1))[:, None] * dw
        rows = torch.arange(H * W, device=self.device)
        pack = torch.cat([pts, zf[:, None], geom.reshape(-1, 1).double(), self.person_t[gl].reshape(-1, 1).double(),
                          (rows // W)[:, None].double(), (rows % W)[:, None].double()], 1)
        return ok.reshape(-1), pack, n_self

    def setup_graph(self):
        self._alloc()
        if not self.use_graph:
            return
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):                                     # warm-up on a side stream before capture
                self._dense()
        torch.cuda.current_stream().wait_stream(st)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.g_ok, self.g_pack, self.g_nself = self._dense()

    def cloud(self, d, origin, R, tick, seed, roi=None):
        """full people / wrist job: render -> GPU noise (CUDA graph) -> compaction -> host dict (HeadCamera.render layout)."""
        s, dev = self.cfg.spec, self.device
        if not hasattr(self, "b_raw"):
            self.setup_graph()
        dep, geom = self.raw(d, origin, R)
        self.b_raw.copy_(torch.from_numpy(np.ascontiguousarray(dep)), non_blocking=True)
        self.b_geom.copy_(torch.from_numpy(np.ascontiguousarray(geom)), non_blocking=True)
        g = _gen(dev, seed, tick, s.name)
        torch.randn(self.b_n.shape, generator=g, device=dev, dtype=torch.float64, out=self.b_n)
        torch.rand(self.b_u.shape, generator=g, device=dev, dtype=torch.float64, out=self.b_u)
        if roi is None:
            self.b_roi.fill_(True)
        else:
            v0, v1, u0, u1 = roi
            self.b_roi.zero_()
            self.b_roi[v0:v1, u0:u1] = True
        self.b_pose[:3].copy_(torch.as_tensor(origin, dtype=torch.float64))
        self.b_pose[3:].copy_(torch.as_tensor(np.asarray(R, float).reshape(-1)))
        if self.graph is not None:
            self.graph.replay()
            ok, pack, nself = self.g_ok, self.g_pack, self.g_nself
        else:
            ok, pack, nself = self._dense()
        sel = ok.nonzero().squeeze(1)
        h = torch.cat([pack[sel].reshape(-1), nself.double().reshape(1)]).cpu().numpy()
        n = (len(h) - 1) // 8
        h, nself = h[:-1].reshape(n, 8), int(h[-1])
        return dict(pts=h[:, :3].copy(), z=h[:, 3].copy(), geom=h[:, 4].astype(np.int32), person=h[:, 5].astype(int), pix=h[:, 6:8].astype(int),
                    n_rays=self.H * self.W, n_self=nself, origin=np.asarray(origin, float).copy(), R=np.asarray(R, float).copy(), tick=int(tick))

    def close(self):
        self.graph = None
        try:
            self.r.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ sensor noise (torch)
def _gen(device, seed, tick, name):
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), zlib.crc32(name.encode())]).generate_state(1, np.uint64)[0] & 0x7FFFFFFFFFFFFFFF))
    return g


def sigma_z(spec, z):
    return torch.sqrt(spec.sigma_floor ** 2 + (z * z / (spec.f_native * spec.baseline) * spec.sigma_disp_px) ** 2)


def depth_cos_incidence(c: "_Cam", z, hit, disc_rel=0.15, disc_abs=0.05):
    """|cos(incidence)| per pixel from the depth image alone (the analytic surface normal of mj_multiRay is not available in a depth buffer).
    P = z * (X, Y, -1); n = dP/du x dP/dv with per-axis one-sided fall-back at depth discontinuities (silhouettes); undefined normal -> 1."""
    P = z[..., None] * torch.stack([c.X, c.Y, -torch.ones_like(c.X)], -1)           # (H,W,3)
    H, W = z.shape

    def diffs(axis):
        # forward / backward differences along axis, with validity (both pixels hit and no depth jump)
        zp, zm = torch.roll(z, -1, axis), torch.roll(z, 1, axis)
        Pp, Pm = torch.roll(P, -1, axis), torch.roll(P, 1, axis)
        hp, hm = torch.roll(hit, -1, axis), torch.roll(hit, 1, axis)
        # wrap-around invalid
        if axis == 1:
            hp[:, -1] = False; hm[:, 0] = False
        else:
            hp[-1, :] = False; hm[0, :] = False
        tol = disc_rel * z + disc_abs
        okp = hp & ((zp - z).abs() < tol)
        okm = hm & ((zm - z).abs() < tol)
        Dp, Dm = Pp - P, P - Pm
        both = okp & okm
        # central if both ok; else the single valid side; else invalid
        Dc = torch.where(both[..., None], 0.5 * (Dp + Dm), torch.where(okp[..., None], Dp, Dm))
        return Dc, okp | okm
    Du, vu = diffs(1)
    Dv, vv = diffs(0)
    n = torch.cross(Du, Dv, dim=-1)
    nn = torch.linalg.norm(n, dim=-1) + 1e-12
    cos = (n * c.dirs).sum(-1).abs() / nn
    return torch.where(vu & vv, cos, torch.ones_like(cos))


class GpuSensor:
    """the noise models.  All tensors stay on the GPU until the (single) compaction/transfer."""

    def __init__(self, device):
        self.device = device

    # mapper: (ok, pts_cam (fwd,left,up), nofree) as HeadDepth.capture, on the GPU (no transfer)
    def map_capture(self, c: _Cam, z, geom, rng_gen, drop_self=True):
        s, cfg = c.cfg.spec, c.cfg
        H, W, dev = c.H, c.W, self.device
        hit = (geom >= 0) & (z > 0)
        slf = torch.zeros_like(hit)
        if c.self_t is not None and drop_self:
            slf = c.self_t[geom.clamp(min=0).long()] & (geom >= 0)
            hit = hit & ~slf
        cosinc = depth_cos_incidence(c, z, hit)
        e = torch.randn((H, W), generator=rng_gen, device=dev, dtype=torch.float64)
        u_out = torch.rand((H, W), generator=rng_gen, device=dev, dtype=torch.float64)
        u_drop = torch.rand((H, W), generator=rng_gen, device=dev, dtype=torch.float64)
        e_out = (0.05 + 0.15 * torch.rand((H, W), generator=rng_gen, device=dev, dtype=torch.float64)) * torch.where(
            torch.rand((H, W), generator=rng_gen, device=dev, dtype=torch.float64) < 0.5, -1.0, 1.0)
        zn = z + sigma_z(s, z) * e
        zn = torch.where(u_out < s.p_out, z + e_out, zn)
        zn = torch.round(zn / s.quant_m) * s.quant_m
        ok = hit & (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        nofree = (~((geom >= 0) & (z > 0))) & (u_drop >= s.p_drop) & ~slf
        return ok.flatten(), zn.flatten(), nofree.flatten()


class SenseCore:
    """persistent renderers + sensor; in-process (tests/benchmarks) or inside the sensing process (proc.py)."""

    def __init__(self, model_path, cam_cfgs, device="cuda"):
        self.device = torch.device(device)
        self.m = mujoco.MjModel.from_binary_path(model_path)
        self.m.vis.quality.offsamples = 0                   # no MSAA: depth/ids are not blended at silhouettes
        self.d = mujoco.MjData(self.m)
        self.cfgs = {c.name: c for c in cam_cfgs}
        self.cams: dict[str, _Cam] = {}
        self.sensor = GpuSensor(self.device)
        self.cam_cfgs = list(cam_cfgs)
        self.warm = {}

    def _cam(self, name):
        c = self.cams.get(name)
        if c is None:
            c = self.cams[name] = _Cam(self.m, self.cfgs[name], self.device)
        return c

    def set_state(self, qpos, mpos, mquat, eq):
        d = self.d
        d.qpos[:] = qpos
        d.mocap_pos[:] = mpos
        d.mocap_quat[:] = mquat
        d.eq_active[:] = eq
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)

    def depth(self, name, origin, R):
        """noise-free (z (H,W) float64 GPU, geom (H,W) int32 GPU) at the current state."""
        return self._cam(name).depth_geom(self.d, np.asarray(origin, float), np.asarray(R, float))

    def run_job(self, job):
        c = self._cam(job["cam"])
        t0 = time.perf_counter()
        o, R = np.asarray(job["origin"], float), np.asarray(job["R"], float)
        out = {"cam": job["cam"], "tick": job["tick"]}
        kind = job.get("kind", "cloud")
        if kind == "segcup":                                      # WristCam.seg_cup: full grid, no noise, no min range
            z, geom = c.depth_geom(self.d, o, R)
            cup = (geom >= 0) & torch.as_tensor(c.cfg.cup_geom, device=self.device)[geom.clamp(min=0).long()]
            n = int(cup.sum())
            if n < 20:
                out["seg"] = None
            else:
                vv, uu = torch.nonzero(cup, as_tuple=True)
                dist = (z / c.cth)[cup]
                out["seg"] = (n, float(uu.double().mean()), float(vv.double().mean()), float(dist.mean()))
        elif kind == "cloud":
            out.update(c.cloud(self.d, o, R, job["tick"], job["seed"], job.get("roi")))
        if job.get("want_static") is not None:               # static-model depth image (people hidden), for background subtraction
            z2, g2 = self._cam(job["want_static"]).depth_geom(self.d, o, R)
            zz = torch.where(z2 > 0, z2, torch.full_like(z2, float("inf")))
            out["static_z"] = zz.cpu().numpy()
        torch.cuda.synchronize()
        out["ms"] = 1e3 * (time.perf_counter() - t0)
        return out

    def process(self, state, jobs):
        self.set_state(*state)
        return [self.run_job(j) for j in jobs]

    def warmup(self, n=3):
        """create every EGL context + framebuffer, build the CUDA graphs, run the render and noise kernels n times.  -> seconds."""
        t0 = time.perf_counter()
        torch.zeros(1, device=self.device)                                            # CUDA context
        mujoco.mj_forward(self.m, self.d)
        o = np.array([0.0, 0.0, 1.0])
        R = np.eye(3)
        for name, cfg in self.cfgs.items():
            c = self._cam(name)
            for i in range(n):
                if cfg.kind == "cloud":
                    c.cloud(self.d, o, R, i, 0)
                else:
                    c.depth_geom(self.d, o, R)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.warm["gl_cuda_s"] = dt
        return dt

    def close(self):
        for c in self.cams.values():
            c.close()
        self.cams.clear()
