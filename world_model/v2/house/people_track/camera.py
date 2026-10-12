"""people_track/camera.py -- the robot's simulated head RGB-D camera, ray-cast on the teleop MuJoCo model.  SIMULATION ONLY.

Spec: perception_bridge._spec_head_new (the ASSUMED torso/mast head: robot-frame (0.05, -0.2096, 1.15 + lift), pitched 36.3 deg down,
fovy 55 deg, 160x90, stereo-law depth noise sigma_z = sqrt(floor^2 + (z^2/(f_native*baseline)*sigma_disp)^2), grazing mask, dropout,
rare outliers, 1 mm quantisation, z in [0.3, 5] m).  The pose rides the (mocap) robot root body, so it moves with the base; the lift
raises it (lift mount).  The noise recipe below is a line-for-line copy of perception_bridge.HouseCamera.capture (that class needs a
HouseScene/Composed model and a tray ROI; the teleop model has neither).

Differences from HouseCamera.capture (documented):
  * ray-cast groups {3,4,5} (robot+wall collision, house hulls+floor, + the person geoms which are moved to group 5 for the cast only);
  * the robot self-filter is by geom id (robot / tray / cup / bead geoms = the robot's own known kinematics) instead of a second cast
    (group 3 of this model also holds walls);
  * the robot still OCCLUDES (a ray that hits the tray/arm stops there).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from world_model.v2.tray_rep.perception import depth_sim as D
from world_model.v2.house import perception_bridge as PB

G_CAST = (3, 4, 5)
CUTOFF_MULT = 1.2
TICK_HZ = 50.0


def head_spec(pitch_deg=PB.HEAD_PITCH_CURRENT, mount="lift", **kw):
    sp = PB._spec_head_new(mount, pitch_deg)
    return replace(sp, **kw) if kw else sp


def _group_mask(groups):
    a = np.zeros(6, np.uint8)
    a[list(groups)] = 1
    return a


class HeadCamera(D.DepthCamera):
    """depth_sim.DepthCamera (rays, noise params) bound to a TeleopSim."""

    def __init__(self, sim, spec: D.CamSpec | None = None):
        spec = spec or head_spec()
        super().__init__(spec)
        self.sim = sim
        m = sim.m
        self.root = m.body("yor_root").id
        self._mask = _group_mask(G_CAST)
        self._mask_static = _group_mask((3, 4))             # people are group 0 (not cast) in the static re-cast
        # person geoms (cast-only group switch) -> person index
        self.person_geoms = {}
        names = [p.name for p in sim.scene.people]
        for g in range(m.ngeom):
            bn = m.body(int(m.geom_bodyid[g])).name
            if bn in names and sim.with_people:
                self.person_geoms[g] = names.index(bn)
        self._pg = np.array(sorted(self.person_geoms), int)
        self._pg_group = m.geom_group[self._pg].copy() if len(self._pg) else np.zeros(0, int)
        # own geoms: robot, tray, cup, beads (known payload) -> dropped as self
        self.self_geom = np.array(sim.robot_geom, bool).copy()
        for g in range(m.ngeom):
            bn = m.body(int(m.geom_bodyid[g])).name or ""
            if bn in ("cup", "tray") or bn.startswith("ball_"):
                self.self_geom[g] = True
        self.n_rays = 0

    # ------------------------------------------------------------------ pose
    def pose(self):
        """(camera centre, camera-to-world rotation) from the CURRENT physics state (robot root + lift)."""
        d = self.sim.d
        Rb = d.xmat[self.root].reshape(3, 3)
        lift = self.sim.lift_actual() if self.spec.mount == "lift" else 0.0
        return d.xpos[self.root] + Rb @ (self._pos + np.array([0.0, 0.0, lift])), Rb @ self._R_cam

    def project(self, P, origin=None, R=None):
        """world points (n,3) -> (u, v, z) pixel coords and z-depth (z<=0: behind camera)."""
        s = self.spec
        if origin is None:
            origin, R = self.pose()
        pc = (np.atleast_2d(P) - origin) @ R
        z = -pc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        u = s.W / 2.0 + s.f_px * pc[:, 0] / zs - 0.5
        v = s.H / 2.0 - s.f_px * pc[:, 1] / zs - 0.5
        return u, v, z

    def in_fov(self, P, origin=None, R=None, margin_px=0.0):
        s = self.spec
        u, v, z = self.project(P, origin, R)
        return (z > s.z_min) & (z < s.z_max) & (u >= margin_px) & (u <= s.W - 1 - margin_px) & (v >= margin_px) & (v <= s.H - 1 - margin_px)

    def static_depth(self, frame, idx):
        """expected z-depth of the STATIC model (no people) along the rays of the returned points `idx` of `frame` (inf: no hit).  The prior
        house model the robot is localised in; same ray, same pose."""
        sim, s = self.sim, self.spec
        rc = frame["pix"][idx]
        dirs = self.dirs[rc[:, 0], rc[:, 1]]
        cth = self.cos_th[rc[:, 0], rc[:, 1]]
        dw = np.ascontiguousarray(dirs @ frame["R"].T)
        n = len(dw)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        mujoco.mj_multiRay(sim.m, sim.d, frame["origin"], dw.reshape(-1), self._mask_static, True, -1, geom, dist, None, n, s.z_max * CUTOFF_MULT * 1.5)
        return np.where((geom >= 0) & (dist > 0), dist * cth, np.inf)

    # ------------------------------------------------------------------ rendering
    def render(self, tick, seed, want_truth=True):
        """-> dict(origin, R, pts (n,3) noisy world points, geom (n,), person (n,) index of the person hit or -1, z (n,) noisy z-depth,
        pix (n,2) (row, col), is_floor-free; n_rays).  Self hits are removed (they still occlude)."""
        sim, m, d, s = self.sim, self.sim.m, self.sim.d, self.spec
        origin, R = self.pose()
        H, W = s.H, s.W
        dirs = self.dirs.reshape(-1, 3)
        cth = self.cos_th.reshape(-1)
        n = len(dirs)
        dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        if len(self._pg):
            m.geom_group[self._pg] = 5
        try:
            mujoco.mj_multiRay(m, d, origin, dw.reshape(-1), self._mask, True, -1, geom, dist, nrm, n, s.z_max * CUTOFF_MULT)
        finally:
            if len(self._pg):
                m.geom_group[self._pg] = self._pg_group
        self.n_rays = n
        hit = (geom >= 0) & (dist > 0)
        idx = np.flatnonzero(hit)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), zlib.crc32(s.name.encode())]))
        k = len(idx)
        z = dist[idx] * cth[idx]
        nh = nrm.reshape(-1, 3)[idx]
        cosinc = np.abs(np.einsum("ij,ij->i", nh, dw[idx]))
        e = rng.normal(size=k)
        u_drop, u_out, e_out = rng.random(k), rng.random(k), rng.uniform(0.05, 0.20, k) * np.where(rng.random(k) < 0.5, -1, 1)
        zn = z + s.sigma_z(z) * e
        zn = np.where(u_out < s.p_out, z + e_out, zn)
        if s.quant_m > 0:
            zn = np.round(zn / s.quant_m) * s.quant_m
        ok = (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        ok &= ~self.self_geom[geom[idx]]                         # robot/tray/cup/bead return: self (known kinematics)
        sel = idx[ok]
        zn = zn[ok]
        pts = origin + (zn / cth[sel])[:, None] * dw[sel]
        gsel = geom[sel].copy()
        person = np.full(len(sel), -1, int)
        if len(self._pg) and want_truth:
            for gi, pi in self.person_geoms.items():
                person[gsel == gi] = pi
        return dict(origin=origin, R=R, pts=pts, geom=gsel, person=person, z=zn, pix=np.stack([sel // W, sel % W], 1), n_rays=n,
                    tick=int(tick))
