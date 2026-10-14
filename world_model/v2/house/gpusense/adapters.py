"""gpusense/adapters.py -- drop-in GPU twins of the CPU ray-cast sensors (SIMULATION ONLY), served by one sensing process (`Hub`).

  GpuHeadCamera(HeadCamera)   people_track/camera.py: same pose / project / in_fov / render() dict / static_depth(); render() = submit + collect
                              (synchronous drop-in); submit()/collect() = the asynchronous pair that hides the render in the modelled latency.
  GpuWristCam(WristCam)       grasp/wrist.py: capture() (-> WristFrame) and seg_cup() on the GPU; rgb_image / seg_cup_proprio unchanged (CPU).
  Hub                         owns the SenseProc, builds every camera config from a TeleopSim, reports READY + warm-up timings.
  MapService                  5 Hz GPU mapping + EDT/gradient in the sensing process; latest_grid() -> OccGrid for sim.occ_ctrl / shield.occ.

Nothing in people_track / grasp / loc / teleop is edited: the owner swaps the class behind a flag (see README_hooks in gpusense/__init__.py).
"""
from __future__ import annotations

import math
import os
import time
import zlib
from dataclasses import replace

import mujoco
import numpy as np

from world_model.v2.tray_rep.perception import depth_sim as D
from world_model.v2.house.people_track import camera as CAM
from world_model.v2.house.grasp import wrist as WR
from . import core as C
from .proc import SenseProc

_HUBS: dict = {}


class Hub:
    """sensing processes for a TeleopSim, sharded by role so the 30 Hz people / wrist streams and the 5 Hz map run in parallel:
    procs["head"] (head + static), procs["wrist"] (both wrist cameras + their cup segmentation), procs["map"] (mapper + EDT).
    shard=False puts everything into one process."""

    def __init__(self, sim, head_spec=None, wrists=("left", "right"), map_cfg=None, tmpdir=None, start=True, shard=True):
        self.sim = sim
        m = sim.m
        self.roles = {"head": [], "wrist": [], "map": []}
        # head: constructed from a CPU HeadCamera (person geoms, self geoms) -- the owner's object, not duplicated logic
        hc = CAM.HeadCamera(sim, head_spec or CAM.head_spec())
        self.head_spec = hc.spec
        pg = np.array(sorted(hc.person_geoms), int)
        person_of = np.full(m.ngeom, -1, int)
        for g, p in hc.person_geoms.items():
            person_of[g] = p
        ov = ((pg, 5),) if len(pg) else ()
        self.roles["head"].append(C.GpuCamCfg("head", hc.spec, CAM.G_CAST, ov, (), CAM.CUTOFF_MULT, hc.self_geom.copy(), person_of))
        self.roles["head"].append(C.GpuCamCfg("head_static", hc.spec, (3, 4), (), (), CAM.CUTOFF_MULT * 1.5, None, None, kind="seg"))
        for side in wrists:
            w = WR.WristCam(m, side)
            body_geoms = tuple(int(g) for g in np.flatnonzero(m.geom_bodyid == w.body_excl))
            self.roles["wrist"].append(C.GpuCamCfg(f"wrist_{side}", w.spec, WR.MASK_GROUPS, (), body_geoms, WR.CUTOFF_MULT, w.is_robot_geom.copy(), None))
            self.roles["wrist"].append(C.GpuCamCfg(f"wrist_{side}_seg", w.spec, WR.MASK_GROUPS, (), body_geoms, 1.0, None, None, cutoff=1.0,
                                                   cup_geom=w.is_cup_geom.copy(), kind="seg"))
        self.map_spec = None
        if map_cfg is not None:
            cfg = map_cfg
            cs = D.CamSpec(name="zed_map", mount="world", pos=(0, 0, 0), R=tuple(np.eye(3).ravel()), fovy=cfg.fovy, W=cfg.W, H=cfg.H, z_max=cfg.z_map_max)
            self.roles["map"].append(C.GpuCamCfg("map", cs, (3, 4), (), (), CAM.CUTOFF_MULT, hc.self_geom.copy(), None, kind="map"))
            ref = sim.occ                                       # truth grid geometry (same extent as the perceived map)
            ny, nx = int(round(ref.occ.shape[0] * ref.res / cfg.res)), int(round(ref.occ.shape[1] * ref.res / cfg.res))
            self.map_spec = dict(cfg=cfg, x0=ref.x0, y0=ref.y0, ny=ny, nx=nx, cam="map", seed=cfg.seed)
        groups = {k: v for k, v in self.roles.items() if v}
        if not shard:
            groups = {"all": [c for v in groups.values() for c in v]}
        self.model_path = SenseProc._save(m, tmpdir or SenseProc.default_tmp())            # one .mjb shared by every process
        self.procs = {k: SenseProc(m, v, self.map_spec if (k == "map" or (k == "all" and self.map_spec)) else None, tmpdir=tmpdir, model_path=self.model_path)
                      for k, v in groups.items()}
        self.ready = None
        if start:
            self.start()

    def proc_for(self, role):
        return self.procs.get(role) or self.procs["all"]

    @property
    def proc(self):                                              # back-compat: the head (or the only) process
        return self.proc_for("head")

    def start(self):
        t0 = time.perf_counter()
        for p in self.procs.values():                           # parallel warm-up
            p.start_async()
        self.ready = {k: p.wait_ready() for k, p in self.procs.items()}
        self.ready["wall_s"] = time.perf_counter() - t0
        print("[gpusense] READY " + "  ".join(f"{k}: pid {v['pid']} warm-up {v['total_s']:.1f}s (spawn+import {v['wall_s'] - v['total_s']:.1f}s)" for k, v in self.ready.items()
                                              if isinstance(v, dict)) + f"  | all up in {self.ready['wall_s']:.1f}s", flush=True)
        return self.ready

    def close(self):
        for p in self.procs.values():
            p.close()
        try:
            os.remove(self.model_path)
        except OSError:
            pass
        _HUBS.pop(id(self.sim), None)


def get_hub(sim, **kw) -> Hub:
    h = _HUBS.get(id(sim))
    if h is None:
        h = _HUBS[id(sim)] = Hub(sim, **kw)
    return h


# ------------------------------------------------------------------------------------------------ head camera
class GpuHeadCamera(CAM.HeadCamera):
    def __init__(self, sim, spec=None, hub: Hub | None = None):
        super().__init__(sim, spec)
        self.hub = hub or get_hub(sim, head_spec=self.spec)
        self._static = {}
        self.want_static = True                         # also render the static-model depth image (background subtraction in DepthDetector)

    def submit(self, tick, seed, want_static=None):
        want_static = self.want_static if want_static is None else want_static
        origin, R = self.pose()
        job = dict(cam="head", kind="cloud", origin=origin, R=R, tick=int(tick), seed=int(seed), want_static="head_static" if want_static else None)
        return self.hub.proc_for("head").submit(self.sim.d, [job]), origin, R

    def collect(self, handle, want_truth=True):
        ticket, origin, R = handle
        r = self.hub.proc_for("head").collect(ticket)[0]
        person = r["person"] if want_truth else np.full(len(r["person"]), -1, int)
        fr = dict(origin=origin, R=R, pts=r["pts"], geom=r["geom"], person=person, z=r["z"], pix=r["pix"], n_rays=r["n_rays"], tick=r["tick"])
        if "static_z" in r:
            fr["_static_z"] = r["static_z"]
        return fr

    def render(self, tick, seed, want_truth=True):
        self.n_rays = self.spec.W * self.spec.H
        return self.collect(self.submit(tick, seed), want_truth)

    def static_depth(self, frame, idx):
        """static-model z-depth along the rays of frame points `idx` (inf: no hit) from the GPU static image rendered with the frame."""
        img = frame.get("_static_z")
        if img is None:
            return super().static_depth(frame, idx)          # CPU fallback (current sim state)
        rc = frame["pix"][idx]
        return img[rc[:, 0], rc[:, 1]]


# ------------------------------------------------------------------------------------------------ wrist camera
class GpuWristCam(WR.WristCam):
    def __init__(self, m, side, hub: Hub, sim=None):
        super().__init__(m, side)
        self.hub, self.sim = hub, sim

    def _job(self, d, tick, seed, roi, kind="cloud"):
        origin, R = self.pose(d)
        return dict(cam=f"wrist_{self.side}" + ("_seg" if kind == "segcup" else ""), kind=kind, origin=origin, R=R, tick=int(tick), seed=int(seed),
                    roi=roi)

    def submit_capture(self, d, tick, seed=0, t=0.0, roi=None):
        """asynchronous half 1: freeze the capture-instant meta (camera / gripper pose) and queue the render.  -> handle for finish_capture."""
        origin, R = self.pose(d)
        gb = int(self.grip_body)
        meta = (origin, R, d.xpos[gb].copy(), d.xmat[gb].reshape(3, 3).copy(), int(tick), float(t))
        return self.hub.proc_for("wrist").submit(d, [self._job(d, tick, seed, roi)]), meta

    def finish_capture(self, handle):
        """asynchronous half 2 (call at the delivery tick): -> WristFrame identical to capture() at the submit instant."""
        ticket, (origin, R, gp, gR, tick, t) = handle
        r = self.hub.proc_for("wrist").collect(ticket)[0]
        return WR.WristFrame(self.side, tick, t, origin, R, gp, gR, r["pts"], r["geom"], r["z"], r["n_rays"], r["n_self"])

    def capture(self, d, tick, seed=0, t=0.0, roi=None):
        origin, R = self.pose(d)
        gb = int(self.grip_body)
        gp, gR = d.xpos[gb].copy(), d.xmat[gb].reshape(3, 3).copy()
        p = self.hub.proc_for("wrist")
        r = p.collect(p.submit(d, [self._job(d, tick, seed, roi)]))[0]
        return WR.WristFrame(self.side, tick, t, origin, R, gp, gR, r["pts"], r["geom"], r["z"], r["n_rays"], r["n_self"])

    def seg_cup(self, d, nominal_range=0.07):
        p = self.hub.proc_for("wrist")
        r = p.collect(p.submit(d, [self._job(d, 0, 0, None, "segcup")]))[0]
        return r["seg"]


# ------------------------------------------------------------------------------------------------ map service
class MapService:
    """5 Hz GPU mapping + EDT/gradient in the sensing process.  update() is non-blocking (submit); poll() returns a fresh OccGrid when one is
    available.  The grid is in the estimated-pose frame (pass sim.ctrl_pose() as est_pose when pose_source='est')."""

    def __init__(self, sim, hub: Hub):
        assert hub.map_spec is not None, "Hub(map_cfg=...) required"
        self.sim, self.hub = sim, hub
        self.pending = []

    def update(self, est_pose=None, publish=True):
        sim = self.sim
        tp = sim.pose()
        t = self.hub.proc_for("map").submit_map(sim.d, tp, tp if est_pose is None else est_pose, publish)
        self.pending.append(t)
        return t

    def poll(self, block=False):
        """-> newest OccGrid or None."""
        grid = None
        while self.pending:
            r = self.hub.proc_for("map").collect(self.pending[0], block=block)
            if r is None:
                break
            self.pending.pop(0)
            grid = True
        return self.hub.proc_for("map").latest_grid() if grid else None


def make_head_camera(sim, spec=None, **kw):
    """flag-gated factory: people_track.perception uses `CAM.HeadCamera(sim, spec)`; with BIMANUAL_GPUSENSE=1 use this."""
    from . import GPUSENSE
    if not GPUSENSE:
        return CAM.HeadCamera(sim, spec)
    return GpuHeadCamera(sim, spec)
