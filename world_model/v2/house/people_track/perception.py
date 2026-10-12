"""people_track/perception.py -- sensor clock + latency + tracker + the sim proxy the shield sees.  SIMULATION ONLY.

PerceptionStack.step(sim) is called once per 50 Hz tick BEFORE the shield (state = start of the tick):
   1. tracker.predict_to(k)                      (every tick, through occlusion)
   2. a camera frame is CAPTURED when the sensor clock fires (rate_hz, sample-and-hold against the 50 Hz tick); detectors run on the capture-time
      physics state and the result is queued with delivery tick  k + latency_ticks  (CamSpec.latency_ticks = 3 = 60 ms by default)
   3. frames whose delivery tick has come are applied at their CAPTURE tick (tracker retrodiction) and the tracks re-predicted to k
   4. the observed-space map is stamped with the analytic frustum (static-map-limited line of sight)
   5. `ground-truth visibility` per person is recorded for the metrics only.
Output: .tracks (confirmed tracks, each with xy, v, P), and .proxy(sim) -> a SimProxy that delegates every attribute to the real TeleopSim except
`people_state` / `scene.people`, which are the TRACKER's (tracks in a fixed order, names trkN).

UNSEEN SPACE policy (documented, deliberately simple): ObservedMap stamps every free grid cell inside the camera's horizontal FOV wedge (range
<= z_max, blocked by the static occupancy map) with the current time.  PerceivedShield caps the translational speed of the operator command at
`v_unseen` whenever >= `frac_unseen` of the free cells along the intended travel direction (0.5 m .. 0.6 + |v| * 1.0 m ahead of the footprint
centre) were not observed within the last `t_stale` seconds.  Moving toward space the camera has not looked at recently (e.g. backwards, or
round a corner) is slow; moving through space it has watched is not.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.house.teleop import geom as GM
from . import camera as CAM
from . import detectors as DET
from .tracker import MultiTracker, TrackerCfg, Det

TICK = 0.02


@dataclass
class PerceptionCfg:
    detector: str = "depth"          # 'depth' | 'seg' | 'param' | 'truth'
    rate_hz: float = 30.0
    latency_ticks: int = 3
    # parametric detector
    sigma: float = 0.10
    p_drop: float = 0.0
    # camera
    W: int = 160
    H: int = 90
    seed: int = 0
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    use_static_map: bool = True
    static_map: str = "raycast"      # 'raycast' (background subtraction vs the house model) | 'occ' (shield's sparse sim.occ EDT only) | 'both'
    r_max: float = 4.5


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class SceneProxy:
    def __init__(self, scene, people):
        self._scene = scene
        self.people = people

    def __getattr__(self, n):
        return getattr(self._scene, n)


class SimProxy:
    """delegates everything to the real TeleopSim except people_state / scene.people (tracker-derived).  `track_info` carries the
    per-track covariance for the uncertainty-aware shield."""

    def __init__(self, sim, tracks):
        self._sim = sim
        self.track_info = [dict(name=t.name, xy=t.x[:2].copy(), v=t.x[2:].copy(), P=t.P.copy(), status=t.status, id=t.id) for t in tracks]
        self.people_state = [(ti["xy"], ti["v"]) for ti in self.track_info]
        self.scene = SceneProxy(sim.scene, [_NS(name=ti["name"], id=ti["id"], dynamic=True) for ti in self.track_info])

    def __getattr__(self, n):
        return getattr(self._sim, n)

    def without_people(self):
        q = object.__new__(SimProxy)
        q._sim = self._sim
        q.track_info = []
        q.people_state = []
        q.scene = SceneProxy(self._sim.scene, [])
        return q


class ObservedMap:
    """time-stamped coarse grid over the static occupancy grid: when was each free cell last inside the camera's view."""

    def __init__(self, occ, res=0.25):
        self.occ = occ
        self.res = res
        ny, nx = occ.occ.shape
        self.x0, self.y0 = occ.x0, occ.y0
        self.nx, self.ny = int(math.ceil(nx * occ.res / res)), int(math.ceil(ny * occ.res / res))
        self.t = np.full((self.ny, self.nx), -1e9)

    def _ij(self, x, y):
        return np.floor((np.asarray(y) - self.y0) / self.res).astype(int), np.floor((np.asarray(x) - self.x0) / self.res).astype(int)

    def stamp(self, origin, yaw_cam, hfov, rmax, t, nb=28, nr=48):
        ang = yaw_cam + np.linspace(-hfov / 2, hfov / 2, nb)
        r = np.linspace(0.3, rmax, nr)
        X = origin[0] + np.outer(np.cos(ang), r)
        Y = origin[1] + np.outer(np.sin(ang), r)
        d = self.occ.lookup(X.ravel(), Y.ravel()).reshape(X.shape)
        blocked = np.minimum.accumulate(np.where(d < 0.05, 0, 1), axis=1) == 0   # first occupied cell blocks the rest of the ray
        vis = ~blocked
        i, j = self._ij(X[vis], Y[vis])
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        self.t[i[ok], j[ok]] = t

    def stale_fraction(self, pts, t, t_stale):
        edt = self.occ.lookup(pts[:, 0], pts[:, 1])
        free = edt > 0.15
        if not free.any():
            return 0.0
        i, j = self._ij(pts[free, 0], pts[free, 1])
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        if not ok.any():
            return 0.0
        age = t - self.t[i[ok], j[ok]]
        return float((age > t_stale).mean())


class PerceptionStack:
    def __init__(self, sim, cfg: PerceptionCfg | None = None):
        self.cfg = cfg or PerceptionCfg()
        c = self.cfg
        self.sim = sim
        self.cam = CAM.HeadCamera(sim, CAM.head_spec(W=c.W, H=c.H, rate_hz=c.rate_hz, latency_ticks=c.latency_ticks))
        self.vis = DET.Visibility(self.cam)
        if c.detector == "seg":
            self.det = DET.SegDetector()
        elif c.detector == "depth":
            self.det = DET.DepthDetector(occ=sim.occ, use_static_map=c.static_map in ("occ", "both"), bg=c.static_map in ("raycast", "both"),
                                         r_max=c.r_max)
        elif c.detector == "param":
            self.det = DET.ParametricDetector(c.sigma, c.p_drop, c.seed)
        else:
            raise ValueError(c.detector)
        self.trk = MultiTracker(c.tracker)
        self.queue: deque = deque()
        self.omap = ObservedMap(sim.occ) if sim.occ is not None else None
        self.hfov = 2 * math.atan(math.tan(math.radians(self.cam.spec.fovy) / 2) * c.W / c.H)
        self.tracks = []
        self.vis_now = []
        self.n_frames = 0
        self.n_dets = 0
        self.render_ms = []
        self.frame_log = []             # (capture tick, [(xy, truth)]) of delivered detections, for metrics
        self.last_truth_ids = None

    def _due(self, k):
        r = self.cfg.rate_hz * TICK
        return math.floor((k + 1) * r) > math.floor(k * r)

    def step(self, sim):
        import time
        k = sim.k
        self.trk.predict_to(k)
        truth = list(sim.people_state)
        self.vis_now = [self.vis.of(sim, xy) for xy, _ in truth]
        if self._due(k):
            t0 = time.perf_counter()
            if self.cfg.detector in ("seg", "depth"):
                frame = self.cam.render(k, self.cfg.seed + 7)
                dets = self.det.detect(frame, self.cam, sim)
            else:
                frame = None
                dets = self.det.detect(None, self.cam, sim, vis=self.vis_now, truth=truth)
            self.render_ms.append((time.perf_counter() - t0) * 1000)
            origin, R = self.cam.pose()
            if self.omap is not None:
                yaw_cam = math.atan2(R[1, 2] * -1.0, R[0, 2] * -1.0)          # camera looks along -z_cam
                self.omap.stamp(origin, yaw_cam, self.hfov, self.cam.spec.z_max, sim.t)
            self.n_frames += 1
            self.queue.append((k + self.cfg.latency_ticks, k, dets))
        delivered = False
        while self.queue and self.queue[0][0] <= k:
            _, kc, dets = self.queue.popleft()
            self.trk.process_frame(dets, kc)
            self.n_dets += len(dets)
            self.frame_log.append((kc, k, [(d.xy.copy(), d.truth) for d in dets]))
            delivered = True
        if not delivered:
            self.trk.predict_frame_tick(k)
        self.tracks = self.trk.output(confirmed_only=True)
        return self.tracks

    def proxy(self, sim):
        return SimProxy(sim, self.tracks)


class TruthPerception:
    """the original behaviour: sim.people_state, exact.  Same interface (for the runner / metrics)."""

    def __init__(self, sim, cfg=None):
        self.sim = sim
        self.cam = CAM.HeadCamera(sim)
        self.vis = DET.Visibility(self.cam)
        self.tracks = []
        self.vis_now = []
        self.omap = None
        self.n_frames = self.n_dets = 0
        self.frame_log = []
        self.render_ms = []
        self.cfg = PerceptionCfg(detector="truth")

    def step(self, sim):
        self.vis_now = [self.vis.of(sim, xy) for xy, _ in sim.people_state]
        return []

    def proxy(self, sim):
        return sim
