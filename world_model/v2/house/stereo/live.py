"""stereo/live.py -- run the stereo sensor beside a live TeleopSim and turn its output into an ocs WorldState for the shield.  SIMULATION ONLY.

    stereo = attach(sim, StereoCfg(...), mode="async")        # once after TeleopSim(...): builds the twin, spawns the sensor process, BLOCKS until READY
    ws = stereo.step(sim)                                     # every 50 Hz tick, after sim.tick():  publish snapshot + run the (cheap) producers
    safe, info = shield.filter(ocs.consumers.ShieldView(sim, ws), cmd)
    stereo.close()

mode  'async'   real-time: the child renders at its own pace from the latest snapshot; the newest delivered frame is used (staleness = its compute time)
      'sync'    deterministic: every due tick the parent publishes and WAITS for that frame, then delivers it ceil(latency_ms / 20 ms) ticks later (nominal, default 150 ms)
      'inproc'  the sensor runs inside this process (blocks the loop; the cost the separate process avoids)

WorldState content: robot (BaseProducer: odometry + encoders), tray (stereo plane o FK prior), cup (stereo cylinder fit; before the first sighting an engineering PRIOR
cup in the nominal tray position with a wide set), people (PeopleProducer on the stereo depth).  Everything Source.STEREO / TRACKER / ODOM / FK / PRIOR: privileged() == [].
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .cfg import StereoCfg
from . import twin as TW, adapters as AD, payload as PL
from .sensor import StereoSensor

TICK = 0.02


class StereoSensors:
    """the `sensors` object of ocs.views.LiveView (head_frame / wrist) backed by the stereo sensor, plus stereo_frame() for the payload producer."""

    def __init__(self, sim, cfg: StereoCfg, mode="async", stride=4, spec=None, wrist_sensors=None, seed=0, twin_path=None, latency_floor_ticks=1, history=32, latency_ms=150.0, static_bg=True):
        assert mode in ("async", "sync", "inproc")
        self.sim, self.cfg, self.mode, self.stride = sim, cfg, mode, stride
        self.static_bg = static_bg     # False: no people-free twin render and NO static-depth fallback cast against the true model (zs = inf): the caller subtracts a PERCEIVED map instead
        self.wrist_sensors = wrist_sensors
        self.rows, self.cols, self.gx, self.gy = AD.grid_rays(cfg, stride)
        self.base_at = {}
        self.enc_at = {}            # tick -> encoders (the FK prior / lift the payload estimator must use is the one at the EXPOSURE time)
        self.est_at = {}            # tick -> odometry pose estimate at that tick (people frames are placed with the estimate AT CAPTURE, not at delivery)
        self.loc = None
        self.last_cap_tick = 0
        self.pending = deque()                      # (deliver_tick, frame)
        self.frame = None                           # newest delivered frame (for stereo_frame())
        self._head = None
        self.lat_ticks = latency_floor_ticks
        self.latency_ms = latency_ms        # sync / inproc: the NOMINAL pair-to-depth latency charged in sim time (measured on an idle node: 3-way 640x360 = 140 ms); async measures the real one
        self.link = self.sens = None
        self.ready_ms = None
        self.n_frames = 0
        self.stale_ms = []
        self.every = max(1, int(round(1.0 / (cfg.rate_hz * TICK))))
        self.cfg_k = dict(cfg.asdict())
        c = StereoCfg.from_dict({**cfg.asdict(), "keep_rgb": True, "self_mask": True, "gt": False, "static_depth": bool(static_bg)})
        self.c = c
        m = TW.build_twin_model(sim, c)
        if mode == "inproc":
            self.sens = StereoSensor(m, c)
            self.sens.load(sim.d.qpos, sim.d.mocap_pos, sim.d.mocap_quat, sim.d.eq_active)
            t0 = time.perf_counter()
            self.sens.warmup()
            self.ready_ms = 1e3 * (time.perf_counter() - t0)
        else:
            from .link import StereoLink
            path = TW.save_model(m, twin_path or TW.OUT / f"twin_live_{id(self)}.mjb")
            t0 = time.perf_counter()
            self.link = StereoLink(sim.m, path, c, sim=sim)
            self.ready_ms = 1e3 * (time.perf_counter() - t0)

    # ------------------------------------------------------------------ per tick
    def tick(self, sim):
        k = sim.k
        due = (k % self.every) == 0
        if self.mode == "inproc":
            if due:
                self.sens.load(sim.d.qpos, sim.d.mocap_pos, sim.d.mocap_quat, sim.d.eq_active)
                fr = self.sens.capture(k, sim.t)
                self._deliver_at(k, fr, sim)
        elif self.mode == "sync":
            if due:
                self.base_at[k] = sim.pose().copy()
                self.link.publish(sim)
                fr = self.link.wait_frame(k, 60.0)
                if fr is not None:
                    self._deliver_at(k, fr, sim)
        else:
            self.link.publish(sim)
            self.base_at[k] = sim.pose().copy()
            seen = getattr(self, "_seen_tick", -1)
            fr = self.link.latest()
            if fr is not None and fr["tick"] > seen:
                self._seen_tick = fr["tick"]
                self.pending.append((k, fr))        # delivered now (its staleness is real wall-clock compute)
                self.stale_ms.append(1e3 * (time.time() - fr["wall_pub"]))
        self._head = None
        while self.pending and self.pending[0][0] <= k:
            _, fr = self.pending.popleft()
            fr["t_deliver"] = sim.t
            fr["enc"] = self.enc_at.get(int(fr["tick"]))
            fr["est_pose"] = self.est_at.get(int(fr["tick"]))
            self.frame = fr
            self._head = fr
            self.n_frames += 1
        if len(self.base_at) > 4096:
            for d_ in (self.base_at, self.enc_at, self.est_at):
                for kk in sorted(d_)[:2048]:
                    del d_[kk]

    def _deliver_at(self, k, fr, sim):
        lat = int(math.ceil(self.latency_ms / (1e3 * TICK)))
        self.base_at[k] = sim.pose().copy()
        self.pending.append((k + max(self.lat_ticks, lat), fr))

    # ------------------------------------------------------------------ LiveView `sensors` protocol
    def head_frame(self, sim):
        fr = self._head
        if fr is None:
            return None
        z = fr
        if "_hf" not in z:
            bt = self.base_at.get(int(fr["tick"]), sim.pose())
            zs_inf = None if self.static_bg else np.full((len(self.rows), len(self.cols)), np.inf)
            z["_hf"] = AD.frame_to_head_frame(fr, self.c, stride=self.stride, sim=sim, base_true=bt, zs_grid=zs_inf)
            if fr.get("est_pose") is not None:
                z["_hf"] = AD.to_est_frame(z["_hf"], bt, fr["est_pose"])      # placed with the odometry estimate AT CAPTURE
        hf = z["_hf"]
        out = dict(hf)
        self.last_cap_tick = int(fr["tick"])
        if self.loc is not None:
            out["base_true"] = np.asarray(self.loc.pose, float).copy()        # PeopleProducer re-expresses  T_est(now) o T_true^-1  -> identity: already placed
        out["tick"] = sim.k                      # the producer clock: the frame is "taken" now (it already carries its own latency)
        return out

    def stereo_frame(self):
        return self._head

    def wrist(self, sim):
        return None if self.wrist_sensors is None else self.wrist_sensors.wrist(sim)

    def close(self):
        if self.link is not None:
            self.link.close()
        if self.sens is not None:
            self.sens.close()


class _CaptureQueue(deque):
    """ocs.PeopleProducer queues (k_now + 3, k_now, dets): replace by (k_now, capture tick, dets) -- the frame is already late; the tracker re-applies it at its
    capture tick (MultiTracker keeps a state history for exactly this) and re-predicts to now."""

    def __init__(self, sensors, latency):
        super().__init__()
        self.sensors, self.lat = sensors, latency

    def append(self, entry):
        dk, kc, dets = entry
        super().append((dk - self.lat, self.sensors.last_cap_tick, dets))


class StereoOCS:
    """pipeline + view bundle: step(sim) -> WorldState."""

    def __init__(self, sim, cfg: StereoCfg, mode="async", calib=None, loc=None, spec_kw=None, people=True, payload=True, seed=0, pcfg=None, det_occ=None):
        """det_occ (opt-in, default None = unchanged): an OccGrid (e.g. the stereo-BUILT perceived map) for the people detector's static-background rejection.  Given: the
        render-based background subtraction (people-free twin = the TRUE house model) is switched OFF and the detector rejects returns near occupied cells of det_occ instead."""
        from world_model.v2.house.ocs.views import LiveView
        from world_model.v2.house.ocs.producers import Pipeline, BaseProducer, TrayFKProducer
        from world_model.v2.house.loc.estimator import PoseEstimator
        self.sim = sim
        self.sensors = StereoSensors(sim, cfg, mode=mode, seed=seed, static_bg=det_occ is None)
        sp = sim.pose()
        self.loc = loc or PoseEstimator(None, seed=seed, true_pose0=(sp[0], sp[1], sp[2]))
        self.sensors.loc = self.loc
        self._own_loc = loc is None
        self.rng = np.random.default_rng(70_000 + seed)
        self.view = _StereoLiveView(sim, self.sensors, self.loc, self.rng)
        ps = [BaseProducer(calib), TrayFKProducer(calib)]
        if payload:
            ps.append(PL.StereoPayloadProducer(self.sensors.c, pcfg))
        if people:
            from world_model.v2.house.people_track.tracker import TrackerCfg
            pp = AD.make_people_producer(AD.stereo_spec(self.sensors.c, self.sensors.stride, **(spec_kw or {})), calib=calib, tracker_cfg=TrackerCfg(hist=96))
            if det_occ is not None:
                pp.det.bg, pp.det.occ, pp.det.use_static_map = False, det_occ, True
            self._pp = pp
            ps.append(pp)
        self.pipe = Pipeline(ps)
        self.pipe.reset(self.view)
        if people:
            self._pp.queue = _CaptureQueue(self.sensors, self._pp.latency)     # (after reset: it rebuilds the queue) frames arrive late by the sensor's own latency: stamp with the CAPTURE tick
        self.ready_ms = self.sensors.ready_ms

    def step(self, sim):
        if self._own_loc:
            self.loc.update(sim.t, sim.pose(), TICK)
        v = self.view
        v.k, v.t = sim.k, sim.t
        self.sensors.enc_at[sim.k] = v.encoders()
        self.sensors.est_at[sim.k] = np.asarray(self.loc.pose, float).copy()
        self.sensors.tick(sim)
        ws = self.pipe.step(v)
        _add_prior_cup(ws, sim.t)
        return ws

    def close(self):
        self.sensors.close()


def _add_prior_cup(ws, t):
    """before the first cup sighting the shield still needs a cup: the engineering prior (cup at the nominal tray position, upright, wide set)."""
    from world_model.v2.tray_rep.sets.ir import Pose, Unc, ZonoShape
    from world_model.v2.tray_rep.sets.zono import Zonotope
    from world_model.v2.house.ocs.state import Source, Kind, ObjectState
    from world_model.v2.house.ocs import util as U
    if "cup" in ws.objects:
        return
    ws.objects["cup"] = ObjectState("cup", Kind.CONTENT, ZonoShape(U.cup_shape()), Pose(np.eye(3), np.array([-0.045, 0.0, 0.0])),
                                    unc=Unc(Zonotope(np.zeros(3), np.diag([0.05, 0.05, 0.003])), 0.15), parent="tray", name="cup", name_source=Source.PRIOR,
                                    dynamics="slip_on_support", source=Source.PRIOR, t_meas=t, visible=False)


def _stereo_view_cls():
    from world_model.v2.house.ocs.views import LiveView

    class StereoLiveView(LiveView):
        def stereo_frame(self):
            return self.sensors.stereo_frame()
    return StereoLiveView


def _StereoLiveView(sim, sensors, loc, rng):
    return _stereo_view_cls()(sim, sensors=sensors, loc=loc, enc_rng=rng)


def attach(sim, cfg: StereoCfg | None = None, mode="async", **kw) -> StereoOCS:
    return StereoOCS(sim, cfg or StereoCfg(mode="3way"), mode=mode, **kw)       # real-time profile: 3-way SGBM (2.2x faster than HH at the same fill, see sweep2)
