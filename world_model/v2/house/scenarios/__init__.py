"""scenarios/ -- scripted, seeded evaluation scenarios for the shared-autonomy base carry (h5).  SIMULATION ONLY (offline MuJoCo, no robot code).

    corridor.py   ONE standing person at a controlled lateral offset d from the carry route centreline (the livelock curve)
    occlusion.py  a person walks out from behind a wall corner into the route at a chosen time / speed (seeded timing jitter)
    run_scen.py   CLI: scenario x param x config x seed grid -> rows.jsonl + summary.json (resumable)

NO teleop/gatekeeper/checkers file is edited.  The teleop entry points are configured from OUTSIDE:
  * a scenario is a SceneSpec factory registered in `teleop.scene.SCENES` under a unique key (`register`), so `TeleopSim(scene_name=key)` /
    `teleop.run.run_episode(scene=key)` / `gatekeeper.run_gk.run_one(scene=key)` build it unchanged (the key encodes every parameter, so the
    two `get_scene` calls per sim build see the same scene);
  * `ScenSim` (a TeleopSim subclass, installed as `teleop.run.TeleopSim` by `patched()`) defaults `people_clock="time"` (a hidden person that
    starts walking at t > 0 needs the time clock; the legacy clock never starts a track whose first segment is stationary) and a
    PersonPolicy, and logs a per-tick trace (stall time, head-camera visibility of the scripted person).
Minimal opt-in teleop hook that would remove the monkeypatch: `run_episode(..., sim_cls=TeleopSim, sim_kw=None)` forwarded to the constructor.
"""
from __future__ import annotations

import contextlib
import json
import math

import numpy as np

from ..teleop import run as RUN
from ..teleop import scene as SC
from ..teleop.sim import TeleopSim, PersonPolicy, TICK

__all__ = ["ScenarioError", "scene_key", "register", "ScenSim", "patched", "dist_to_route", "route_point"]

REGISTRY: dict = {}          # scene key -> dict(policy=PersonPolicy|None, watch=bool)
WATCH_EVERY = 5              # head-camera visibility log every N ticks (0.1 s)


class ScenarioError(ValueError):
    pass


def scene_key(kind, **params):
    return f"h5@{kind}:" + json.dumps(params, sort_keys=True, default=float)


def register(key, factory, policy=None, watch=False):
    """make `key` resolvable by teleop.scene.get_scene (module-level SCENES registry; process-local, call it in every worker)."""
    SC.SCENES[key] = factory
    REGISTRY[key] = dict(policy=policy, watch=watch)
    return key


def route_point(route, s):
    """(xy, unit tangent) at arc length s along the polyline `route` (clamped to its ends)."""
    R = np.asarray(route, float)
    seg = np.hypot(*np.diff(R, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1))
    t = (R[i + 1] - R[i]) / seg[i]
    return R[i] + t * (s - cum[i]), t


def dist_to_route(xy, route):
    """distance from a point to the route polyline [m]."""
    R = np.asarray(route, float)
    p = np.asarray(xy, float)
    best = math.inf
    for a, b in zip(R[:-1], R[1:]):
        ab = b - a
        u = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + u * ab))))
    return best


class ScenSim(TeleopSim):
    """TeleopSim + people_clock="time" + the registered PersonPolicy; per-tick trace (`trace`) and, for watched scenarios, the head-camera
    visibility of the scripted person (`vis_log` rows (t, in_fov, visible, range))."""

    LAST = None

    def __init__(self, scene_name="h5", *a, **k):
        reg = REGISTRY.get(scene_name)
        if reg is not None:
            k.setdefault("people_clock", "time")
            if reg.get("policy") is not None:
                k.setdefault("people_policy", reg["policy"])
        super().__init__(scene_name, *a, **k)
        self._watch = bool(reg and reg.get("watch")) and self.with_people
        self._vis = None
        self.trace, self.vis_log = [], []
        ScenSim.LAST = self

    def _visibility(self):
        if self._vis is None:
            from ..people_track.camera import HeadCamera
            from ..people_track.detectors import Visibility
            self._vis = Visibility(HeadCamera(self))
        return self._vis

    def tick(self, cmd):
        o = super().tick(cmd)
        self.trace.append((o["t"], float(np.hypot(*o["v"][:2])), float(o["pdist"])))
        if self._watch and self.k % WATCH_EVERY == 0 and self.people_state:
            f, v, r = self._visibility().of(self, self.people_state[0][0])
            self.vis_log.append((o["t"], bool(f), bool(v), float(r)))
        return o

    def extras(self, done_t=None):
        """episode extras from the trace: stall time (base speed < 0.03 m/s before the task is done) and first head-camera sighting."""
        tr = np.array(self.trace) if self.trace else np.zeros((0, 3))
        end = done_t if done_t is not None else (tr[-1, 0] if len(tr) else 0.0)
        pre = tr[tr[:, 0] <= end] if len(tr) else tr
        out = dict(stall_s=float(np.sum(pre[:, 1] < 0.03) * TICK) if len(pre) else 0.0, sim_s=float(tr[-1, 0]) if len(tr) else 0.0)
        if self._watch:
            seen = [r[0] for r in self.vis_log if r[2]]
            out["first_visible_t"] = float(seen[0]) if seen else None
        return out


@contextlib.contextmanager
def patched():
    """install ScenSim as teleop.run.TeleopSim (run_episode / run_gk.run_one construct it by that module global); restores on exit."""
    old = RUN.TeleopSim
    RUN.TeleopSim = ScenSim
    try:
        yield ScenSim
    finally:
        RUN.TeleopSim = old
