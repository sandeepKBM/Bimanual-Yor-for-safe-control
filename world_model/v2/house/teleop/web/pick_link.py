"""teleop/web/pick_link.py -- bridge between the browser teleop and house/pick PickSession (PROTOCOL_BETA.md section 2).  SIMULATION ONLY.

Threading: websocket handlers only call enqueue_*() (thread-safe queue).  Everything that touches the PickSession runs on the SIM THREAD:
    pre_tick(sim, enabled, latched, watchdog, ep_state, ready)   drain the queue, cancel-on-stop guard, cache the status
    interlock(cmd)                                               zero base twist / lift / jog while a skill runs (returns a COPY)
    step(sim)                                                    ps.step(sim) -- once per tick, before sim.tick()
    telemetry(pose, lift_h, ep_state, enabled)                   JSON-safe dict for the `pick` key
    on_disconnect()                                              cancel a running skill (client gone)
"""
from __future__ import annotations

import dataclasses
import math
import queue

import numpy as np

FRAME = (640, 360)                 # head camera pixels (boxes are in these)
STALE_POS_M = 0.02                 # ASSUMED, user-approved 2026-11-12: base moved more than this since the snapshot -> boxes stale
STALE_YAW_DEG = 1.0                # ASSUMED, user-approved 2026-11-12
STALE_LIFT_M = 0.01                # ASSUMED, user-approved 2026-11-12
RUNNING = ("planning", "ready", "executing", "cancelling")        # `ready` = plan done, skill not yet instantiated
INTERLOCK = ("planning", "ready", "executing", "cancelling")      # statuses that zero the operator's motion (PROTOCOL 2; `ready` too: the skill starts in this tick's ps.step and expects a still base)
SELECTABLE = ("idle", "failed", "holding", "done")
PROP_KEYS = ("id", "label", "box2d", "mode", "arm", "feasible", "why")


def zero_cmd(cmd, motion_only=True):
    """copy of `cmd` with base twist, lift velocity and tray jog zeroed (and grasp/release too when motion_only is False).  Never mutates `cmd`."""
    kw = dict(twist=np.zeros_like(np.asarray(cmd.twist, float)), lift_v=0.0, jog=np.zeros_like(np.asarray(cmd.jog, float)))
    if not motion_only:
        kw.update(grasp=False, release=False)
    return dataclasses.replace(cmd, **kw)


def clean(o):
    """recursively JSON-safe: numpy -> python, non-finite floats -> None, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PickLink:
    def __init__(self, session):
        self.ps = session
        self.q: queue.Queue = queue.Queue()
        self.last_reject = ""
        self.snap_pose = None            # (x, y, yaw, lift) when the last propose was requested
        self.status = "idle"             # cached on the sim thread

    # ------------------------------------------------------------------ any thread
    def enqueue_propose(self):
        self.q.put(("propose", None))

    def enqueue_select(self, obj_id):
        self.q.put(("select", obj_id))

    def enqueue_cancel(self):
        self.q.put(("cancel", None))

    # ------------------------------------------------------------------ sim thread
    def _status(self):
        try:
            return str(self.ps.state().get("status", "idle"))
        except Exception:      # noqa: BLE001
            return "idle"

    def running(self):
        return self.status in RUNNING

    def is_stale(self, pose, lift_h):
        if self.snap_pose is None:
            return bool(self.ps.proposals())
        x, y, yaw, lift = self.snap_pose
        return bool(math.hypot(pose[0] - x, pose[1] - y) > STALE_POS_M or abs(math.degrees(_wrap(pose[2] - yaw))) > STALE_YAW_DEG
                    or abs(lift_h - lift) > STALE_LIFT_M)

    def _reject(self, why):
        self.last_reject = why

    def _select(self, sim, obj_id, enabled, ep_state, ready):
        """accept/reject rules of PROTOCOL 2; the reason of a rejection goes to last_reject."""
        if not ready:
            return self._reject("not ready")
        if ep_state is not None and ep_state != "ACTIVE":
            return self._reject(f"episode not active ({ep_state})")
        if not enabled:
            return self._reject("not armed")
        if self.status not in SELECTABLE:
            return self._reject(f"pick busy ({self.status})")
        prop = next((p for p in self.ps.proposals() if p.get("id") == obj_id), None)
        if prop is None:
            return self._reject(f"no proposal {obj_id}")
        if not prop.get("feasible"):
            return self._reject(f"infeasible: {prop.get('why', '')}")
        if self.is_stale(sim.pose(), sim.lift_actual()):
            return self._reject("stale proposals: scan again")
        if not self.ps.request_with_snapshot(sim, obj_id):
            return self._reject("session refused (busy)")
        self.last_reject = ""

    def pre_tick(self, sim, enabled, latched=False, watchdog=False, ep_state=None, ready=True):
        self.status = self._status()                          # fresh before the select rules look at it
        while True:
            try:
                kind, arg = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "propose":
                self.snap_pose = (*[float(v) for v in sim.pose()[:3]], float(sim.lift_actual()))
                self.ps.propose(sim, blocking=False)
            elif kind == "select":
                self._select(sim, arg, bool(enabled), ep_state, ready)
            elif kind == "cancel":
                self.ps.cancel()
        self.status = self._status()
        if self.running() and self.status != "cancelling" and (not enabled or latched or watchdog):
            self.ps.cancel()                                  # Esc / disarm / watchdog: release + retreat
            self.status = self._status()

    def on_disconnect(self):
        if self.running() and self.status != "cancelling":
            self.ps.cancel()
            self.status = self._status()

    def interlock(self, cmd):
        return zero_cmd(cmd) if self.status in INTERLOCK else cmd

    def step(self, sim):
        self.ps.step(sim)
        self.status = self._status()

    def telemetry(self, pose, lift_h, ep_state=None, enabled=True):
        st = self.ps.state()
        props = [{k: p.get(k) for k in PROP_KEYS} for p in st.get("proposals", [])]
        stale = self.is_stale(pose, lift_h)
        ready = bool((ep_state is None or ep_state == "ACTIVE") and enabled and st.get("status") == "idle"
                     and any(p["feasible"] for p in props) and not stale)
        return clean(dict(status=st.get("status"), selected=st.get("selected"), error=st.get("error"), skill=st.get("skill"), proposals=props,
                          frame=list(FRAME), snap_pose=None if self.snap_pose is None else list(self.snap_pose), stale=stale, ready_to_pick=ready,
                          last_reject=self.last_reject, backend=st.get("backend"), planner=st.get("planner")))

    def close(self):
        try:
            self.ps.close()
        except Exception:      # noqa: BLE001
            pass
