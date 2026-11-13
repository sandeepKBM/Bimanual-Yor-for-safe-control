"""PickLink with a FAKE PickSession (no MuJoCo): thread rule, interlock, select rules, cancel-on-stop, stale, telemetry."""
import dataclasses
import json
import threading

import numpy as np

from world_model.v2.house.teleop.web import pick_link as PL


@dataclasses.dataclass
class Cmd:
    twist: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.3, 0.1, 0.2]))
    lift_v: float = 0.05
    jog: np.ndarray = dataclasses.field(default_factory=lambda: np.ones(6))
    grasp: bool = True
    release: bool = False


class FakeSim:
    def __init__(self):
        self.p = [1.0, 2.0, 0.5]
        self.lift = 0.30

    def pose(self):
        return np.array(self.p)

    def lift_actual(self):
        return self.lift


class FakePS:
    def __init__(self):
        self.calls, self.threads = [], set()
        self.status, self.selected, self.error = "idle", None, None
        self.props = []
        self.accept = True

    def _c(self, name, *a):
        self.calls.append((name,) + a)
        self.threads.add(threading.get_ident())

    def propose(self, sim, blocking=True, timeout=120.0):
        self._c("propose", blocking)
        self.props = [dict(id=1, box2d=(10, 20, 30, 40), label="mug", mode="single", arm="left", feasible=True, why="", extra=1),
                      dict(id=2, box2d=(50, 20, 80, 40), label="crate", mode="infeasible", arm=None, feasible=False, why="too wide")]
        return self.props

    def proposals(self):
        return list(self.props)

    def request_with_snapshot(self, sim, i):
        self._c("request", i)
        if not self.accept:
            return False
        self.selected, self.status = i, "planning"
        return True

    def cancel(self):
        self._c("cancel")
        self.status = "cancelling" if self.status in ("planning", "ready", "executing") else "idle"

    def step(self, sim):
        self._c("step")

    def state(self):
        return dict(status=self.status, selected=self.selected, error=self.error, backend="mock", planner="rule", skill=None,
                    proposals=[dict(p) for p in self.props])

    def close(self):
        self._c("close")


def make():
    ps, sim = FakePS(), FakeSim()
    return PL.PickLink(ps), ps, sim


def tick(link, sim, enabled=True, **kw):
    link.pre_tick(sim, enabled, ep_state=kw.pop("ep_state", "ACTIVE"), **kw)


def scan(link, sim):
    link.enqueue_propose()
    tick(link, sim)


def test_queue_is_drained_only_by_the_sim_thread():
    link, ps, sim = make()
    t = threading.Thread(target=lambda: (link.enqueue_propose(), link.enqueue_select(1), link.enqueue_cancel()))
    t.start(); t.join()
    assert ps.calls == []                                         # handler threads touched nothing
    me = threading.get_ident()
    tick(link, sim)
    assert [c[0] for c in ps.calls] == ["propose", "cancel"] or ps.calls[0] == ("propose", False)
    assert ps.threads == {me} and ps.calls[0] == ("propose", False)       # non-blocking propose, on this thread


def test_interlock_zeroes_motion_only_while_running_and_does_not_mutate():
    link, ps, sim = make()
    for stt, zero in (("idle", False), ("planning", True), ("ready", True), ("executing", True), ("cancelling", True), ("holding", False), ("failed", False), ("done", False)):
        ps.status = stt
        tick(link, sim)
        c = Cmd()
        out = link.interlock(c)
        if zero:
            assert not out.twist.any() and out.lift_v == 0 and not out.jog.any() and out.grasp is True and out.release is False
            assert c.twist.any() and c.lift_v == 0.05 and c.jog.all()          # original untouched
        else:
            assert out is c


def test_select_accept_and_reject_rules():
    link, ps, sim = make()
    link.enqueue_select(1); tick(link, sim)
    assert link.last_reject.startswith("no proposal") and not [c for c in ps.calls if c[0] == "request"]
    scan(link, sim)
    link.enqueue_select(2); tick(link, sim)
    assert link.last_reject == "infeasible: too wide"
    link.enqueue_select(1); tick(link, sim, enabled=False)
    assert link.last_reject == "not armed"
    link.enqueue_select(1); tick(link, sim, ep_state="IDLE")
    assert link.last_reject.startswith("episode not active")
    link.enqueue_select(1); tick(link, sim, ready=False)
    assert link.last_reject == "not ready"
    ps.status = "executing"
    link.enqueue_select(1); tick(link, sim)
    assert link.last_reject == "pick busy (executing)"
    ps.status = "idle"
    sim.p[0] += 0.05
    link.enqueue_select(1); tick(link, sim)
    assert link.last_reject.startswith("stale")
    assert not [c for c in ps.calls if c[0] == "request"]
    sim.p[0] -= 0.05
    ps.accept = False
    link.enqueue_select(1); tick(link, sim)
    assert link.last_reject == "session refused (busy)"
    ps.accept = True
    link.enqueue_select(1); tick(link, sim)
    assert link.last_reject == "" and ("request", 1) in ps.calls and ps.status == "planning"


def test_select_allowed_from_failed_holding_done_and_without_episode():
    for stt in ("failed", "holding", "done"):
        link, ps, sim = make()
        scan(link, sim)
        ps.status = stt
        link.enqueue_select(1); tick(link, sim, ep_state=None)           # --episode off
        assert ("request", 1) in ps.calls, stt


def test_cancel_request_and_cancel_on_stop_conditions():
    link, ps, sim = make()
    ps.status = "executing"
    link.enqueue_cancel(); tick(link, sim)
    assert ps.calls.count(("cancel",)) == 1
    for kw in (dict(enabled=False), dict(latched=True), dict(watchdog=True)):
        link, ps, sim = make()
        ps.status = "executing"
        tick(link, sim, **kw)
        assert ps.calls.count(("cancel",)) == 1 and ps.status == "cancelling", kw
        tick(link, sim, **kw)                                           # already cancelling: no repeated cancel
        assert ps.calls.count(("cancel",)) == 1
    link, ps, sim = make()
    ps.status = "holding"                                                 # holding is not a running skill: disarm does not drop the object
    tick(link, sim, enabled=False)
    assert ("cancel",) not in ps.calls
    link, ps, sim = make()
    ps.status = "planning"
    tick(link, sim)
    link.on_disconnect()
    assert ps.calls.count(("cancel",)) == 1


def test_stale_thresholds():
    link, ps, sim = make()
    scan(link, sim)
    assert not link.is_stale(sim.pose(), sim.lift)
    assert not link.is_stale([1.0 + 0.0199, 2.0, 0.5], 0.30)
    assert link.is_stale([1.0 + 0.0201, 2.0, 0.5], 0.30)
    assert link.is_stale([1.0, 2.0, 0.5 + np.radians(1.1)], 0.30)
    assert not link.is_stale([1.0, 2.0, 0.5 + np.radians(0.9)], 0.30)
    assert link.is_stale([1.0, 2.0, 0.5], 0.30 + 0.011)
    assert not link.is_stale([1.0, 2.0, 0.5], 0.30 + 0.009)
    assert link.is_stale([1.0, 2.0, 0.5 + 2 * np.pi - 0.5 * np.pi], 0.30)       # yaw wraps
    assert not link.is_stale([1.0, 2.0, 0.5 + 2 * np.pi], 0.30)


def test_telemetry_keys_ready_flag_and_json():
    link, ps, sim = make()
    t = link.telemetry(sim.pose(), sim.lift, "ACTIVE", True)
    assert t["proposals"] == [] and t["ready_to_pick"] is False and t["snap_pose"] is None and t["frame"] == [640, 360]
    scan(link, sim)
    t = link.telemetry(sim.pose(), sim.lift, "ACTIVE", True)
    assert set(t) == {"status", "selected", "error", "skill", "proposals", "frame", "snap_pose", "stale", "ready_to_pick", "last_reject", "backend", "planner"}
    assert set(t["proposals"][0]) == {"id", "label", "box2d", "mode", "arm", "feasible", "why"} and t["proposals"][0]["box2d"] == [10, 20, 30, 40]
    assert t["ready_to_pick"] is True and t["stale"] is False and t["snap_pose"] == [1.0, 2.0, 0.5, 0.30]
    assert link.telemetry(sim.pose(), sim.lift, "IDLE", True)["ready_to_pick"] is False
    assert link.telemetry(sim.pose(), sim.lift, "ACTIVE", False)["ready_to_pick"] is False
    assert link.telemetry([1.1, 2.0, 0.5], sim.lift, "ACTIVE", True)["ready_to_pick"] is False       # stale
    ps.status = "executing"
    assert link.telemetry(sim.pose(), sim.lift, "ACTIVE", True)["ready_to_pick"] is False
    ps.status = "idle"
    ps.props[0]["feasible"] = False
    assert link.telemetry(sim.pose(), sim.lift, "ACTIVE", True)["ready_to_pick"] is False
    ps.props[0]["feasible"] = True
    ps.props[0]["box2d"] = (np.float32(1.0), np.int64(2), float("nan"), 4)
    ps.error = "x"
    ps.status = "failed"
    t = link.telemetry(sim.pose(), sim.lift, None, True)
    json.dumps(t, allow_nan=False)
    assert t["proposals"][0]["box2d"] == [1.0, 2, None, 4]


def test_step_calls_session_step_and_refreshes_status():
    link, ps, sim = make()
    link.step(sim)
    assert ("step",) in ps.calls and link.status == "idle"
