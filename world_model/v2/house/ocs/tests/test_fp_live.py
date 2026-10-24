"""tests for ocs/fp_live.py: async latest-completed semantics, capture-time stamping, staleness inflation.  MOCKED FP backend / client (no GPU, no GL).  SIMULATION ONLY."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from world_model.v2.house.ocs import fp_live as FL, anchored as AN, producers as P, util as U
from world_model.v2.house.ocs.state import Kind, WorldState, RobotState


class FakeTruth:
    def __init__(self):
        self.base = np.zeros(3); self.lift = 0.1; self.qL = np.zeros(6); self.qR = np.zeros(6)
        self.tray_p = np.zeros(3); self.tray_q = np.array([1.0, 0, 0, 0]); self.cup_p = np.zeros(3); self.cup_q = np.array([1.0, 0, 0, 0])


class FakeView:
    def __init__(self, k):
        self.k, self.t, self.truth = k, k * 0.02, FakeTruth()

    def encoders(self):
        return dict(qL=np.zeros(6), qR=np.zeros(6), qdL=np.zeros(6), qdR=np.zeros(6), lift=0.1)


class MockBackend:
    """process() returns a pose that encodes the capture tick (x = 1 mm * k) so the tests can tell WHICH frame a measurement came from; gate = threading.Event to hold a job."""

    def __init__(self, delay=0.0, ok=True, gate=None):
        self.delay, self.ok, self.gate, self.calls = delay, ok, gate, []
        self.opened = self.closed = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def process(self, s):
        self.calls.append(s["k"])
        if self.gate is not None:
            self.gate.wait(10)
        time.sleep(self.delay)
        p = np.array([1e-3 * s["k"], 0.0, 0.0])
        return dict(ok=self.ok, mode=2, p=p, R=np.eye(3), ms=1.0, mask_px=500, render_ms=1.0, mask_ms=1.0, fp_ms=1.0)


class FakeClock:
    """deterministic wall clock: advance() by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _model():
    return AN.ObjectModel("cup", Kind.CONTENT, U.cup_shape("sim"), parent="tray", on_support=True)


def _wait(cond, timeout=5.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, "timeout"
        time.sleep(0.005)


def _src(backend, **kw):
    return FL.LiveFPSource(backend, **kw).start(timeout=10)


def test_start_opens_backend_in_worker_and_stop_closes():
    b = MockBackend()
    s = _src(b)
    assert b.opened
    s.stop()
    assert b.closed


def test_due_ticks_match_recorded_head_frames():
    s = _src(MockBackend())
    due = [k for k in range(30) if s._due(k)]
    s.stop()
    assert due == [4, 9, 14, 19, 24, 29]                       # the ticks of the recorded head frames / fp_s*.npz


def test_capture_time_stamp_not_receive_time():
    clk = FakeClock()
    b = MockBackend()
    s = _src(b, clock="lockstep", wall=clk)
    m = _model()
    assert s.measure(FakeView(4), None, m, 0.08) is None          # due: submitted
    _wait(lambda: not s._done.empty())
    clk.t = 0.12                                                   # the worker's job spanned wall 0 -> 0.12 s   (done stamped by the FAKE clock at completion)
    # completion stamp was taken when the job finished (clock 0.0), so re-stamp deterministically through the event
    r = s._done.get()
    r["wall_done"] = 0.12
    s._done.put(r)
    got = None
    for k in range(5, 20):
        got = s.measure(FakeView(k), None, m, k * 0.02)
        if got is not None:
            break
    s.stop()
    assert got is not None
    assert got.t == pytest.approx(0.08)                            # CAPTURE time (tick 4), not the receive time
    assert got.extra["fp_t"] == pytest.approx(0.08)
    assert got.extra["fp_recv_t"] == pytest.approx(0.20)           # t >= 0.08 + 0.12 simulated latency, first tick: k = 10
    assert got.extra["fp_latency"] == pytest.approx(0.12)
    assert got.p[0] == pytest.approx(1e-3 * 4) and got.p[2] == 0.0  # pose of the captured frame (tick 4), on-support z = 0


def test_never_blocks_on_slow_fp():
    gate = threading.Event()                                       # the job never finishes until we release it
    b = MockBackend(gate=gate)
    s = _src(b)
    m = _model()
    t0 = time.perf_counter()
    outs = [s.measure(FakeView(k), None, m, k * 0.02) for k in range(0, 60)]
    dt = time.perf_counter() - t0
    assert all(o is None for o in outs) and dt < 0.5               # 60 ticks (3 s sim) with FP stuck: loop was not blocked
    assert s.n_submitted == 1 and s.n_skipped >= 1                 # busy worker: later due frames skipped, not queued
    gate.set()
    s.stop()


def test_latest_completed_wins_older_superseded():
    clk = FakeClock()
    b = MockBackend()
    s = _src(b, clock="wall", wall=clk)                            # wall clock: arrival = first tick the worker is done
    m = _model()
    # three completed jobs captured at ticks 4, 9, 14 sit un-read (the control loop did not call measure in between)
    for i, k in enumerate((4, 9, 14)):
        s._q.put(FL.LiveFPSource._snapshot(FakeView(k), clk))
        _wait(lambda i=i: s._done.qsize() == i + 1)
    got = s.measure(FakeView(15), None, m, 0.30)
    s.stop()
    assert got is not None and got.t == pytest.approx(14 * 0.02)   # the newest COMPLETED frame
    assert got.p[0] == pytest.approx(14e-3)
    assert s.n_superseded == 2


def test_stale_or_failed_results_are_not_redelivered_or_delivered():
    b = MockBackend(ok=False)
    s = _src(b, clock="wall")
    m = _model()
    s.measure(FakeView(4), None, m, 0.08)
    _wait(lambda: not s._done.empty())
    assert s.measure(FakeView(5), None, m, 0.10) is None          # FP failed (ok=False): no measurement
    s.stop()
    assert s.n_delivered == 0 and len(s.events) == 1 and s.events[0]["delivered"] is False

    b = MockBackend()
    s = _src(b, clock="wall")
    s.measure(FakeView(4), None, m, 0.08)
    _wait(lambda: not s._done.empty())
    assert s.measure(FakeView(5), None, m, 0.10) is not None
    assert s.measure(FakeView(6), None, m, 0.12) is None          # same completed pose is delivered ONCE (re-delivery would re-anchor the bank on a stale pose)
    s.stop()


def test_lockstep_busy_until_arrival_skips_due_frames():
    clk = FakeClock()
    b = MockBackend()
    s = _src(b, clock="lockstep", wall=clk)
    m = _model()
    s.measure(FakeView(4), None, m, 0.08)                          # submitted at wall 0.0
    _wait(lambda: not s._done.empty())
    r = s._done.get(); r["wall_done"] = 0.30; s._done.put(r)       # a 300 ms job: the simulated worker is busy until t = 0.38
    s.measure(FakeView(5), None, m, 0.10)                          # drain -> free_t = 0.38
    s.measure(FakeView(9), None, m, 0.18)                          # due (k=9) but busy -> skipped
    assert s.n_submitted == 1 and s.n_skipped == 1
    got = s.measure(FakeView(19), None, m, 0.40)                   # arrived (0.38 <= 0.40)
    assert got is not None and got.t == pytest.approx(0.08)
    s.measure(FakeView(24), None, m, 0.50)                         # free again -> next due frame is submitted
    assert s.n_submitted == 2
    s.stop()


def test_staleness_inflation_formula():
    assert FL.slide_bound(0.0, 0.3, 3.5) == 0.0
    assert FL.slide_bound(0.1, 0.05, 3.5) == pytest.approx(0.05 * 0.1 + 0.5 * 3.5 * 0.01)
    assert FL.slide_bound(-1.0, 1.0, 1.0) == 0.0                   # no negative latency


def test_radius_inflated_by_latency_and_bank_speed():
    clk = FakeClock()
    s = _src(MockBackend(), clock="lockstep", wall=clk, pos_prior=0.006, a_rel_max=2.0, v_rel_max=0.1)
    m = _model()
    s.measure(FakeView(4), None, m, 0.08)
    _wait(lambda: not s._done.empty())
    r = s._done.get(); r["wall_done"] = 0.10; s._done.put(r)       # 100 ms latency
    got = None
    for k in range(5, 12):
        got = s.measure(FakeView(k), None, m, k * 0.02) or got
    s.stop()
    L = got.extra["fp_latency"]
    assert L >= 0.1 - 1e-9
    assert got.r == pytest.approx(0.006 + 0.1 * L + 0.5 * 2.0 * L * L)    # no producer bound -> v_rel_max
    assert got.extra["fp_r_stale"] == pytest.approx(got.r - 0.006)
    # with a bound producer the bank speed is used
    s2 = _src(MockBackend(), clock="lockstep", wall=clk, pos_prior=0.006, a_rel_max=0.0)
    s2.bind(SimpleNamespace(H=SimpleNamespace(V=np.array([[0.0, 0.0], [0.3, 0.4]]))))
    assert s2._v_bound() == pytest.approx(0.5)
    s2.stop()


def test_pipeline_branch_additive_and_hold_widened():
    src = FL.LiveFPSource(MockBackend(), hold_extra_s=0.25)
    pipe = P.perception_pipeline(None, people=False, base=False, names=False, cup_kind="sim", cup_tracker="anchored_fp_live", fp_live=src)
    prod = pipe.producers[-1]
    assert prod.sources[0] is src and src.producer is prod and prod.hold_s == pytest.approx(0.15 + 0.25)
    with pytest.raises(ValueError):
        P.perception_pipeline(None, people=False, cup_tracker="anchored_fp_live")
    # defaults unchanged
    d = P.perception_pipeline(None, people=False, base=False, names=False, cup_kind="sim")
    assert type(d.producers[-1]).__name__ == "CupWristProducer"
    a = P.perception_pipeline(None, people=False, base=False, names=False, cup_kind="sim", cup_tracker="anchored")
    assert a.producers[-1].hold_s == 0.15 and len(a.producers[-1].sources) == 1


def test_fp_step_register_then_track_and_loss_rules():
    class C:
        def __init__(self):
            self.log = []
            self.p = np.array([0.0, 0.0, 0.05])

        def _pose(self):
            T = np.eye(4); T[:3, 3] = self.p; return T

        def register(self, name, rgb, depth, mask, K, iters=5, up_cam=None):
            self.log.append("reg"); return self._pose(), 1.0, 10.0

        def track(self, name, rgb, depth, K, iters=2):
            self.log.append("trk"); return self._pose(), 1.0, 5.0

    o = dict(K=np.eye(3), rgb=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.float32))
    mask = np.ones((4, 4), bool)
    pts = np.array([[-0.01, -0.01, 0.0], [0.01, 0.01, 0.1]])
    st = dict(tracking=False, last_p=None, gap=0)
    c = C()
    bd = lambda p, P_: float(np.linalg.norm(np.maximum(np.maximum(P_.min(0) - p, p - P_.max(0)), 0.0)))
    r1 = FL.fp_step(c, "cup", o, mask, pts, np.eye(4), st, 2, 1, bd)
    r2 = FL.fp_step(c, "cup", o, mask, pts, np.eye(4), st, 2, 1, bd)
    assert c.log == ["reg", "trk"] and r1["ok"] and r2["ok"] and r1["mode"] == 1 and r2["mode"] == 2
    c.p = np.array([0.2, 0.0, 0.05])                               # jump > 3 cm while tracking and far outside the mask box -> re-register on this frame, then lost
    r3 = FL.fp_step(c, "cup", o, mask, pts, np.eye(4), st, 2, 1, bd)
    assert c.log[-2:] == ["trk", "reg"] and not r3["ok"] and st["tracking"] is False
    empty = FL.fp_step(c, "cup", o, np.zeros((4, 4), bool), pts, np.eye(4), st, 2, 1, bd)
    assert not empty["ok"] and empty["mode"] == 0 and st["gap"] == 1


def test_stale_bound_uses_measured_support_acceleration():
    s = _src(MockBackend(), mu_lo=0.18, a_margin=0.3)
    s.bind(SimpleNamespace(H=SimpleNamespace(V=np.zeros((3, 2)))))
    t_cap, t = 1.00, 1.20
    # cruising base: |a_support| = 0.5 < mu_lo*g = 1.77 -> the cup sticks, only the margin remains
    s._acc.extend((t_cap + i * 0.02, 0.5) for i in range(11))
    cruise = s.stale_bound(t_cap, t)
    s._acc.clear()
    # hard braking: |a_support| = 4 m/s^2 -> relative acceleration 4 - 1.77 + 0.3
    s._acc.extend((t_cap + i * 0.02, 4.0) for i in range(11))
    brake = s.stale_bound(t_cap, t)
    s.stop()
    assert cruise < brake
    assert cruise == pytest.approx(sum(0.3 * 0.02 * (t - (t_cap + i * 0.02)) for i in range(11)))     # margin only
    assert brake == pytest.approx(sum((4.0 - 0.18 * 9.81 + 0.3) * 0.02 * (t - (t_cap + i * 0.02)) for i in range(11)))
    # no history -> constant worst case a_hw - mu_lo*g
    s2 = _src(MockBackend())
    assert s2.stale_bound(1.0, 1.2) == pytest.approx(FL.slide_bound(0.2, s2.v_rel_max, 4.0 - 0.18 * 9.81))
    s2.stop()


def test_radius_capped_at_tray_and_old_results_expire():
    clk = FakeClock()
    s = _src(MockBackend(), clock="lockstep", wall=clk, pos_prior=0.006, max_age_s=0.5, r_cap=0.18)
    m = _model()
    s.measure(FakeView(4), None, m, 0.08)
    _wait(lambda: not s._done.empty())
    r = s._done.get(); r["wall_done"] = 0.45; s._done.put(r)       # 450 ms latency: allowed (< 0.5) but the worst-case slide exceeds the tray
    got = None
    for k in range(5, 40):
        got = s.measure(FakeView(k), None, m, k * 0.02) or got
    assert got is not None and got.r == pytest.approx(0.18)        # capped at the tray half-diagonal (r_cap), never above
    s.stop()

    s = _src(MockBackend(), clock="lockstep", wall=clk, max_age_s=0.5)
    s.measure(FakeView(4), None, m, 0.08)
    _wait(lambda: not s._done.empty())
    r = s._done.get(); r["wall_done"] = 2.0; s._done.put(r)        # a 2 s latency (e.g. the first register): dropped, not delivered with a whole-tray radius
    outs = [s.measure(FakeView(k), None, m, k * 0.02) for k in range(5, 140)]
    s.stop()
    assert all(o is None for o in outs) and s.n_expired == 1 and s.n_delivered == 0


def test_prewarm_runs_one_synchronous_job_and_is_not_a_measurement():
    b = MockBackend()
    s = _src(b, clock="wall")
    r = s.prewarm(FakeView(0))
    assert r is not None and r["ok"] and b.calls == [0]
    assert s._done.empty() and s.n_submitted == 0 and s.events == []        # not counted, not delivered
    assert s.measure(FakeView(1), None, _model(), 0.02) is None
    s.stop()


def test_holdoff_suppresses_blob_only_after_fp_delivery():
    calls = []

    class Blob(AN.MeasurementSource):
        name = "blob"

        def measure(self, view, ws, model, t):
            calls.append(t)
            return AN.Measurement(t, np.zeros(3), 0.02)

    s = _src(MockBackend(), clock="wall")
    h = FL.HoldoffSource(Blob(), s, holdoff_s=0.5)
    m = _model()
    assert h.measure(None, None, m, 0.1) is not None                # FP never delivered: the blob is the fallback
    s.last_recv_t = 1.0
    assert h.measure(None, None, m, 1.3) is None and len(calls) == 1  # FP delivered 0.3 s ago: blob suppressed (and not even evaluated)
    assert h.measure(None, None, m, 1.6) is not None                # FP silent > 0.5 s: blob is back
    s.stop()
    src = FL.LiveFPSource(MockBackend(), blob_holdoff_s=0.5)
    pipe = P.perception_pipeline(None, people=False, base=False, names=False, cup_kind="sim", cup_tracker="anchored_fp_live", fp_live=src)
    srcs = pipe.producers[-1].sources
    assert srcs[0] is src and isinstance(srcs[1], FL.HoldoffSource)
    src2 = FL.LiveFPSource(MockBackend(), blob_holdoff_s=None)
    srcs2 = P.perception_pipeline(None, people=False, base=False, names=False, cup_kind="sim", cup_tracker="anchored_fp_live", fp_live=src2).producers[-1].sources
    assert not isinstance(srcs2[1], FL.HoldoffSource)
