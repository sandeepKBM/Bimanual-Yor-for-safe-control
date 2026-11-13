"""ocs/fp_live.py -- LIVE (closed-loop, in-simulation) FoundationPose measurement source for ocs.anchored.  SIMULATION ONLY.

LiveFPSource(MeasurementSource) is the online counterpart of fp_source.RecordedFPSource:
    at the perception rate (10 Hz, same ticks as the recorded head frames: floor((k+1)*rate*TICK) crossing)
      1. the control thread SNAPSHOTS the live sim state (a few small arrays; microseconds) and hands it to a worker thread -- it never waits for it;
      2. the worker renders the torso ZED left eye from that state (fp_render.FPRender, head cam, stereo-law depth noise, robot self-filter),
         builds the NON-PRIVILEGED tray-plane mask from ENCODER FK (fp_render.tray_gate_mask / cv_cam_in_tray, as ocs/fp_pass.py --cam head),
         runs FoundationPose register/track (foundation.pose_fp.Client, the same gating / loss / re-register rules as fp_pass.run) and converts to the FK tray frame;
      3. measure() returns the newest COMPLETED pose that has ARRIVED, stamped with its CAPTURE time (Measurement.t = capture t, not the receive time).
Truth enters ONLY as what the renderer draws (the camera image); the output never reads truth.  Eval error vs truth is computed by the runner, not here.

STALENESS.  ocs/anchored.AnchoredProducer does NOT handle a measurement older than `now`: _condition() re-anchors every hypothesis at meas.p as if it were
current, and Measurement.t is only used for the `visible` hold window and t_meas.  A pose captured L seconds ago is a pose of the cup-in-tray L seconds ago; the cup
has slid relative to the tray since.  Handled HERE (anchored.py untouched): the delivered radius is inflated by a bound on that relative slide,
        r_eff = min(r_cal + v_b * L + SUM_i a_rel,i * dt * (t - t_i), r_cap)        (L = receive t - capture t)
v_b = max |v_rel| over the producer's hypothesis bank (bind(producer); else `v_rel_max`).  a_rel,i = the relative acceleration a Coulomb-sliding cup can have at tick i of the
latency window = max(0, min(|a_support,i|, a_hw) - mu_lo * g) + a_margin, with a_support from the SAME measured odometry twist / cup lever arm the AnchoredProducer uses (a cup sticks while
|a_support| < mu_lo * g, so a cruising base gives only the margin).  Without a bound producer / odometry it falls back to the constant worst case a_rel = a_hw - mu_lo*g (slide_bound).
r_cap = tray half-diagonal (the cup is on the tray); a result older than max_age_s is dropped (counted `expired`): its radius would be the whole tray.  The tray-frame measurement is insensitive to the tray's own motion (it is expressed in the tray frame).  The orientation bound is NOT
inflated (cup tilt rate relative to the tray is not bounded here).  producers.perception_pipeline widens AnchoredProducer.hold_s by `hold_extra_s` so a fresh-but-late
pose still counts as `visible`.

CLOCK.  The sim cannot run in real time with FP in the loop (the TeleopSim + shield + FL-MPC loop itself runs ~5-10x slower than real time), so
  clock="lockstep" (default): FP latency is SIMULATED in sim time.  The worker really runs; its measured WALL latency (submit -> done) * lat_scale is mapped to sim time:
      a result is delivered at sim time >= t_capture + latency, and the simulated worker is busy until then (a due frame while busy is skipped, counted).  A result
      that is not finished in wall time when its sim arrival time comes is delivered at the first later tick (never blocks the loop).
  clock="wall": arrival = the first tick at which the worker has finished (real-time paced sims only; latency = sim time elapsed meanwhile).
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .state import Source
from .anchored import Measurement, MeasurementSource
from . import fp_render as FR, util as U

TICK = 0.02
MESH = Path("data/molmospaces/ocs/fp/cup_sim_mesh/cup_sim.obj")
GAP_REG, BOX_TOL, JUMP_TOL = 5, 0.02, 0.03          # = fp_pass.GAP_REG / BOX_TOL / JUMP_TOL (imported lazily in HeadFPBackend to keep this module light)


G = 9.81
A_HW = 4.0                                           # base hardware slew limit [m/s^2] (teleop.base.A_HW)
R_CAP = float(np.hypot(U.HX, U.HY))                  # tray half-diagonal [m]


def slide_bound(L: float, v_b: float, a_rel: float) -> float:
    """bound on the cup-in-tray displacement over a latency L [s] for a relative speed <= v_b and relative acceleration <= a_rel."""
    L = max(float(L), 0.0)
    return float(v_b) * L + 0.5 * float(a_rel) * L * L


# ====================================================================================== FoundationPose register / track state machine (same rules as ocs/fp_pass.run)
def fp_step(client, name, o, mask, pts, T_tc, st, iters=2, min_px=150, box_dist=None, gap_reg=GAP_REG, box_tol=BOX_TOL, jump_tol=JUMP_TOL):
    """one frame of the fp_pass rules.  st = dict(tracking, last_p, gap) (mutated).  -> dict(ok, mode (0 none / 1 register / 2 track), p (3,), R (3,3) canonical, ms, mask_px)."""
    if box_dist is None:
        from .fp_pass import box_dist
    px = int(mask.sum())
    out = dict(ok=False, mode=0, p=None, R=None, ms=0.0, mask_px=px)
    if px < min_px:
        st["gap"] += 1
        return out
    K = o["K"]
    up_cam = T_tc[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    need_reg = (not st["tracking"]) or st["gap"] > gap_reg
    try:
        if need_reg:
            pose, _, ms = client.register(name, o["rgb"], o["depth"], mask, K, iters=5, up_cam=up_cam)
            used = 1
        else:
            pose, _, ms = client.track(name, o["rgb"], o["depth"], K, iters=iters)
            used = 2
        out["ms"] += ms
    except RuntimeError:
        st["tracking"] = False
        st["gap"] += 1
        return out
    st["gap"] = 0

    def to_tray(pose_):
        T = T_tc @ pose_
        return T[:3, 3].copy(), T[:3, :3].copy()

    p, R = to_tray(pose)
    lost = box_dist(p, pts) > box_tol or (st["last_p"] is not None and used == 2 and np.linalg.norm(p - st["last_p"]) > jump_tol)
    if lost and used == 2:
        try:
            pose, _, ms = client.register(name, o["rgb"], o["depth"], mask, K, iters=5, up_cam=up_cam)
            out["ms"] += ms
            used = 1
            p, R = to_tray(pose)
            lost = box_dist(p, pts) > box_tol
        except RuntimeError:
            lost = True
    out["mode"] = used
    if lost:
        st["tracking"] = False
    else:
        st["tracking"], st["last_p"] = True, p
        out.update(ok=True, p=p, R=FR.canonical_yaw(R))
    return out


# ====================================================================================== worker backend (GPU / EGL; lives in the worker thread)
class HeadFPBackend:
    """render + mask + FoundationPose for the head ZED.  open() builds everything (it MUST run in the thread that will call process(): the EGL context is thread-bound)."""

    def __init__(self, seed, W=640, H=360, mesh=MESH, gpu=5, iters=2, min_px=150, client=None):
        self.seed, self.W, self.H, self.mesh, self.gpu, self.iters, self.min_px = seed, W, H, str(mesh), gpu, iters, min_px
        self.client, self._own_client = client, client is None
        self.st = dict(tracking=False, last_p=None, gap=0)
        self.name = "cup_H"

    def open(self):
        from .views import ArmFK
        from . import fp_pass as FP
        self._box_dist = FP.box_dist
        self.gap_reg, self.box_tol, self.jump_tol = FP.GAP_REG, FP.BOX_TOL, FP.JUMP_TOL
        self.fr = FR.FPRender(self.seed, self.W, self.H, cam="head")
        self.fk = ArmFK(self.fr.sim)                                    # the worker's OWN encoder FK (ArmFK mutates its MjData: not shared with the control thread)
        self.cup_r, self.cup_h = U.cup_dims("sim")
        if self.client is None:
            from world_model.v2.house.foundation.pose_fp import Client
            self.client = Client(sizes=[f"{self.W}x{self.H}"], gpu=self.gpu)
        self.client.warmup()
        self.client.load_mesh(self.name, self.mesh)

    def close(self):
        try:
            if self._own_client and self.client is not None:
                self.client.close()
        finally:
            try:
                self.fr.close()
            except Exception:
                pass

    def process(self, s: dict) -> dict:
        """s = snapshot (render truth + encoders at capture).  -> result dict (no truth)."""
        import cv2
        tm = {}
        t0 = time.perf_counter()
        self.fr.set_state(s["base"], s["lift"], s["tL"], s["tR"], s["tray_p"], s["tray_q"], s["cup_p"], s["cup_q"])
        o = self.fr.render("head", s["k"], seed=self.seed)
        tm["render_ms"] = 1e3 * (time.perf_counter() - t0)
        t1 = time.perf_counter()
        pt, Rt = self.fk.tray(s["qL"], s["qR"], s["lift_enc"])
        org, Rc = FR.head_cam_base(s["lift_enc"])
        T_tc = FR.cv_cam_in_tray(Rt, pt, org, Rc)
        dm = cv2.medianBlur(o["depth"], 3)
        mask, pts = FR.tray_gate_mask(dm, o["K"], T_tc, U.HX, U.HY, shrink=0.02, h_lo=0.005, h_hi=self.cup_h + 0.03, min_px=self.min_px)
        tm["mask_ms"] = 1e3 * (time.perf_counter() - t1)
        t2 = time.perf_counter()
        r = fp_step(self.client, self.name, o, mask, pts, T_tc, self.st, self.iters, self.min_px, self._box_dist, self.gap_reg, self.box_tol, self.jump_tol)
        tm["fp_ms"] = 1e3 * (time.perf_counter() - t2)
        r.update(tm)
        return r


# ====================================================================================== the source
class LiveFPSource(MeasurementSource):
    name = "foundationpose_live"
    source = Source.WRIST

    def __init__(self, backend, rate_hz: float = 10.0, clock: str = "lockstep", lat_scale: float = 1.0, calib=None, prefix: str = "cup",
                 pos_prior: float = 0.006, rot_prior: float = 0.05, a_rel_max: float | None = None, v_rel_max: float = 0.2, hold_extra_s: float = 0.25,
                 mu_lo: float = 0.18, a_margin: float = 0.3, r_cap: float = R_CAP, max_age_s: float = 0.6, blob_holdoff_s: float | None = 0.5,
                 wall=time.perf_counter):
        assert clock in ("lockstep", "wall")
        self.backend, self.rate, self.clock, self.lat_scale = backend, rate_hz * TICK, clock, lat_scale
        self.calib, self.prefix, self.pos_prior, self.rot_prior = calib, prefix, pos_prior, rot_prior
        self.mu_lo, self.a_margin, self.r_cap, self.max_age_s = mu_lo, a_margin, r_cap, max_age_s
        self.a_rel_max = max(A_HW - mu_lo * G, 0.0) if a_rel_max is None else a_rel_max      # worst-case relative acceleration (hardware slew minus the lowest friction)
        self.v_rel_max, self.hold_extra_s, self.blob_holdoff_s = v_rel_max, hold_extra_s, blob_holdoff_s
        self._wall = wall
        self.producer = None
        self._q = queue.Queue(maxsize=1)
        self._done = queue.Queue()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._err = None
        self._thr = None
        self.reset()

    # ---- lifecycle
    def start(self, timeout: float = 1800.0):
        """spawn the worker (opens the backend in it) and block until READY.  Call BEFORE the control loop starts."""
        if self._thr is None:
            self._thr = threading.Thread(target=self._main, name="fp_live_worker", daemon=True)
            self._thr.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("FP worker never became ready")
        if self._err is not None:
            raise RuntimeError(f"FP worker failed to start: {self._err!r}")
        return self

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thr is not None:
            self._thr.join(timeout=30)
        self._thr = None

    def bind(self, producer):
        """the AnchoredProducer that owns this source: its hypothesis-bank velocities bound the relative slide speed."""
        self.producer = producer
        return self

    def reset(self, view=None):
        self._pending = deque()                 # completed in wall, not yet arrived in sim: dicts with avail_t
        self._free_t = -1e9                     # lockstep: sim time the simulated worker is free again
        self._inflight = False
        self._last_cap = -1e9
        self.last_recv_t = -1e9                  # sim time of the last delivered FP measurement (HoldoffSource)
        self._vbuf = deque(maxlen=6)             # base twist history (as AnchoredProducer._support_ctx) -> support acceleration at the cup
        self._acc = deque(maxlen=400)            # (t, |a_support| at the cup) per tick
        self.n_expired = 0
        self.submit_ms = []                     # control-thread cost of one snapshot + hand-off [ms]
        self.events = []                        # every completed job (arrived or not), in capture order, for logs / evaluation
        self.n_due = self.n_submitted = self.n_skipped = self.n_delivered = self.n_superseded = self.n_late = 0

    def _main(self):
        try:
            self.backend.open()
        except BaseException as e:               # noqa: BLE001 -- surfaced by start()
            self._err = e
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                s = self._q.get()
                if s is None:
                    break
                t_start = self._wall()
                try:
                    r = self.backend.process(s)
                except BaseException as e:       # noqa: BLE001 -- a failed frame is a missing measurement, the loop goes on
                    r = dict(ok=False, mode=0, p=None, R=None, ms=0.0, mask_px=0, error=repr(e))
                r.update(k=s["k"], t_cap=s["t"], wall_submit=s["wall_submit"], wall_start=t_start, wall_done=self._wall())
                self._done.put(r)
        finally:
            try:
                self.backend.close()
            except Exception:
                pass

    def prewarm(self, view):
        """BEFORE the control loop: register once on the current (standstill) frame, synchronously, so the first in-loop frames TRACK (the ~2 s register never lands in the loop).
        The result is discarded (not a measurement).  Must not be called once the loop runs: it uses the backend from the calling thread's side while the worker idles."""
        assert self._inflight is False and self._q.empty()
        done = threading.Event()
        out = {}
        orig = self.backend.process

        def job(s):                                       # run in the WORKER (EGL / FP are thread-bound), the caller waits
            r = orig(s)
            out["r"] = r
            done.set()
            return r
        self.backend.process = job
        try:
            self._q.put(self._snapshot(view, self._wall))
            done.wait(600)
        finally:
            self.backend.process = orig
        self._done.get(timeout=5)                          # drop the worker's completion record
        return out.get("r")

    # ---- control-thread side (never blocks)
    def _due(self, k: int) -> bool:
        return int(np.floor((k + 1) * self.rate)) > int(np.floor(k * self.rate))

    @staticmethod
    def _snapshot(view, wall) -> dict:
        tr, enc = view.truth, view.encoders()
        return dict(k=int(view.k), t=float(view.t), wall_submit=wall(), base=np.asarray(tr.base, float).copy(), lift=float(tr.lift), tL=np.asarray(tr.qL, float).copy(),
                    tR=np.asarray(tr.qR, float).copy(), tray_p=np.asarray(tr.tray_p, float).copy(), tray_q=np.asarray(tr.tray_q, float).copy(),
                    cup_p=np.asarray(tr.cup_p, float).copy(), cup_q=np.asarray(tr.cup_q, float).copy(),
                    qL=np.asarray(enc["qL"], float).copy(), qR=np.asarray(enc["qR"], float).copy(), lift_enc=float(enc["lift"]))

    def _drain(self, t):
        while True:
            try:
                r = self._done.get_nowait()
            except queue.Empty:
                return
            self._inflight = False
            r["latency_wall"] = r["wall_done"] - r["wall_submit"]
            r["latency_sim"] = r["latency_wall"] * self.lat_scale
            r["avail_t"] = r["t_cap"] + r["latency_sim"] if self.clock == "lockstep" else t
            if self.clock == "lockstep":
                self._free_t = r["avail_t"]
            self._pending.append(r)
            self.events.append(r)

    def _arrived(self, t):
        """pop every pending result whose arrival time has come; -> the newest by capture time (older ones are superseded), or None."""
        got = [r for r in self._pending if r["avail_t"] <= t + 1e-9]
        if not got:
            return None
        for r in got:
            self._pending.remove(r)
            r["recv_t"] = t
            r["late"] = bool(self.clock == "lockstep" and t > r["avail_t"] + TICK + 1e-9)      # finished in wall only after its sim arrival time
            self.n_late += int(r["late"])
        got.sort(key=lambda r: r["t_cap"])
        self.n_superseded += len(got) - 1
        for r in got[:-1]:
            r["superseded"] = True
        return got[-1]

    def _track_support_acc(self, ws, t):
        """|support acceleration| at the cup from the measured odometry twist (same estimate the producer's slide model uses); appended per tick."""
        rob = getattr(ws, "robot", None)
        v = None if rob is None else rob.base_vel
        if v is None:
            return
        self._vbuf.append(np.asarray(v, float))
        if len(self._vbuf) < 6:
            return
        vb = list(self._vbuf)
        v_now, a = np.mean(vb[1:], 0), (np.mean(vb[1:], 0) - np.mean(vb[:-1], 0)) / TICK
        r_b = np.zeros(2)
        try:
            from world_model.v2.house.teleop import base as BS
            H = getattr(self.producer, "H", None)
            m = self.producer.model
            if H is not None and m.parent in ws.objects:
                Ps = ws.world_pose(m.parent)
                pw = np.asarray(Ps.R, float) @ H.P[0] + np.asarray(Ps.p, float)
                r_b = (np.asarray(rob.base.R, float).T @ (pw - np.asarray(rob.base.p, float)))[:2]
            self._acc.append((float(t), float(np.linalg.norm(BS.cup_accel_body(v_now, a, r_b)))))
        except Exception:                                         # a missing lever arm only loosens the bound (fallback below)
            pass

    def stale_bound(self, t_cap: float, t: float) -> float:
        """bound [m] on the cup-in-tray slide between the capture and now (see module docstring)."""
        L = max(float(t - t_cap), 0.0)
        v_b = self._v_bound()
        hist = [(ti, a) for ti, a in self._acc if t_cap - 1e-9 <= ti <= t + 1e-9]
        if len(hist) < max(2, int(0.5 * L / TICK)):
            return slide_bound(L, v_b, self.a_rel_max)
        s = v_b * L
        for ti, a in hist:
            s += (max(0.0, min(a, A_HW) - self.mu_lo * G) + self.a_margin) * TICK * (t - ti)
        return float(s)

    def _v_bound(self) -> float:
        P = self.producer
        H = getattr(P, "H", None)
        if H is None:
            return self.v_rel_max
        return float(max(np.linalg.norm(H.V, axis=1).max(), 0.0)) if len(H.V) else self.v_rel_max

    def measure(self, view, ws, model, t):
        self._track_support_acc(ws, t)
        self._drain(t)
        r = self._arrived(t)
        if self._due(int(view.k)):
            self.n_due += 1
            if self._inflight or (self.clock == "lockstep" and t < self._free_t - 1e-9):
                self.n_skipped += 1
            else:
                try:
                    t_s = time.perf_counter()
                    self._q.put_nowait(self._snapshot(view, self._wall))
                    self.submit_ms.append(1e3 * (time.perf_counter() - t_s))
                    self._inflight = True
                    self.n_submitted += 1
                except queue.Full:
                    self.n_skipped += 1
        if r is None or not r["ok"] or r["t_cap"] <= self._last_cap:
            if r is not None:
                r["delivered"] = False
            return None
        if t - r["t_cap"] > self.max_age_s:
            r["delivered"], r["expired"] = False, True
            self.n_expired += 1
            return None
        self._last_cap = r["t_cap"]
        self.last_recv_t = float(t)
        r["delivered"] = True
        self.n_delivered += 1
        p = np.asarray(r["p"], float).copy()
        R = np.asarray(r["R"], float).copy()
        if model.on_support:
            p = np.array([p[0], p[1], 0.0])
        c = self.calib
        r0 = c.get(f"{self.prefix}_fp_pos", self.pos_prior) if c is not None else self.pos_prior
        rot = c.get(f"{self.prefix}_fp_rot", self.rot_prior) if c is not None else self.rot_prior
        L = float(t - r["t_cap"])
        infl = max(min(self.stale_bound(r["t_cap"], t), self.r_cap - r0), 0.0)
        r["r_eff"] = float(r0 + infl)
        r["latency"] = L
        return Measurement(r["t_cap"], p, float(r0 + infl), R, float(rot), self.source,
                           dict(fp_p=p.copy(), fp_R=R.copy(), fp_t=float(r["t_cap"]), fp_recv_t=float(t), fp_latency=L, fp_r_stale=float(infl),
                                fp_mode=[int(r["mode"])], fp_mask_px=int(r["mask_px"]), fp_sides=["head"]))


class HoldoffSource(MeasurementSource):
    """wrap a FALLBACK source (the wrist blob): it measures only when the primary (LiveFPSource) has delivered nothing for `holdoff_s`.  AnchoredProducer takes the FIRST source that
    measures on a tick; with a latency-delayed primary that lands off the blob's 10 Hz grid, an un-gated blob would override FP at most ticks (radius 20 mm instead of FP's) -- see
    the live runs.  Skipping the blob also skips its (expensive) ray-cast."""

    def __init__(self, inner: MeasurementSource, primary: LiveFPSource, holdoff_s: float = 0.5):
        self.inner, self.primary, self.holdoff_s = inner, primary, holdoff_s
        self.name, self.source = inner.name, inner.source

    def reset(self, view=None):
        self.inner.reset(view)

    def measure(self, view, ws, model, t):
        if t - self.primary.last_recv_t <= self.holdoff_s + 1e-9:
            return None
        return self.inner.measure(view, ws, model, t)
