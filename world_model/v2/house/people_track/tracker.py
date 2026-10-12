"""people_track/tracker.py -- multi-person constant-velocity Kalman tracker.  Pure numpy/scipy.  SIMULATION ONLY.

State per track  x = [px, py, vx, vy] (world, m, m/s), covariance P (4x4).
  * predict   x <- F x,  P <- F P F' + Q(dt),  Q = q_acc * [[dt^3/3, dt^2/2], [dt^2/2, dt]] per axis (white-noise acceleration, q_acc = (m/s^2)^2/Hz).
              Run EVERY 50 Hz tick, including through occlusion: P grows, so the keep-out inflation grows with it.
  * associate gated nearest neighbour, solved jointly by the Hungarian algorithm on the Mahalanobis cost  d2 = y' S^-1 y,  S = H P H' + R_det,
              gate d2 < gate_chi2 AND |y| < gate_m.
  * update    Joseph-form KF update with the detection's own 2x2 covariance R_det.
  * latency   a detection frame captured at tick kc and delivered at tick k > kc is applied at kc (the state history of the last `hist` ticks
              is stored) and the track is re-predicted to k  (exact out-of-sequence-free retrodiction for FIFO delivery).
  * lifecycle birth from an unmatched detection (tentative; P_v = sigma_v0^2), confirmed after `confirm_hits` updates, a tentative track dies after
              `tent_max_miss` consecutive missed frames, a confirmed track coasts for at most `coast_s` seconds (or until sigma_pos > max_sigma)
              without an update; duplicate tracks closer than `merge_m` are merged (the older survives).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

TICK = 0.02


@dataclass
class TrackerCfg:
    q_acc: float = 0.03              # white-noise acceleration spectral density [(m/s^2)^2 s]  (0.03 ~ 0.4 m/s^2 rms over 0.2 s)
    sigma_v0: float = 1.0            # initial velocity std [m/s]
    gate_chi2: float = 13.8          # 99.9% chi2, 2 dof
    gate_m: float = 1.5              # hard Euclidean gate [m]
    confirm_hits: int = 2
    tent_max_miss: int = 2
    coast_s: float = 6.0
    max_sigma: float = 1.5           # delete a coasting track whose position std exceeds this [m]
    merge_m: float = 0.25
    hist: int = 12                   # ticks of state history kept for latency compensation


@dataclass
class Det:
    xy: np.ndarray                   # world position [m]
    cov: np.ndarray                  # 2x2 measurement covariance [m^2]
    n_pts: int = 0
    truth: int = -1                  # EVALUATION ONLY (never read by the tracker)
    kind: str = ""


@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    born: int
    status: str = "tent"            # 'tent' | 'conf'
    hits: int = 1
    miss_run: int = 0
    last_upd: int = 0
    first_conf: int | None = None

    @property
    def name(self):
        return f"trk{self.id}"

    def sigma_pos(self):
        return float(np.sqrt(max(np.trace(self.P[:2, :2]) / 2.0, 0.0)))


def F_mat(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def Q_mat(dt, q):
    Q = np.zeros((4, 4))
    for a in (0, 1):
        Q[a, a] = q * dt ** 3 / 3.0
        Q[a, a + 2] = Q[a + 2, a] = q * dt ** 2 / 2.0
        Q[a + 2, a + 2] = q * dt
    return Q


def predict_xP(x, P, dt, q):
    F = F_mat(dt)
    return F @ x, F @ P @ F.T + Q_mat(dt, q)


def kf_update(x, P, z, R):
    """Joseph-form update with H = [I2 0]."""
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
    y = np.asarray(z, float) - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(4) - K @ H
    return x + K @ y, I_KH @ P @ I_KH.T + K @ R @ K.T


class MultiTracker:
    def __init__(self, cfg: TrackerCfg | None = None):
        self.cfg = cfg or TrackerCfg()
        self.tracks: list[Track] = []
        self.k = -1                                # last tick the tracker has been predicted to
        self._next_id = 0
        self.hist: dict[int, dict[int, tuple]] = {}    # tick -> {track id: (x, P)} (state after the predict of that tick)
        self.n_births = 0
        self.n_deaths = 0

    # ------------------------------------------------------------------ time
    def _snapshot(self, k):
        self.hist[k] = {t.id: (t.x.copy(), t.P.copy()) for t in self.tracks}
        for old in [j for j in self.hist if j < k - self.cfg.hist]:
            del self.hist[old]

    def predict_to(self, k):
        """advance all tracks to tick k (one TICK per step)."""
        while self.k < k:
            if self.k >= 0:
                for t in self.tracks:
                    t.x, t.P = predict_xP(t.x, t.P, TICK, self.cfg.q_acc)
            self.k += 1
            self._snapshot(self.k)

    # ------------------------------------------------------------------ frame
    def process_frame(self, dets: list[Det], kc: int):
        """apply the detection frame captured at tick kc (<= current tick self.k)."""
        cfg = self.cfg
        now = self.k
        snap = self.hist.get(kc)
        replay = set()                             # ids whose state sits at tick kc (to be re-predicted to the present)
        if kc < now:
            if snap is None:                       # older than the history: apply at the present state (graceful)
                kc, snap = now, None
            else:
                for t in self.tracks:
                    if t.id in snap:
                        t.x, t.P = snap[t.id][0].copy(), snap[t.id][1].copy()
                        replay.add(t.id)
        live = [t for t in self.tracks if (snap is None or t.id in snap) or kc == now]
        nT, nD = len(live), len(dets)
        matched_t, matched_d = set(), set()
        if nT and nD:
            C = np.full((nT, nD), 1e6)
            H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
            for i, t in enumerate(live):
                for j, dt in enumerate(dets):
                    y = dt.xy - t.x[:2]
                    S = H @ t.P @ H.T + dt.cov
                    d2 = float(y @ np.linalg.solve(S, y))
                    if d2 < cfg.gate_chi2 and np.linalg.norm(y) < cfg.gate_m:
                        C[i, j] = d2
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] < 1e5:
                    t = live[i]
                    t.x, t.P = kf_update(t.x, t.P, dets[j].xy, dets[j].cov)
                    t.hits += 1
                    t.miss_run = 0
                    t.last_upd = kc
                    if t.status == "tent" and t.hits >= cfg.confirm_hits:
                        t.status = "conf"
                        t.first_conf = kc
                    matched_t.add(i); matched_d.add(j)
        for i, t in enumerate(live):
            if i not in matched_t:
                t.miss_run += 1
        for j, dt in enumerate(dets):
            if j in matched_d:
                continue
            x0 = np.array([dt.xy[0], dt.xy[1], 0.0, 0.0])
            P0 = np.zeros((4, 4)); P0[:2, :2] = dt.cov; P0[2, 2] = P0[3, 3] = cfg.sigma_v0 ** 2
            tr = Track(self._next_id, x0, P0, born=kc, last_upd=kc)
            self._next_id += 1
            self.n_births += 1
            self.tracks.append(tr)
            live.append(tr)
            replay.add(tr.id)
        # the retrodicted states must be re-predicted to the present (replay), writing the history
        if kc < now:
            for j in range(kc + 1, now + 1):
                for t in self.tracks:
                    if t.id in replay:
                        t.x, t.P = predict_xP(t.x, t.P, TICK, cfg.q_acc)
                self._snapshot(j)
        else:
            self._snapshot(now)
        self._lifecycle(now)

    def _lifecycle(self, now):
        cfg = self.cfg
        keep = []
        for t in self.tracks:
            if t.status == "tent" and t.miss_run >= cfg.tent_max_miss:
                self.n_deaths += 1
                continue
            if t.status == "conf" and ((now - t.last_upd) * TICK > cfg.coast_s or t.sigma_pos() > cfg.max_sigma):
                self.n_deaths += 1
                continue
            keep.append(t)
        keep.sort(key=lambda t: t.born)               # merge duplicates: older survives
        out = []
        for t in keep:
            if any(np.linalg.norm(t.x[:2] - o.x[:2]) < cfg.merge_m for o in out):
                self.n_deaths += 1
                continue
            out.append(t)
        self.tracks = out

    def predict_frame_tick(self, k):
        """per-tick maintenance when no frame was processed: expire coasting tracks."""
        self._lifecycle(k)

    # ------------------------------------------------------------------ output
    def output(self, confirmed_only=True):
        return [t for t in self.tracks if (t.status == "conf" or not confirmed_only)]
