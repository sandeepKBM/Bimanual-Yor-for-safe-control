"""grasp/place.py -- torque-based touchdown detection for the friction-grasp PLACE / RELEASE sequence.  SIMULATION ONLY.

The robot lowers the held tray with the lift axis at a slow constant speed until the tray meets a surface.  Touchdown is detected from what the
arms feel -- the measured joint torque (JointState.torque stand-in = the applied FL-MPC torque, noisy, quantised) -- NOT from simulator contacts:

    r_k  = tau_meas_k - (qfrc_bias - qfrc_passive)_k          torque the hand carries beyond holding its own arm still      (6,)
    j_k  = J_pt_k^T [0, 0, g]                                  torque a unit mass at the tray centre would need              (6,)
    m_k  = <j_k, r_k> / <j_k, j_k>                             load the hand is carrying [kg]

While the tray hangs in the hands m_k ~ half the payload; when the surface takes the weight m_k falls to ~0 (and below if the lift keeps pushing).
The detector calibrates on a 0.3 s hold dwell BEFORE the descent (`m_ref`; the load estimate at descent speed 2.5 cm/s is the same to within the noise,
measured) and fires when the summed load drops below `frac` * m_ref for `n_hold` ticks after `min_desc` of descent.  Joint friction (0.3 Nm per joint)
is why the criterion is RELATIVE to the carried load.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import proprio as PR


class TouchdownDetector:
    def __init__(self, m, ld, rd, gb, seed=0, frac=0.5, n_hold=3, t_cal=(0.1, 0.3), min_desc=0.003):
        self.m = m
        self.dofs = (np.asarray(ld), np.asarray(rd))
        self.gb = gb
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 6613]))
        self.frac, self.n_hold, self.t_cal, self.min_desc = frac, n_hold, t_cal, min_desc
        self._jp = np.zeros((3, m.nv))
        self.reset()

    def reset(self):
        self.tau_acc = [np.zeros(6), np.zeros(6)]
        self.n_acc = 0
        self.t = 0.0
        self.cal = []
        self.m_ref = None
        self.m_hist = []
        self.below = 0
        self.fired = False
        self.m_lp = None

    def accumulate_substep(self, d):
        self.tau_acc[0] += d.qfrc_applied[self.dofs[0]]
        self.tau_acc[1] += d.qfrc_applied[self.dofs[1]]
        self.n_acc += 1

    def end_tick(self, d, pt, descended, dt=0.02):
        """close the tick average and update the detector.  pt: world point of the payload (3,) or (2,3).  descended: lowered distance so far [m].
        Returns the per-hand load estimate (2,) [kg]."""
        self.t += dt
        if self.n_acc == 0:
            return np.full(2, np.nan)
        tau = [t / self.n_acc for t in self.tau_acc]
        self.tau_acc = [np.zeros(6), np.zeros(6)]
        self.n_acc = 0
        pt = np.asarray(pt, float)
        mh = np.zeros(2)
        for k in (0, 1):
            dofs = self.dofs[k]
            tm = PR._q(tau[k] + self.rng.normal(0, PR.TAU_SIGMA, 6), PR.TAU_QUANT)
            c = d.qfrc_bias[dofs] - d.qfrc_passive[dofs]
            p = pt if pt.ndim == 1 else pt[k]
            mujoco.mj_jac(self.m, d, self._jp, None, p, self.gb[k])
            j = self._jp[:, dofs].T @ np.array([0.0, 0.0, -float(self.m.opt.gravity[2])])
            mh[k] = float(j @ (tm - c) / (j @ j + 1e-12))
        ms = float(mh.sum())
        self.m_lp = ms if self.m_lp is None else 0.6 * self.m_lp + 0.4 * ms           # ~2-tick smoothing of the noisy torque residual
        self.m_hist.append((self.t, ms, self.m_lp, descended))
        if self.m_ref is None:
            if self.t_cal[0] <= self.t < self.t_cal[1]:
                self.cal.append(ms)
            elif self.t >= self.t_cal[1] and self.cal:
                self.m_ref = float(np.mean(self.cal))
            return mh
        if descended >= self.min_desc and self.m_ref > 0.05:
            self.below = self.below + 1 if self.m_lp < self.frac * self.m_ref else 0
            if self.below >= self.n_hold:
                self.fired = True
        return mh
