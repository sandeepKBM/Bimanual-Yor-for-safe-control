"""fl_log.py -- FL-MPC fallback counters and TRUE constraint residuals per episode/segment (11-03 F2; LOGGING ONLY, no control change).

Why: FLMPC's `feasible` flag is not proof that the constraints held (the QP rows are sphere proxies, the dual ascent has a tolerance, and the
fallbacks run on an infeasible problem).  This module re-measures against MuJoCo truth, once per control step.

What is recorded (all MEASURED in the sim, per reset() window; one sample per FLMPC.step call):
  counts (FLMPC attributes, deltas since reset): n_infeasible = steps whose hard QP reported feasible=False (== n_relaxed + n_zero),
    n_relaxed = steps answered by `_relaxed_fallback` (collision rows slackened, torque box hard), n_zero = steps answered by a=0,
    n_scale_called = steps that ran `_torque_safe_scale` (it runs on EVERY infeasible step), n_scale_shrunk = of those, steps where it actually
    shrank the commanded acceleration (lam < 1; the only counter added to mpc_fl.py: n_scale_shrunk, a counter, no logic change),
    n_taumax_relaxed = steps on which |c_j| > tau_max_j forced a box row open.
  residuals (true geometry, state at the START of each Ctl.tau call = the result of the previous step's torque; the last step of a window is
    not checked): res_armarm_min_m = min mj_geomDistance over left x right arm collision geoms (hull meshes, A.true_min_dist rules, dmax 0.5:
    0.5 means nothing closer), res_static_min_m = min over arm collision geoms x world-body (static) collision geoms (floor, table, counter...,
    dmax 0.3), res_n_armarm_pen / res_n_static_pen = steps with that distance < 0, res_tau_ratio_max = max |tau|/limit of the FL-MPC output
    (te, ti as returned by FLMPC, BEFORE any driver-side friction compensation or clip), res_tau_joint = its argmax (0-6 left, 7-13 right),
    res_tau_ratio_joint = per-joint maxima (summary() only), res_n_tau_over = steps with a ratio > 1 + 1e-9.
The torque limit used is the one handed to FLMPC (tau_max_vec); `BIMANUAL_FL_RESIDUAL=0` turns the geometry checks off (counters stay).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from . import adapters as A

ARMARM_DMAX = 0.5       # m, A.true_min_dist default
STATIC_DMAX = 0.3       # m
ENABLED = os.environ.get("BIMANUAL_FL_RESIDUAL", "1") != "0"
_CNT = ("n_relaxed", "n_zero_fallback", "n_scale_shrunk", "n_taumax_relaxed", "n_limit_slack", "n_limit_fallback", "n_limit_steps")


def _min_dist(m, d, ga, gb, dmax):
    ft = np.zeros(6)
    best = dmax
    for a in ga:
        for b in gb:
            dist = mujoco.mj_geomDistance(m, d, a, b, dmax, ft)
            if dist < dmax and abs(np.linalg.norm(ft[3:] - ft[:3]) - abs(dist)) > 1e-4:
                continue                            # spurious mesh-mesh value (see A.true_min_dist)
            if dist == 0.0:
                continue
            best = min(best, float(dist))
    return best


class FLRecorder:
    def __init__(self, m, fl, tau_lim):
        self.m, self.fl = m, fl
        lim = np.broadcast_to(np.asarray(tau_lim, float).ravel(), (7,))
        self.lim = np.concatenate([lim, lim])
        # own selection, NOT A.arm_geoms: that one keys on geom_contype != 0, and in the contents scenes the right arm's collision geoms are
        # (contype 0, conaffinity 65) so A.arm_geoms(m, "right") is EMPTY there (a true_min_dist over it silently returns dmax).
        self.gl, self.gr = (self._geoms(m, "left"), self._geoms(m, "right"))
        arm = set(self.gl) | set(self.gr)
        self.gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom_contype[g] or m.geom_conaffinity[g]) and g not in arm]
        self.reset()

    @staticmethod
    def _geoms(m, side):
        return [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"{side}_arm_") and (m.geom_contype[g] or m.geom_conaffinity[g])]

    _RK = ("n_q_past", "tab_cand", "tab_pruned", "q_cand", "q_pruned", "qd_cand", "qd_pruned")

    def _rstats(self):
        r = getattr(self.fl, "_rules", None)         # contents/rules.Rules (rules='full'), attached by CarryEpisode
        return None if r is None else {k: int(r.stats[k]) for k in self._RK}

    def reset(self):
        self.base = {k: int(getattr(self.fl, k, 0)) for k in _CNT}
        self.rbase = self._rstats()
        self.n = 0
        self.armarm_min, self.static_min = ARMARM_DMAX, STATIC_DMAX
        self.n_armarm_pen = self.n_static_pen = self.n_tau_over = 0
        self.tau_joint = np.zeros(14)

    def pre(self, d):
        """true residuals of the current state (call right after mj_forward, before fl.step)."""
        if not ENABLED:
            return
        m = self.m
        a = _min_dist(m, d, self.gl, self.gr, ARMARM_DMAX)
        s = _min_dist(m, d, self.gl + self.gr, self.gs, STATIC_DMAX) if self.gs else STATIC_DMAX
        self.armarm_min, self.static_min = min(self.armarm_min, a), min(self.static_min, s)
        self.n_armarm_pen += int(a < 0.0)
        self.n_static_pen += int(s < 0.0)

    def post(self, te, ti):
        """FL-MPC output torque residual (call right after fl.step)."""
        self.n += 1
        r = np.abs(np.concatenate([te, ti])) / self.lim
        self.tau_joint = np.maximum(self.tau_joint, r)
        self.n_tau_over += int(r.max() > 1.0 + 1e-9)

    def _delta(self, k):
        return int(getattr(self.fl, k, 0)) - self.base[k]

    def scalars(self):
        """flat floats/ints for the per-segment `track` dicts (every key prefixed fl_)."""
        rel, zero = self._delta("n_relaxed"), self._delta("n_zero_fallback")
        extra = {}
        if int(getattr(self.fl, "n_limit_steps", 0)) > 0:      # 11-03 opt-in joint position/speed rows (rules='full'); keys absent otherwise
            extra = dict(fl_n_limit_steps=self._delta("n_limit_steps"), fl_n_limit_slack=self._delta("n_limit_slack"),
                         fl_n_limit_fallback=self._delta("n_limit_fallback"))
        rs = self._rstats()
        if rs is not None:                                     # rule-row bookkeeping (rules.Rules.stats), deltas over this window
            dl = {k: rs[k] - self.rbase[k] for k in rs}
            extra.update(fl_n_q_past_lim=dl["n_q_past"], fl_rows_cand=dl["tab_cand"] + dl["q_cand"] + dl["qd_cand"],
                         fl_rows_pruned=dl["tab_pruned"] + dl["q_pruned"] + dl["qd_pruned"],
                         fl_tab_cand=dl["tab_cand"], fl_tab_pruned=dl["tab_pruned"], fl_q_cand=dl["q_cand"], fl_q_pruned=dl["q_pruned"],
                         fl_qd_cand=dl["qd_cand"], fl_qd_pruned=dl["qd_pruned"])
        return dict(**extra, fl_steps=self.n, fl_n_infeasible=rel + zero, fl_n_relaxed=rel, fl_n_zero=zero, fl_n_scale_called=rel + zero,
                    fl_n_scale_shrunk=self._delta("n_scale_shrunk"), fl_n_taumax_relaxed=self._delta("n_taumax_relaxed"),
                    fl_res_armarm_min_m=self.armarm_min, fl_res_static_min_m=self.static_min,
                    fl_res_n_armarm_pen=self.n_armarm_pen, fl_res_n_static_pen=self.n_static_pen,
                    fl_res_tau_ratio_max=float(self.tau_joint.max()), fl_res_tau_joint=int(np.argmax(self.tau_joint)),
                    fl_res_n_tau_over=self.n_tau_over)

    def summary(self):
        s = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.scalars().items()}
        s["fl_res_tau_ratio_joint"] = [round(float(x), 4) for x in self.tau_joint]
        s["fl_res_geometry_checked"] = bool(ENABLED)
        return s


def aggregate(items):
    """over a list of per-segment scalar dicts (keys fl_*: a track/nominal dict or FLRecorder.scalars()): sums of the counts, min of the
    distances, max of the torque ratio.  Missing keys (records from before 11-03) are skipped; n_with = items that carried the fields."""
    it = [x for x in items if x and x.get("fl_steps") is not None]
    if not it:
        return dict(n_with=0)
    S = lambda k: int(sum(x[k] for x in it))
    return dict(n_with=len(it), steps=S("fl_steps"), n_infeasible=S("fl_n_infeasible"), n_relaxed=S("fl_n_relaxed"), n_zero=S("fl_n_zero"),
                n_scale_shrunk=S("fl_n_scale_shrunk"), n_taumax_relaxed=S("fl_n_taumax_relaxed"),
                armarm_min_m=float(min(x["fl_res_armarm_min_m"] for x in it)), static_min_m=float(min(x["fl_res_static_min_m"] for x in it)),
                n_armarm_pen=S("fl_res_n_armarm_pen"), n_static_pen=S("fl_res_n_static_pen"),
                tau_ratio_max=float(max(x["fl_res_tau_ratio_max"] for x in it)), n_tau_over=S("fl_res_n_tau_over"))


_CLS = {}


def recording_class(base):
    """subclass of FLMPC whose step() calls self._rec.pre/post around the UNCHANGED base step.  A class (not an instance-attribute closure) so that
    copy.deepcopy(ctl) in the episodes' snapshot/restore keeps the recorder bound to the copied controller."""
    if base not in _CLS:
        class RecFLMPC(base):
            _rec = None

            def step(self, model, data, *a, **k):
                r = self._rec
                if r is not None:
                    r.pre(data)
                out = super().step(model, data, *a, **k)
                if r is not None:
                    r.post(out[0], out[1])
                return out
        _CLS[base] = RecFLMPC
    return _CLS[base]
