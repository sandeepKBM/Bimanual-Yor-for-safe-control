"""world_model/v2/eval/predictors.py -- reference predictors for the harness.

  OraclePredictor      LEAKS the label (y_H itself). Sanity ceiling / unit tests only.
  PostOnsetDetector    fires only at/after onset: a pure DETECTOR. Must score ~0.5
                       pre-onset AUROC and 0 detection -- proves the harness cannot
                       be fooled by detection.
  ConstantPredictor    chance floor.
  SloshPhysicsPredictor  adapter to experiments/slosh/slosh_models.py analytic
                       models (a0 static tilt, a1 effective gravity, b1 world
                       pendulum, b2 cup-frame oscillator) from CUP KINEMATICS
                       (ep["cup_pos"] (T,3), ep["cup_quat"] (T,4) wxyz). Uses
                       CAUSAL backward differences (the slosh study's central
                       differences peek ahead). If ep["kin_hi"] = {hz, pos, quat,
                       idx} (a native-rate stream + low-rate->hi index map) is
                       present, the model runs at the high rate and is sampled at
                       idx -- far better accelerations than 15 Hz differencing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import metrics as M

_SLOSH = Path(__file__).resolve().parents[3] / "experiments" / "slosh"


class OraclePredictor:
    probabilistic = True

    def __init__(self, horizons_s=M.DEFAULT_HORIZONS_S):
        self.horizons_s = tuple(horizons_s); self.name = "oracle(LEAKS LABEL)"

    def predict(self, ep):
        T = M.episode_len(ep); on = M.episode_onset(ep); t0 = int(ep.get("t_start", 0))
        return np.stack([M.labels(on, T, t0, M.horizon_steps(h, ep["hz"]))[0].astype(float)
                         for h in self.horizons_s], 1)


class PostOnsetDetector:
    probabilistic = False
    horizons_s = None
    name = "post-onset detector"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, ep):
        T = M.episode_len(ep); on = M.episode_onset(ep)
        s = self.rng.normal(0, 0.05, T)
        if on >= 0:
            s[on:] += 1.0
        return s


class ConstantPredictor:
    probabilistic = False
    horizons_s = None
    name = "constant"

    def predict(self, ep):
        return np.zeros(M.episode_len(ep))


# --------------------------------------------------------------------------
def _causal_kin(pos, quat, dt, acc=None):
    """Causal (backward-difference) versions of the slosh_models.kinematics fields
    the analytic models consume. pos (T,3), quat (T,4) wxyz."""
    if str(_SLOSH) not in sys.path:
        sys.path.insert(0, str(_SLOSH))
    import slosh_models as sm
    pos = np.asarray(pos, float)[None]; n = sm.quat_axis(np.asarray(quat, float))[None]
    bd = lambda x: np.concatenate([np.zeros_like(x[:, :1]), np.diff(x, axis=1) / dt], 1)
    v = bd(pos); a = bd(v); a[:, :2] = 0.0; v[:, :1] = 0.0
    if acc is not None:                       # logged (FK, 200 Hz-averaged) cup accel: v2 data
        a = np.asarray(acc, float)[None, :, :3]
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    f = a - sm.GVEC; R = sm.swing_frames(n)
    fn = np.linalg.norm(f, axis=-1) + 1e-9
    K = dict(n=n, f=f, R=R, fnorm=fn, f_c=np.einsum("...ji,...j->...i", R, f),
             theta0=np.arccos(np.clip(n[..., 2], -1, 1)),
             theta_eff=np.arccos(np.clip(np.sum(n * f, -1) / fn, -1, 1)))
    return sm, K


class SloshPhysicsPredictor:
    """model in {a0, a1, b1, b2, b3}; params default to the CURRENT fitted values in
    experiments/slosh/data/scores_phys.npz (read at construction, so re-fits propagate).
    NOTE: the study fit them on central-difference kinematics; here kinematics are
    causal backward differences (a few-ms phase lag at 200 Hz)."""
    probabilistic = False
    horizons_s = None
    KEYS = {"b1": "b1_world_pendulum", "b2": "b2_cup_oscillator", "b3": "b3_world_lti"}

    @classmethod
    def fitted(cls, model):
        try:
            import json as _j
            P = _j.loads(str(np.load(_SLOSH / "data" / "scores_phys.npz", allow_pickle=True)["params"]))
            return {k: v for k, v in P.get(cls.KEYS.get(model, ""), {}).items() if k in ("w", "zeta", "kappa")}
        except Exception:
            return {}

    def __init__(self, model="b3", **params):
        self.model = model; self.p = {**self.fitted(model), **params}
        self.name = f"slosh-physics[{model}]"

    def _score(self, pos, quat, hz, acc=None):
        dt = 1.0 / hz
        sm, K = _causal_kin(pos, quat, dt, acc)
        n_sub = 1
        if self.model in ("b1", "b2", "b3") and self.p["w"] * dt > 0.5:
            # semi-implicit Euler is unstable for w*dt >~ 2 (w=35 rad/s at 15 Hz: 2.3):
            # zero-order-hold the inputs and sub-step the filter, sample at tick ends
            n_sub = int(np.ceil(self.p["w"] * dt / 0.5))
            K = {k: np.repeat(v, n_sub, axis=1) for k, v in K.items()}
            dt = dt / n_sub
        if self.model == "a0":
            phi = K["theta0"]
        elif self.model == "a1":
            phi = K["theta_eff"]
        elif self.model == "b1":
            phi = sm.pendulum_world(K, self.p["w"], self.p["zeta"], dt)
        elif self.model == "b2":
            phi = sm.pendulum_cup(K, self.p["w"], self.p["zeta"], dt)
        elif self.model == "b3":
            phi = sm.lti_world(K, self.p["w"], self.p["zeta"], dt)
        else:
            raise ValueError(self.model)
        if "kappa" in self.p:
            phi = sm.risk(phi, self.p["kappa"], dt)
        return phi[0, n_sub - 1::n_sub]

    def predict(self, ep):
        hi = ep.get("kin_hi")
        if hi is not None:
            return self._score(hi["pos"], hi["quat"], float(hi["hz"]))[np.asarray(hi["idx"])]
        if ep.get("cup_pos") is None:
            raise KeyError("SloshPhysicsPredictor needs ep['cup_pos'], ep['cup_quat'] (or ep['kin_hi'])")
        return self._score(ep["cup_pos"], ep["cup_quat"], float(ep["hz"]), ep.get("cup_acc"))


class ClockBaseline:
    """SCRIPT-PHASE CONFOUND CHECK: P(spill within H | time since t_start) from the
    TRAIN episodes' empirical rate (valid steps only), ignoring the observation.
    If a learned model does not beat this, it learned *when the script spills*,
    not slosh. (v1 WM audit: the latent value learned script phase.)"""
    probabilistic = True

    def __init__(self, train_eps, horizons_s=M.DEFAULT_HORIZONS_S, smooth=3, prior=0.5):
        self.horizons_s = tuple(horizons_s); self.name = "clock (time-only) baseline"
        Tm = max(M.episode_len(e) for e in train_eps) + 1
        num = np.zeros((Tm, len(self.horizons_s))); den = np.zeros(Tm)
        for e in train_eps:
            T = M.episode_len(e); on = M.episode_onset(e); t0 = int(e.get("t_start", 0))
            for j, h in enumerate(self.horizons_s):
                y, v = M.labels(on, T, t0, M.horizon_steps(h, float(e["hz"])))
                k = np.nonzero(v)[0] - t0
                num[k, j] += y[v]
                if j == 0:
                    den[k] += 1
        ker = np.ones(2 * smooth + 1)
        cs = lambda x: np.convolve(x, ker, mode="same")
        rate = np.stack([(cs(num[:, j]) + prior * 0.01) / (cs(den) + 0.01) for j in range(len(self.horizons_s))], 1)
        self.rate = np.clip(rate, 0, 1)

    def predict(self, ep):
        T = M.episode_len(ep); t0 = int(ep.get("t_start", 0))
        k = np.clip(np.arange(T) - t0, 0, len(self.rate) - 1)
        return self.rate[k]
