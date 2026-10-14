"""robust/rfilters.py -- the five filters of TEST 2, all on the same learned value. SIMULATION / SAVED DATA ONLY (torch CPU).

Every filter is a callable  f(obs_hist, act_hist, u_nom, t) -> u  (bridge convention). obs_hist is the CORRUPTED observation
stream. The runner sets `f.info` (the corruptor's info for this tick: acc, age, phase) before each call.
Each filter reduces the tick to ONE number, a safety SCORE s (higher = safer). The shared rule is

        s >= thr  ->  u = u_nom                (no intervention)
        s <  thr  ->  u = the filter's fallback action

so "matched conservatism" is a statement about `thr` only: `thr` is tuned on VAL until all five filters intervene on the same
fraction of ticks. thr = -inf never intervenes (the PROFILE mode: it only records the scores and the latent residuals).

  plain  (i)   s = V(f(z, u_nom))                       the F3 filter (bridge), one latent
  conf   (ii)  s = V(z)                                 the state value, one global threshold (Kim et al. style); `tau`
               can come from `conformal_tau` on VAL (P(fail | V >= tau) <= eps) or be matched; both are reported
  epi    (iii) s = -std_j V_j(f(z_j, u_nom))            disagreement over the 3 seed models (UNISafe style); s = -inf when the
               ensemble mean is below `tau0` (0 = HJ sign: certainly unsafe). Fallback: argmax of the ensemble-mean Q
  aci    (iv)  s = min over 9 latents within radius r_t of V(f(z_i, u_nom)); r_t from the WM residual ||dec(h) - no(obs)|| with
               adaptive conformal inference (alpha_{t+1} = alpha_t + gamma (alpha - err_t); r = (1 - alpha_t) quantile of a
               calibration pool + the recent residuals). The 9 latents: z and z +- r e_k on the 4 ball columns of o_n.
               stale=skip : no ACI update on ticks without a fresh sample (the generous reading: it uses the tracker's flag)
               stale=hold : update every tick (the plain reading: the held value is an observation)
  rob    (v)   s = min over 9 ball observations from the TRACKER set of V(f(z_i, u_nom)). The set (corrupt.TrackerSet) is built
               from the tracker's own covariance: position std grows with (latency + time since the last good sample); the
               set is the centre plus 8 boundary points. kappa scales it (calibrate it for coverage on VAL)
For aci and rob the fallback is argmax_u min_i V(f(z_i, u)) over the same 9 latents (the set-robust greedy); for the others it
is argmax_u V(f(z, u)) (ensemble mean for epi).
Spec strings:  "plain:arm=Rc3k,thr=0.1"  "conf:arm=N0,thr=0.0"  "epi:arm=Rc3k,thr=-0.02,tau0=0"
               "aci:arm=Rc3k,thr=0.1,alpha=0.1,gamma=0.02,stale=skip,calib=/path/pool.npz"  "rob:arm=Rc3k,thr=0.1,scale=0.4"
"""
from __future__ import annotations

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.bridge import filters as BF
from world_model.v2.tray_rep.robust import corrupt as CR
from world_model.v2.tray_rep.robust import lat as LT

NEG_INF = -1e9


class RFilter:
    name = "R?"
    n_models = 1

    def __init__(self, renv, tracks, cond, thr=0.0, record=False, **p):
        self.renv, self.tracks, self.cond = renv, tracks, cond
        self.thr = float(thr)
        self.record = bool(record)
        self.p = p
        self.info = None
        self.begin({})

    def begin(self, ctx):
        for lt in self.tracks:
            lt.begin()
        self.scores, self.resids = [], []
        self.n_fb = 0
        self._begin()

    def _begin(self):
        pass

    def stats(self):
        s = np.array(self.scores) if self.scores else np.zeros(1)
        d = dict(x_fallback_ticks=int(self.n_fb), x_score_min=float(s.min()), x_score_mean=float(s.mean()))
        if self.record:
            d["x_scores"] = [round(float(x), 5) for x in self.scores]
            d["x_resid"] = [round(float(x), 5) for x in self.resids]
        return d

    def score(self, u_nom):
        """-> (s, fallback_fn) ; fallback_fn() -> u. Subclasses."""
        raise NotImplementedError

    def __call__(self, obs_hist, act_hist, u_nom, t):
        for lt in self.tracks:
            lt.sync(obs_hist, act_hist)
        s, fb = self.score(np.asarray(u_nom, np.float64))
        self.scores.append(s)
        if s >= self.thr:
            return u_nom
        self.n_fb += 1
        return fb()


class Plain(RFilter):
    name = "plain"

    def score(self, u_nom):
        q = self.tracks[0].q(None, u_nom)                     # (1, 86)
        return float(q[0, 0]), lambda: self.tracks[0].U_np[int(np.argmax(q[0, 1:]))]


class Conf(RFilter):
    name = "conf"

    def score(self, u_nom):
        lt = self.tracks[0]
        v = float(lt.v_state()[0])
        return v, lambda: lt.U_np[int(np.argmax(lt.q()[0]))]


class Epi(RFilter):
    name = "epi"
    n_models = 3

    def score(self, u_nom):
        qs = np.stack([lt.q(None, u_nom)[0] for lt in self.tracks])      # (M, 86)
        v = qs[:, 0]
        tau0 = float(self.p.get("tau0", 0.0))
        s = NEG_INF if v.mean() < tau0 else -float(v.std())
        return s, lambda: self.tracks[0].U_np[int(np.argmax(qs[:, 1:].mean(0)))]


class Aci(RFilter):
    name = "aci"

    def _begin(self):
        self.alpha_t = float(self.p.get("alpha", 0.1))
        self.r = 0.0
        calib = self.p.get("calib", "")
        self.pool = list(np.load(calib)["resid"]) if calib else []
        self.win = []
        self.n_err = self.n_upd = 0

    def _quantile_radius(self):
        pool = np.array(self.pool[-4000:] + self.win[-100:]) if (self.pool or self.win) else np.zeros(0)
        if len(pool) < 30:
            return 0.0
        lev = 1.0 - self.alpha_t
        return float(np.quantile(pool, min(max(lev, 0.0), 1.0)) * max(1.0, lev))

    def score(self, u_nom):
        lt = self.tracks[0]
        res = lt.resid
        info = self.info or {"acc": True}
        if self.record:
            self.resids.append(res if np.isfinite(res) else 0.0)
        if np.isfinite(res):
            fresh = info.get("acc", True) or self.p.get("stale", "skip") == "hold"
            if fresh:
                err = float(res > self.r)
                self.alpha_t += float(self.p.get("gamma", 0.02)) * (float(self.p.get("alpha", 0.1)) - err)
                self.n_err += err
                self.n_upd += 1
                self.win.append(res)
                self.r = self._quantile_radius()
        ov = lt.radial_variants(self.r)
        q = lt.q(ov, u_nom)                                         # (9, 86)
        return float(q[:, 0].min()), lambda: lt.U_np[int(np.argmax(q[:, 1:].min(0)))]

    def stats(self):
        d = super().stats()
        d.update(x_aci_r=float(self.r), x_aci_alpha=float(self.alpha_t), x_aci_err=float(self.n_err / max(self.n_upd, 1)))
        return d


class Rob(RFilter):
    name = "rob"

    def _begin(self):
        self.ts = CR.TrackerSet(self.cond, kappa=float(self.p.get("kappa", 1.0)), scale=float(self.p.get("scale", 1.0)),
                                sigma_a=float(self.p.get("sigma_a", 1.0)),
                                vmax=float(self.p.get("vmax", 0.6)), lam=float(self.p.get("lam", 0.5)))
        self._rad = []

    def score(self, u_nom):
        lt = self.tracks[0]
        info = self.info
        self.ts.step(info)
        ball = lt.o_n[0, 9:13] * lt.wm.o_sd[9:13] + lt.wm.o_mu[9:13]          # the observed ball (raw units)
        sup = self.ts.support(ball.cpu().numpy().astype(np.float64), info["age"])
        self._rad.append(self.ts.radius(info["age"]))
        q = lt.q(lt.ball_variants(sup), u_nom)                                 # (9, 86)
        return float(q[:, 0].min()), lambda: lt.U_np[int(np.argmax(q[:, 1:].min(0)))]

    def stats(self):
        d = super().stats()
        d.update(x_rad_mean=float(np.mean(self._rad)) if self._rad else 0.0, x_rad_max=float(np.max(self._rad)) if self._rad else 0.0)
        return d


KINDS = dict(plain=Plain, conf=Conf, epi=Epi, aci=Aci, rob=Rob)


def make_rfilter(spec, renv, setup, seed, cond, runs_root, dev="cpu", record=False, tracks=None):
    """spec 'kind:arm=Rc3k,thr=..'. -> RFilter. `seed` = the model seed (epi loads seeds 0, 1, 2 whatever it is)."""
    kind, p = BF.parse_spec(spec)
    cls = KINDS[kind]
    seeds = (0, 1, 2) if cls.n_models == 3 else (int(seed),)
    if tracks is None:
        tracks = LT.load_tracks(runs_root, setup, str(p.pop("arm")), seeds, renv.U, renv.alpha, dev)
    else:
        p.pop("arm", None)
    thr = p.pop("thr", 0.0)
    return cls(renv, tracks, cond, thr=thr, record=record, **p)


def spec_str(kind, arm, thr, **kw):
    s = f"{kind}:arm={arm},thr={thr:.6g}"
    for k, v in kw.items():
        s += f",{k}={v:g}" if isinstance(v, (int, float)) else f",{k}={v}"
    return s
