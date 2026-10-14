"""bridge/filters.py -- the safety filters of the bridge experiment. SIMULATION / SAVED DATA ONLY (no hardware, no network).

Every filter is a callable   f(obs_hist, act_hist, u_nom, t) -> u   (6,) = [tray accel (3), tray tilt rate (3)]
  obs_hist (n, 13) float32  the observation stream so far (row -1 = now; the ball part is the tracker output in trk mode)
  act_hist (n-1, 6)         the REALISED actions so far (act_hist[i] was applied between obs_hist[i] and obs_hist[i+1])
  u_nom    (6,)             the nominal operator command for this tick
  t                         tick index (0-based)
and has begin(ctx) (once per carry; ctx = dict(renv, hid, seed, T)) and stats() (extras for the record).
Every output is projected to the Piper-feasible box (|a_xy| <= a_max, |a_z| <= a_max, |w| <= w_max per axis). The env then applies
its own tilt-rate slew limit (alpha), exactly as for the nominal command.

  F0  nominal      u = u_nom
  F1  keypoint     SafeBimanual-style cost family, one-step guided selection (see README for the faithfulness note)
  F2  hand rule    clamp the tilt, cap the acceleration
  F3  latent-HJ    least-restrictive filter on a learned WM latent + value (our method)
  F4  oracle       privileged MuJoCo look-ahead (upper bound; expensive)
  F5  set-reach    analytic backup-reachability filter on zonotope sets of the ball (sets/filter.py); no learned part, no hidden mu_roll
Spec strings:  "F1:l_align=3,l_prox=10"   "F3:arm=Rc3k,tau=0.0"   "F2:theta_max=0.15,a_cap=0.7"   "F5:H=60,k_sigma=2,eps_scale=1"   "F0"
"""
from __future__ import annotations

import re
import time

import numpy as np

from world_model.v2.tray_rep import common as K          # thread pins, sys.path
from world_model.v2.tray_rep import env as E
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY

DT = E.DT_WM
H = DT / 4.0                         # physics step (4 per tick)
Z = np.array([0.0, 0.0, 1.0])


# ===================================================================================== helpers
def parse_spec(s):
    """'F1:l_align=3,l_prox=10' -> ('F1', {'l_align': 3.0, 'l_prox': 10.0}). Values: float if possible, else str."""
    name, _, rest = s.partition(":")
    p = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        try:
            p[k] = float(v)
        except ValueError:
            p[k] = v
    return name, p


def spec_tag(s):
    """filesystem-safe tag of a spec string."""
    name, p = parse_spec(s)
    return name + ("__" + "_".join(f"{k}-{v:g}" if isinstance(v, float) else f"{k}-{v}" for k, v in sorted(p.items())) if p else "")


def project_box(u, a_max, w_max):
    """project rows of u (..., 6) to the Piper box: |a_xy| <= a_max (vector), |a_z| <= a_max, |w_i| <= w_max."""
    u = np.array(u, np.float64, copy=True)
    na = np.hypot(u[..., 0], u[..., 1])
    s = np.where(na > a_max, a_max / np.maximum(na, 1e-12), 1.0)
    u[..., 0] *= s
    u[..., 1] *= s
    u[..., 2] = np.clip(u[..., 2], -a_max, a_max)
    u[..., 3:] = np.clip(u[..., 3:], -w_max, w_max)
    return u


def predict_tray(c, v, w_prev, Ucand, alpha):
    """one-tick TRAY-only kinematics for N candidate commands (the same maths as the env: slew limit, 4 substeps of
    c += h (w x c), semi-implicit position). Returns c_next (N,3), dx (N,3) world displacement of the tray origin."""
    Ucand = np.atleast_2d(Ucand)
    N = len(Ucand)
    if alpha is None:
        w = Ucand[:, 3:]
    else:
        lim = alpha * DT
        w = w_prev + np.clip(Ucand[:, 3:] - w_prev, -lim, lim)
    cc = np.tile(c, (N, 1))
    for _ in range(4):
        cc = cc + H * np.cross(w, cc)
        cc /= np.linalg.norm(cc, axis=1, keepdims=True)
    dx = 4 * H * v + 10 * H * H * Ucand[:, :3]
    return cc, dx


def rot_from_axis(c):
    """(N,3) unit axes -> (N,3,3) minimal rotation taking z to c (yaw ignored)."""
    c = np.atleast_2d(c)
    Kx = np.zeros((len(c), 3, 3))
    Kx[:, 0, 2], Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = c[:, 0], c[:, 1], -c[:, 0], -c[:, 1]      # [z x c]_x
    return np.eye(3) + Kx + (Kx @ Kx) / (1.0 + c[:, 2])[:, None, None]


def edge_keypoints(step=0.02):
    """tray-edge keypoints (tray frame, xy) along the rectangle perimeter, spacing `step`."""
    xs = np.linspace(-HX, HX, int(round(2 * HX / step)) + 1)
    ys = np.linspace(-HY, HY, int(round(2 * HY / step)) + 1)
    pts = [(x, -HY) for x in xs] + [(x, HY) for x in xs] + [(-HX, y) for y in ys[1:-1]] + [(HX, y) for y in ys[1:-1]]
    return np.array(pts)


# ===================================================================================== base
class Filter:
    name = "F?"

    def __init__(self, renv, **p):
        self.a_max, self.w_max, self.alpha, self.U = renv.a_max, renv.w_max, renv.alpha, renv.U
        self.p = p
        self._x = {}

    def begin(self, ctx):
        self._x = {}

    def stats(self):
        return dict(self._x)

    def __call__(self, obs_hist, act_hist, u_nom, t):
        raise NotImplementedError


class F0(Filter):
    name = "F0"

    def __call__(self, obs_hist, act_hist, u_nom, t):
        return u_nom


# ===================================================================================== F1
class F1Keypoint(Filter):
    """SafeBimanual-style myopic keypoint cost, one-step guided selection.
    candidates = {u_nom} + the scaled 85-set + a local grid around u_nom (3^4 - 1 steps in a_x, a_y, w_x, w_y, size
    grid_frac * limit), all in the box. Pick  argmin ||u-u_nom||^2 + l_align C_align + l_prox C_prox  (the first minimum wins,
    u_nom is first, so ties keep u_nom).
      ||u - u_nom||^2   accel / a_max and tilt rate / w_max, so the two parts are comparable
      C_align = (tilt'/theta_ref)^2          tilt' = angle of the NEXT-tick tray normal from vertical (tray kinematics only)
      C_prox  = (relu(d0 - d')/d0)^2         d' = signed distance (inside > 0) from the perceived ball keypoint to the nearest
                                             tray-edge keypoint, after the tray moves one tick. The ball keypoint is held FIXED
                                             IN THE WORLD during that tick (a keypoint, not a ball model): no friction, no
                                             rolling, no gravity along a tilt. It is the obs ball (tracker output in trk mode).
    vlook (default 0): optional constant-velocity extrapolation of the ball keypoint by `vlook` seconds from the observed
    tray-frame ball velocity. 0 = the faithful myopic baseline; > 0 is an ablation (still no payload dynamics)."""
    name = "F1"

    def __init__(self, renv, l_align=1.0, l_prox=10.0, d0=0.05, theta_ref=0.2, grid_frac=0.25, vlook=0.0, **kw):
        super().__init__(renv, **kw)
        self.l_align, self.l_prox, self.d0, self.theta_ref, self.vlook = float(l_align), float(l_prox), float(d0), \
            float(theta_ref), float(vlook)
        g = np.array(np.meshgrid(*[[-1.0, 0.0, 1.0]] * 4, indexing="ij")).reshape(4, -1).T
        g = g[np.any(g != 0, axis=1)]
        off = np.zeros((len(g), 6))
        off[:, 0], off[:, 1], off[:, 3], off[:, 4] = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
        self.off = off * float(grid_frac) * np.array([self.a_max] * 3 + [self.w_max] * 3)
        self.kp = edge_keypoints()
        self.zc = E.RADIUS + 0.0005                       # ball centre height above the tray plane

    def cost(self, obs, Ucand, u_nom):
        c, v, w_prev = obs[0:3].astype(np.float64), obs[3:6].astype(np.float64), obs[6:9].astype(np.float64)
        bx, by = float(obs[9]) + self.vlook * float(obs[11]), float(obs[10]) + self.vlook * float(obs[12])
        d = Ucand - u_nom
        c_dev = ((d[:, :3] / self.a_max) ** 2).sum(1) + ((d[:, 3:] / self.w_max) ** 2).sum(1)
        cn, dx = predict_tray(c, v, w_prev, Ucand, self.alpha)
        tilt = np.arccos(np.clip(cn[:, 2], -1.0, 1.0))
        c_align = (tilt / self.theta_ref) ** 2
        R0 = rot_from_axis(c)[0]
        q = R0 @ np.array([bx, by, self.zc]) - dx                       # ball keypoint (world-fixed), relative to the new tray origin
        p = np.einsum("nji,nj->ni", rot_from_axis(cn), q)[:, :2]       # in the NEW tray frame
        dk = np.linalg.norm(p[:, None, :] - self.kp[None], axis=2).min(1)
        inside = (np.abs(p[:, 0]) <= HX) & (np.abs(p[:, 1]) <= HY)
        dsig = np.where(inside, dk, -dk)
        c_prox = (np.clip(self.d0 - dsig, 0.0, 2 * self.d0) / self.d0) ** 2
        return c_dev + self.l_align * c_align + self.l_prox * c_prox, c_dev, c_align, c_prox

    def __call__(self, obs_hist, act_hist, u_nom, t):
        cand = np.concatenate([u_nom[None], self.U, project_box(u_nom[None] + self.off, self.a_max, self.w_max)], 0)
        tot, _, ca, cp = self.cost(obs_hist[-1], cand, u_nom)
        i = int(np.argmin(tot))
        self._x["x_cprox_sum"] = self._x.get("x_cprox_sum", 0.0) + float(cp[0])       # C_prox of u_nom (is the hinge ever active?)
        self._x["x_prox_active"] = self._x.get("x_prox_active", 0) + int(cp[0] > 0)
        return cand[i]


# ===================================================================================== F2
class F2HandRule(Filter):
    """'carry it level and gently': cap the horizontal acceleration at a_cap * a_max, and keep the tray tilt <= theta_max.
    If the next-tick tilt (tray kinematics) would exceed theta_max, the tilt command is replaced by a levelling rate
    w = gain * (c x z), clipped to the box. No ball information is used."""
    name = "F2"

    def __init__(self, renv, theta_max=0.15, a_cap=1.0, gain=3.0, **kw):
        super().__init__(renv, **kw)
        self.theta_max, self.a_cap, self.gain = float(theta_max), float(a_cap), float(gain)

    def __call__(self, obs_hist, act_hist, u_nom, t):
        o = obs_hist[-1].astype(np.float64)
        u = np.array(u_nom, np.float64)
        cap = self.a_cap * self.a_max
        na = float(np.hypot(u[0], u[1]))
        if na > cap:
            u[:2] *= cap / na
        u[2] = np.clip(u[2], -cap, cap)
        cn, _ = predict_tray(o[0:3], o[3:6], o[6:9], u[None], self.alpha)
        if np.arccos(np.clip(cn[0, 2], -1.0, 1.0)) > self.theta_max:
            u[3:] = np.clip(self.gain * np.cross(o[0:3], Z), -self.w_max, self.w_max)
            self._x["x_clamp_ticks"] = self._x.get("x_clamp_ticks", 0) + 1
        return project_box(u, self.a_max, self.w_max)


# ===================================================================================== F3
ARM_RE = re.compile(r"^(N0|R|Rc)(\d+)?(k)?$")


def parse_arm(s):
    """'N0' -> ('N0', 0); 'Rc3k' -> ('Rc', 3000); 'Rc300' -> ('Rc', 300)."""
    m = ARM_RE.match(str(s))
    if not m:
        raise ValueError(f"bad arm {s!r} (N0, R3k, Rc3k, Rc1k, Rc300 ...)")
    arm, b, k = m.group(1), m.group(2), m.group(3)
    if arm == "N0":
        return "N0", 0
    return arm, int(b) * (1000 if k else 1)


def load_arm(runs_root, model_setup, arm, seed, obs_mode="full", dev="cpu"):
    """-> (wm, net, dir). Reads <runs_root>/<model_setup>/<ARM>_<mode>_b<budget>_s<seed>/{wm.pt, value/value.pt}."""
    from pathlib import Path
    from world_model.v2.tray_rep.wm import model as M
    from world_model.v2.tray_rep.wm import value as V
    a, b = parse_arm(arm)
    d = Path(runs_root) / model_setup / f"{a}_{obs_mode}_b{b}_s{int(seed)}"
    for f in (d / "wm.pt", d / "value" / "value.pt"):
        if not f.exists():
            raise FileNotFoundError(f"{f} (model not trained yet?)")
    wm = M.load_wm(d / "wm.pt", dev)
    net, _ = V.load_value(d / "value" / "value.pt", dev)
    return wm, net, d


class F3LatentHJ(Filter):
    """Least-restrictive latent-HJ filter (ours). A running WM latent is filtered on the real observation stream (the
    incremental filter of wm/policy.py). Each tick:  v_nom = V(f_WM(z, u_nom)).
        v_nom >= tau  ->  u = u_nom
        else          ->  u = argmax_{u in scaled 85-set} V(f_WM(z, u))
    tau = inf gives the always-fallback WM-greedy policy of phase 1b (a reference)."""
    name = "F3"

    def __init__(self, renv, wm=None, net=None, arm="N0", tau=0.0, seed=0, runs_root=None, model_setup="piper", dev="cpu", **kw):
        super().__init__(renv, **kw)
        import torch
        torch.set_num_threads(1)
        from world_model.v2.tray_rep.wm.policy import WMGreedy
        if wm is None:
            wm, net, _ = load_arm(runs_root, model_setup, arm, int(seed), "full", dev)
        self.torch, self.tau = torch, float(tau)
        self.wg = WMGreedy(wm, net, dev, renv.U, renv.alpha)
        self.arm = arm

    def begin(self, ctx):
        super().begin(ctx)
        self.wg.begin(ctx)

    def __call__(self, obs_hist, act_hist, u_nom, t):
        wg, torch = self.wg, self.torch
        wg._sync(obs_hist, act_hist)
        cand = np.concatenate([np.asarray(u_nom, np.float64)[None], wg.U_np], 0)
        with torch.no_grad():
            ct = torch.as_tensor(cand, dtype=torch.float32, device=wg.dev)
            nxt = wg.dyn.step(wg.z.expand(len(ct), -1), ct)
            V = wg.net(nxt, wg.dyn.margin(nxt)).cpu().numpy()
        v_nom = float(V[0])
        self._x["x_vnom_min"] = min(self._x.get("x_vnom_min", 1e9), v_nom)
        if v_nom >= self.tau:
            return u_nom
        self._x["x_fallback_ticks"] = self._x.get("x_fallback_ticks", 0) + 1
        return wg.U_np[int(np.argmax(V[1:]))]


# ===================================================================================== F4
class F4OracleLookahead(Filter):
    """Privileged MuJoCo look-ahead (upper bound). Every `check_every` ticks, on a shadow env restored from the true state:
      1. hold u_nom for `la` ticks, then ask the oracle (oracle.certify, the library search, horizon `horizon`) whether that
         state is certified (label hold_ok or savable). Not certified or a drop in the look-ahead = unsafe.
      2. safe   -> apply u_nom until the next check.
         unsafe -> certify the CURRENT state and execute the best library sequence (hold-still if hold_ok, brake-only if
                   uncertified) until the next check.
    It reads the hidden mu_roll and the true MuJoCo state. It is NOT a real-robot option; it bounds what a look-ahead can do."""
    name = "F4"

    def __init__(self, renv, check_every=5, la=5, horizon=100, **kw):
        super().__init__(renv, **kw)
        from world_model.v2.tray_rep import oracle as OR
        self.OR = OR
        self.check_every, self.la, self.horizon = int(check_every), int(la), int(horizon)
        self.sh = E.TrayRepEnv("full", renv.a_scale, renv.lip, w_scale=renv.w_scale, alpha=renv.alpha)   # shadow env, no tracker
        self.mode, self.seq, self.k = "nom", None, 0

    def begin(self, ctx):
        super().begin(ctx)
        self.renv, self.hid = ctx["renv"], ctx["hid"]
        self.mode, self.seq, self.k = "nom", None, 0

    def _check(self, u_nom):
        OR, sh = self.OR, self.sh
        snap0 = self.renv.snapshot()
        sh.restore(snap0, mu_roll=self.hid)
        ok = True
        for _ in range(self.la):
            r = sh.tick(u_nom)
            if r["failed"] or r["tunnel"]:
                ok = False
                break
        if ok:
            ok = OR.certify(sh, sh.snapshot(), (OR.D_HOLD,), K=self.horizon)["label"] != OR.UNCERT
            self._x["x_certs"] = self._x.get("x_certs", 0) + 1
        self._x["x_checks"] = self._x.get("x_checks", 0) + 1
        if ok:
            self.mode = "nom"
            return
        sh.restore(snap0, mu_roll=self.hid)
        rec = OR.certify(sh, snap0, (OR.D_HOLD,), K=self.horizon)
        self._x["x_certs"] = self._x.get("x_certs", 0) + 1
        self._x["x_unsafe_checks"] = self._x.get("x_unsafe_checks", 0) + 1
        n = len(sh.U)
        sn = snap0.snap
        if rec["label"] == OR.HOLD_OK:
            self.seq = OR.hold_sequence(self.horizon, snap0.w_slew, sh.alpha)
        else:
            dh = rec["best_dh"] if rec["best_dh"] > 0 else OR.D_HOLD
            seqs = OR.best_sequence(sn.kin[:3].copy(), sn.kin[7:10].copy(), sn.kin[3:7].copy(), sh.U, dh, self.horizon,
                                    sh.a_max, sh.w_max, snap0.w_slew, sh.alpha)
            i = int(rec["best_i"])
            self.seq = seqs[i] if (rec["label"] == OR.SAVABLE and 0 <= i < n + 2) else seqs[n + 1]
        self.mode, self.k = "seq", 0

    def __call__(self, obs_hist, act_hist, u_nom, t):
        if t % self.check_every == 0:
            self._check(np.asarray(u_nom, np.float64))
        if self.mode == "nom":
            return u_nom
        u = self.seq[min(self.k, len(self.seq) - 1)]
        self.k += 1
        return np.asarray(u, np.float64)


# ===================================================================================== factory
def make_filter(spec, renv, seed=0, runs_root=None, model_setup="piper", dev="cpu"):
    name, p = parse_spec(spec)
    if name == "F0":
        return F0(renv)
    if name == "F1":
        return F1Keypoint(renv, **p)
    if name == "F2":
        return F2HandRule(renv, **p)
    if name == "F3":
        return F3LatentHJ(renv, seed=seed, runs_root=runs_root, model_setup=model_setup, dev=dev, **p)
    if name == "F4":
        return F4OracleLookahead(renv, **p)
    if name == "F5":                                   # analytic set-reachability filter (sets/filter.py); lazy import: no cost for F0-F4
        from world_model.v2.tray_rep.sets.filter import F5SetFilter
        return F5SetFilter(renv, **p)
    if name == "F5P":                                  # Experiment B: F5 + synthetic VLM prior / online belief resistance set (sets/vlmprior/filter.py)
        from world_model.v2.tray_rep.sets.vlmprior.filter import F5PFilter
        return F5PFilter(renv, **p)
    raise ValueError(f"unknown filter {spec!r}")
