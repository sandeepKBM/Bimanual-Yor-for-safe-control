"""cost.py -- CarryProblem: the batched cost of a base-twist sequence for a carried cup / tray.  SIMULATION ONLY.

cost(U) = progress | intent  +  clearance (min over robot points, hinge + collision)  +  spill risk (CVaR over sampled mu)  +  smoothness

 * progress (S2):  path frame (p0, t_hat); running (s_end - s_t)+ (linear => minimum-time flavour) + lateral^2 + yaw^2 + terminal;  NON-CONVEX
   clearance means "stay behind the bottle" and "go around it" are both just different low-cost sequences.
 * intent (teleop): sum ((u_t - u_int)/V)^2 with u_int = the operator's stick held over the horizon.
 * clearance: h_t = min_i [field(p_i(t)) - r_i - margin];  hinge w_soft relu(d0 - h)^2  +  w_col [h < 0] + w_pen relu(-h)  (+ person discs).
 * spill: ratio_{b,t,m} = |g_in| / (min(kappa mu_m, tan theta) g)  for mu_m sampled from [mu_lo, mu_hi] (M draws / plan);  per sample
   R_m = sum_t relu(ratio - s0)^2;  cost = CVaR_alpha over m (alpha = 1 -> mean, -> 0 worst case).  mu_nominal mode: M = 1 at the nominal value.
All tensors the cost reads from live in `self.S` (static buffers, updated in place by `set_state`) so the planner can capture one CUDA graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import dynamics as DY
from . import fields as FL


@dataclass
class CostCfg:
    # progress (S2)
    w_prog: float = 1.0
    w_term: float = 6.0
    w_lat: float = 0.5
    w_yaw: float = 0.5
    prog_tol: float = 0.02
    # intent (teleop)
    w_intent: float = 1.0
    # clearance
    margin: float = 0.03
    d0: float = 0.12
    w_soft: float = 60.0
    w_cbf: float = 40.0            # penalty on approaching faster than the hard filter's class-K bound (hdot >= -gamma h): keeps proposals inside the filter's feasible set
    gamma: float = 1.5
    a_brk: float = 0.4
    d_act: float = 0.4
    w_col: float = 40.0
    w_pen: float = 800.0
    # people
    person_r: float = 0.40
    w_person: float = 20.0
    # spill
    use_cup: bool = True
    w_spill: float = 30.0
    s0: float = 0.9
    kappa: float = 0.8
    theta_spill: float = 0.30
    mu_lo: float = 0.18
    mu_hi: float = 0.35
    n_mu: int = 8
    cvar_alpha: float = 0.25
    mu_nominal: float | None = None      # None: robust (sample [mu_lo, mu_hi]); a number: ONE mu (the non-robust comparison)
    # smoothness
    w_du: float = 1.0
    w_jerk: float = 0.2
    jerk_max: float = 15.0          # m/s^3 (shield.ShieldCfg.jerk_max): soft hinge at jerk_frac of it on the planned cup acceleration
    jerk_frac: float = 0.7
    w_jerk_hinge: float = 0.3
    # tilt effort
    w_tilt: float = 0.5


class CarryProblem:
    """mode: "path" (S2) or "intent" (teleop)."""

    def __init__(self, model: DY.BaseModel, H: int, pts_local, pts_r, pts_z, field, cfg: CostCfg, mode="path", device="cuda",
                 r_cup=(0.0, 0.0), n_people=2, v_scale=(0.9, 0.45, 1.2), wall_extra=None):
        self.model, self.H, self.cfg, self.mode, self.dev = model, H, cfg, mode, device
        f32 = dict(dtype=torch.float32, device=device)
        self.field = field
        self.pl = torch.as_tensor(np.asarray(pts_local, float).reshape(-1, 2), **f32)        # (P,2) base-frame xy
        self.pr = torch.as_tensor(np.asarray(pts_r, float), **f32)
        self.pz = torch.as_tensor(np.asarray(pts_z, float), **f32)
        self.nu = 5 if model.use_tilt else 3
        self.dtv = model.dt_vec(H, device)
        self.t_axis = torch.cumsum(self.dtv, 0)
        self.vs = torch.as_tensor(v_scale, **f32)
        S = {}
        S["pose0"] = torch.zeros(3, **f32); S["v0"] = torch.zeros(3, **f32); S["u_prev"] = torch.zeros(self.nu, **f32)
        S["tilt0"] = torch.zeros(2, **f32); S["r_cup"] = torch.as_tensor(r_cup, **f32)
        S["p0"] = torch.zeros(2, **f32); S["t_hat"] = torch.tensor([0.0, 1.0], **f32); S["s_end"] = torch.ones((), **f32)
        S["yaw_ref"] = torch.zeros((), **f32)
        S["u_int"] = torch.zeros(self.nu, **f32)
        S["mu"] = torch.full((cfg.n_mu if cfg.mu_nominal is None else 1,), 0.25, **f32)
        S["people"] = torch.tensor([[1e3, 1e3, 0.0, 0.0]] * n_people, **f32)
        S["a_prev"] = torch.zeros(2, **f32)
        S["d0"] = torch.full((), cfg.d0, **f32)                    # soft clearance zone (shrunk near a dock)
        S["c_world"] = torch.zeros(2, **f32)                      # measured horizontal tray axis (world), added to the rollout's steered tilt
        S["margin_add"] = torch.zeros((), **f32)                  # extra clearance (e.g. dock-reduced wall buffer: negative)
        # pack every state buffer into ONE flat device tensor (S[k] are views): a plan then needs a single host->device copy
        self._layout, off = {}, 0
        for k_, t_ in S.items():
            self._layout[k_] = (off, tuple(t_.shape)); off += t_.numel()
        self.flat = torch.zeros(off, **f32)
        self._host = np.zeros(off, np.float32)
        for k_, t_ in list(S.items()):
            o_, sh_ = self._layout[k_]
            v_ = self.flat[o_:o_ + t_.numel()].view(sh_) if sh_ else self.flat[o_:o_ + 1].view(())
            v_.copy_(t_); self._host[o_:o_ + t_.numel()] = t_.cpu().numpy().reshape(-1); S[k_] = v_
        self.S = S
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- state I/O (host side, cheap copies)
    def _put(self, k, val):
        o, sh = self._layout[k]
        n = int(np.prod(sh)) if sh else 1
        self._host[o:o + n] = np.asarray(val, np.float32).reshape(-1)

    def flush(self):
        self.flat.copy_(torch.from_numpy(self._host))

    def set_state(self, pose, v, u_prev=None, a_prev=None, d0=None, tilt=None, r_cup=None, p0=None, t_hat=None, s_end=None, yaw_ref=None, u_int=None, people=None,
                  margin_add=None, c_world=None, flush=True):
        put = self._put
        put("pose0", pose); put("v0", v)
        if u_prev is not None:
            up = np.zeros(self.nu); u = np.asarray(u_prev, float).reshape(-1); up[:min(self.nu, len(u))] = u[:self.nu]; put("u_prev", up)
        if a_prev is not None: put("a_prev", a_prev)
        if d0 is not None: put("d0", d0)
        if tilt is not None: put("tilt0", tilt)
        if r_cup is not None: put("r_cup", r_cup)
        if p0 is not None: put("p0", p0)
        if t_hat is not None: put("t_hat", t_hat)
        if s_end is not None: put("s_end", s_end)
        if yaw_ref is not None: put("yaw_ref", yaw_ref)
        if c_world is not None: put("c_world", c_world)
        if u_int is not None:
            ui = np.zeros(self.nu); u = np.asarray(u_int, float).reshape(-1); ui[:len(u)] = u; put("u_int", ui)
        if people is not None:
            P = np.zeros(self.S["people"].shape, np.float32); P[:, :2] = 1e3
            for i, p in enumerate(people[:P.shape[0]]):
                P[i] = p
            put("people", P)
        if margin_add is not None: put("margin_add", margin_add)
        if flush:
            self.flush()

    def resample_mu(self):
        """draw the mu samples for this plan (stratified over [mu_lo, mu_hi] + jitter), or the single nominal value."""
        cfg = self.cfg
        if cfg.mu_nominal is not None:
            m = np.array([cfg.mu_nominal], np.float32)
        else:
            u = (np.arange(cfg.n_mu) + self.rng.random(cfg.n_mu)) / cfg.n_mu
            m = (cfg.mu_lo + (cfg.mu_hi - cfg.mu_lo) * u).astype(np.float32)
            m[0], m[-1] = cfg.mu_lo, cfg.mu_hi                         # always include both interval ends
        self._put("mu", m)

    # ---------------------------------------------------------------- the cost
    def world_points(self, pose):
        """pose (B,H,3) -> (B,H,P,2) world xy of the robot points."""
        c = torch.cos(pose[..., 2])[..., None]; s = torch.sin(pose[..., 2])[..., None]
        px, py = self.pl[:, 0], self.pl[:, 1]
        return torch.stack([pose[..., 0:1] + c * px - s * py, pose[..., 1:2] + s * px + c * py], -1)

    def cost(self, U, return_parts=False):
        cfg, S, model = self.cfg, self.S, self.model
        ro = DY.rollout(model, S["pose0"], S["v0"], U, S["r_cup"], S["tilt0"] if model.use_tilt else None)
        pose = ro["pose"]
        B, H = pose.shape[:2]
        parts = {}
        # ---- clearance
        pw = self.world_points(pose)
        d = self.field.query(pw, self.pz)                                      # (B,H,P)
        h = (d - self.pr - cfg.margin - S["margin_add"]).amin(-1)             # (B,H)
        hp_ = d - self.pr - cfg.margin - S["margin_add"]                        # (B,H,P) per-point clearance along the rollout
        h0p = self.field.query(self.world_points(S["pose0"].reshape(1, 1, 3)), self.pz) - self.pr - cfg.margin - S["margin_add"]   # (1,1,P)
        hh = torch.cat([h0p.expand(B, 1, -1), hp_], 1)                          # (B,H+1,P)
        dh = (hh[:, 1:] - hh[:, :-1]) / self.dtv[None, :, None]
        hb = hh[:, :-1]
        # the hard filter's per-point law on its active facet:  hdot >= -min(gamma h, sqrt(2 a_brk h)) (h >= 0) -> penalise faster approach (max over points)
        beta = torch.minimum(cfg.gamma * torch.relu(hb), torch.sqrt(2.0 * cfg.a_brk * torch.relu(hb)))
        viol = torch.relu(-dh - beta) * (hb < cfg.d_act).float()
        c_cbf = cfg.w_cbf * viol.amax(-1).pow(2).sum(1)
        c_clr = c_cbf + (cfg.w_soft * torch.relu(S["d0"] - h) ** 2 + cfg.w_col * (h < 0).float() + cfg.w_pen * torch.relu(-h)).sum(1)
        pe = FL.people_clearance(pw, self.t_axis, S["people"], cfg.person_r)
        hp = pe.amin(-1)
        c_clr = c_clr + cfg.w_person * (torch.relu(0.15 - hp) ** 2 * 10 + 10.0 * torch.relu(-hp)).sum(1)
        parts["clear"] = c_clr; parts["h_min"] = h.amin(1)
        # ---- spill
        if cfg.use_cup:
            ratio = DY.slip_ratio(ro["a"], ro["c"] + S["c_world"], S["mu"], cfg.kappa, cfg.theta_spill)       # (B,H,M)
            R = (torch.relu(ratio - cfg.s0) ** 2).sum(1)                                      # (B,M)
            M = R.shape[1]
            kk = max(1, int(math.ceil(cfg.cvar_alpha * M)))
            c_sp = cfg.w_spill * torch.topk(R, kk, dim=1).values.mean(1)
            parts["spill"] = c_sp; parts["ratio_max"] = ratio.amax((1, 2))
        else:
            c_sp = torch.zeros(B, device=U.device)
        # ---- goal / intent
        if self.mode == "path":
            rel = pose[..., :2] - S["p0"]
            s = rel @ S["t_hat"]
            lat = rel[..., 0] * (-S["t_hat"][1]) + rel[..., 1] * S["t_hat"][0]
            rem = torch.relu(S["s_end"] - cfg.prog_tol - s)
            c_g = cfg.w_prog * rem.sum(1) + cfg.w_term * H * rem[:, -1] + cfg.w_lat * (lat ** 2).sum(1) + cfg.w_yaw * ((pose[..., 2] - S["yaw_ref"]) ** 2).sum(1)
        else:
            e = (U[..., :3] - S["u_int"][:3]) / self.vs
            c_g = cfg.w_intent * (e ** 2).sum((1, 2))
        parts["goal"] = c_g
        # ---- smoothness (on the commanded sequence), tilt effort
        du = torch.cat([U[:, :1, :3] - S["u_prev"][:3], U[:, 1:, :3] - U[:, :-1, :3]], 1) / self.vs
        c_s = cfg.w_du * (du ** 2).sum((1, 2))
        a = ro["a"]
        jerk = torch.cat([a[:, :1] - S["a_prev"], a[:, 1:] - a[:, :-1]], 1) / self.dtv[None, :, None]                 # (B,H,2) m/s^3
        jn = torch.linalg.norm(jerk, dim=-1)
        c_s = c_s + cfg.w_jerk_hinge * (torch.relu(jn / (cfg.jerk_frac * cfg.jerk_max) - 1.0) ** 2).sum(1) + cfg.w_jerk * (jn ** 2).sum(1) * 1e-3
        if model.use_tilt:
            c_s = c_s + cfg.w_tilt * (ro["c"] ** 2).sum((1, 2)) * 10
        tot = c_g + c_clr + c_sp + c_s
        if return_parts:
            parts.update(total=tot, rollout=ro)
            return tot, parts
        return tot
