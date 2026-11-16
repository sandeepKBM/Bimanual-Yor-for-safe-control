"""vdirect/cl_eval.py -- closed-loop ground truth + winner's-curse diagnostics. SIMULATION ONLY.

Replicates world_model/v2/wm_audit/closed_loop.py exactly (same 2000 held-out TEST states: 1500 near the physics
boundary + 500 clearly safe, seed 0; 50 ticks; WM filtering the TRUE pendulum observation stream), but
  * stores the PER-STATE outcome (for the hierarchical bootstrap), and caches the WM-independent references
    (phys_greedy, hold) once;
  * takes a generic policy (action set + Q function) so the M (in-data actions) and P (uncertainty-pessimistic)
    arms use their own max;
  * logs, at t = 0 of every state:
      winner's curse  V_model(argmax) - V_true(argmax), V_true = physics value of the TRUE one-tick successor
                      (same object phys_greedy maximises), vs the same gap for the LOGGED dataset action;
      regret          max_u V_true(u) - V_true(argmax);
      one-step model error (raw-unit error norms of the axis / velocity / angular-rate blocks, |margin error|)
      under the logged action vs under the argmax (imagination step = the value's f, commanded u as action),
      and the 10-tick held-action margin error (logged 10-tick sequence vs held argmax).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.wm_audit.value_gate import ProjWMDyn


def cl_states(dev, cache, n=2000, seed=0):
    """closed_loop.py's state pick (depends only on data + the physics value)."""
    import torch
    cache = Path(cache)
    if cache.exists():
        f = np.load(cache)
        return {k: f[k] for k in f.files}
    D = C.load_split("test")
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    rng = np.random.default_rng(seed)
    cand = np.array([(e, t) for e in range(len(D["lens"])) for t in range(20, D["lens"][e] - 1, 2)])
    s0 = D["start"][cand[:, 0]] + cand[:, 1]
    T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    with torch.no_grad():
        c_all = T_(D["obs"][s0, :3]); c_all = c_all / c_all.norm(dim=-1, keepdim=True)
        Vp_all = C.phys_V(vnet, vdyn, T_(D["n"][s0]), T_(D["nd"][s0]), c_all).cpu().numpy()
    l_all = D["l"][s0]
    near = np.flatnonzero((l_all > 0) & (Vp_all > -0.3) & (Vp_all < 0.4))
    safe = np.flatnonzero((l_all > 0) & (Vp_all >= 0.4))
    n_near = min(len(near), int(0.75 * n))
    pick = np.concatenate([rng.choice(near, n_near, replace=False), rng.choice(safe, n - n_near, replace=False)])
    out = dict(cand=cand[pick], s0=s0[pick], Vp0=Vp_all[pick], n_near=np.array(n_near))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


class Policy:
    """greedy fallback over action set U: q(z) (B, nu) = V(f(z,u)); z = [h, o_n]."""

    def __init__(self, wm, net, dev, U_idx=None, pess=None):
        import torch
        self.wm, self.net, self.dev = wm, net, dev
        self.base = ProjWMDyn(wm, None, dev, project=True)
        self.idx = torch.arange(len(self.base.U), device=dev) if U_idx is None else torch.as_tensor(U_idx, device=dev)
        self.U = self.base.U[self.idx]
        self.pess = pess                                   # PessDyn or None

    def q(self, z, chunk=4096):
        import torch
        out = []
        nu = len(self.U)
        for i in range(0, len(z), chunk):
            zz = z[i:i + chunk]
            b = len(zz)
            if self.pess is None:
                nxt = self.base.step(zz[:, None, :].expand(-1, nu, -1), self.U[None].expand(b, -1, -1))
                out.append(self.net(nxt, self.base.margin(nxt)))
            else:
                za = torch.cat([zz, torch.zeros(b, 1, device=self.dev)], -1)
                nxt = self.pess.step(za[:, None, :].expand(-1, nu, -1), self.U[None].expand(b, -1, -1))
                out.append(self.net(self.pess.features(nxt), self.pess.margin(nxt)))
        return torch.cat(out)


def _rollout(D, st, dev, T, choose, wm=None, z0=None):
    """closed_loop.py loop; choose(z or None, n, nd, c) -> u (B,6). Returns per-state min true margin."""
    import torch
    s0 = st["s0"]
    T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    n = T_(D["n"][s0]); nd = T_(D["nd"][s0]); c = T_(D["obs"][s0, :3]); c = c / c.norm(dim=-1, keepdim=True)
    v = T_(D["obs"][s0, 3:6])
    B = len(s0)
    if wm is not None:
        h = z0[:, :wm.d_h].clone(); o_n = z0[:, wm.d_h:].clone()
    lmin = torch.full((B,), 1.0, device=dev)
    with torch.no_grad():
        for k in range(T):
            z = torch.cat([h, o_n], -1) if wm is not None else None
            u = choose(z, n, nd, c)
            l1, weff, n, nd, c = C.pend_rollout_torch(n, nd, c, u[:, None, :3], u[:, None, 3:], project=True,
                                                      return_state=True)
            lmin = torch.minimum(lmin, l1[:, 0])
            w_obs = weff[:, 0]
            if wm is not None:
                act = torch.cat([u[:, :3], w_obs], -1)
                h, _ = wm.step(h, o_n, act)
                v = v + C.DT_WM * u[:, :3]
                o_n = wm.no(torch.cat([c, v, w_obs], -1))
    return lmin.cpu().numpy()


def reference_outcomes(dev, cache_states, cache_ref, T=50):
    """phys_greedy and hold per-state spill (WM independent) -- cached."""
    import torch
    cache_ref = Path(cache_ref)
    if cache_ref.exists():
        f = np.load(cache_ref)
        return {k: f[k] for k in f.files}
    st = cl_states(dev, cache_states)
    D = C.load_split("test")
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    U = torch.tensor(C.fallback_U(), dtype=torch.float32, device=dev)
    nu = len(U)

    def phys(z, n, nd, c):
        B = len(n)
        rep = lambda x: x.repeat_interleave(nu, 0)
        uu = U.repeat(B, 1)
        n2, nd2, c2 = C.pend_state_after(rep(n), rep(nd), rep(c), uu[:, None, :3], uu[:, None, 3:], project=True)
        return U[C.phys_V(vnet, vdyn, n2, nd2, c2).view(B, nu).argmax(1)]

    hold = lambda z, n, nd, c: torch.zeros(len(n), 6, device=dev)
    out = dict(phys_greedy=_rollout(D, st, dev, T, phys), hold=_rollout(D, st, dev, T, hold))
    np.savez(cache_ref, **out)
    return out


def phys_succ_values(D, st, U, dev, vnet, vdyn):
    """V_phys of the TRUE one-tick successor for every u in U at the states -> (B, nu)."""
    import torch
    T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    s0 = st["s0"]
    n = T_(D["n"][s0]); nd = T_(D["nd"][s0]); c = T_(D["obs"][s0, :3]); c = c / c.norm(dim=-1, keepdim=True)
    B, nu = len(s0), len(U)
    rep = lambda x: x.repeat_interleave(nu, 0)
    uu = U.repeat(B, 1)
    with torch.no_grad():
        n2, nd2, c2 = C.pend_state_after(rep(n), rep(nd), rep(c), uu[:, None, :3], uu[:, None, 3:], project=True)
        return C.phys_V(vnet, vdyn, n2, nd2, c2).view(B, nu)


def evaluate(wm, pol, dev, cache_states, cache_ref, T=50, max_states=None):
    """-> (summary dict, per-state dict). max_states: SMOKE ONLY (evenly spaced subset)."""
    import torch
    st = cl_states(dev, cache_states)
    ref = reference_outcomes(dev, cache_states, cache_ref, T)
    if max_states is not None and max_states < len(st["s0"]):
        j = np.linspace(0, len(st["s0"]) - 1, max_states).astype(int)
        st = dict(cand=st["cand"][j], s0=st["s0"][j], Vp0=st["Vp0"][j], n_near=st["n_near"])
        ref = {k: v[j] for k, v in ref.items()}
    D = C.load_split("test")
    eps = np.unique(st["cand"][:, 0])
    Zep = dict(zip(eps.tolist(), C.filtered_latents(wm, D, dev, eps)))
    z0 = torch.tensor(np.stack([Zep[e][t] for e, t in st["cand"]]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Vw0 = pol.net(z0, wm.margin(z0[:, :wm.d_h], z0[:, wm.d_h:])).cpu().numpy()

    def greedy(z, n, nd, c):
        return pol.U[pol.q(z).argmax(1)]

    lmin = _rollout(D, st, dev, T, greedy, wm=wm, z0=z0)
    spill = lmin < 0
    Vp0 = st["Vp0"]
    sav = Vp0 > 0
    res = dict(n=int(len(spill)), n_near=int(st["n_near"]), n_savable=int(sav.sum()),
               frac_Vphys_le0=float((Vp0 <= 0).mean()))
    for name, sp_ in (("wm_greedy", spill), ("phys_greedy", ref["phys_greedy"] < 0), ("hold", ref["hold"] < 0)):
        res[name] = dict(spill_rate_all=float(sp_.mean()), spill_rate_phys_savable=float(sp_[sav].mean()),
                         spill_rate_phys_doomed=float(sp_[~sav].mean()) if (~sav).any() else float("nan"))
    res["wm_greedy"].update(
        spill_rate_given_Vwm0_safe=float(spill[Vw0 > 0].mean()) if (Vw0 > 0).any() else float("nan"),
        spill_rate_given_Vwm0_unsafe=float(spill[Vw0 <= 0].mean()) if (Vw0 <= 0).any() else float("nan"))
    res["frac_Vwm0_safe"] = float((Vw0 > 0).mean())
    res["frac_Vwm0_safe_but_phys_doomed"] = float(((Vw0 > 0) & (Vp0 <= 0)).mean())
    # ------------------------------------------------ winner's curse + one-step model error at t = 0
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    with torch.no_grad():
        q = pol.q(z0)                                         # (B, nu_pol)
        ist = q.argmax(1)
        Vt = phys_succ_values(D, st, pol.U, dev, vnet, vdyn)  # (B, nu_pol)
        ar = torch.arange(len(ist), device=dev)
        vm_arg, vt_arg = q[ar, ist], Vt[ar, ist]
        s0 = st["s0"]
        a_log = torch.tensor(D["act"][s0], dtype=torch.float32, device=dev)
        # model value of the LOGGED action (same Q construction as the policy)
        zl = z0
        if pol.pess is None:
            nl = pol.base.step(zl, a_log); vm_log = pol.net(nl, pol.base.margin(nl))
        else:
            za = torch.cat([zl, torch.zeros(len(zl), 1, device=dev)], -1)
            nl = pol.pess.step(za, a_log); vm_log = pol.net(pol.pess.features(nl), pol.pess.margin(nl))
        T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        n = T_(D["n"][s0]); nd = T_(D["nd"][s0]); c = T_(D["obs"][s0, :3]); c = c / c.norm(dim=-1, keepdim=True)
        v0 = T_(D["obs"][s0, 3:6])
        # true successor under the logged action = the logged next state
        vt_log_n = T_(D["n"][s0 + 1]); vt_log_nd = T_(D["nd"][s0 + 1])
        c1 = T_(D["obs"][s0 + 1, :3]); c1 = c1 / c1.norm(dim=-1, keepdim=True)
        vt_log = C.phys_V(vnet, vdyn, vt_log_n, vt_log_nd, c1)
        # one-step error: imagination step (the value's f) vs truth
        u_arg = pol.U[ist]
        z_arg = pol.base.step(z0, u_arg)
        l1, weff, n1, nd1, c1a = C.pend_rollout_torch(n, nd, c, u_arg[:, None, :3], u_arg[:, None, 3:], project=True,
                                                      return_state=True)
        o_true_arg = torch.cat([c1a, v0 + C.DT_WM * u_arg[:, :3], weff[:, 0]], -1)
        dh = wm.d_h
        raw = lambda zz: zz[:, dh:] * wm.o_sd + wm.o_mu
        blk = lambda a, b: [((a[:, i:i + 3] - b[:, i:i + 3]) ** 2).sum(-1).sqrt() for i in (0, 3, 6)]
        err_c_arg, err_v_arg, err_w_arg = blk(raw(z_arg), o_true_arg)
        err_mar_arg = (pol.base.margin(z_arg) - l1[:, 0]).abs()
        z_log = pol.base.step(z0, a_log)
        err_c_log, err_v_log, err_w_log = blk(raw(z_log), T_(D["obs"][s0 + 1]))
        err_mar_log = (pol.base.margin(z_log) - T_(D["l"][s0 + 1])).abs()
        # 10-tick: logged sequence vs held argmax
        K = 10
        ok10 = st["cand"][:, 1] + K < D["lens"][st["cand"][:, 0]]
        idx = np.minimum(s0[:, None] + np.arange(K)[None], len(D["l"]) - 1 - K)
        a_seq = T_(D["act"][idx])
        zk = z0
        for k in range(K):
            zk = pol.base.step(zk, a_seq[:, k])
        err10_log = (pol.base.margin(zk) - T_(D["l"][np.minimum(s0 + K, len(D["l"]) - 1)])).abs()
        err10_log = torch.where(torch.tensor(ok10, device=dev), err10_log, torch.full_like(err10_log, float("nan")))
        lK, _ = C.pend_rollout_torch(n, nd, c, u_arg[:, None, :3].expand(-1, K, -1).contiguous(),
                                     u_arg[:, None, 3:].expand(-1, K, -1).contiguous(), project=True)
        zk = z0
        for k in range(K):
            zk = pol.base.step(zk, u_arg)
        err10_arg = (pol.base.margin(zk) - lK[:, -1]).abs()
    g = lambda x: x.detach().cpu().numpy()
    vm_arg, vt_arg, vm_log, vt_log = g(vm_arg), g(vt_arg), g(vm_log), g(vt_log)
    regret = g(Vt.max(1).values) - vt_arg
    err = dict(axis_log=g(err_c_log), axis_arg=g(err_c_arg), vel_log=g(err_v_log), vel_arg=g(err_v_arg),
               angvel_log=g(err_w_log), angvel_arg=g(err_w_arg), mar_log=g(err_mar_log), mar_arg=g(err_mar_arg),
               mar10_log=g(err10_log), mar10_arg=g(err10_arg))
    for nm, m in (("savable", sav), ("all", np.ones_like(sav))):
        res[f"winners_curse_{nm}"] = dict(
            gap_argmax_mean=float((vm_arg - vt_arg)[m].mean()), gap_argmax_median=float(np.median((vm_arg - vt_arg)[m])),
            gap_logged_mean=float((vm_log - vt_log)[m].mean()),
            Vmodel_argmax_mean=float(vm_arg[m].mean()), Vtrue_argmax_mean=float(vt_arg[m].mean()),
            regret_true_mean=float(regret[m].mean()), frac_Vtrue_argmax_le0=float((vt_arg[m] <= 0).mean()),
            **{f"err_{k}_mean": float(np.nanmean(v[m])) for k, v in err.items()})
    per_state = dict(spill=spill, lmin=lmin, Vw0=Vw0, Vp0=Vp0, vm_arg=vm_arg, vt_arg=vt_arg, vm_log=vm_log,
                     vt_log=vt_log, regret=regret, argmax=g(pol.idx[ist]), **{f"err_{k}": v for k, v in err.items()})
    return res, per_state
