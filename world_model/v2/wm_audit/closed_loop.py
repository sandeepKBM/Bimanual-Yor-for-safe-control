"""wm_audit/closed_loop.py -- SIMULATOR ground truth for a latent reachability value: run its greedy
fallback policy u* = argmax_u V(f(z,u)) in closed loop on the TRUE liquid proxy (the same pendulum
the labels come from), with the WM filtering the true observations every tick (KinWMObserver
semantics). SIMULATION ONLY.

From held-out TEST states near the physics boundary (l > 0, V_phys in (-0.3, 0.4)) plus clearly-safe
states, 1 s (50 ticks) of:
  wm_greedy    argmax over the 85-action fallback set of V_wm(f_WM(z, u))
  phys_greedy  argmax of V_phys(true one-tick successor)   (oracle-dynamics reference)
  hold         u = 0 (stop the cup)
and report the realized spill rate (min true margin < 0), split by the physics verdict and by the
WM value's own verdict at t=0 (closed-loop false-safe = V_wm(z0) > 0 but its own fallback spills).

  CUDA_VISIBLE_DEVICES=2 python -m world_model.v2.wm_audit.closed_loop --wm <wm.pt> --value <value.pt> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.wm_audit.value_gate import ProjWMDyn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from world_model.v2.latent_hj import kin_wm, reach_value as rv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = kin_wm.load_wm(a.wm, dev)
    net, _ = rv.load(a.value, dev)
    for p in net.parameters():
        p.requires_grad_(False)
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    D = C.load_split("test")
    rng = np.random.default_rng(a.seed)
    Z = C.filtered_latents(wm, D, dev)
    cand = np.array([(e, t) for e in range(len(D["lens"])) for t in range(20, D["lens"][e] - 1, 2)])
    s0 = D["start"][cand[:, 0]] + cand[:, 1]
    T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    with torch.no_grad():
        c_all = T_(D["obs"][s0, :3]); c_all = c_all / c_all.norm(dim=-1, keepdim=True)
        Vp_all = C.phys_V(vnet, vdyn, T_(D["n"][s0]), T_(D["nd"][s0]), c_all).cpu().numpy()
    l_all = D["l"][s0]
    near = np.flatnonzero((l_all > 0) & (Vp_all > -0.3) & (Vp_all < 0.4))
    safe = np.flatnonzero((l_all > 0) & (Vp_all >= 0.4))
    n_near = min(len(near), int(0.75 * a.n))
    pick = np.concatenate([rng.choice(near, n_near, replace=False), rng.choice(safe, a.n - n_near, replace=False)])
    cand, s0 = cand[pick], s0[pick]
    B = len(pick)
    z0 = T_(np.stack([Z[e][t] for e, t in cand]))
    Vp0 = Vp_all[pick]
    dyn = ProjWMDyn(wm, None, dev, project=True)
    U = dyn.U
    nu = len(U)
    with torch.no_grad():
        Vw0 = net(z0, wm.margin(z0[:, :wm.d_h], z0[:, wm.d_h:])).cpu().numpy()
    res = dict(n=int(B), n_near=int(n_near), frac_Vphys_le0=float((Vp0 <= 0).mean()))
    for policy in ("wm_greedy", "phys_greedy", "hold"):
        n = T_(D["n"][s0]); nd = T_(D["nd"][s0]); c = T_(D["obs"][s0, :3]); c = c / c.norm(dim=-1, keepdim=True)
        v = T_(D["obs"][s0, 3:6])
        h = z0[:, :wm.d_h].clone(); o_n = z0[:, wm.d_h:].clone()
        lmin = torch.full((B,), 1.0, device=dev)
        with torch.no_grad():
            for k in range(a.T):
                if policy == "wm_greedy":
                    z = torch.cat([h, o_n], -1)
                    nxt = dyn.step(z[:, None, :].expand(-1, nu, -1), U[None].expand(B, -1, -1))
                    vv = net(nxt, dyn.margin(nxt))
                    u = U[vv.argmax(1)]
                elif policy == "phys_greedy":
                    rep = lambda x: x.repeat_interleave(nu, 0)
                    uu = U.repeat(B, 1)
                    n2, nd2, c2 = C.pend_state_after(rep(n), rep(nd), rep(c), uu[:, None, :3], uu[:, None, 3:], project=True)
                    vv = C.phys_V(vnet, vdyn, n2, nd2, c2).view(B, nu)
                    u = U[vv.argmax(1)]
                else:
                    u = torch.zeros(B, 6, device=dev)
                l1, weff, n, nd, c = C.pend_rollout_torch(n, nd, c, u[:, None, :3], u[:, None, 3:], project=True,
                                                          return_state=True)
                lmin = torch.minimum(lmin, l1[:, 0])
                w_obs = weff[:, 0]
                act = torch.cat([u[:, :3], w_obs], -1)
                h, _ = wm.step(h, o_n, act)                      # WM filter on the TRUE observation stream
                v = v + C.DT_WM * u[:, :3]
                o_n = wm.no(torch.cat([c, v, w_obs], -1))
        sp_ = (lmin < 0).cpu().numpy()
        res[policy] = dict(
            spill_rate_all=float(sp_.mean()),
            spill_rate_phys_savable=float(sp_[Vp0 > 0].mean()),
            spill_rate_phys_doomed=float(sp_[Vp0 <= 0].mean()) if (Vp0 <= 0).any() else float("nan"),
            spill_rate_given_Vwm0_safe=float(sp_[Vw0 > 0].mean()) if (Vw0 > 0).any() else float("nan"),
            spill_rate_given_Vwm0_unsafe=float(sp_[Vw0 <= 0].mean()) if (Vw0 <= 0).any() else float("nan"))
        print(policy, json.dumps(res[policy]), flush=True)
    res["frac_Vwm0_safe"] = float((Vw0 > 0).mean())
    res["frac_Vwm0_safe_but_phys_doomed"] = float(((Vw0 > 0) & (Vp0 <= 0)).mean())
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
