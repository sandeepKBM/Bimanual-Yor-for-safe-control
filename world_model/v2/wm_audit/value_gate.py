"""wm_audit/value_gate.py -- LatentSafe/Fisac-style discounted reachability value in a learned
kin_wm latent, gated against the physics-latent value and the simulator ground truth.
SIMULATION / SAVED DATA ONLY.

  train : V(z) = (1-g) l^(z) + g min(l^(z), max_{u in U} V(f_WM(z,u)))   (reach_value.train, max over
          the 85-element discretised fallback set, greedy K-step rollout bootstrap, gamma 0.8 -> 0.98 per
          20 ms tick == Stage-0's 0.99 per 10 ms), states = filtered TRAIN latents + imagined fallback
          rollouts; optional tilt projection of the fallback (same 1.5 rad limit as the physics Dyn).
  gate  : on held-out TEST states with the privileged pendulum state:
          doomed recall  P(V_wm <= 0 | l > 0, V_phys <= 0)      (V_phys = Stage-0 value, IoU .99 vs grid HJ)
          false-safe     P(V_wm > 0 | V_phys <= 0)
          safe-set IoU   {V_wm > 0} vs {V_phys > 0}
          AUROC(-V_wm) for doomed among l>0 (threshold-free), vs the margin head l^
          episode lead-time gate (eps from VAL non-spill eps at a fixed FA budget)
          FL-MPC row gradient cosine: d V(F^K(z; u)) / du through WM+V_wm vs pendulum+V_phys

  CUDA_VISIBLE_DEVICES=2 python -m world_model.v2.wm_audit.value_gate --wm <wm.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C


class ProjWMDyn:
    """kin_wm.WMDyn with the fallback's tilt projection: a tilt-rate that would push the WM's own
    predicted cup axis beyond tilt_lim is zeroed (the physics Dyn clamps the axis there)."""

    def __init__(self, wm, bank, device, project=True):
        import torch
        self.wm, self.dev, self.bank, self.project = wm, device, bank, project
        self.U = torch.tensor(C.fallback_U(), dtype=torch.float32, device=device)
        self.dh = wm.d_h
        self.P = C.P0

    def sample_states(self, B):
        import torch
        return self.bank[torch.randint(len(self.bank), (B,), device=self.dev)]

    def step(self, z, u):
        import torch
        shp = torch.broadcast_shapes(z.shape[:-1], u.shape[:-1])
        z = z.expand(*shp, z.shape[-1]).reshape(-1, z.shape[-1])
        u = u.expand(*shp, u.shape[-1]).reshape(-1, u.shape[-1])
        h, o_n = z[:, :self.dh], z[:, self.dh:]
        if self.project:
            c = o_n[:, :3] * self.wm.o_sd[:3] + self.wm.o_mu[:3]
            c = c / (c.norm(dim=-1, keepdim=True) + 1e-9)
            tilt = torch.arccos(c[:, 2].clamp(-1, 1))
            dcz = u[:, 3] * c[:, 1] - u[:, 4] * c[:, 0]           # (w x c)_z; < 0 increases tilt
            block = (tilt >= self.P.tilt_lim - 0.02) & (dcz < 0)
            u = u.clone(); u[block, 3:] = 0.0
        h2, o2 = self.wm.step(h, o_n, u)
        return torch.cat([h2, o2], -1).view(*shp, -1)

    def margin(self, z):
        return self.wm.margin(z[..., :self.dh], z[..., self.dh:])

    def features(self, z):
        return z


def build_bank(wm, dev, stride=2, n_imag=200000, imag_k=10, seed=0, project=True):
    import torch
    D = C.load_split("train")
    Z = C.filtered_latents(wm, D, dev)
    Z = torch.tensor(np.concatenate([z[::stride] for z in Z], 0), device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    dyn = ProjWMDyn(wm, Z, dev, project)
    zi = Z[torch.randint(len(Z), (n_imag // imag_k,), device=dev, generator=g)]
    out = [Z]
    with torch.no_grad():
        for _ in range(imag_k):
            u = dyn.U[torch.randint(len(dyn.U), (len(zi),), device=dev, generator=g)]
            zi = dyn.step(zi, u)
            out.append(zi)
    return torch.cat(out, 0)


def gate(wm, net, dev, out, fa_budgets=(0.02, 0.05, 0.10), n_grad=2000, grad_K=(5, 10), stride=2):
    import torch
    from world_model.v2.latent_hj.validate_stage0 import gate_eval
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    res = {}
    rows = {}
    per_state = dict(l=[], lh=[], Vw=[], Vp=[])
    for split in ("val", "test"):
        D = C.load_split(split)
        Z = C.filtered_latents(wm, D, dev)
        R = []
        for e, z in enumerate(Z):
            s, L = D["start"][e], D["lens"][e]
            with torch.no_grad():
                zt = torch.tensor(z, device=dev)
                lh = wm.margin(zt[:, :wm.d_h], zt[:, wm.d_h:])
                Vw = net(zt, lh).cpu().numpy(); lh = lh.cpu().numpy()
            r = dict(onset=int(D["onset"][e]), stride=1, l=D["l"][s:s + L], lhat=lh, V=Vw)
            R.append(r)
            if split == "test":
                ix = np.arange(s, s + L, stride)
                n = torch.tensor(D["n"][ix], device=dev); nd = torch.tensor(D["nd"][ix], device=dev)
                c = torch.tensor(D["obs"][ix, :3], device=dev); c = c / c.norm(dim=-1, keepdim=True)
                with torch.no_grad():
                    Vp = C.phys_V(vnet, vdyn, n, nd, c).cpu().numpy()
                per_state["l"].append(D["l"][ix]); per_state["lh"].append(lh[::stride])
                per_state["Vw"].append(Vw[::stride]); per_state["Vp"].append(Vp)
        rows[split] = R
    S = {k: np.concatenate(v) for k, v in per_state.items()}
    l, lh, Vw, Vp = S["l"], S["lh"], S["Vw"], S["Vp"]
    doomed = (l > 0) & (Vp <= 0)
    pos = l > 0
    res["states"] = dict(
        n=int(len(l)), frac_doomed_among_lpos=float(doomed.sum() / pos.sum()),
        doomed_recall_Vwm=float((Vw[doomed] <= 0).mean()),
        doomed_recall_lhat=float((lh[doomed] <= 0).mean()),
        false_safe_rate_Vwm=float((Vw[Vp <= 0] > 0).mean()),
        false_safe_rate_lhat=float((lh[Vp <= 0] > 0).mean()),
        false_unsafe_rate_Vwm=float((Vw[Vp > 0] <= 0).mean()),
        iou_safe_Vwm=float(((Vw > 0) & (Vp > 0)).sum() / max(((Vw > 0) | (Vp > 0)).sum(), 1)),
        iou_safe_lhat=float(((lh > 0) & (Vp > 0)).sum() / max(((lh > 0) | (Vp > 0)).sum(), 1)),
        auroc_doomed_given_lpos_Vwm=C.auroc(-Vw[pos], doomed[pos]),
        auroc_doomed_given_lpos_lhat=C.auroc(-lh[pos], doomed[pos]),
        auroc_doomed_given_lpos_ltrue=C.auroc(-l[pos], doomed[pos]),
        mae_Vwm_vs_Vphys=float(np.abs(Vw - Vp).mean()),
        collapse_mean_abs_V_minus_lhat=float(np.abs(Vw - lh).mean()),
        collapse_frac_V_below_lhat_by_0p05=float((Vw < lh - 0.05).mean()),
        phys_frac_V_below_l_by_0p05=float((Vp < l - 0.05).mean()),
    )
    # best threshold-shifted recall at matched false-unsafe (is it only an offset problem?)
    thr = np.quantile(Vw[(Vp > 0) & pos], 0.05)
    res["states"]["doomed_recall_Vwm_at_5pct_false_unsafe"] = float((Vw[doomed] <= thr).mean())
    thr_l = np.quantile(lh[(Vp > 0) & pos], 0.05)
    res["states"]["doomed_recall_lhat_at_5pct_false_unsafe"] = float((lh[doomed] <= thr_l).mean())
    for fa in fa_budgets:
        for key in ("l", "lhat", "V"):
            g = gate_eval(rows["val"], rows["test"], key, fa, dt=C.DT_WM)
            res[f"lead/{key}/fa{fa}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()}
    # ---------------------------------------------------------------- FL-MPC row gradients
    D = C.load_split("test")
    rng = np.random.default_rng(0)
    Z = C.filtered_latents(wm, D, dev)
    cand = []
    for e in range(len(D["lens"])):
        s, L = D["start"][e], D["lens"][e]
        for t in range(20, L - max(grad_K) - 1, 3):
            cand.append((e, t))
    cand = np.array(cand)
    s0 = D["start"][cand[:, 0]] + cand[:, 1]
    with torch.no_grad():
        n = torch.tensor(D["n"][s0], device=dev); nd = torch.tensor(D["nd"][s0], device=dev)
        c = torch.tensor(D["obs"][s0, :3], device=dev); c = c / c.norm(dim=-1, keepdim=True)
        Vp0 = C.phys_V(vnet, vdyn, n, nd, c).cpu().numpy()
    act_ok = (D["l"][s0] > 0) & (np.abs(Vp0) < 0.5)
    sel = rng.choice(np.flatnonzero(act_ok), min(n_grad, act_ok.sum()), replace=False)
    cand, s0 = cand[sel], s0[sel]
    z0 = torch.tensor(np.stack([Z[e][t] for e, t in cand]), device=dev)
    scale = torch.tensor([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], device=dev)
    for K in grad_K:
        idx = s0[:, None] + np.arange(K)[None]
        u = torch.tensor(D["act"][idx], device=dev).requires_grad_(True)
        lw, zs = C.wm_rollout(wm, z0, u[..., :3], u[..., 3:])
        zK = zs[:, -1]
        lhK = wm.margin(zK[:, :wm.d_h], zK[:, wm.d_h:])
        gw = torch.autograd.grad(net(zK, lhK).sum(), u)[0]
        u2 = torch.tensor(D["act"][idx], device=dev).requires_grad_(True)
        lw2, zs2 = C.wm_rollout(wm, z0, u2[..., :3], u2[..., 3:])
        zK2 = zs2[:, -1]
        gl = torch.autograd.grad(wm.margin(zK2[:, :wm.d_h], zK2[:, wm.d_h:]).sum(), u2)[0]
        u3 = torch.tensor(D["act"][idx], device=dev).requires_grad_(True)
        nK, ndK, cK = C.pend_state_after(n[sel], nd[sel], c[sel], u3[..., :3], u3[..., 3:], project=False)
        gp = torch.autograd.grad(C.phys_V(vnet, vdyn, nK, ndK, cK).sum(), u3)[0]
        u4 = torch.tensor(D["act"][idx], device=dev).requires_grad_(True)
        nK4, ndK4, cK4 = C.pend_state_after(n[sel], nd[sel], c[sel], u4[..., :3], u4[..., 3:], project=False)
        glp = torch.autograd.grad(vdyn.margin(torch.cat([nK4, ndK4, cK4], -1)).sum(), u4)[0]
        f = lambda g: (g * scale).reshape(len(g), -1)
        def cos(a, b):
            a, b = f(a), f(b)
            na, nb = a.norm(dim=1), b.norm(dim=1)
            ok = (na > 1e-8) & (nb > 1e-8)
            cs = ((a * b).sum(1)[ok] / (na[ok] * nb[ok])).cpu().numpy()
            return dict(median=float(np.median(cs)), frac_pos=float((cs > 0).mean()), frac_gt_0p7=float((cs > 0.7).mean()),
                        n=int(ok.sum()))
        res[f"grad_cos_K{K}"] = dict(Vwm_vs_Vphys=cos(gw, gp), lhat_vs_Vphys=cos(gl, gp),
                                     lhat_vs_lphys=cos(gl, glp), lphys_vs_Vphys=cos(glp, gp))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "gate.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--value", default=None, help="skip training, gate this value.pt")
    ap.add_argument("--log-every", type=int, default=1000)
    a = ap.parse_args()
    import torch
    from world_model.v2.latent_hj import kin_wm, reach_value as rv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = kin_wm.load_wm(a.wm, dev)
    t0 = time.time()
    if a.value:
        net, meta = rv.load(a.value, dev)
    else:
        bank = build_bank(wm, dev, project=not a.no_project)
        print(f"[value] bank {tuple(bank.shape)} ({time.time()-t0:.0f}s)", flush=True)
        dyn = ProjWMDyn(wm, bank, dev, project=not a.no_project)
        net, curve = rv.train(dyn, a.out, iters=a.iters, batch=a.batch, gamma0=0.8, gamma_final=a.gamma,
                              K=a.K, lam=0.5, device=dev, log_every=a.log_every)
        meta = dict(dx=int(bank.shape[1]), hidden=256, layers=3, gamma=a.gamma, K=a.K, kind="kin_wm_latent_audit",
                    wm=str(a.wm), P=C.P0.as_dict() | dict(dt=C.DT_WM), project=not a.no_project)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        rv.save(net.cpu(), Path(a.out) / "value.pt", meta)
        (Path(a.out) / "curve.json").write_text(json.dumps(curve))
        net = net.to(dev)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    res = gate(wm, net, dev, a.out)
    res["train_sec"] = time.time() - t0
    (Path(a.out) / "gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
