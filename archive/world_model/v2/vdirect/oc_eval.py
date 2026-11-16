"""vdirect/oc_eval.py -- outcome-level conformal audit of a run's learned value(s) (PREREG_ADDENDUM_F, secondary).
SIMULATION / SAVED DATA ONLY. Reads the run's files read-only; writes one json under --out.

Lin & Bansal-style OUTCOME calibration of the certified set C(tau) = {z : V(z) >= tau}:
  calibration states: N_CAL = 2000 VAL states drawn by cl_eval's protocol (75% near the physics boundary, 25% clearly
     safe), seed 1, split into halves A / B (alternating). Outcome = the value's OWN greedy fallback rolled 50 ticks on
     the TRUE pendulum with the WM filtering the true stream (cl_eval._rollout); fail = true margin < 0.
  threshold: candidate taus from half A only (tau_m = the m-th largest V(z0) of A, m = 100, 150, ..., 1000, i.e.
     descending tau), tested in that FIXED order on half B (Learn-then-Test fixed sequence): H0_m 'P(fail | V >= tau_m)
     > eps' is rejected iff the one-sided Clopper-Pearson (1 - beta) upper bound of the fail rate among half-B states
     with V >= tau_m is <= eps. tau* = the last tau of the leading run of rejections; if the first is not rejected,
     tau* = +inf (nothing certified). eps = 0.05, beta = 0.05.
  test (no new rollouts): on the 2000 cl_eval TEST states, from the run's value_<kind>/cl_states.npz: certified fraction
     of physics-savable states (permissiveness), failure rate among certified states, certified fraction of clearly
     safe states (V_phys >= 0.4) and of all states.
  B0 runs: kinds std (N0), M (in-data max), P (pessimistic; its bootstrap heads are NOT saved by the grid, so they are
     refit with the run's seed -- same recipe, not bit-identical on GPU; flagged in the output).

  python -m world_model.v2.vdirect.oc_eval --run <run dir> --out <json> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.vdirect import cl_eval as CE

EPS, BETA, SEED, T_CL = 0.05, 0.05, 1, 50
CFG = dict(N_CAL=2000, M_GRID=list(range(100, 1001, 50)))
SMOKE = dict(N_CAL=200, M_GRID=list(range(10, 101, 5)))
CACHE = Path(os.environ.get("VD_OC_CACHE", "/common/users/ss5772/bimanual_data/vdirect_f/shared_oc"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cal_states(dev, n, split="val", seed=SEED):
    """cl_eval.cl_states' pick on another split (depends only on data + the physics value); atomic cache write."""
    import torch
    cache = CACHE / f"cal_states_{split}_n{n}_seed{seed}.npz"
    if cache.exists():
        f = np.load(cache)
        return {k: f[k] for k in f.files}
    D = C.load_split(split)
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
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **out)
    os.replace(tmp, cache)
    return out


def cp_upper(k, n, beta=BETA):
    """one-sided Clopper-Pearson (1 - beta) upper confidence bound of a binomial rate, k events in n trials."""
    if n == 0 or k >= n:
        return 1.0
    try:
        from scipy.stats import beta as Bd
        return float(Bd.ppf(1 - beta, k + 1, n - k))
    except ImportError:
        def cdf(p):
            return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1))
        lo, hi = k / n, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cdf(mid) > beta else (lo, mid)
        return hi


def calibrate(V, fail, m_grid):
    A = (np.arange(len(V)) % 2) == 0
    Bm = ~A
    VA = np.sort(V[A])[::-1]
    steps, tau_star = [], float("inf")
    for m in m_grid:
        if m > len(VA):
            break
        tau = float(VA[m - 1])
        c = Bm & (V >= tau)
        n, k = int(c.sum()), int((fail & c).sum())
        ub = cp_upper(k, n)
        rej = ub <= EPS
        steps.append(dict(m=m, tau=tau, n_B=n, k_B=k, cp_upper=ub, reject=bool(rej)))
        if not rej:
            break
        tau_star = tau
    return tau_star, steps


def test_metrics(per, tau):
    cert = per["Vw0"] >= tau
    sav, safe = per["Vp0"] > 0, per["Vp0"] >= 0.4
    spill = per["spill"].astype(bool)
    f = lambda m: float(cert[m].mean()) if m.any() else float("nan")
    return dict(tau=tau, n_cert=int(cert.sum()), cert_frac_all=float(cert.mean()), cert_frac_savable=f(sav),
                cert_frac_clearly_safe=f(safe), fail_rate_certified=float(spill[cert].mean()) if cert.any() else None,
                n_cert_doomed=int((cert & ~sav).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch
    from world_model.v2.latent_hj import kin_wm, reach_value as rv
    from world_model.v2.wm_audit import value_gate as VG
    from world_model.v2.vdirect import pess as PS
    cfg = dict(CFG) | (SMOKE if a.smoke else {})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    t_start = time.time()
    run = Path(a.run)
    seed = int(run.name.split("_s")[-1])
    kinds = ["std", "M", "P"] if run.name.replace("smoke_", "").startswith("B0") else ["std"]
    st = cal_states(dev, cfg["N_CAL"])
    D = C.load_split("val")
    wm = kin_wm.load_wm(str(run / "wm.pt"), dev)
    eps_ = np.unique(st["cand"][:, 0])
    Zep = dict(zip(eps_.tolist(), C.filtered_latents(wm, D, dev, eps_)))
    z0 = torch.tensor(np.stack([Zep[e][t] for e, t in st["cand"]]), dtype=torch.float32, device=dev)
    del Zep
    out = dict(run=str(run), eps=EPS, beta=BETA, n_cal=int(cfg["N_CAL"]), m_grid=cfg["M_GRID"], dev=dev, kinds={})
    for kind in kinds:
        t0 = time.time()
        net, _ = rv.load(str(run / f"value_{kind}" / "value.pt"), dev)
        net = net.to(dev).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        U_idx, pess, hinfo = None, None, None
        if kind == "M":
            U_idx, _ = PS.in_data_actions()
        if kind == "P":
            with torch.enable_grad():
                heads, hinfo = PS.fit_heads(wm, dev, E=5, steps=3000, seed=seed)
            pess = PS.PessDyn(VG.ProjWMDyn(wm, None, dev, project=True), heads)
            hinfo = dict(hinfo, refit=True)
        pol = CE.Policy(wm, net, dev, U_idx=U_idx, pess=pess)
        V0 = pol.net(z0, wm.margin(z0[:, :wm.d_h], z0[:, wm.d_h:])).cpu().numpy()
        greedy = lambda z, n, nd, c: pol.U[pol.q(z).argmax(1)]
        lmin = CE._rollout(D, st, dev, T_CL, greedy, wm=wm, z0=z0)
        fail = lmin < 0
        tau, steps = calibrate(V0, fail, cfg["M_GRID"])
        per = np.load(run / f"value_{kind}" / "cl_states.npz")
        res = dict(tau_star=tau, steps=steps, heads=hinfo,
                   cal=dict(n=int(len(V0)), fail_rate=float(fail.mean()), frac_V_pos=float((V0 > 0).mean()),
                            fail_rate_given_V_pos=float(fail[V0 > 0].mean()) if (V0 > 0).any() else None,
                            fail_rate_savable=float(fail[st["Vp0"] > 0].mean())),
                   test=test_metrics(per, tau), test_at_tau0=test_metrics(per, 0.0), sec=time.time() - t0)
        out["kinds"][kind] = res
        log(f"[oc {run.name} {kind}] tau*={tau:.4f} cert_savable={res['test']['cert_frac_savable']:.4f} "
            f"fail_cert={res['test']['fail_rate_certified']} cal_fail={fail.mean():.4f} ({time.time()-t0:.0f}s)")
    out["wall_sec"] = time.time() - t_start
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, a.out)
    Path(str(a.out) + "._DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))


if __name__ == "__main__":
    main()
