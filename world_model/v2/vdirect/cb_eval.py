"""vdirect/cb_eval.py -- CB arm + the conservatism evaluation for every arm (PREREG_ADDENDUM_CB). CPU, SAVED DATA
+ SIMULATION ONLY. Reads the grid's model files (read-only), writes only under --out.

  cons   --run <grid run dir> --out <npz>     per-state TEST (stride-2) arrays: V(z_t), V(f(z_t,a_t)) for every value
                                              kind of the run; for B0 also CB's Vbar (full-latent ball = paper; o-block
                                              ball = secondary) with its ACI radius and coverage along TEST episodes
  cl     --run <B0 run dir> --policy {nominal,robust} --part k --nparts n --out <npz>
                                              CB closed loop on the SAME 2000 TEST states (cl_eval protocol), ACI warm-
                                              started on the logged prefix; nominal = the paper's pi_safe (argmax V),
                                              robust = argmax of Vbar(f(z,u)) (secondary, not in the paper)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from world_model.v2.wm_audit import common as C
from world_model.v2.vdirect import cb as CB

ILAB = Path("/common/users/ss5772/bimanual_data/vdirect_ilab")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_run(run, kind, dev="cpu"):
    from world_model.v2.latent_hj import kin_wm, reach_value as rv
    wm = kin_wm.load_wm(str(Path(run) / "wm.pt"), dev)
    net, meta = rv.load(str(Path(run) / f"value_{kind}" / "value.pt"), dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return wm, net, meta


def test_states(dev="cpu", split="test", max_eps=None):
    """the value_gate stride-2 TEST state set + privileged physics value (WM independent). split/max_eps: SMOKE only."""
    import torch
    D = C.load_split(split)
    if max_eps is not None:
        D = C.subset_eps(D, np.arange(min(max_eps, len(D["lens"]))))
    vnet, vdyn, _ = C.load_phys_value(device=dev)
    E, T, S = [], [], []
    for e in range(len(D["lens"])):
        s, L = D["start"][e], D["lens"][e]
        t = np.arange(0, L, 2)
        E.append(np.full(len(t), e)); T.append(t); S.append(s + t)
    E, T, S = map(np.concatenate, (E, T, S))
    with torch.no_grad():
        c = torch.tensor(D["obs"][S, :3]); c = c / c.norm(dim=-1, keepdim=True)
        Vp = C.phys_V(vnet, vdyn, torch.tensor(D["n"][S]), torch.tensor(D["nd"][S]), c).numpy()
    return D, dict(ep=E, t=T, s=S, l=D["l"][S], Vp=Vp)


def cmd_cons(a):
    import torch
    from world_model.v2.wm_audit.value_gate import ProjWMDyn
    torch.set_grad_enabled(False)
    run = Path(a.run)
    D, st = test_states(split=a.split, max_eps=a.max_eps)
    kinds = ["std", "M", "P"] if run.name.startswith("B0") else ["std"]
    out = dict(ep=st["ep"], t=st["t"], l=st["l"], Vp=st["Vp"])
    t0 = time.time()
    wm, _, _ = load_run(run, "std")
    Z = C.filtered_latents(wm, D, "cpu")
    z = torch.tensor(np.stack([Z[e][t] for e, t in zip(st["ep"], st["t"])]), dtype=torch.float32)
    del Z
    base = ProjWMDyn(wm, None, "cpu", project=True)
    a_log = torch.tensor(D["act"][st["s"]], dtype=torch.float32)
    zn = torch.cat([base.step(z[i:i + 65536], a_log[i:i + 65536]) for i in range(0, len(z), 65536)])
    for kind in kinds:
        _, net, _ = load_run(run, kind)
        out[f"V_now_{kind}"] = torch.cat([CB.V_of(net, wm, z[i:i + 65536]) for i in range(0, len(z), 65536)]).numpy()
        out[f"V_1step_{kind}"] = torch.cat([CB.V_of(net, wm, zn[i:i + 65536]) for i in range(0, len(zn), 65536)]).numpy()
        log(f"[cons] {run.name} {kind}: {len(z)} states ({time.time()-t0:.0f}s)")
        if kind == "std" and run.name.startswith("B0"):
            qmax, cal = CB.calibrate_qmax(wm, "cpu", seed=0)
            log(f"[cb] qmax={qmax:.4f} {cal}")
            E = CB.episode_errors(wm, D, np.arange(len(D["lens"])), "cpu")
            r_next = np.zeros(len(D["l"]), np.float32)          # radius of U_{t+1} formed after tick t
            cov = dict(n=0, err=0, cov_bar=0, r_sum=0.0)
            for e, ee in enumerate(E):
                s = D["start"][e]
                aci = CB.ACI(qmax)
                r_next[s] = aci.radius()                         # t = 0: no error yet, q_1 = inf -> qmax
                for t in range(1, len(ee)):
                    aci.update(float(ee[t]))
                    r_next[s + t] = aci.radius()
                cov["n"] += aci.n; cov["err"] += aci.n_err; cov["cov_bar"] += aci.n_cov_bar; cov["r_sum"] += aci.r_sum
            r = r_next[st["s"]]
            with torch.enable_grad():
                out["Vbar_CB"] = CB.pgd_min(net, wm, zn, r).detach().numpy()
                out["Vbar_CBo"] = CB.pgd_min(net, wm, zn, r, block="o").detach().numpy()
            out["r_CB"] = r
            out["e_test_all"] = np.concatenate([x[1:] for x in E]).astype(np.float32)
            meta = dict(qmax=qmax, cal=cal, aci_test=dict(n_steps=cov["n"], miscoverage_q=cov["err"] / cov["n"],
                                                          coverage_qbar=cov["cov_bar"] / cov["n"],
                                                          mean_radius=cov["r_sum"] / cov["n"]),
                        alpha=CB.ALPHA, lam=CB.LAM, alpha_traj=CB.ALPHA_TRAJ, alpha_cal=CB.ALPHA_CAL,
                        pgd=dict(steps=CB.PGD_STEPS, frac=CB.PGD_FRAC))
            out["cb_meta"] = np.array(json.dumps(meta))
            log(f"[cb] ACI on TEST: {meta['aci_test']}  ({time.time()-t0:.0f}s)")
    np.savez_compressed(a.out, **out)
    Path(str(a.out) + "._DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))


def cmd_cl(a):
    import torch
    from world_model.v2.wm_audit.value_gate import ProjWMDyn
    torch.set_grad_enabled(False)
    run = Path(a.run)
    wm, net, _ = load_run(run, "std")
    st = np.load(ILAB / "shared" / "cl_states.npz")
    idx = np.array_split(np.arange(len(st["s0"])), a.nparts)[a.part]
    cand, s0 = st["cand"][idx], st["s0"][idx]
    D = C.load_split("test")
    qmax, cal = CB.calibrate_qmax(wm, "cpu", seed=0)
    eps = np.unique(cand[:, 0])
    Zep = dict(zip(eps.tolist(), C.filtered_latents(wm, D, "cpu", eps)))
    Eep = dict(zip(eps.tolist(), CB.episode_errors(wm, D, eps, "cpu")))
    B = len(idx)
    acis = []
    for e, t in cand:                                     # deployment warm start: ACI along the logged prefix
        aci = CB.ACI(qmax)
        for k in range(1, t + 1):
            aci.update(float(Eep[e][k]))
        acis.append(aci)
    pre = [(x.n, x.n_err, x.n_cov_bar, x.r_sum) for x in acis]
    z0 = torch.tensor(np.stack([Zep[e][t] for e, t in cand]), dtype=torch.float32)
    base = ProjWMDyn(wm, None, "cpu", project=True)
    U = base.U; nu = len(U)
    T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    n = T_(D["n"][s0]); nd = T_(D["nd"][s0]); c = T_(D["obs"][s0, :3]); c = c / c.norm(dim=-1, keepdim=True)
    v = T_(D["obs"][s0, 3:6])
    h = z0[:, :wm.d_h].clone(); o_n = z0[:, wm.d_h:].clone()
    lmin = torch.full((B,), 1.0)
    r0 = np.array([x.radius() for x in acis])
    t0 = time.time()
    for k in range(a.T):
        z = torch.cat([h, o_n], -1)
        nxt = base.step(z[:, None, :].expand(-1, nu, -1), U[None].expand(B, -1, -1)).reshape(B * nu, -1)
        if a.policy == "nominal":
            q = CB.V_of(net, wm, nxt).view(B, nu)
        else:
            rr = np.repeat(np.array([x.radius() for x in acis]), nu)
            with torch.enable_grad():
                q = CB.pgd_min(net, wm, nxt, rr).detach().view(B, nu)
        u = U[q.argmax(1)]
        l1, weff, n, nd, c = C.pend_rollout_torch(n, nd, c, u[:, None, :3], u[:, None, 3:], project=True,
                                                  return_state=True)
        lmin = torch.minimum(lmin, l1[:, 0])
        w_obs = weff[:, 0]
        act = torch.cat([u[:, :3], w_obs], -1)
        h, _ = wm.step(h, o_n, act)
        v = v + C.DT_WM * u[:, :3]
        o_n = wm.no(torch.cat([c, v, w_obs], -1))
        e_cl = (o_n - wm.dec(h)).norm(dim=-1).numpy()
        for i, x in enumerate(acis):
            x.update(float(e_cl[i]))
        if k % 10 == 0:
            log(f"[cl {a.policy} part {a.part}] tick {k} ({time.time()-t0:.0f}s)")
    post = np.array([(x.n - p[0], x.n_err - p[1], x.n_cov_bar - p[2], x.r_sum - p[3]) for x, p in zip(acis, pre)])
    np.savez_compressed(a.out, idx=idx, spill=(lmin < 0).numpy(), lmin=lmin.numpy(), r0=r0, cl_n=post[:, 0],
                        cl_err=post[:, 1], cl_cov_bar=post[:, 2], cl_r_sum=post[:, 3], qmax=qmax,
                        policy=np.array(a.policy))
    Path(str(a.out) + "._DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))
    log(f"[cl {a.policy} part {a.part}] done: spill(all states in part)={(lmin < 0).float().mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cons"); p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.add_argument("--split", default="test"); p.add_argument("--max-eps", type=int, default=None)
    q = sub.add_parser("cl"); q.add_argument("--run", required=True); q.add_argument("--out", required=True)
    q.add_argument("--policy", choices=["nominal", "robust"], default="nominal")
    q.add_argument("--part", type=int, default=0); q.add_argument("--nparts", type=int, default=1)
    q.add_argument("--T", type=int, default=50)
    a = ap.parse_args()
    dict(cons=cmd_cons, cl=cmd_cl)[a.cmd](a)


if __name__ == "__main__":
    main()
