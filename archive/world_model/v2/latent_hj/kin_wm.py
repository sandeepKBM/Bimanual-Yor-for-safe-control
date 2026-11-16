"""kin_wm.py -- Stage 1: LEARNED latent world model + latent reachability value.

The liquid proxy's state (slosh angle / rate) is NOT observed -- neither in the
images (which show beads) nor in proprio. A world model therefore has to infer it
from the motion history; that is exactly what a recurrent latent does. This is a
compact deterministic recurrent latent WM (PlaNet deterministic path / the GRU
core of wm_dino.DinoLatentWM, without the DINO token branch -- tokens carry no
information about the INVISIBLE liquid proxy, see report) at 50 Hz:

    obs  o_t = [cup axis c (3), cup lin vel (3), cup ang vel (3)]      (FK / proprio)
    act  a_t = [cup-origin lin accel (3), cup ang vel (3)] averaged over the next
               20 ms (the FL-MPC plan determines both -- see flmpc_value.HorizonMaps)
    h_{t+1} = GRU([enc(o_t), a_t], h_t)       latent z_t = [h_t, o_t]
    o^_{t+1} = dec(h_{t+1}),  l^_t = tanh(m(h_t, o_t))  (spill margin head, trained on
               the liquid-proxy margin l = (phi* - phi)/phi*, replayed with h = 4 cm)
    Imagination feeds o^ back.  Losses: filtered + K-step open-loop overshooting
    (dataset actions) on obs and margin.

Latent reachability: reach_value.train on the WM Dyn (step = one imagined WM step
under a candidate action, U = the SAME fallback control set as Stage 0, margin =
learned head), states sampled from filtered latents of TRAIN episodes plus
imagined rollouts under random fallback actions (off-data coverage).

Gate (same held-out TEST shards, same eps-from-VAL rule as Stage 0): lead time of
V_latent vs l^_latent (learned classifier-like margin) vs true l.

  CUDA_VISIBLE_DEVICES=6 python -m world_model.v2.latent_hj.kin_wm build  --out $S/wmdata
  CUDA_VISIBLE_DEVICES=6 python -m world_model.v2.latent_hj.kin_wm train  --data $S/wmdata --out $S/wm --steps 20000
  CUDA_VISIBLE_DEVICES=6 python -m world_model.v2.latent_hj.kin_wm value  --data $S/wmdata --wm $S/wm/wm.pt --out $S/wm_v
  python -m world_model.v2.latent_hj.kin_wm gate --data $S/wmdata --wm $S/wm/wm.pt --value $S/wm_v/value.pt
SIMULATION / SAVED DATA ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.latent_hj import slosh_pend as sp  # noqa: E402

STRIDE = 4            # 200 Hz -> 50 Hz
DT_WM = 0.005 * STRIDE
SCR = Path(os.environ.get("LATENT_HJ_SCR", "/common/home/ss5772/.claude/jobs/f1580deb/tmp/latent_hj"))


# =============================================================================== data
def _episode_arrays(ep, P):
    from world_model.v2.latent_hj import spill_data as sd
    r = sd.replay_episode(ep, P)
    ax = ep["axis"] / np.linalg.norm(ep["axis"], axis=1, keepdims=True)
    v = np.gradient(ep["pos"], 0.005, axis=0)
    acc = np.gradient(v, 0.005, axis=0)
    w = np.cross(ax[:-1], ax[1:]) / 0.005
    w = np.concatenate([w[:1], w], 0)
    T = (len(ax) // STRIDE) * STRIDE
    idx = np.arange(0, T, STRIDE)
    obs = np.concatenate([ax[idx], v[idx], w[idx]], 1)
    act = np.concatenate([acc[:T].reshape(-1, STRIDE, 3).mean(1), w[:T].reshape(-1, STRIDE, 3).mean(1)], 1)
    # latent-truth for analysis only (never a WM input): pendulum state at the WM ticks
    return dict(obs=obs.astype(np.float32), act=act.astype(np.float32), l=r["l"][idx].astype(np.float32),
                n=r["n"][idx].astype(np.float32), nd=r["nd"][idx].astype(np.float32),
                onset=int(r["onset"] // STRIDE) if r["onset"] >= 0 else -1, onset200=r["onset"])


def _build_shard(args):
    path, h = args
    from world_model.v2.latent_hj import spill_data as sd
    P = sp.PendParams(h=h)
    out = []
    for ep in sd.load_dense(path):
        d = _episode_arrays(ep, P)
        d["seed"] = ep["seed"]; d["shard"] = Path(path).name
        out.append(d)
    return out


def cmd_build(a):
    import multiprocessing as mp
    from world_model.v2.latent_hj import spill_data as sd
    split = sd.shard_split(sd.list_shards())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, paths in split.items():
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=3) as pool:
            res = pool.map(_build_shard, [(p, a.h) for p in paths[: a.max_shards]], chunksize=1)
        eps = [e for r in res for e in r]
        lens = np.array([len(e["obs"]) for e in eps])
        cat = lambda k: np.concatenate([e[k] for e in eps], 0)
        np.savez(out / f"{name}.npz", obs=cat("obs"), act=cat("act"), l=cat("l"), n=cat("n"), nd=cat("nd"),
                 lens=lens, onset=np.array([e["onset"] for e in eps]), onset200=np.array([e["onset200"] for e in eps]),
                 seed=np.array([e["seed"] for e in eps]))
        print(f"[build] {name}: {len(eps)} eps, {lens.sum()} steps, spill eps {np.mean([e['onset'] >= 0 for e in eps]):.2f}",
              flush=True)


def load_split(data, name):
    f = np.load(Path(data) / f"{name}.npz")
    lens = f["lens"]; st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return dict(obs=f["obs"], act=f["act"], l=f["l"], n=f["n"], nd=f["nd"], lens=lens, start=st,
                onset=f["onset"], seed=f["seed"])


# =============================================================================== model
def build_model(d_obs=9, d_act=6, d_h=192, d_e=128):
    import torch
    import torch.nn as nn

    class KinWM(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_obs, d_e), nn.SiLU(), nn.Linear(d_e, d_e), nn.SiLU())
            self.cell = nn.GRUCell(d_e + d_act, d_h)
            self.dec = nn.Sequential(nn.Linear(d_h, 256), nn.SiLU(), nn.Linear(256, d_obs))
            self.mhead = nn.Sequential(nn.Linear(d_h + d_obs, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                       nn.Linear(256, 1))
            self.h0 = nn.Parameter(torch.zeros(d_h))
            self.d_h, self.d_obs, self.d_act = d_h, d_obs, d_act
            self.register_buffer("o_mu", torch.zeros(d_obs)); self.register_buffer("o_sd", torch.ones(d_obs))
            self.register_buffer("a_mu", torch.zeros(d_act)); self.register_buffer("a_sd", torch.ones(d_act))

        # normalized-space helpers
        def no(self, o):
            return (o - self.o_mu) / self.o_sd

        def na(self, a):
            return (a - self.a_mu) / self.a_sd

        def step(self, h, o_n, a):
            """h (B,dh), o_n normalized obs (B,do), a RAW action (B,da) -> h', o^'_n."""
            h2 = self.cell(torch.cat([self.enc(o_n), self.na(a)], -1), h)
            return h2, self.dec(h2)

        def margin(self, h, o_n):
            return torch.tanh(self.mhead(torch.cat([h, o_n], -1))[..., 0])

        def filter(self, obs, act):
            """obs (B,T,do) raw, act (B,T,da) raw -> H (B,T,dh) with H[:,t] having seen o_<=t-1,a_<=t-1
            and margin l^_t = m(H_t, o_t)."""
            B, T, _ = obs.shape
            h = self.h0.expand(B, -1)
            on = self.no(obs)
            Hs = []
            for t in range(T):
                Hs.append(h)
                h, _ = self.step(h, on[:, t], act[:, t])
            return torch.stack(Hs, 1), on

    return KinWM()


def cmd_train(a):
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = load_split(a.data, "train"), load_split(a.data, "val")
    wm = build_model().to(dev)
    wm.o_mu.copy_(torch.tensor(tr["obs"].mean(0))); wm.o_sd.copy_(torch.tensor(tr["obs"].std(0) + 1e-3))
    wm.a_mu.copy_(torch.tensor(tr["act"].mean(0))); wm.a_sd.copy_(torch.tensor(tr["act"].std(0) + 1e-3))
    opt = torch.optim.AdamW(wm.parameters(), lr=a.lr, weight_decay=1e-5)
    rng = np.random.default_rng(a.seed)
    W, burn, K = a.window, a.burn, a.K

    def batch(D, B):
        ok = np.flatnonzero(D["lens"] > W + 1)
        # oversample spill episodes 2x
        wts = np.where(D["onset"][ok] >= 0, 2.0, 1.0); wts /= wts.sum()
        e = rng.choice(ok, B, p=wts)
        s0 = np.array([0 if rng.random() < 0.3 else rng.integers(0, D["lens"][i] - W) for i in e])
        ix = (D["start"][e] + s0)[:, None] + np.arange(W)[None]
        g = lambda k: torch.tensor(D[k][ix], device=dev)
        return g("obs"), g("act"), g("l"), torch.tensor(s0 == 0, device=dev)

    def loss_fn(obs, act, l, from_start):
        Hs, on = wm.filter(obs, act)
        B, T, _ = obs.shape
        m = torch.ones(B, T, device=dev); m[~from_start, :burn] = 0.0
        # one-step obs prediction + margin on filtered states
        L_obs = (((wm.dec(Hs[:, 1:]) - on[:, 1:]) ** 2).mean(-1) * m[:, 1:]).sum() / m[:, 1:].sum()
        lh = wm.margin(Hs, on)
        L_mar = (((lh - l) ** 2) * m).sum() / m.sum()
        # K-step overshooting from a random subset of filtered states (dataset actions)
        ts = torch.randint(burn, T - K - 1, (a.n_os,), device=dev)
        bs = torch.randint(0, B, (a.n_os,), device=dev)
        h = Hs[bs, ts]; o_n = on[bs, ts]
        L_os_o, L_os_m = 0.0, 0.0
        for k in range(K):
            h, o_hat = wm.step(h, o_n, act[bs, ts + k])
            L_os_o = L_os_o + ((o_hat - on[bs, ts + k + 1]) ** 2).mean()
            L_os_m = L_os_m + ((wm.margin(h, o_hat) - l[bs, ts + k + 1]) ** 2).mean()
            o_n = o_hat
        return L_obs + L_os_o / K + a.w_mar * (L_mar + L_os_m / K), dict(obs=float(L_obs), mar=float(L_mar),
                                                                          os_o=float(L_os_o / K), os_m=float(L_os_m / K))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    curve = []; t0 = time.time()
    for it in range(1, a.steps + 1):
        obs, act, l, fs = batch(tr, a.batch)
        loss, lg = loss_fn(obs, act, l, fs)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 10.0); opt.step()
        if it % a.log_every == 0 or it == 1:
            with torch.no_grad():
                vl = loss_fn(*batch(va, a.batch))[1]
            curve.append(dict(it=it, train=lg, val=vl, sec=time.time() - t0))
            print(f"[wm] it {it} ({time.time()-t0:.0f}s) train " + " ".join(f"{k}={v:.4f}" for k, v in lg.items())
                  + " | val " + " ".join(f"{k}={v:.4f}" for k, v in vl.items()), flush=True)
    torch.save(dict(state=wm.cpu().state_dict(), args=vars(a)), out / "wm.pt")
    (out / "curve.json").write_text(json.dumps(curve))


def load_wm(path, device="cpu"):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    wm = build_model().to(device); wm.load_state_dict(ck["state"]); wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


class KinWMObserver:
    """Online posterior filter for the controller (200 Hz sim -> 50 Hz WM ticks).
    Mirrors _episode_arrays: obs at every STRIDE-th sim step, action = mean over the
    following STRIDE steps of (cup-origin accel, axis angular velocity). Online the
    accel is a backward 2nd difference (training used central differences)."""

    def __init__(self, wm, dt=0.005):
        import torch
        self.t, self.wm, self.dt = torch, wm, dt
        self.reset()

    def reset(self):
        self.k = 0
        self.p = []; self.ax_prev = None
        self.acc_sum = np.zeros(3); self.w_sum = np.zeros(3); self.n_sum = 0
        self.h = self.wm.h0.detach()[None].clone(); self.o_prev = None
        self.z = None

    def update(self, pos, axis):
        t = self.t
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.p.append(np.asarray(pos, float).copy()); self.p = self.p[-3:]
        v = (self.p[-1] - self.p[-2]) / self.dt if len(self.p) >= 2 else np.zeros(3)
        acc = (self.p[-1] - 2 * self.p[-2] + self.p[-3]) / self.dt ** 2 if len(self.p) >= 3 else np.zeros(3)
        w = np.cross(self.ax_prev, axis) / self.dt if self.ax_prev is not None else np.zeros(3)
        self.ax_prev = axis
        if self.k % STRIDE == 0:
            o = t.tensor(np.concatenate([axis, v, w]), dtype=t.float32)[None]
            on = self.wm.no(o)
            with t.no_grad():
                if self.o_prev is not None and self.n_sum > 0:
                    a = t.tensor(np.concatenate([self.acc_sum, self.w_sum]) / self.n_sum, dtype=t.float32)[None]
                    self.h, _ = self.wm.step(self.h, self.o_prev, a)
            self.o_prev = on
            self.z = t.cat([self.h, on], -1)[0].clone()
            self.acc_sum[:] = 0; self.w_sum[:] = 0; self.n_sum = 0
        self.acc_sum += acc; self.w_sum += w; self.n_sum += 1
        self.k += 1
        return self.z


# =============================================================================== latent Dyn
class WMDyn:
    """reach_value Dyn over the learned latent z = [h (dh), o_n (do)]."""

    def __init__(self, wm, P, device, bank=None):
        import torch
        self.wm, self.P, self.dev = wm, P, device
        U = sp.control_set(P)                              # (a (3), w (3)) -- WM action space units
        self.U = torch.tensor(U, dtype=torch.float32, device=device)
        self.bank = bank                                   # (N, dz) filtered latents (+ imagined)
        self.dh = wm.d_h

    def sample_states(self, B):
        import torch
        z = self.bank[torch.randint(len(self.bank), (B,), device=self.dev)]
        return z

    def step(self, z, u):
        import torch
        shp = torch.broadcast_shapes(z.shape[:-1], u.shape[:-1])
        z = z.expand(*shp, z.shape[-1]).reshape(-1, z.shape[-1])
        u = u.expand(*shp, u.shape[-1]).reshape(-1, u.shape[-1])
        h, o_n = z[:, :self.dh], z[:, self.dh:]
        h2, o2 = self.wm.step(h, o_n, u)
        return torch.cat([h2, o2], -1).view(*shp, -1)

    def margin(self, z):
        return self.wm.margin(z[..., :self.dh], z[..., self.dh:])

    def features(self, z):
        return z


def latent_bank(wm, D, device, max_eps=None, stride=1, imag_k=10, n_imag=200000, seed=0):
    """filtered latents of the given split + imagined random-fallback-action rollouts."""
    import torch
    rng = np.random.default_rng(seed)
    Z = []
    eps = np.arange(len(D["lens"]))[: max_eps]
    with torch.no_grad():
        for i0 in range(0, len(eps), 64):
            ii = eps[i0:i0 + 64]
            T = int(D["lens"][ii].max())
            obs = np.zeros((len(ii), T, 9), np.float32); act = np.zeros((len(ii), T, 6), np.float32)
            for j, e in enumerate(ii):
                s, L = D["start"][e], D["lens"][e]
                obs[j, :L] = D["obs"][s:s + L]; act[j, :L] = D["act"][s:s + L]
                obs[j, L:] = obs[j, L - 1]
            H, on = wm.filter(torch.tensor(obs, device=device), torch.tensor(act, device=device))
            for j, e in enumerate(ii):
                L = D["lens"][e]
                Z.append(torch.cat([H[j, :L:stride], on[j, :L:stride]], -1))
    Z = torch.cat(Z, 0)
    # imagined off-data coverage: random fallback-control sequences from random bank states
    P = sp.PendParams()
    U = torch.tensor(sp.control_set(P), dtype=torch.float32, device=device)
    dyn = WMDyn(wm, P, device)
    zi = Z[torch.randint(len(Z), (n_imag // imag_k,), device=device)]
    out = [Z]
    with torch.no_grad():
        for k in range(imag_k):
            u = U[torch.randint(len(U), (len(zi),), device=device)]
            zi = dyn.step(zi, u)
            out.append(zi)
    return torch.cat(out, 0)


def cmd_value(a):
    import torch
    from world_model.v2.latent_hj import reach_value as rv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_wm(a.wm, dev)
    tr = load_split(a.data, "train")
    bank = latent_bank(wm, tr, dev, max_eps=a.max_eps, stride=2)
    print(f"[wm-value] bank {tuple(bank.shape)}", flush=True)
    P = sp.PendParams()
    P.dt = DT_WM                                        # one WM step per Bellman backup
    dyn = WMDyn(wm, P, dev, bank)
    net, curve = rv.train(dyn, a.out, iters=a.iters, batch=a.batch, gamma0=0.8, gamma_final=a.gamma,
                          K=a.K, lam=0.5, device=dev, log_every=a.log_every)
    meta = dict(dx=int(bank.shape[1]), hidden=256, layers=3, gamma=a.gamma, K=a.K, kind="kin_wm_latent",
                wm=str(a.wm), P=P.as_dict())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rv.save(net.cpu(), Path(a.out) / "value.pt", meta)
    (Path(a.out) / "curve.json").write_text(json.dumps(curve))


def cmd_gate(a):
    import torch
    from world_model.v2.latent_hj import reach_value as rv
    from world_model.v2.latent_hj.validate_stage0 import gate_eval
    wm = load_wm(a.wm, "cpu")
    net, meta = rv.load(a.value) if a.value else (None, None)
    rows = {}
    for name in ("val", "test"):
        D = load_split(a.data, name)
        R = []
        with torch.no_grad():
            for e in range(len(D["lens"])):
                s, L = D["start"][e], D["lens"][e]
                obs = torch.tensor(D["obs"][s:s + L])[None]; act = torch.tensor(D["act"][s:s + L])[None]
                H, on = wm.filter(obs, act)
                lh = wm.margin(H, on)[0].numpy()
                r = dict(onset=int(D["onset"][e]), stride=1, l=D["l"][s:s + L], lhat=lh)
                if net is not None:
                    z = torch.cat([H[0], on[0]], -1)
                    r["V"] = net(z, wm.margin(H[0], on[0])).numpy()
                R.append(r)
        rows[name] = R
    out = dict(wm=a.wm, value=a.value, n_val=len(rows["val"]), n_test=len(rows["test"]))
    keys = ["l", "lhat"] + (["V"] if net is not None else [])
    for fa in (0.02, 0.05, 0.10):
        for key in keys:
            g = gate_eval(rows["val"], rows["test"], key, fa, dt=DT_WM)
            out[f"{key}/fa{fa}"] = g
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}), flush=True)
    # margin-head fidelity (how well the latent recovers the invisible slosh margin)
    lt = np.concatenate([r["l"] for r in rows["test"]]); lp = np.concatenate([r["lhat"] for r in rows["test"]])
    out["margin_head_test"] = dict(mae=float(np.abs(lt - lp).mean()), corr=float(np.corrcoef(lt, lp)[0, 1]))
    print(json.dumps(out["margin_head_test"]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", default=str(SCR / "wmdata"))
    b.add_argument("--workers", type=int, default=16); b.add_argument("--h", type=float, default=0.04)
    b.add_argument("--max-shards", type=int, default=100000)
    t = sub.add_parser("train")
    t.add_argument("--data", default=str(SCR / "wmdata")); t.add_argument("--out", default=str(SCR / "wm"))
    t.add_argument("--steps", type=int, default=20000); t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--window", type=int, default=160)
    t.add_argument("--burn", type=int, default=60); t.add_argument("--K", type=int, default=10)
    t.add_argument("--n-os", type=int, default=1024); t.add_argument("--w-mar", type=float, default=2.0)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--log-every", type=int, default=500)
    v = sub.add_parser("value")
    v.add_argument("--data", default=str(SCR / "wmdata")); v.add_argument("--wm", default=str(SCR / "wm" / "wm.pt"))
    v.add_argument("--out", default=str(SCR / "wm_v")); v.add_argument("--iters", type=int, default=20000)
    v.add_argument("--batch", type=int, default=2048); v.add_argument("--gamma", type=float, default=0.98)
    v.add_argument("--K", type=int, default=4); v.add_argument("--max-eps", type=int, default=None)
    v.add_argument("--log-every", type=int, default=1000)
    g = sub.add_parser("gate")
    g.add_argument("--data", default=str(SCR / "wmdata")); g.add_argument("--wm", default=str(SCR / "wm" / "wm.pt"))
    g.add_argument("--value", default=None); g.add_argument("--out", default=str(SCR / "stage1_gate.json"))
    a = ap.parse_args()
    dict(build=cmd_build, train=cmd_train, value=cmd_value, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    main()
