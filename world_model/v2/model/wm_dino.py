"""world_model/v2/model/wm_dino.py -- OPTION B: frozen DINOv2-small encoder +
action-conditioned latent predictor (DINO-WM / arXiv 2509.14758 style), with
multi-horizon spill hazard + SPILL-ONLY Fisac value trained in imagination.

Pipeline (two stages; stage 1 is the only one touching pixels):
  1. cache : frozen DINOv2 ViT-S/14 (torch.hub facebookresearch/dinov2, 22M params)
             on every camera frame -> CLS + (pool x pool) avg-pooled patch tokens,
             fp16, one npz per prepared episode. Run once per dataset.
     python -m world_model.v2.model.wm_dino cache --prep $PREP/ds --out $PREP/ds_dino --device cuda
  2. train : recurrent latent predictor over the cached tokens (+ proprio + FK cup
             kinematics), conditioned on actions.
     python -m world_model.v2.model.wm_dino train --prep $PREP/ds --feats $PREP/ds_dino \
         --out $RUN/dino --steps 30000 --members 5 --device cuda

Model (per ensemble member; deterministic, ~3.7M params):
  z_t   = Enc(DINO tokens_t (all cams), proprio/cup-kin_t)       observation latent
  h_t   = GRU([z_{t-1}, a_{t-1}], h_{t-1})   (cuDNN over the sequence; a_t applied AFTER obs t)
  z^_t  = Prior(h_t)                         predicted latent WITHOUT obs_t
  f     = [h, z]  (z observed when filtering, z^ when imagining: h_{t+1} = GRU([z^_t, a_t], h_t))
  Losses: decode frozen DINO tokens + proprio from f_t AND from K-step open-loop
  imagined f^_{t+k} (overshooting, k=1..K, dataset actions) -- what makes rollouts
  under a PLANNED action sequence meaningful; consistency |z^_t - sg(z_t)|^2;
  hazard + margin heads on observed and imagined features; Fisac value on imagined
  rollouts (dataset actions + noise-perturbed copies for off-data plan coverage).
  Deviation from DINO-WM: pooled tokens (CLS + pool x pool patches) and a recurrent
  predictor instead of a ViT over full patch grids -- sized for a 15 Hz shield.
Object emphasis: per-camera loss weights (--cam-weight zed=2) and the
cup-centred crop from the v2 dataset; DINO patch tokens keep small objects that a
64x64 pixel decoder blurs away.

Shield API (15 Hz): DinoWMPredictor.observe_step(...) -> state; .imagine(state,
action_seqs (N,K,A)) -> hazard/value/margin per member (ensemble mean/std).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.model import common as C  # noqa: E402
from world_model.v2.model import data as D  # noqa: E402

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ==========================================================================
# stage 1: frozen DINOv2 featurizer + cache
# ==========================================================================
class DinoFeaturizer:
    def __init__(self, device="cuda", img_px=126, pool=3, arch="dinov2_vits14"):
        self.m = torch.hub.load("facebookresearch/dinov2", arch, verbose=False).to(device).eval()
        self.device, self.img_px, self.pool = device, img_px, pool
        self.dim = self.m.embed_dim

    @torch.no_grad()
    def __call__(self, imgs_uint8: np.ndarray, bs=256) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N, 1+pool^2, dim) float16."""
        out = []
        g = self.img_px // 14
        for i in range(0, len(imgs_uint8), bs):
            x = torch.as_tensor(imgs_uint8[i:i + bs], device=self.device).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(self.img_px, self.img_px), mode="bilinear", align_corners=False, antialias=True)
            x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                o = self.m.forward_features(x)
            cls = o["x_norm_clstoken"].float()
            pt = o["x_norm_patchtokens"].float().view(len(x), g, g, -1).permute(0, 3, 1, 2)
            pt = F.adaptive_avg_pool2d(pt, self.pool).flatten(2).transpose(1, 2)
            out.append(torch.cat([cls[:, None], pt], 1).half().cpu().numpy())
        return np.concatenate(out, 0)


def build_cache(prep_dir, out, device="cuda", img_px=126, pool=3, cams=None):
    prep_dir, out = Path(prep_dir), Path(out); out.mkdir(parents=True, exist_ok=True)
    fz = DinoFeaturizer(device, img_px, pool)
    index = json.loads((prep_dir / "index.json").read_text())
    t0 = time.time(); n = 0
    for r in index["episodes"]:
        fn = out / r["file"]
        if fn.exists():
            continue
        ep = D.load_canonical(prep_dir / r["file"], True, cams)
        np.savez(fn, **{c: fz(im) for c, im in sorted(ep["images"].items())})
        n += ep["T"] * len(ep["images"])
    (out / "cache_meta.json").write_text(json.dumps(dict(img_px=img_px, pool=pool, dim=fz.dim, arch="dinov2_vits14")))
    print(f"[dino cache] {n} frames in {time.time()-t0:.0f}s -> {out}", flush=True)


# ==========================================================================
# stage 2: latent predictor
# ==========================================================================
class DinoLatentWM(nn.Module):
    """Recurrent latent predictor. z_t = Enc(obs_t) is the observation latent;
    h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1}) summarises the past (cuDNN over the whole
    sequence when filtering); z^_t = Prior(h_t) predicts z_t WITHOUT seeing obs_t.
    Feature f = [h, z] (z = z_t when observed, z^_t when imagined). Imagination
    feeds its own predictions back: h_{t+1} = GRU([z^_t, a_t], h_t)."""

    def __init__(self, n_cams, n_tok, tok_dim, d_low, d_act, d_tok=64, d_emb=256, d_h=256, d_s=None,
                 n_h=len(C.HORIZONS_S)):
        super().__init__()
        self.n_cams, self.n_tok, self.tok_dim, self.d_low = n_cams, n_tok, tok_dim, d_low
        self.tok_in = nn.Linear(tok_dim, d_tok)
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, n_tok, d_tok))
        self.enc = nn.Sequential(C.mlp(n_cams * n_tok * d_tok + d_low, d_emb, d_emb), nn.LayerNorm(d_emb))
        self.gru = nn.GRU(d_emb + d_act, d_h, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(d_h))
        self.prior = nn.Sequential(C.mlp(d_h, d_emb, d_emb), nn.LayerNorm(d_emb))
        d_f = d_h + d_emb
        self.dec_tok = C.mlp(d_f, n_cams * n_tok * d_tok, 512)
        self.tok_out = nn.Linear(d_tok, tok_dim)
        self.dec_low = C.mlp(d_f, d_low, 256)
        self.heads = C.SpillHeads(d_f, n_h)
        self.d_f, self.d_h, self.d_emb = d_f, d_h, d_emb

    def encode(self, tok, low):
        """tok (...,cams,n_tok,dim) normalized; low (...,d_low) -> z (...,d_emb)."""
        z = self.tok_in(tok) + self.cam_emb
        return self.enc(torch.cat([z.flatten(-3), low], -1))

    def decode(self, f):
        t = self.dec_tok(f).view(*f.shape[:-1], self.n_cams, self.n_tok, -1)
        return self.tok_out(F.silu(t)), self.dec_low(f)

    def filter(self, z, act):
        """z (B,T,E) observed latents, act (B,T,A) -> h (B,T,dh) (h_t sees z/a up to t-1)
        and zq (B,T,E) = Prior(h_t)."""
        B = z.shape[0]
        x = torch.cat([z[:, :-1], act[:, :-1]], -1)
        h0 = self.h0.expand(1, B, -1).contiguous()
        hs, _ = self.gru(x, h0) if x.shape[1] else (z[:, :0, :0], None)
        h = torch.cat([h0.transpose(0, 1), hs], 1)
        return h, self.prior(h)

    def step(self, h, z, a):
        """one recurrent step: (N,dh),(N,E),(N,A) -> next h (N,dh)."""
        _, hn = self.gru(torch.cat([z, a], -1)[:, None], h[None].contiguous())
        return hn[0]

    def imagine(self, h, z, actions):
        """From (h_t, z_t) roll K steps under actions (N,K,A) -> feats (N,K,d_f);
        feats[:,k] = imagined state after applying actions[:,k]."""
        out = []
        for k in range(actions.shape[1]):
            h = self.step(h, z, actions[:, k]); z = self.prior(h)
            out.append(torch.cat([h, z], -1))
        return torch.stack(out, 1)


# ==========================================================================
# data plumbing
# ==========================================================================
def load_split(prep_dir, feats_dir, split, cams=None):
    eps = D.load_prepared(prep_dir, split, with_images=False)
    index = json.loads((Path(prep_dir) / "index.json").read_text())
    fmap = {r["id"]: r["file"] for r in index["episodes"]}
    for e in eps:
        with np.load(Path(feats_dir) / fmap[e["id"]]) as f:
            cs = sorted(f.files) if cams is None else list(cams)
            e["dino"] = np.stack([f[c] for c in cs], 1)       # (T,cams,n_tok,dim) fp16
            e["dino_cams"] = cs
    return eps


class Batcher:
    """GPU-resident padded tensors (tokens fp16, normalized once) -> batches are
    pure GPU gathers (the v0 per-step numpy normalization was CPU-bound)."""

    def __init__(self, eps, low_norm, act_norm, tok_mean, tok_std, horizons, device, T_max=160):
        self.dev = device
        lens = [min(e["T"], T_max) for e in eps]; Tm = max(lens)
        tm, ts = torch.tensor(tok_mean, device=device), torch.tensor(tok_std, device=device)
        self.tok = torch.stack([F.pad(((torch.tensor(e["dino"][:Tm], device=device).float() - tm) / ts).half(),
                                      (0, 0, 0, 0, 0, 0, 0, Tm - l)) for e, l in zip(eps, lens)])
        pb = lambda A: torch.tensor(C.pad_batch([np.asarray(x, np.float32)[:Tm] for x in A], Tm), device=device)
        tg = [C.episode_targets(e, horizons) for e in eps]
        self.low = pb([low_norm(C.step_features(e, use_action=False)) for e in eps])
        self.act = pb([act_norm(np.asarray(e["action"], np.float32)) for e in eps])
        self.y, self.valid = pb([t["y"] for t in tg]), pb([t["valid"] for t in tg])
        self.margin, self.scored = pb([t["margin"] for t in tg]), pb([t["scored"] for t in tg])
        self.mask = pb([np.ones(l) for l in lens])
        self.tgt = tg

    def get(self, idx, T_max=None):
        i = torch.as_tensor(np.asarray(idx), device=self.dev)
        return dict(tok=self.tok[i].float(), low=self.low[i], act=self.act[i], y=self.y[i], valid=self.valid[i],
                    margin=self.margin[i], scored=self.scored[i], mask=self.mask[i])


def _shift(x, k):
    """x (B,T,...) -> x[:, t+k] aligned at t (zero-padded)."""
    if k == 0:
        return x
    pad = torch.zeros_like(x[:, :k])
    return torch.cat([x[:, k:], pad], 1)


def member_loss(wm: DinoLatentWM, b, pos_weight, cam_w, K=8, K_val=15, gamma=0.95, act_noise=0.3,
                n_val_starts=256, w=None):
    w = w or dict(tok=1.0, low=0.5, cons=0.5, haz=1.0, mar=0.5, val=1.0)
    s = wm.encode(b["tok"], b["low"])
    h, sq = wm.filter(s, b["act"])
    f = torch.cat([h, s], -1)
    m = b["mask"]
    tok_hat, low_hat = wm.decode(f)
    cw = cam_w.view(1, 1, -1, 1, 1)
    msum = m.sum().clamp_min(1)
    L = {}
    L["tok"] = (((tok_hat - b["tok"]) ** 2 * cw).mean((-1, -2, -3)) * m).sum() / msum
    L["low"] = (((low_hat - b["low"]) ** 2).mean(-1) * m).sum() / msum
    L["cons"] = (((sq - s.detach()) ** 2).mean(-1) * m).sum() / msum
    haz, mar = wm.heads.losses(f, b["y"], b["valid"] * m, b["margin"], b["scored"] * m, pos_weight)
    # ---- K-step overshooting from every posterior state (dataset actions)
    B, T, _ = f.shape
    hN, sN = h.reshape(B * T, -1), s.reshape(B * T, -1)
    acts = torch.stack([_shift(b["act"], k) for k in range(K)], 2).reshape(B * T, K, -1)
    fi = wm.imagine(hN, sN, acts).view(B, T, K, -1)
    ot, ol, oh, om = 0.0, 0.0, 0.0, 0.0
    for k in range(1, K + 1):
        mk = _shift(m, k) * m
        fk = fi[:, :, k - 1]
        th, lh = wm.decode(fk)
        ot = ot + (((th - _shift(b["tok"], k)) ** 2 * cw).mean((-1, -2, -3)) * mk).sum() / mk.sum().clamp_min(1)
        ol = ol + (((lh - _shift(b["low"], k)) ** 2).mean(-1) * mk).sum() / mk.sum().clamp_min(1)
        hz_k, mr_k = wm.heads.losses(fk, _shift(b["y"], k), _shift(b["valid"], k) * mk,
                                     _shift(b["margin"], k), _shift(b["scored"], k) * mk, pos_weight)
        oh, om = oh + hz_k, om + mr_k
    L["tok"] = L["tok"] + ot / K; L["low"] = L["low"] + ol / K
    L["haz"] = haz + oh / K; L["mar"] = mar + om / K
    # ---- Fisac value in imagination (dataset actions + noisy copies)
    valid_starts = torch.nonzero(m.reshape(-1) > 0)[:, 0]
    pick = valid_starts[torch.randint(len(valid_starts), (min(n_val_starts, len(valid_starts)),), device=f.device)]
    va = torch.stack([_shift(b["act"], k) for k in range(K_val)], 2).reshape(B * T, K_val, -1)[pick]
    va = torch.cat([va, va + act_noise * torch.randn_like(va)], 0)
    h0, s0 = hN[pick].detach().repeat(2, 1), sN[pick].detach().repeat(2, 1)
    with torch.no_grad():
        fv = wm.imagine(h0, s0, va)
    L["val"] = wm.heads.value_loss(torch.cat([torch.cat([h0, s0], -1)[:, None], fv], 1), gamma)
    total = sum(w[k] * v for k, v in L.items())
    return total, {k: float(v.detach()) for k, v in L.items()}


# ==========================================================================
# training
# ==========================================================================
def train(prep_dir, feats_dir, out, steps=500, members=1, batch=16, lr=3e-4, device="cuda", K=8, K_val=15,
          gamma=0.95, cam_weight=None, seed=0, log_every=50, T_max=160, horizons=C.HORIZONS_S, act_noise=0.3, action_key="action"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tr = load_split(prep_dir, feats_dir, "train"); va = load_split(prep_dir, feats_dir, "val")
    for e in tr + va:                      # WM action space: joint refs | commanded cup accel | both
        e["action"] = C.get_action(e, action_key)
    cams = tr[0]["dino_cams"]
    sample = np.concatenate([e["dino"][::4].astype(np.float32) for e in tr], 0)
    tok_mean, tok_std = sample.mean(0), sample.std(0) + 1e-3
    low_norm = C.Normalizer().fit([C.step_features(e, use_action=False) for e in tr])
    act_norm = C.Normalizer().fit([np.asarray(e["action"], np.float32) for e in tr])
    Btr = Batcher(tr, low_norm, act_norm, tok_mean, tok_std, horizons, device, T_max)
    Bva = Batcher(va, low_norm, act_norm, tok_mean, tok_std, horizons, device, T_max)
    pw = torch.tensor(C.auto_pos_weight([t["y"] for t in Btr.tgt], [t["valid"] for t in Btr.tgt]), device=device)
    cw = torch.ones(len(cams), device=device)
    for kv in (cam_weight or []):
        c, v = kv.split("="); cw[cams.index(c)] = float(v)
    n_cams, n_tok, tok_dim = tr[0]["dino"].shape[1:]
    d_low, d_act = Btr.low[0].shape[1], Btr.act[0].shape[1]
    arch = dict(n_cams=n_cams, n_tok=n_tok, tok_dim=tok_dim, d_low=d_low, d_act=d_act, n_h=len(horizons))
    C.seed_all(seed)
    wms = nn.ModuleList([DinoLatentWM(**arch) for _ in range(members)]).to(device)
    opts = [torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=1e-4) for w in wms]
    rngs = [np.random.default_rng(seed + 100 * i) for i in range(members)]
    print(f"[dino-wm] members={members} params/member={sum(p.numel() for p in wms[0].parameters())/1e6:.2f}M "
          f"train_eps={len(tr)} val_eps={len(va)} cams={cams} tokens={n_tok}x{tok_dim}", flush=True)
    curve = []; t0 = time.time()
    for it in range(1, steps + 1):
        logs = []
        for i, (wm, opt) in enumerate(zip(wms, opts)):
            idx = rngs[i].integers(0, len(tr), batch)                  # bootstrap per member
            loss, lg = member_loss(wm, Btr.get(idx, T_max), pw, cw, K, K_val, gamma, act_noise)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt.step()
            wm.heads.update_target(); logs.append(lg)
        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                vb = Bva.get(np.arange(min(len(va), 32)), T_max)
                vl, vlg = member_loss(wms[0], vb, pw, cw, K, K_val, gamma, act_noise)
            tl = {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}
            curve.append(dict(step=it, train=tl, val=vlg, sec=time.time() - t0))
            print(f"[dino-wm] step {it} ({time.time()-t0:.0f}s) train " + " ".join(f"{k}={v:.3f}" for k, v in tl.items())
                  + " | val " + " ".join(f"{k}={v:.3f}" for k, v in vlg.items()), flush=True)
    ck = dict(kind="dino_wm", arch=arch, cams=cams, horizons_s=list(horizons), gamma=gamma,
              tok_mean=tok_mean, tok_std=tok_std, low_norm=low_norm.state(), act_norm=act_norm.state(), action_key=action_key,
              feats_dir=str(feats_dir),
              members=[{k: v.cpu() for k, v in w.state_dict().items()} for w in wms],
              feats_meta=json.loads((Path(feats_dir) / "cache_meta.json").read_text()), prep_dir=str(prep_dir))
    torch.save(ck, out / "model.pt"); (out / "curves.json").write_text(json.dumps(curve))
    return out / "model.pt", curve


# ==========================================================================
# predictor + shield API
# ==========================================================================
class DinoWMPredictor:
    """Harness predictor (predict(ep) uses the cached-or-live DINO features) +
    streaming / imagination API for the shield."""
    probabilistic = True

    def __init__(self, ckpt, device="cpu", featurizer: DinoFeaturizer | None = None, feats_dir=None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.ck, self.dev = ck, device
        self.horizons_s = tuple(ck["horizons_s"]); self.cams = ck["cams"]
        self.wms = []
        for sd in ck["members"]:
            w = DinoLatentWM(**ck["arch"]); w.load_state_dict(sd); self.wms.append(w.to(device).eval())
        self.low_norm = C.Normalizer(ck["low_norm"]["mean"], ck["low_norm"]["std"])
        self.act_norm = C.Normalizer(ck["act_norm"]["mean"], ck["act_norm"]["std"])
        feats_dir = feats_dir or ck.get("feats_dir")
        self.tm = torch.tensor(ck["tok_mean"], device=device); self.ts = torch.tensor(ck["tok_std"], device=device)
        self.fz = featurizer; self.feats_dir = feats_dir
        self.name = f"dino-WM(x{len(self.wms)})"; self.last_std = None

    def _tokens(self, ep):
        if "dino" in ep:
            return ep["dino"]
        if self.feats_dir is not None:
            index = json.loads((Path(self.ck["prep_dir"]) / "index.json").read_text())
            f = {r["id"]: r["file"] for r in index["episodes"]}.get(ep["id"])
            if f and (Path(self.feats_dir) / f).exists():
                with np.load(Path(self.feats_dir) / f) as z:
                    return np.stack([z[c] for c in self.cams], 1)
        if self.fz is None:
            m = self.ck["feats_meta"]
            self.fz = DinoFeaturizer(self.dev if self.dev != "cpu" else "cpu", m["img_px"], m["pool"])
        return np.stack([self.fz(ep["images"][c]) for c in self.cams], 1)

    @torch.no_grad()
    def predict(self, ep):
        tok = (torch.tensor(self._tokens(ep).astype(np.float32), device=self.dev)[None] - self.tm) / self.ts
        low = torch.tensor(self.low_norm(C.step_features(ep, use_action=False)), device=self.dev)[None]
        act = torch.tensor(self.act_norm(C.get_action(ep, self.ck.get("action_key", "action"))), device=self.dev)[None]
        P = []
        for w in self.wms:
            z = w.encode(tok, low); h, _ = w.filter(z, act)
            P.append(w.heads.hazard(torch.cat([h, z], -1))[0])
        P = torch.stack(P)
        self.last_std = P.std(0).cpu().numpy() if len(P) > 1 else None
        return P.mean(0).cpu().numpy()

    # ---- streaming: one call per vision tick ---------------------------------
    def reset(self):
        self._st = [None] * len(self.wms); self._a = None

    @torch.no_grad()
    def observe_step(self, tok_t: np.ndarray, low_t: np.ndarray, a_prev: np.ndarray | None):
        """tok_t (cams,n_tok,dim) DINO tokens, low_t raw step features (proprio+cup kin),
        a_prev = action applied since the last tick. Returns per-member (h,s)."""
        tok = (torch.tensor(tok_t.astype(np.float32), device=self.dev)[None] - self.tm) / self.ts
        low = torch.tensor(self.low_norm(low_t[None].astype(np.float32)), device=self.dev)
        out = []
        for i, w in enumerate(self.wms):
            z = w.encode(tok, low)
            if self._st[i] is None:
                h = w.h0[None]
            else:
                hp, zp = self._st[i]
                ap = torch.tensor(self.act_norm(np.asarray(a_prev, np.float32))[None], device=self.dev)
                h = w.step(hp, zp, ap)
            self._st[i] = (h, z); out.append((h, z))
        return out

    @torch.no_grad()
    def imagine(self, action_seqs: np.ndarray) -> dict:
        """Roll N candidate plans (N,K,A) from the current filtered state through
        every member. Returns hazard (M,N,K,nH), value (M,N,K), margin (M,N,K),
        plus ensemble mean/std of min-over-horizon value."""
        a = torch.tensor(self.act_norm(np.asarray(action_seqs, np.float32)), device=self.dev)
        N = a.shape[0]; Hz, V, Mg = [], [], []
        for (h, s), w in zip(self._st, self.wms):
            f = w.imagine(h.expand(N, -1), s.expand(N, -1), a)
            Hz.append(w.heads.hazard(f)); V.append(w.heads.value(f)[..., 0]); Mg.append(w.heads.margin(f)[..., 0])
        Hz, V, Mg = torch.stack(Hz), torch.stack(V), torch.stack(Mg)
        vmin = V.min(-1).values
        return dict(hazard=Hz.cpu().numpy(), value=V.cpu().numpy(), margin=Mg.cpu().numpy(),
                    vmin_mean=vmin.mean(0).cpu().numpy(), vmin_std=vmin.std(0).cpu().numpy() if len(V) > 1 else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--prep", required=True); c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda"); c.add_argument("--img-px", type=int, default=126)
    c.add_argument("--pool", type=int, default=3)
    t = sub.add_parser("train")
    t.add_argument("--prep", required=True); t.add_argument("--feats", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=500); t.add_argument("--members", type=int, default=1)
    t.add_argument("--batch", type=int, default=16); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--K", type=int, default=8); t.add_argument("--K-val", type=int, default=15)
    t.add_argument("--gamma", type=float, default=0.95)
    t.add_argument("--act-noise", type=float, default=0.3, help="std (normalized action units) of perturbed imagination plans")
    t.add_argument("--cam-weight", action="append", default=[], help="e.g. zed=2")
    t.add_argument("--device", default="cuda"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--action-key", default="action", choices=["action", "action_cup", "both"])
    t.add_argument("--log-every", type=int, default=50); t.add_argument("--T-max", type=int, default=160)
    mg = sub.add_parser("merge", help="merge independently trained member ckpts into one ensemble ckpt")
    mg.add_argument("--out", required=True); mg.add_argument("ckpts", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        print(f"[merge] {C.merge_member_ckpts(a.ckpts, a.out)} members -> {a.out}"); return
    if a.cmd == "cache":
        build_cache(a.prep, a.out, a.device, a.img_px, a.pool)
    else:
        train(a.prep, a.feats, a.out, a.steps, a.members, a.batch, a.lr, a.device, a.K, a.K_val, a.gamma,
              a.cam_weight, a.seed, a.log_every, a.T_max, act_noise=a.act_noise, action_key=a.action_key)


if __name__ == "__main__":
    main()
