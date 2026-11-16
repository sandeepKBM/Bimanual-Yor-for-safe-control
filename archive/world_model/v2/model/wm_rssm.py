"""world_model/v2/model/wm_rssm.py -- OPTION A: DreamerV3 RSSM (vendored
/common/users/ss5772/dreamerv3, imported READ-ONLY) with an OBJECT-WEIGHTED
image reconstruction loss so the 1-3 px balls matter, plus multi-horizon spill
hazard + SPILL-ONLY Fisac value trained in imagination.

What changed vs world_model/train_wm.py (v1):
  * image = channel-concat of the chosen cameras (v2: cup-centred zed crop +
    wrists) resized to --img (64 or 128; the vendored ConvEncoder needs a power of 2);
  * recon loss sum_px w(px) (x^ - x)^2 with w = 1 + alpha*ball_mask(target) +
    beta*|x_t - x_{t-1}|: ball pixels (blue balls, rgba 0.15 0.55 0.9 -- detector
    below, or a dataset-provided mask) and moving pixels dominate the loss
    instead of the static table;
  * CORRECT action alignment: dreamer's obs_step consumes the action that LED to
    obs t, so prev_action[t] = a[t-1] (v1 fed a[t]);
  * no reward/cont heads; spill heads (common.SpillHeads) on posterior features
    AND on K-step imagined priors (so hazard/value are valid in imagination);
    Fisac value on imagined rollouts under dataset + noise-perturbed actions;
  * sized for 50-100k steps (--steps), resumable (latest.pt every --ckpt-every).

  python -m world_model.v2.model.wm_rssm train --prep $PREP/ds --out $RUN/rssm \
      --steps 100000 --img 64 --batch 16 --seq 64 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DREAMERV3_ROOT = "/common/users/ss5772/dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)
import networks  # noqa: E402  (vendored, read-only)

from world_model.v2.model import common as C  # noqa: E402
from world_model.v2.model import data as D  # noqa: E402

BASE_CFG = REPO / "world_model" / "train_wm_config.yaml"     # read-only defaults


def ball_mask(img01: torch.Tensor, blue_ratio=0.65, gb_lo=0.4, gb_hi=0.8, b_min=0.35) -> torch.Tensor:
    """(...,H,W,3k) in [0,1] -> (...,H,W,3k) {0,1}: pixels of the saturated-blue balls
    (scene ball_mat rgba 0.15 0.55 0.9: (b-r)/b ~0.83, g/b ~0.61), dilated 1 px and
    broadcast to the 3 channels of each camera. The desaturated blue-grey floor
    ((b-r)/b ~0.55) and sky haze (~0.57) are rejected by `blue_ratio`. Scene-
    specific: re-tune if contents colours are randomised (v2 data)."""
    shp = img01.shape
    x = img01.reshape(-1, shp[-3], shp[-2], shp[-1] // 3, 3)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    bb = b.clamp_min(1e-3)
    m = ((b - r) / bb > blue_ratio) & (g / bb > gb_lo) & (g / bb < gb_hi) & (b > b_min)
    m = m.float().permute(0, 3, 1, 2)                               # (N,k,H,W)
    m = F.max_pool2d(m, 3, 1, 1).permute(0, 2, 3, 1)[..., None].expand(-1, -1, -1, -1, 3)
    return m.reshape(shp)


class SpillRSSM(nn.Module):
    def __init__(self, cfg, img_shape, d_low, d_act, n_h=len(C.HORIZONS_S)):
        super().__init__()
        shapes = {"image": tuple(img_shape), "proprio": (d_low,)}
        self.cfg = cfg
        self.encoder = networks.MultiEncoder(shapes, **cfg.encoder)
        self.dynamics = networks.RSSM(cfg.dyn_stoch, cfg.dyn_deter, cfg.dyn_hidden, cfg.dyn_rec_depth,
                                      cfg.dyn_discrete, cfg.act, cfg.norm, cfg.dyn_mean_act, cfg.dyn_std_act,
                                      cfg.dyn_min_std, cfg.unimix_ratio, cfg.initial, d_act,
                                      self.encoder.outdim, cfg.device)
        d_f = cfg.dyn_stoch * cfg.dyn_discrete + cfg.dyn_deter if cfg.dyn_discrete else cfg.dyn_stoch + cfg.dyn_deter
        self.decoder = networks.MultiDecoder(d_f, shapes, **cfg.decoder)
        self.heads = C.SpillHeads(d_f, n_h)
        self.d_f = d_f

    def embed(self, image01, low):
        return self.encoder({"image": image01.clone(), "proprio": low})

    def observe(self, image01, low, act, is_first):
        prev_a = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)   # a[t-1] led to obs t
        post, prior = self.dynamics.observe(self.embed(image01, low), prev_a, is_first)
        return post, prior


def load_cfg(device, overrides=None):
    cfg = dict(yaml.safe_load(open(BASE_CFG)))
    cfg["device"] = device
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return types.SimpleNamespace(**cfg)


class SeqBatcher:
    def __init__(self, eps, cams, img, low_norm, act_norm, horizons, device):
        self.eps, self.cams, self.img, self.dev = eps, cams, img, device
        self.low = [low_norm(C.step_features(e, use_action=False)).astype(np.float32) for e in eps]
        self.act = [act_norm(np.asarray(e["action"], np.float32)).astype(np.float32) for e in eps]
        self.tgt = [C.episode_targets(e, horizons) for e in eps]

    def _img(self, e, s, t):
        ims = [torch.as_tensor(e["images"][c][s:t], device=self.dev) for c in self.cams]
        x = torch.cat(ims, -1).float() / 255.0                                  # (L,H,W,3k)
        if x.shape[1] != self.img:
            x = F.interpolate(x.permute(0, 3, 1, 2), size=(self.img, self.img), mode="bilinear",
                              align_corners=False, antialias=True).permute(0, 2, 3, 1)
        return x

    def get(self, idx, L, rng):
        out = {k: [] for k in ("image", "low", "act", "y", "valid", "margin", "scored", "mask", "is_first")}
        for i in idx:
            e = self.eps[i]; T = e["T"]
            s = int(rng.integers(0, max(1, T - L + 1))) if T > L else 0
            t = min(T, s + L); n = t - s
            pad = lambda a: np.concatenate([a, np.zeros((L - len(a),) + a.shape[1:], a.dtype)], 0)
            im = self._img(e, s, t)
            out["image"].append(torch.cat([im, torch.zeros((L - n,) + im.shape[1:], device=self.dev)], 0))
            out["low"].append(pad(self.low[i][s:t])); out["act"].append(pad(self.act[i][s:t]))
            tg = self.tgt[i]
            out["y"].append(pad(tg["y"][s:t])); out["valid"].append(pad(tg["valid"][s:t].astype(np.float32)))
            out["margin"].append(pad(tg["margin"][s:t])); out["scored"].append(pad(tg["scored"][s:t].astype(np.float32)))
            out["mask"].append(pad(np.ones(n, np.float32)))
            f = np.zeros(L, np.float32); f[0] = 1.0; out["is_first"].append(f)
        b = {k: (torch.stack(v) if k == "image" else torch.tensor(np.stack(v), device=self.dev)) for k, v in out.items()}
        return b


def _shift(x, k):
    return x if k == 0 else torch.cat([x[:, k:], torch.zeros_like(x[:, :k])], 1)


def rssm_loss(wm: SpillRSSM, b, pos_weight, alpha=20.0, beta=5.0, K=8, K_val=15, gamma=0.95, act_noise=0.3,
              n_val_starts=256, w=None):
    w = w or dict(kl=1.0, img=1.0, low=1.0, haz=1.0, mar=0.5, val=1.0)
    cfg = wm.cfg
    post, prior = wm.observe(b["image"], b["low"], b["act"], b["is_first"])
    kl, kl_value, dyn, rep = wm.dynamics.kl_loss(post, prior, cfg.kl_free, cfg.dyn_scale, cfg.rep_scale)
    feat = wm.dynamics.get_feat(post)
    m = b["mask"]; msum = m.sum().clamp_min(1)
    dists = wm.decoder(feat)
    x = b["image"]; xh = dists["image"].mode()
    with torch.no_grad():
        wmap = 1.0 + alpha * ball_mask(x)
        if beta > 0:
            dx = (x - torch.cat([x[:, :1], x[:, :-1]], 1)).abs()
            wmap = wmap + beta * dx * (b["is_first"][..., None, None, None] == 0).float()
    L = {}
    L["kl"] = (kl * m).sum() / msum
    L["img"] = ((((xh - x) ** 2) * wmap).sum((-1, -2, -3)) * m).sum() / msum
    L["low"] = ((-dists["proprio"].log_prob(b["low"])) * m).sum() / msum
    haz, mar = wm.heads.losses(feat, b["y"], b["valid"] * m, b["margin"], b["scored"] * m, pos_weight)
    # K-step imagined priors from every posterior state -> heads (dataset actions)
    B, T = m.shape
    flat = {k: v.reshape((B * T,) + v.shape[2:]) for k, v in post.items()}
    acts = torch.stack([_shift(b["act"], k) for k in range(K)], 2).reshape(B * T, K, -1)
    imag = wm.dynamics.imagine_with_action(acts, {k: v.detach() for k, v in flat.items()})
    fi = wm.dynamics.get_feat(imag).view(B, T, K, -1)
    oh, om = 0.0, 0.0
    for k in range(1, K + 1):
        mk = _shift(m, k) * m
        h_k, m_k = wm.heads.losses(fi[:, :, k - 1], _shift(b["y"], k), _shift(b["valid"], k) * mk,
                                   _shift(b["margin"], k), _shift(b["scored"], k) * mk, pos_weight)
        oh, om = oh + h_k, om + m_k
    L["haz"] = haz + oh / K; L["mar"] = mar + om / K
    # Fisac value in imagination
    vs = torch.nonzero(m.reshape(-1) > 0)[:, 0]
    pick = vs[torch.randint(len(vs), (min(n_val_starts, len(vs)),), device=vs.device)]
    va = torch.stack([_shift(b["act"], k) for k in range(K_val)], 2).reshape(B * T, K_val, -1)[pick]
    va = torch.cat([va, va + act_noise * torch.randn_like(va)], 0)
    st = {k: v[pick].detach().repeat((2,) + (1,) * (v.dim() - 1)) for k, v in flat.items()}
    with torch.no_grad():
        fv = wm.dynamics.get_feat(wm.dynamics.imagine_with_action(va, st))
        f0 = wm.dynamics.get_feat(st)
    L["val"] = wm.heads.value_loss(torch.cat([f0[:, None], fv], 1), gamma)
    total = sum(w[k] * v for k, v in L.items())
    return total, {k: float(v.detach()) for k, v in L.items()}


def train(prep_dir, out, steps=500, batch=16, seq=64, img=64, lr=1e-4, device="cuda", cams=None, alpha=20.0,
          beta=5.0, K=8, K_val=15, gamma=0.95, seed=0, log_every=50, ckpt_every=1000, horizons=C.HORIZONS_S,
          cfg_overrides=None, action_key="action"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tr = D.load_prepared(prep_dir, "train"); va = D.load_prepared(prep_dir, "val")
    for e in tr + va:                      # WM action space: joint refs | commanded cup accel | both
        e["action"] = C.get_action(e, action_key)
    cams = cams or sorted(tr[0]["images"])
    low_norm = C.Normalizer().fit([C.step_features(e, use_action=False) for e in tr])
    act_norm = C.Normalizer().fit([np.asarray(e["action"], np.float32) for e in tr])
    Btr = SeqBatcher(tr, cams, img, low_norm, act_norm, horizons, device)
    Bva = SeqBatcher(va, cams, img, low_norm, act_norm, horizons, device)
    pw = torch.tensor(C.auto_pos_weight([t["y"] for t in Btr.tgt], [t["valid"] for t in Btr.tgt]), device=device)
    cfg = load_cfg(device, cfg_overrides)
    C.seed_all(seed)
    arch = dict(img_shape=(img, img, 3 * len(cams)), d_low=Btr.low[0].shape[1], d_act=Btr.act[0].shape[1], n_h=len(horizons))
    wm = SpillRSSM(cfg, **arch).to(device)
    opt = torch.optim.Adam([p for p in wm.parameters() if p.requires_grad], lr=lr, eps=1e-8)
    step0 = 0; curve = []
    if (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=device, weights_only=False)
        wm.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]; curve = ck.get("curve", [])
        print(f"[rssm] resumed at step {step0}", flush=True)
    print(f"[rssm] params={sum(p.numel() for p in wm.parameters())/1e6:.1f}M cams={cams} img={img} "
          f"train_eps={len(tr)} val_eps={len(va)}", flush=True)
    rng = np.random.default_rng(seed + step0); t0 = time.time()

    def save(name, step):
        torch.save(dict(kind="rssm_wm", state=wm.state_dict(), opt=opt.state_dict(), step=step, curve=curve,
                        arch=arch, cams=cams, img=img, cfg=vars(cfg), horizons_s=list(horizons),
                        low_norm=low_norm.state(), act_norm=act_norm.state(), action_key=action_key, prep_dir=str(prep_dir)), out / name)

    for it in range(step0 + 1, steps + 1):
        b = Btr.get(rng.integers(0, len(tr), batch), seq, rng)
        loss, lg = rssm_loss(wm, b, pw, alpha, beta, K, K_val, gamma)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(wm.parameters(), 1000.0); opt.step()
        wm.heads.update_target()
        if it % log_every == 0 or it == step0 + 1:
            with torch.no_grad():
                vb = Bva.get(np.arange(min(len(va), batch)), seq, np.random.default_rng(0))
                _, vlg = rssm_loss(wm, vb, pw, alpha, beta, K, K_val, gamma)
            curve.append(dict(step=it, train=lg, val=vlg, sec=time.time() - t0))
            print(f"[rssm] step {it} ({time.time()-t0:.0f}s) train " + " ".join(f"{k}={v:.3f}" for k, v in lg.items())
                  + " | val " + " ".join(f"{k}={v:.3f}" for k, v in vlg.items()), flush=True)
        if it % ckpt_every == 0:
            save("latest.pt", it)
    save("model.pt", steps); save("latest.pt", steps)
    (out / "curves.json").write_text(json.dumps(curve))
    return out / "model.pt", curve


class RSSMPredictor:
    """ckpt = one path or a comma-separated list (ensemble of independently trained RSSMs)."""
    probabilistic = True

    def __init__(self, ckpt, device="cpu"):
        paths = [c for c in str(ckpt).split(",") if c]
        self.wms = []
        for pth in paths:
            ck = torch.load(pth, map_location="cpu", weights_only=False)
            cfg = types.SimpleNamespace(**{**ck["cfg"], "device": device})
            w = SpillRSSM(cfg, **ck["arch"]); w.load_state_dict(ck["state"]); self.wms.append(w.to(device).eval())
        self.ck, self.dev = ck, device
        self.wm = self.wms[0]
        self.horizons_s = tuple(ck["horizons_s"]); self.cams = ck["cams"]; self.img = ck["img"]
        self.low_norm = C.Normalizer(ck["low_norm"]["mean"], ck["low_norm"]["std"])
        self.act_norm = C.Normalizer(ck["act_norm"]["mean"], ck["act_norm"]["std"])
        self.name = f"rssm-WM(x{len(self.wms)})"; self._st = None; self.last_std = None

    def _img(self, ims):
        x = torch.cat([torch.as_tensor(ims[c], device=self.dev) for c in self.cams], -1).float() / 255.0
        if x.shape[-2] != self.img:
            x = F.interpolate(x.permute(0, 3, 1, 2), size=(self.img, self.img), mode="bilinear",
                              align_corners=False, antialias=True).permute(0, 2, 3, 1)
        return x

    @torch.no_grad()
    def predict(self, ep):
        x = self._img(ep["images"])[None]
        low = torch.tensor(self.low_norm(C.step_features(ep, use_action=False)), device=self.dev)[None]
        act = torch.tensor(self.act_norm(C.get_action(ep, self.ck.get("action_key", "action"))), device=self.dev)[None]
        isf = torch.zeros(1, len(act[0]), device=self.dev); isf[:, 0] = 1
        P = []
        for w in self.wms:
            post, _ = w.observe(x, low, act, isf)
            P.append(w.heads.hazard(w.dynamics.get_feat(post))[0])
        P = torch.stack(P)
        self.last_std = P.std(0).cpu().numpy() if len(P) > 1 else None
        return P.mean(0).cpu().numpy()

    # ---- streaming + imagination (shield API) -------------------------------
    def reset(self):
        self._st = [None] * len(self.wms)

    @torch.no_grad()
    def observe_step(self, images_t: dict, low_t: np.ndarray, a_prev: np.ndarray | None):
        x = self._img({c: images_t[c][None] for c in self.cams})
        low = torch.tensor(self.low_norm(low_t[None].astype(np.float32)), device=self.dev)
        for i, w in enumerate(self.wms):
            emb = w.embed(x, low)
            first = self._st[i] is None
            a = torch.zeros(1, self.ck["arch"]["d_act"], device=self.dev) if first else \
                torch.tensor(self.act_norm(np.asarray(a_prev, np.float32))[None], device=self.dev)
            self._st[i], _ = w.dynamics.obs_step(self._st[i], a, emb,
                                                 torch.tensor([1.0 if first else 0.0], device=self.dev), sample=False)
        return self._st

    @torch.no_grad()
    def imagine(self, action_seqs: np.ndarray) -> dict:
        a = torch.tensor(self.act_norm(np.asarray(action_seqs, np.float32)), device=self.dev)
        N = a.shape[0]; Hz, V, Mg = [], [], []
        for st0, w in zip(self._st, self.wms):
            st = {k: v.expand((N,) + v.shape[1:]).contiguous() for k, v in st0.items()}
            f = w.dynamics.get_feat(w.dynamics.imagine_with_action(a, st))
            Hz.append(w.heads.hazard(f)); V.append(w.heads.value(f)[..., 0]); Mg.append(w.heads.margin(f)[..., 0])
        Hz, V, Mg = torch.stack(Hz), torch.stack(V), torch.stack(Mg)
        vmin = V.min(-1).values
        return dict(hazard=Hz.cpu().numpy(), value=V.cpu().numpy(), margin=Mg.cpu().numpy(),
                    vmin_mean=vmin.mean(0).cpu().numpy(), vmin_std=vmin.std(0).cpu().numpy() if len(V) > 1 else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--prep", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=500); t.add_argument("--batch", type=int, default=16)
    t.add_argument("--seq", type=int, default=64); t.add_argument("--img", type=int, default=64, choices=[64, 128])
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--device", default="cuda")
    t.add_argument("--action-key", default="action", choices=["action", "action_cup", "both"])
    t.add_argument("--cams", default=None, help="comma list; default all prepared cams")
    t.add_argument("--alpha", type=float, default=20.0, help="ball-pixel loss weight")
    t.add_argument("--beta", type=float, default=5.0, help="motion-pixel loss weight")
    t.add_argument("--K", type=int, default=8); t.add_argument("--K-val", type=int, default=15)
    t.add_argument("--gamma", type=float, default=0.95); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50); t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--small", action="store_true", help="half-width RSSM (deter 256, cnn_depth 16) for quick runs")
    a = ap.parse_args(argv)
    ov = None
    if a.small:
        base = dict(yaml.safe_load(open(BASE_CFG)))
        enc, dec = dict(base["encoder"]), dict(base["decoder"])
        enc.update(cnn_depth=16, mlp_units=256, mlp_layers=2); dec.update(cnn_depth=16, mlp_units=256, mlp_layers=2)
        ov = dict(dyn_deter=256, dyn_hidden=256, units=256, encoder=enc, decoder=dec)
    train(a.prep, a.out, a.steps, a.batch, a.seq, a.img, a.lr, a.device, a.cams.split(",") if a.cams else None,
          a.alpha, a.beta, a.K, a.K_val, a.gamma, a.seed, a.log_every, a.ckpt_every, cfg_overrides=ov,
          action_key=a.action_key)


if __name__ == "__main__":
    main()
