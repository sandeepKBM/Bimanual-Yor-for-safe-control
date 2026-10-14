"""world_model/v2/dinowm/train.py -- cache DINOv2 patch tokens + train the official-style
DINO-WM predictor with a spill head on gen_spill_data.py shards.

SIMULATION DATA ONLY. No robot, no network (DINOv2 from the local torch-hub cache).

  # 1) frozen-encoder token cache (GPU; idempotent per shard; ~49 KB/frame at 8x8 fp16)
  python -m world_model.v2.dinowm.train cache --data $SHARDS --cache $CACHE --img-px 112 --device cuda
  # 2) train (single GPU)
  python -m world_model.v2.dinowm.train train --data $SHARDS --cache $CACHE --out $RUN/dinowm \
      --steps 30000 --batch 32 --num-hist 3 --rollout-k 4 --label spill --device cuda
Split = shards.split_of(ep_seed) (70/15/15, deterministic per seed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from world_model.v2.dinowm import shards as S  # noqa: E402
from world_model.v2.dinowm.model import DinoWMSpill, FrozenDino, n_params  # noqa: E402
from world_model.v2.model import common as C  # noqa: E402


def cache_tag(img_px, pool, cam):
    return f"{cam}_px{img_px}" + (f"_pool{pool}" if pool else "")


# =========================================================================== cache
def build_cache(data, cache, img_px=112, pool=None, cam="zed", device="cuda", bs=256, limit_shards=None):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    enc = FrozenDino(img_px, pool).to(device)
    tag = cache_tag(img_px, pool, cam); t0 = time.time(); n = 0
    for p in S.shard_paths(data)[:limit_shards]:
        fn = cache / f"{Path(p).stem}.{tag}.npy"
        if fn.exists():
            continue
        with np.load(p) as f:
            im = f[f"img_{cam}"]
        out = np.empty((len(im), enc.n_patches, enc.dim), np.float16)
        for i in range(0, len(im), bs):
            out[i:i + bs] = enc(torch.as_tensor(im[i:i + bs], device=device)).half().cpu().numpy()
        tmp = fn.with_suffix(".tmp.npy"); np.save(tmp, out); os.replace(tmp, fn)
        n += len(im)
    (cache / f"meta.{tag}.json").write_text(json.dumps(dict(img_px=img_px, pool=pool, cam=cam, n_patches=enc.n_patches,
                                                             dim=enc.dim, arch="dinov2_vits14")))
    print(f"[dinowm cache] {n} new frames in {time.time()-t0:.0f}s -> {cache} ({tag})", flush=True)


# =========================================================================== data
class WindowData:
    """Episodes (low-dim, from shards) + memmapped token cache -> random windows."""

    def __init__(self, eps, tok_mm, low_norm, act_norm, action_key, horizons=C.HORIZONS_S):
        self.eps, self.tok = eps, tok_mm
        self.low = [low_norm(C.step_features(e, use_action=False)).astype(np.float32) for e in eps]
        self.act = [act_norm(C.get_action(e, action_key)).astype(np.float32) for e in eps]
        self.tg = [C.episode_targets(e, horizons) for e in eps]
        self.T = np.array([e["T"] for e in eps])

    def tokens(self, i, a, b):
        e = self.eps[i]; sh, st = e["_tok"]
        return np.asarray(self.tok[sh][st + a:st + b], np.float32)

    def batch(self, rng, B, L, device, pos_frac=0.0, pre_s=1.5):
        """pos_frac of the windows END in [onset - pre_s, onset + 3] of a spilling episode
        (spill windows are rare under uniform sampling; the hazard head needs them)."""
        ok = np.flatnonzero(self.T >= L)
        okp = np.array([i for i in ok if 0 <= self.eps[i]["onset"]], int)
        n_pos = int(round(B * pos_frac)) if len(okp) else 0
        idx = np.r_[rng.choice(okp, n_pos) if n_pos else np.zeros(0, int), rng.choice(ok, B - n_pos)].astype(int)
        V, Lo, A, Y, Va, Mg, Sc = [], [], [], [], [], [], []
        for j, i in enumerate(idx):
            if j < n_pos:
                on = self.eps[i]["onset"]; hz = self.eps[i]["hz"]
                end = int(rng.integers(max(L - 1, on - int(pre_s * hz)), min(self.T[i] - 1, on + 3) + 1))
                t0 = int(np.clip(end - L + 1, 0, self.T[i] - L))
            else:
                t0 = int(rng.integers(0, self.T[i] - L + 1))
            s = slice(t0, t0 + L)
            V.append(self.tokens(i, t0, t0 + L)); Lo.append(self.low[i][s]); A.append(self.act[i][s])
            g = self.tg[i]; Y.append(g["y"][s]); Va.append(g["valid"][s]); Mg.append(g["margin"][s]); Sc.append(g["scored"][s])
        f = lambda X, dt=torch.float32: torch.as_tensor(np.stack(X), dtype=dt, device=device)
        return dict(vis=f(V), low=f(Lo), act=f(A), y=f(Y), valid=f(np.array(Va, np.float32)), margin=f(Mg),
                    scored=f(np.array(Sc, np.float32)))


def attach_tokens(eps, cache, tag):
    """ep['_tok'] = (shard key, start frame) into per-shard memmaps."""
    mm = {}
    for e in eps:
        stem, j = e["id"].split("#")
        if stem not in mm:
            mm[stem] = np.load(Path(cache) / f"{stem}.{tag}.npy", mmap_mode="r")
        e["_tok"] = (stem, e["_start"])
    return mm


def load_eps(data, label, kinds, limit_shards=None, cache=None, tag=None):
    ps = S.shard_paths(data)[:limit_shards]
    if cache is not None:
        ps = [p for p in ps if (Path(cache) / f"{Path(p).stem}.{tag}.npy").exists()]
    eps = []
    for p in ps:
        es = S.load_shard(p, False, label=label)
        st = 0
        for e in es:
            e["_start"] = st; st += e["T"]
        eps += [e for e in es if e["meta"]["kind"] in kinds]
    return eps


# =========================================================================== loss
def wm_loss(wm: DinoWMSpill, b, pw, R, w=None):
    w = w or dict(tf=1.0, ro=1.0, haz=1.0, mar=0.5)
    H = wm.H
    z = wm.encode(b["vis"], b["low"], b["act"])                       # (B,L,P,dim), L = H+R
    L = {}
    zp = wm.predict(z[:, :H])                                         # official teacher-forced 1-step
    L["tf"] = wm.latent_loss(zp, z[:, 1:H + 1])
    zi = wm.rollout(z[:, :H], b["act"][:, H:H + R])                   # open-loop frames H..H+R-1
    L["ro"] = wm.latent_loss(zi, z[:, H:H + R]) if R > 1 else zi.sum() * 0
    # spill head on observed windows (targets at window end H-1..L-1) and imagined ones (H..L-1)
    zo = wm.windows(z)                                                # (B,R+1,H,P,dim)
    zim = wm.windows(torch.cat([z[:, :H], zi], 1))[:, 1:]             # (B,R,H,P,dim)
    tgt = lambda k, a: b[k][:, a:]
    fo, fi = wm.head_feat(zo), wm.head_feat(zim)
    ho, mo = wm.head.heads.losses(fo, tgt("y", H - 1), tgt("valid", H - 1), tgt("margin", H - 1), tgt("scored", H - 1), pw)
    hi, mi = wm.head.heads.losses(fi, tgt("y", H), tgt("valid", H), tgt("margin", H), tgt("scored", H), pw)
    L["haz"] = 0.5 * (ho + hi); L["mar"] = 0.5 * (mo + mi)
    tot = sum(w[k] * v for k, v in L.items())
    return tot, {k: float(v.detach()) for k, v in L.items()}


# =========================================================================== train
def train(a):
    dev = a.device; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = cache_tag(a.img_px, a.pool, a.cam)
    meta = json.loads((Path(a.cache) / f"meta.{tag}.json").read_text())
    kinds = tuple(a.kinds.split(","))
    eps = load_eps(a.data, a.label, kinds, a.limit_shards, a.cache, tag)
    mm = attach_tokens(eps, a.cache, tag)
    spl = S.by_split(eps)
    tr, va = spl["train"], spl["val"]
    low_norm = C.Normalizer().fit([C.step_features(e, use_action=False) for e in tr])
    act_norm = C.Normalizer().fit([C.get_action(e, a.action_key) for e in tr])
    Dtr = WindowData(tr, mm, low_norm, act_norm, a.action_key); Dva = WindowData(va, mm, low_norm, act_norm, a.action_key)
    pw = torch.tensor(C.auto_pos_weight([g["y"] for g in Dtr.tg], [g["valid"] for g in Dtr.tg]), device=dev)
    arch = dict(n_patches=meta["n_patches"], d_low=Dtr.low[0].shape[1], d_act=Dtr.act[0].shape[1], num_hist=a.num_hist,
                depth=a.depth, heads=a.heads, mlp_dim=a.mlp_dim)
    C.seed_all(a.seed)
    wm = DinoWMSpill(**arch).to(dev)
    opt = torch.optim.AdamW([{"params": wm.predictor.parameters(), "lr": a.lr},
                             {"params": list(wm.proprio_encoder.parameters()) + list(wm.action_encoder.parameters()), "lr": a.lr},
                             {"params": wm.head.parameters(), "lr": a.lr_head}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    rng = np.random.default_rng(a.seed); vrng = np.random.default_rng(12345)
    L = a.num_hist + a.rollout_k
    print(f"[dinowm] params={n_params(wm)/1e6:.2f}M arch={arch} train_eps={len(tr)} val_eps={len(va)} "
          f"label={a.label} kinds={kinds} window={L} tokens/frame={meta['n_patches']} amp={a.amp}", flush=True)
    vb = Dva.batch(vrng, min(64, 4 * a.batch), L, dev, a.pos_frac)
    with torch.no_grad():                        # reference: "copy the last observed frame" latent MSE
        zc = vb["vis"]; copy_mse = float(((zc[:, a.num_hist:] - zc[:, a.num_hist - 1:-1]) ** 2).mean())
    print(f"[dinowm] val copy-last-frame visual MSE (1-step) = {copy_mse:.4f}", flush=True)
    curve = []; t0 = time.time()
    for it in range(1, a.steps + 1):
        wm.train()
        b = Dtr.batch(rng, a.batch, L, dev, a.pos_frac)
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
            loss, lg = wm_loss(wm, b, pw, a.rollout_k)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(wm.parameters(), 1.0); scaler.step(opt); scaler.update()
        wm.head.heads.update_target()
        if it % a.log_every == 0 or it == 1 or it == a.steps:
            wm.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                _, vl = wm_loss(wm, vb, pw, a.rollout_k)
            curve.append(dict(step=it, train=lg, val=vl, sec=time.time() - t0))
            print(f"[dinowm] step {it} ({time.time()-t0:.0f}s) train " + " ".join(f"{k}={v:.4f}" for k, v in lg.items())
                  + " | val " + " ".join(f"{k}={v:.4f}" for k, v in vl.items()), flush=True)
        if a.save_every and it % a.save_every == 0:
            save(out, wm, arch, a, low_norm, act_norm, meta, curve, tag)
    save(out, wm, arch, a, low_norm, act_norm, meta, curve, tag)
    return out / "model.pt"


def save(out, wm, arch, a, low_norm, act_norm, meta, curve, tag):
    ck = dict(kind="dinowm_official", arch=arch, state=wm.state_dict(), low_norm=low_norm.state(), act_norm=act_norm.state(),
              action_key=a.action_key, label=a.label, kinds=a.kinds, enc=dict(img_px=a.img_px, pool=a.pool, cam=a.cam),
              cache=str(a.cache), cache_tag=tag, data=str(a.data), horizons_s=list(C.HORIZONS_S), token_meta=meta,
              rollout_k=a.rollout_k)
    torch.save(ck, out / "model.pt"); (out / "curves.json").write_text(json.dumps(curve))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cache", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=str(S.DATA)); p.add_argument("--cache", required=True)
        p.add_argument("--img-px", type=int, default=112); p.add_argument("--pool", type=int, default=None)
        p.add_argument("--cam", default="zed"); p.add_argument("--device", default="cuda")
        p.add_argument("--limit-shards", type=int, default=None)
    t = sub.choices["train"]
    t.add_argument("--out", required=True); t.add_argument("--steps", type=int, default=500)
    t.add_argument("--batch", type=int, default=32); t.add_argument("--lr", type=float, default=5e-4)   # official predictor_lr
    t.add_argument("--lr-head", type=float, default=3e-4)
    t.add_argument("--num-hist", type=int, default=3); t.add_argument("--rollout-k", type=int, default=4)
    t.add_argument("--depth", type=int, default=6); t.add_argument("--heads", type=int, default=16)
    t.add_argument("--mlp-dim", type=int, default=2048)
    t.add_argument("--label", default="spill", choices=("spill", "pend"))
    t.add_argument("--kinds", default="beads,particles")
    t.add_argument("--action-key", default="action_cup", choices=("action", "action_cup", "both"))
    t.add_argument("--pos-frac", type=float, default=0.25, help="share of windows sampled around spill onset")
    t.add_argument("--amp", action="store_true"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50); t.add_argument("--save-every", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "cache":
        build_cache(a.data, a.cache, a.img_px, a.pool, a.cam, a.device, limit_shards=a.limit_shards)
    else:
        train(a)


if __name__ == "__main__":
    main()
