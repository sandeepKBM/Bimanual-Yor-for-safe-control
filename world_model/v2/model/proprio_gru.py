"""world_model/v2/model/proprio_gru.py -- proprio-history GRU spill-hazard BASELINE.

The bar any image world model must beat: a causal GRU over robot-knowable
signals only -- proprio (q, qd, grippers), CUP KINEMATICS FROM FK (axis, v, a,
specific force in the cup frame, tilt, effective tilt; see common.cup_kin_features)
and the commanded action -- predicting P(spill within H) for H in {0.25,0.5,1.0} s
through a monotone hazard head. No images, no GT object state. Trains in minutes.

  train : python -m world_model.v2.model.proprio_gru train --prep $PREP/ds --out $RUN/gru \
              --steps 3000 --members 5 --device cuda
  eval  : python -m world_model.v2.eval.run_eval --model gru:$RUN/gru/model.pt --prep $PREP/ds

Ensemble: --members M trains M independently-seeded members on bootstrap
resamples of the TRAIN episodes; predict() returns the member mean, `.last_std`
holds the epistemic spread.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.model import common as C  # noqa: E402
from world_model.v2.model import data as D  # noqa: E402


class HazardGRU(nn.Module):
    def __init__(self, d_in, hidden=128, n_h=len(C.HORIZONS_S)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = C.MonotoneHazard(hidden, n_h)

    def forward(self, x, h=None):
        z, h = self.gru(self.inp(x), h)
        return self.head.cumhaz(z), h


def _prep_eps(eps, norm, use_action, use_cup_kin, horizons):
    X, Y, V = [], [], []
    for e in eps:
        X.append(norm(C.step_features(e, use_action, use_cup_kin)).astype(np.float32))
        t = C.episode_targets(e, horizons); Y.append(t["y"]); V.append(t["valid"])
    return X, Y, V


def train(prep_dir, out, steps=3000, members=1, batch=32, lr=1e-3, hidden=128, device="cuda",
          use_action=True, use_cup_kin=True, horizons=C.HORIZONS_S, seed=0, log_every=100):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tr = D.load_prepared(prep_dir, "train", with_images=False)
    va = D.load_prepared(prep_dir, "val", with_images=False)
    norm = C.Normalizer().fit([C.step_features(e, use_action, use_cup_kin) for e in tr])
    Xtr, Ytr, Vtr = _prep_eps(tr, norm, use_action, use_cup_kin, horizons)
    Xva, Yva, Vva = _prep_eps(va, norm, use_action, use_cup_kin, horizons)
    pw = torch.tensor(C.auto_pos_weight(Ytr, Vtr), device=device)
    d_in = Xtr[0].shape[1]
    tens = lambda A: torch.tensor(C.pad_batch(A), device=device)
    Xv, Yv, Vv = tens(Xva), tens(Yva), tens([v.astype(np.float32) for v in Vva])
    states, curves = [], []
    for m in range(members):
        C.seed_all(seed + m)
        rng = np.random.default_rng(seed + m)
        pool = rng.integers(0, len(Xtr), len(Xtr)) if members > 1 else np.arange(len(Xtr))   # bootstrap
        net = HazardGRU(d_in, hidden, len(horizons)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        curve = []; t0 = time.time()
        for it in range(1, steps + 1):
            b = rng.choice(pool, min(batch, len(pool)))
            x, y = tens([Xtr[i] for i in b]), tens([Ytr[i] for i in b])
            v = tens([Vtr[i].astype(np.float32) for i in b])
            lam, _ = net(x)
            loss = (C.MonotoneHazard.bce(lam, y, pw) * v[..., None]).sum() / v.sum().clamp_min(1) / len(horizons)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
            if it % log_every == 0 or it == 1:
                with torch.no_grad():
                    lv, _ = net(Xv)
                    vl = (C.MonotoneHazard.bce(lv, Yv, pw) * Vv[..., None]).sum() / Vv.sum().clamp_min(1) / len(horizons)
                curve.append(dict(step=it, train=float(loss.detach()), val=float(vl), sec=time.time() - t0))
                print(f"[gru m{m}] step {it} train {float(loss.detach()):.4f} val {float(vl):.4f} ({time.time()-t0:.0f}s)", flush=True)
        states.append({k: v.cpu() for k, v in net.state_dict().items()}); curves.append(curve)
    ck = dict(kind="proprio_gru", d_in=d_in, hidden=hidden, horizons_s=list(horizons), use_action=use_action,
              use_cup_kin=use_cup_kin, norm=norm.state(), members=states, prep_dir=str(prep_dir))
    torch.save(ck, out / "model.pt")
    (out / "curves.json").write_text(json.dumps(curves))
    return out / "model.pt", curves


class GRUPredictor:
    probabilistic = True

    def __init__(self, ckpt, device="cpu"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.ck = ck; self.device = device
        self.horizons_s = tuple(ck["horizons_s"])
        self.norm = C.Normalizer(ck["norm"]["mean"], ck["norm"]["std"])
        self.nets = []
        for sd in ck["members"]:
            n = HazardGRU(ck["d_in"], ck["hidden"], len(self.horizons_s)); n.load_state_dict(sd)
            self.nets.append(n.to(device).eval())
        self.name = f"proprio-GRU(x{len(self.nets)})"
        self.last_std = None; self._h = None

    @torch.no_grad()
    def predict(self, ep):
        x = torch.tensor(self.norm(C.step_features(ep, self.ck["use_action"], self.ck["use_cup_kin"])),
                         device=self.device)[None]
        P = torch.stack([-torch.expm1(-n(x)[0][0]) for n in self.nets])
        self.last_std = P.std(0).cpu().numpy() if len(self.nets) > 1 else None
        return P.mean(0).cpu().numpy()

    # ---- streaming (one call per 15 Hz tick) --------------------------------
    def reset(self):
        self._h = [None] * len(self.nets)

    @torch.no_grad()
    def step(self, feat_row: np.ndarray) -> np.ndarray:
        x = torch.tensor(self.norm(feat_row[None, None].astype(np.float32)), device=self.device)
        ps = []
        for i, n in enumerate(self.nets):
            lam, self._h[i] = n(x, self._h[i]); ps.append(-torch.expm1(-lam[0, 0]))
        return torch.stack(ps).mean(0).cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--prep", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000); p.add_argument("--members", type=int, default=1)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128); p.add_argument("--device", default="cuda")
    p.add_argument("--no-action", action="store_true")
    p.add_argument("--action-key", default="action", choices=["action", "action_cup", "both"]); p.add_argument("--no-cup-kin", action="store_true")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--log-every", type=int, default=100)
    mg = sub.add_parser("merge", help="merge independently trained member ckpts into one ensemble ckpt")
    mg.add_argument("--out", required=True); mg.add_argument("ckpts", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        print(f"[merge] {C.merge_member_ckpts(a.ckpts, a.out)} members -> {a.out}"); return
    train(a.prep, a.out, a.steps, a.members, a.batch, a.lr, a.hidden, a.device,
          (a.action_key if not a.no_action else False), not a.no_cup_kin, seed=a.seed, log_every=a.log_every)


if __name__ == "__main__":
    main()
