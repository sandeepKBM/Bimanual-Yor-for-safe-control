"""world_model/v2/model/common.py -- shared pieces for v2 spill models.

  episode_targets   per-step multi-horizon labels y_H, valid mask (pre-onset
                    only), SPILL-ONLY safety margin l(t) (+1 before onset, -1 from
                    onset on; collision / cube-drop are NEVER mixed in).
  cup_kin_features  causal cup-kinematics features (FK cup pose -> axis, v, a,
                    specific force in the cup swing frame, tilt, effective tilt),
                    computed at the native rate when `kin_hi` exists and
                    window-averaged down to the step rate.
  MonotoneHazard    head whose P(spill within H_k) is monotone in H by
                    construction: Lambda_k = sum_{j<=k} softplus(r_j),
                    p_k = 1 - exp(-Lambda_k); BCE computed stably in Lambda.
  fisac_backup      discounted safety Bellman (Fisac et al. 2019) over an
                    imagined horizon: V_k = (1-g) l_k + g min(l_k, V_{k+1}).
  Normalizer, pad_batch, seed_all.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.v2.eval import metrics as M

HORIZONS_S = (0.25, 0.5, 1.0)
G = np.array([0.0, 0.0, -9.81])


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def episode_targets(ep: dict, horizons_s=HORIZONS_S):
    T = M.episode_len(ep); on = M.episode_onset(ep); t0 = int(ep.get("t_start", 0)); hz = float(ep["hz"])
    ys = []
    for h in horizons_s:
        y, valid = M.labels(on, T, t0, M.horizon_steps(h, hz)); ys.append(y)
    y = np.stack(ys, 1).astype(np.float32)
    post = np.zeros(T, bool)
    if on >= 0:
        post[on:] = True
    margin = np.where(post, -1.0, 1.0).astype(np.float32)
    scored = np.arange(T) >= t0
    return dict(y=y, valid=valid, margin=margin, post=post, scored=scored)


# --------------------------------------------------------------------------
# cup kinematics features (causal)
# --------------------------------------------------------------------------
def _quat_axis(q):
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def _swing_RT_apply(n, v):
    """R^T v with R the minimal rotation taking z to n (slosh_models.swing_frames)."""
    c = np.clip(n[..., 2], -0.999, None)
    vx_, vy_ = -n[..., 1], n[..., 0]
    K = np.zeros(n.shape[:-1] + (3, 3))
    K[..., 0, 2] = vy_; K[..., 1, 2] = -vx_; K[..., 2, 0] = -vy_; K[..., 2, 1] = vx_
    R = np.eye(3) + K + (K @ K) / (1 + c)[..., None, None]
    return np.einsum("...ji,...j->...i", R, v)


def _kin_raw(pos, quat, dt):
    pos = np.asarray(pos, float); n = _quat_axis(quat)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    bd = lambda x: np.concatenate([np.zeros_like(x[:1]), np.diff(x, axis=0) / dt], 0)
    v = bd(pos); v[:1] = 0; a = bd(v); a[:2] = 0; ndot = bd(n); ndot[:1] = 0
    f = a - G
    fn = np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9
    fc = _swing_RT_apply(n, f / fn)
    th0 = np.arccos(np.clip(n[:, 2], -1, 1))[:, None]
    theff = np.arccos(np.clip(np.sum(n * f, -1, keepdims=True) / fn, -1, 1))
    return np.concatenate([n, v, a / 9.81, fc, th0, theff, np.linalg.norm(ndot, axis=-1, keepdims=True),
                           np.linalg.norm(a, axis=-1, keepdims=True) / 9.81], 1)


CUP_KIN_DIM = 16


def _kin_from_logged(ep):
    """v2 data logs FK cup vel/acc (acc = mean over the preceding 200 Hz interval):
    use them instead of 15 Hz finite differences."""
    n = _quat_axis(ep["cup_quat"]); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    v, w = ep["cup_vel"][:, :3].astype(float), ep["cup_vel"][:, 3:6].astype(float)
    a = ep["cup_acc"][:, :3].astype(float)
    f = a - G; fn = np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9
    fc = _swing_RT_apply(n, f / fn)
    th0 = np.arccos(np.clip(n[:, 2], -1, 1))[:, None]
    theff = np.arccos(np.clip(np.sum(n * f, -1, keepdims=True) / fn, -1, 1))
    return np.concatenate([n, v, a / 9.81, fc, th0, theff, np.linalg.norm(w, axis=-1, keepdims=True),
                           np.linalg.norm(a, axis=-1, keepdims=True) / 9.81], 1).astype(np.float32)


def get_action(ep: dict, kind: str = "action") -> np.ndarray:
    """'action' (joint refs) | 'action_cup' (commanded cup accel, v2) | 'both'."""
    if kind == "both":
        return np.concatenate([np.asarray(ep["action"], np.float32), np.asarray(ep["action_cup"], np.float32)], 1)
    return np.asarray(ep[kind], np.float32)


def cup_kin_features(ep: dict) -> np.ndarray | None:
    if ep.get("cup_vel") is not None and ep.get("cup_acc") is not None and ep.get("cup_quat") is not None:
        return _kin_from_logged(ep)
    hi = ep.get("kin_hi")
    if hi is not None:
        raw = _kin_raw(hi["pos"], hi["quat"], 1.0 / float(hi["hz"]))
        idx = np.asarray(hi["idx"]); out = np.empty((len(idx), raw.shape[1]))
        prev = -1
        for k, i in enumerate(idx):            # causal window average (prev, i]
            out[k] = raw[prev + 1:i + 1].mean(0) if i > prev else raw[i]; prev = i
        out[:, :3] = raw[idx, :3]; out[:, 12] = raw[idx, 12]   # axis / tilt: instantaneous
        return out.astype(np.float32)
    if ep.get("cup_pos") is not None:
        return _kin_raw(ep["cup_pos"], ep["cup_quat"], 1.0 / float(ep["hz"])).astype(np.float32)
    return None


def step_features(ep: dict, use_action=True, use_cup_kin=True) -> np.ndarray:
    parts = [np.asarray(ep["proprio"], np.float32)]
    if use_cup_kin:
        k = cup_kin_features(ep)
        if k is not None:
            parts.append(k)
    if use_action:
        parts.append(get_action(ep, use_action if isinstance(use_action, str) else "action"))
    return np.concatenate(parts, 1)


class Normalizer:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, arrays):
        x = np.concatenate(arrays, 0).astype(np.float64)
        self.mean = x.mean(0).astype(np.float32); self.std = (x.std(0) + 1e-3).astype(np.float32)
        return self

    def __call__(self, x):
        return (x - self.mean) / self.std

    def state(self):
        return dict(mean=self.mean, std=self.std)


def pad_batch(arrs, T=None, value=0.0):
    T = T or max(len(a) for a in arrs)
    out = np.full((len(arrs), T) + arrs[0].shape[1:], value, dtype=arrs[0].dtype)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a[:T]
    return out


# --------------------------------------------------------------------------
# heads / losses
# --------------------------------------------------------------------------
class MonotoneHazard(nn.Module):
    def __init__(self, d_in, n_h=len(HORIZONS_S), hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, n_h))
        nn.init.constant_(self.net[-1].bias, -4.0)

    def cumhaz(self, x):
        return torch.cumsum(F.softplus(self.net(x)), -1)

    def forward(self, x):
        return -torch.expm1(-self.cumhaz(x))           # P(spill within H_k)

    @staticmethod
    def bce(lam, y, pos_weight=1.0):
        # log p = log(1-exp(-lam)), log(1-p) = -lam
        logp = torch.log(-torch.expm1(-lam.clamp_min(1e-6)))
        return -(pos_weight * y * logp - (1 - y) * lam)


def fisac_backup(margin, v_boot, gamma: float):
    """margin (B,K) l(s_k), v_boot (B,) V(s_K) -> targets (B,K) (no grad)."""
    B, K = margin.shape
    tgt = torch.empty_like(margin); nxt = v_boot
    for k in range(K - 1, -1, -1):
        l = margin[:, k]
        nxt = (1 - gamma) * l + gamma * torch.minimum(l, nxt)
        tgt[:, k] = nxt
    return tgt


def auto_pos_weight(ys, valids, cap=20.0):
    y = np.concatenate([a[v] for a, v in zip(ys, valids)], 0)
    p = y.mean(0).clip(1e-4, 1)
    return np.minimum((1 - p) / p, cap).astype(np.float32)


# --------------------------------------------------------------------------
# shared WM heads (hazard + spill-only margin + Fisac value w/ target net)
# --------------------------------------------------------------------------
def mlp(d_in, d_out, hidden=256, layers=2):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]; d = hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


class SpillHeads(nn.Module):
    """On a WM feature f: P(spill within H) (monotone), spill-only margin l(f) in
    [-1,1] (+1 = cup still holds its contents), and the Fisac discounted safety
    value V(f) in [-1,1] (V<0 = a spill is predicted along the imagined future).
    Trained on posterior AND imagined features so they are valid in imagination."""

    def __init__(self, d_feat, n_h=len(HORIZONS_S), hidden=256):
        super().__init__()
        self.hazard = MonotoneHazard(d_feat, n_h, hidden)
        self.margin = nn.Sequential(mlp(d_feat, 1, hidden), nn.Tanh())
        self.value = nn.Sequential(mlp(d_feat, 1, hidden), nn.Tanh())
        self.value_targ = nn.Sequential(mlp(d_feat, 1, hidden), nn.Tanh())
        self.value_targ.load_state_dict(self.value.state_dict())
        for p in self.value_targ.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, tau=0.02):
        for p, q in zip(self.value.parameters(), self.value_targ.parameters()):
            q.mul_(1 - tau).add_(tau * p)

    def losses(self, feat, y, valid, margin_t, scored, pos_weight):
        """feat (...,D); y (...,nH); valid/scored (...) float masks; margin_t (...)."""
        lam = self.hazard.cumhaz(feat)
        n_h = y.shape[-1]
        l_haz = (MonotoneHazard.bce(lam, y, pos_weight) * valid[..., None]).sum() / valid.sum().clamp_min(1) / n_h
        l_mar = (((self.margin(feat)[..., 0] - margin_t) ** 2) * scored).sum() / scored.sum().clamp_min(1)
        return l_haz, l_mar

    def value_loss(self, imag_feat, gamma):
        """imag_feat (B,K,D) imagined under GIVEN actions -> Fisac backup target."""
        with torch.no_grad():
            l = self.margin(imag_feat)[..., 0]
            boot = self.value_targ(imag_feat[:, -1])[..., 0]
            tgt = fisac_backup(l, boot, gamma)
        v = self.value(imag_feat.detach())[..., 0]
        return ((v - tgt) ** 2).mean()


def merge_member_ckpts(paths, out):
    """Ensemble = N independent single-GPU jobs (different --seed) -> one ckpt.
    Members must share arch + normalizers (deterministic given the same prep dir)."""
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    base = dict(cks[0])
    for c in cks[1:]:
        assert c.get("arch", c.get("d_in")) == base.get("arch", base.get("d_in")), "arch mismatch"
        for k in ("norm", "low_norm", "act_norm"):
            if k in base:
                for kk in base[k]:
                    assert np.allclose(base[k][kk], c[k][kk], atol=1e-4), f"{k} mismatch -- different prep dirs?"
    base["members"] = [m for c in cks for m in c["members"]]
    base["merged_from"] = [str(p) for p in paths]
    torch.save(base, out)
    return len(base["members"])
