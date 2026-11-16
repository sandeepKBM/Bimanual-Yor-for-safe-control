"""wm_adapter.py -- one interface for every latent WM the reachability learner and the
FL-MPC value rows consume:

    class LatentWM(Protocol):
        def encode(self, obs_hist, act_hist) -> z0            # filtered posterior latent (torch, (B,dz))
        def imagine(self, z0, actions) -> z                   # actions (B,K,da) -> latents (B,K,dz), differentiable
        def spill_margin(self, z) -> l                        # (...,) in [-1,1], >0 = contents safe
        action_space: str                                     # 'cup_accel_angvel' (a_lin(3), w(3)) | 'action_cup' | ...

Adapters:
  KinWMAdapter   -- kin_wm.KinWM (ours, Stage 1 here)
  DinoWMAdapter  -- world_model.v2.model.wm_dino.DinoLatentWM (tokens + low-dim; action
                    = the WM's own action space, e.g. 'action_cup' = cup lin accel (3) +
                    ang accel (3), world, at 15 Hz). Uses the member's margin head.
  dinowm_api()   -- world_model.v2.dinowm.api (parallel DINO-WM agent) if it exists:
                    expects encode / imagine / spill_margin; wrapped as-is.

AUDIT of wm_dino's value head (world_model/v2/model/common.SpillHeads.value_loss):
  target = fisac_backup(l(imagined feats under GIVEN actions), V_targ(last)) i.e.
  V(f_k) = (1-g) l_k + g min(l_k, V(f_{k+1})) along DATASET (+noise) action sequences.
  That is a proper discounted-safety Bellman BACKUP (not the F2 sign(l) classifier),
  but it is POLICY EVALUATION of the data-collection policy: there is no max over
  actions, so V answers "does the logged/behaviour motion spill?", not "does a
  recovery exist?". It is also action-unconditioned V(f) trained on a MIXTURE of
  action sequences (dataset + noisy), so it averages over behaviours. As a
  reachability value (LatentSafe) it is the wrong fixed point; reach_value.train
  (max over the fallback control set + greedy K-step rollout bootstrap) is the fix
  and runs unchanged on any adapter below via LatentDyn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class KinWMAdapter:
    action_space = "cup_accel_angvel"

    def __init__(self, wm):
        self.wm = wm
        self.dh = wm.d_h

    def encode(self, obs_hist, act_hist):
        H, on = self.wm.filter(obs_hist, act_hist)
        # latent after the last observation: one more step with the last action is NOT
        # taken -- z0 = [h_T-1, o_T-1] is the state at the last observed tick.
        import torch
        return torch.cat([H[:, -1], on[:, -1]], -1)

    def imagine(self, z0, actions):
        import torch
        h, o = z0[:, :self.dh], z0[:, self.dh:]
        out = []
        for k in range(actions.shape[1]):
            h, o = self.wm.step(h, o, actions[:, k])
            out.append(torch.cat([h, o], -1))
        return torch.stack(out, 1)

    def spill_margin(self, z):
        return self.wm.margin(z[..., :self.dh], z[..., self.dh:])


class DinoWMAdapter:
    """wm_dino.DinoLatentWM member -> LatentWM. z = [h, z_emb] (the WM's own feature f)."""
    action_space = "wm_dino"

    def __init__(self, member, act_norm=None):
        self.m = member
        self.act_norm = act_norm

    def encode(self, tok, low, act):
        s = self.m.encode(tok, low)
        h, _ = self.m.filter(s, act)
        import torch
        return torch.cat([h[:, -1], s[:, -1]], -1)

    def imagine(self, z0, actions):
        d_h = self.m.d_h
        a = actions if self.act_norm is None else self.act_norm(actions)
        return self.m.imagine(z0[:, :d_h], z0[:, d_h:], a)

    def spill_margin(self, z):
        return self.m.heads.margin(z)[..., 0]


def dinowm_api():
    """Return the parallel DINO-WM agent's API module if present, else None."""
    p = Path(__file__).resolve().parents[1] / "dinowm" / "api.py"
    if not p.exists():
        return None
    import importlib
    mod = importlib.import_module("world_model.v2.dinowm.api")
    if not hasattr(mod, "SpillWM"):
        raise AttributeError("dinowm.api lacks SpillWM -- interface changed")
    for fn in ("encode", "imagine", "spill_margin"):
        if not hasattr(mod.SpillWM, fn):
            raise AttributeError(f"dinowm.api.SpillWM lacks {fn}() -- interface changed")
    return mod


class DinoWMApiAdapter:
    """world_model.v2.dinowm.api.SpillWM -> LatentWM. Its latent is a window
    z (B, H, P=64, D=404) of DINO patch tokens + proprio/action embeddings; imagine()
    returns history + K frames (sliding window). For reach_value we step ONE action at
    a time and keep the last H frames; features(z) for the value net = mean-pooled
    patches of the last frame (D) concatenated with the margin (the net adds l itself).
    Action space = the ckpt's action_key (default 'action_cup' = cup lin accel (3) +
    ang accel (3)) -- NOT the Stage-0/kin_wm (a_lin, w) space: map the fallback set /
    plan accordingly. NOTE (DINO-WM agent's gate): it is trained on BEAD spill labels,
    where the contents are unobservable and b3 physics already reaches 0.998 ep-AUROC;
    for the liquid-proxy (pend) labels it would need retraining. Not run here."""
    action_space = "action_cup"

    def __init__(self, spill_wm):
        self.w = spill_wm
        self.H = spill_wm.H

    def encode(self, obs_history):
        return self.w.encode(obs_history)

    def imagine(self, z0, actions):
        """z0 (B,H,P,D), actions (B,K,6) -> (B,K,H,P,D) window after each action."""
        import torch
        zs = self.w.imagine(z0, actions)                 # (B, H+K, P, D)
        K = actions.shape[1]
        return torch.stack([zs[:, k + 1:k + 1 + self.H] for k in range(K)], 1)

    def spill_margin(self, z):
        shp = z.shape[:-3]
        return self.w.spill_margin(z.reshape(-1, *z.shape[-3:])).view(shp)

    @staticmethod
    def features(z):
        return z[..., -1, :, :].mean(-2)


class LatentDyn:
    """reach_value Dyn over ANY LatentWM adapter (one imagined WM step per backup)."""

    def __init__(self, adapter, U, bank, device="cpu"):
        import torch
        self.a, self.U, self.bank, self.dev = adapter, U, bank, device

    def sample_states(self, B):
        import torch
        return self.bank[torch.randint(len(self.bank), (B,), device=self.bank.device)]

    def step(self, z, u):
        import torch
        shp = torch.broadcast_shapes(z.shape[:-1], u.shape[:-1])
        zz = z.expand(*shp, z.shape[-1]).reshape(-1, z.shape[-1])
        uu = u.expand(*shp, u.shape[-1]).reshape(-1, 1, u.shape[-1])
        return self.a.imagine(zz, uu)[:, 0].view(*shp, -1)

    def margin(self, z):
        return self.a.spill_margin(z)

    def features(self, z):
        return z


class WMContents:
    """flmpc_value contents model over a LatentWM adapter (Stage 1 in the controller):
    plan (a_piv(t), w(t)) at the 200 Hz sim rate -> WM actions (block means over
    `stride` sim steps) -> imagine -> V at WM checkpoints. Same rollout_scores()
    signature as flmpc_value.PendContents, so ValueConstraint uses it unchanged.
    z0 here is the WM posterior latent (torch (dz,)), maintained by the caller."""

    def __init__(self, adapter, value_net, stride=4, score="value"):
        import torch
        self.t, self.a, self.net, self.stride, self.score = torch, adapter, value_net, stride, score

    def _s(self, z):
        l = self.a.spill_margin(z)
        return l if (self.score == "margin" or self.net is None) else self.net(z, l)

    def rollout_scores(self, z0, A0, A1, W0, W1, x, ck, need_grad=True):
        t = self.t
        xt = t.tensor(x, dtype=t.float32, requires_grad=need_grad)
        A0t, A1t, W0t, W1t = (t.tensor(v, dtype=t.float32) for v in (A0, A1, W0, W1))
        a = A0t + t.einsum("tik,k->ti", A1t, xt)
        w = W0t + t.einsum("tik,k->ti", W1t, xt)
        NP = a.shape[0]; K = NP // self.stride
        act = t.cat([a[:K * self.stride].view(K, self.stride, 3).mean(1),
                     w[:K * self.stride].view(K, self.stride, 3).mean(1)], -1)[None]
        z = self.a.imagine(t.as_tensor(z0, dtype=t.float32)[None], act)[0]      # (K, dz)
        kk = sorted(set(int(c) // self.stride for c in ck))
        s = self._s(z[kk])
        if not need_grad:
            return s.detach().numpy().astype(float), None
        J = t.stack([t.autograd.grad(s[i], xt, retain_graph=i < len(kk) - 1)[0] for i in range(len(kk))])
        return s.detach().numpy().astype(float), J.numpy().astype(float)
