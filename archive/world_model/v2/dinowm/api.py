"""world_model/v2/dinowm/api.py -- the latent world-model interface for the spill
shield and the latent HJ-reachability agent (Nakamura et al.-style latent safety filter).

SIMULATION / OFFLINE ONLY: pure torch over a checkpoint; nothing here opens a socket,
serial port, ROS node or robot SDK.

    wm = SpillWM("/path/model.pt", device="cuda")             # dinowm/train.py checkpoint
    z   = wm.encode(obs_history)                               # (B, H, P, D) latent state
    zs  = wm.imagine(z, actions)                               # (B, H+K, P, D) history + K imagined frames
    l   = wm.spill_margin(zs)                                  # (B, K+1) margins, window ends t..t+K
    p   = wm.hazard(zs)                                        # (B, K+1, 3) P(spill within 0.25/0.5/1 s)

LATENT STATE z
  A window of the last H = num_hist frames (official DINO-WM state). Each frame is
  P DINOv2 ViT-S/14 patch tokens (384-d, frozen) concatenated (DINO-WM concat_dim=1)
  with a 10-d proprio embedding and a 10-d action embedding -> D = 404.
  P = 64 for the default 112 px crop (8x8 grid). z is a plain torch tensor.

encode(obs_history) -> z  (B, H, P, D)
  obs_history: dict with
    'images'  (B, H, 112, 112, 3) uint8 cup-centred ZED crops (img_zed), oldest first,
              OR 'tokens' (B, H, P, 384) precomputed frozen-DINO patch tokens;
    'low'     (B, H, d_low) RAW robot-knowable features per frame =
              common.step_features(ep, use_action=False) rows: proprio (26) + FK cup
              kinematics (16);
    'actions' (B, H, d_act) RAW actions applied after each history frame (ckpt
              'action_key', default action_cup = commanded cup accel [lin 3, ang 3]).
              Row H-1 (the action about to be applied) is a placeholder: imagine()
              overwrites it with actions[:, 0]. (B, H-1, d_act) is accepted too.
  Missing history at episode start: repeat the first frame (what the harness does).

imagine(z, actions) -> z_seq  (B, H+K, P, D)
  actions (B, K, d_act) RAW planned actions a_t .. a_{t+K-1}. If z has batch 1 and
  actions batch N, z is broadcast (N candidate plans from one state). Frame H-1+k of
  the output is the latent after applying a_t..a_{t+k-1}. The official DINO-WM
  rollout (sliding window, action dims of each new frame overwritten by the next
  action). Differentiable w.r.t. actions (and z): use it inside a torch graph for
  gradient-based MPC / HJ value learning. No torch.no_grad inside.

spill_margin(z) -> l
  z (B, H, P, D) -> (B,)   or   z (B, T>=H, P, D) -> (B, T-H+1) (one margin per window end).
  l in [-1, 1], +1 = cup keeps its contents, trained toward -1 after spill onset (the
  failure-set margin l(z) of the Fisac/Nakamura HJ formulation: failure <=> l < 0).
hazard(z) -> P(spill within {0.25, 0.5, 1.0} s), same shapes with a trailing 3.

All four accept/return torch tensors, batch-first, on wm.device, and are
differentiable (encode is differentiable w.r.t. low/actions; the DINOv2 encoder itself
is frozen and run without grad). Set wm.net.eval() (default) for deterministic dropout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.dinowm.model import DinoWMSpill, FrozenDino  # noqa: E402
from world_model.v2.model import common as C  # noqa: E402


class SpillWM:
    def __init__(self, ckpt, device="cuda", encoder: FrozenDino | None = None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.ck, self.device = ck, torch.device(device)
        self.net = DinoWMSpill(**ck["arch"]); self.net.load_state_dict(ck["state"]); self.net.to(self.device).eval()
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)
        self.low_mu, self.low_sd = t(ck["low_norm"]["mean"]), t(ck["low_norm"]["std"])
        self.act_mu, self.act_sd = t(ck["act_norm"]["mean"]), t(ck["act_norm"]["std"])
        self.H, self.P = self.net.H, self.net.P
        self.action_key = ck["action_key"]; self.horizons_s = tuple(ck["horizons_s"])
        self._enc = encoder

    @property
    def encoder(self):
        if self._enc is None:
            e = self.ck["enc"]; self._enc = FrozenDino(e["img_px"], e["pool"]).to(self.device)
        return self._enc

    # ------------------------------------------------------------------ helpers
    def _t(self, x):
        return x.to(self.device) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), device=self.device)

    def norm_act(self, a):
        return (self._t(a).float() - self.act_mu) / self.act_sd

    def norm_low(self, x):
        return (self._t(x).float() - self.low_mu) / self.low_sd

    # ------------------------------------------------------------------ API
    def encode(self, obs_history: dict) -> torch.Tensor:
        if "tokens" in obs_history:
            vis = self._t(obs_history["tokens"]).float()
        else:
            im = self._t(obs_history["images"])
            B, H = im.shape[:2]
            vis = self.encoder(im.reshape(B * H, *im.shape[2:])).view(B, H, self.P, -1)
        B, H = vis.shape[:2]
        assert H == self.H, f"need exactly num_hist={self.H} frames, got {H}"
        act = self.norm_act(obs_history["actions"])
        if act.shape[1] == H - 1:
            act = torch.cat([act, torch.zeros_like(act[:, :1])], 1)
        return self.net.encode(vis, self.norm_low(obs_history["low"]), act)

    def imagine(self, z: torch.Tensor, actions) -> torch.Tensor:
        a = self.norm_act(actions)
        if z.shape[0] == 1 and a.shape[0] > 1:
            z = z.expand(a.shape[0], *z.shape[1:])
        z = torch.cat([z[:, :-1], self.net.replace_actions(z[:, -1:], a[:, :1])], 1)   # a_t on the current frame
        nxt = torch.cat([a[:, 1:], a[:, -1:]], 1)            # action after each imagined frame (last: hold)
        zi = self.net.rollout(z, nxt)
        return torch.cat([z, zi], 1)

    def _windows(self, z):
        if z.shape[1] == self.H:
            return z[:, None], True
        return self.net.windows(z), False

    def spill_margin(self, z: torch.Tensor) -> torch.Tensor:
        w, single = self._windows(z)
        m = self.net.margin(w)
        return m[:, 0] if single else m

    def hazard(self, z: torch.Tensor) -> torch.Tensor:
        w, single = self._windows(z)
        h = self.net.hazard(w)
        return h[:, 0] if single else h


# ============================================================================ harness adapter
class DinoWMOfficialPredictor:
    """eval/harness.py predictor: causal sliding windows over the OBSERVED episode
    (no future actions used; row t uses frames <= t and the action logged at t, which
    is the action about to be applied -- the same information the shield has).
    Tokens from ep['_tok'] (train.attach_tokens memmaps) if present, else encoded
    from ep['images'][cam]."""
    probabilistic = True

    def __init__(self, ckpt, device="cuda", mm=None, chunk=256):
        self.wm = SpillWM(ckpt, device); self.mm = mm; self.chunk = chunk
        self.horizons_s = self.wm.horizons_s
        self.name = f"DINO-WM official-style (P={self.wm.P}, H={self.wm.H})"

    def _tokens(self, ep):
        if self.mm is not None and "_tok" in ep:
            sh, st = ep["_tok"]
            return torch.as_tensor(np.asarray(self.mm[sh][st:st + ep["T"]], np.float32), device=self.wm.device)
        cam = self.wm.ck["enc"]["cam"]
        im = torch.as_tensor(ep["images"][cam], device=self.wm.device)
        return torch.cat([self.wm.encoder(im[i:i + 256]) for i in range(0, len(im), 256)])

    @torch.no_grad()
    def predict(self, ep):
        H = self.wm.H; T = ep["T"]
        vis = self._tokens(ep)
        low = self.wm.norm_low(C.step_features(ep, use_action=False).astype(np.float32))
        act = self.wm.norm_act(C.get_action(ep, self.wm.action_key))
        z = self.wm.net.encode(vis[None], low[None], act[None])[0]          # (T,P,D)
        z = torch.cat([z[:1].expand(H - 1, -1, -1), z], 0)                  # pad start by repeating frame 0
        out = []
        for s in range(0, T, self.chunk):
            idx = torch.arange(s, min(T, s + self.chunk), device=z.device)
            w = torch.stack([z[i:i + H] for i in idx.tolist()])             # windows ending at frame i
            out.append(self.wm.net.hazard(w))
        return torch.cat(out).float().cpu().numpy()
