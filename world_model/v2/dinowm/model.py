"""world_model/v2/dinowm/model.py -- official-style DINO-WM (Zhou et al., arXiv 2411.04983)
adapted to cup-spill prediction.

SIMULATION / OFFLINE ONLY. No robot, no network (DINOv2 is loaded from the local
torch-hub cache; see `load_dinov2`).

What is ported from github.com/gaoyuezhou/dino_wm (MIT License, (c) 2025 gaoyuezhou;
checkout: /common/users/ss5772/external/dino_wm @ 0a9492f):
  * models/vit.py  ViTPredictor / Transformer / Attention / FeedForward and the
    frame-block-causal mask (`generate_mask_matrix`). Changes: the mask is a
    registered buffer built from (num_patches, num_frames) instead of module
    globals + a hard-coded `.to('cuda')` (so CPU / several instances work), and
    attention uses F.scaled_dot_product_attention (same math, less memory).
  * models/proprio.py ProprioceptiveEmbedding (Conv1d k=1 == Linear) for proprio
    AND action embeddings (conf/action_encoder/proprio.yaml).
  * models/visual_world_model.py concat_dim=1 token layout: every DINO patch token
    is concatenated with the frame's proprio embedding and action embedding
    (dim = 384 + 10 + 10); predict() over a window of num_hist frames; rollout()
    slides the window and overwrites the action dims of each new frame with the
    next planned action (`replace_actions_from_z`); latent loss = MSE on the
    visual+proprio dims vs. the (frozen) encoded next frames.
Our additions (NOT in DINO-WM): a spill head (attention-pooled window ->
common.SpillHeads monotone hazard P(spill within H) + margin l in [-1,1]), trained on
observed AND imagined windows; optional multi-step open-loop rollout loss
(official trains 1-step teacher-forced only); ImageNet normalisation of the input
(official feeds [-1,1]-normalised pixels to DINOv2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.model import common as C  # noqa: E402

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HUB_DIR = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"


# =========================================================================== encoder
def load_dinov2(arch="dinov2_vits14"):
    """Local-cache load only (source='local'); weights from ~/.cache/torch/hub/checkpoints."""
    return torch.hub.load(str(HUB_DIR), arch, source="local", verbose=False)


class FrozenDino(nn.Module):
    """uint8 (N,H,W,3) or float (N,3,H,W) in [0,1] -> x_norm_patchtokens (N, g*g, 384).
    img_px=112 -> 8x8 patches (native res of our 112 px crop); 224 -> 16x16 (official
    DINO-WM uses 224 -> 196 px -> 14x14). pool=g' avg-pools the grid to g' x g'."""

    def __init__(self, img_px=112, pool=None, arch="dinov2_vits14"):
        super().__init__()
        self.m = load_dinov2(arch).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.img_px, self.pool = img_px, pool
        self.grid = img_px // 14
        self.n_patches = (pool or self.grid) ** 2
        self.dim = self.m.embed_dim

    @torch.no_grad()
    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.img_px:
            x = F.interpolate(x, size=(self.img_px, self.img_px), mode="bilinear", align_corners=False, antialias=True)
        x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            pt = self.m.forward_features(x)["x_norm_patchtokens"]
        pt = pt.float()
        if self.pool and self.pool != self.grid:
            g = self.grid
            pt = pt.view(len(x), g, g, -1).permute(0, 3, 1, 2)
            pt = F.adaptive_avg_pool2d(pt, self.pool).flatten(2).transpose(1, 2)
        return pt


# =========================================================================== ViT predictor (ported)
def generate_mask_matrix(npatch, nwindow):
    """[dino_wm models/vit.py] frame-block-causal mask: tokens of frame i attend to
    every patch of frames <= i."""
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        rows.append(torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1))
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.p = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        B, T, _ = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, :, :T, :T],
                                           dropout_p=self.p if self.training else 0.0)
        return self.to_out(o.transpose(1, 2).reshape(B, T, -1))


class ViTPredictor(nn.Module):
    """[dino_wm models/vit.py ViTPredictor] learned pos-emb over num_frames*num_patches
    tokens + pre-norm transformer with the block-causal mask."""

    def __init__(self, *, num_patches, num_frames, dim, depth=6, heads=16, mlp_dim=2048, dim_head=64,
                 dropout=0.1, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, dim_head, dropout),
                                                    FeedForward(dim, mlp_dim, dropout)]) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("mask", generate_mask_matrix(num_patches, num_frames).bool(), persistent=False)

    def forward(self, x):
        n = x.shape[1]
        x = self.dropout(x + self.pos_embedding[:, :n])
        for attn, ff in self.layers:
            x = attn(x, self.mask) + x
            x = ff(x) + x
        return self.norm(x)


# =========================================================================== spill head (ours)
class WindowSpillHead(nn.Module):
    """Attention-pool a window of frame latents (B,H,P,Dz) (action dims excluded) ->
    feature (B,d) -> common.SpillHeads (monotone hazard, margin, Fisac value)."""

    def __init__(self, d_in, n_frames, n_patches, d=256, n_q=4, n_h=len(C.HORIZONS_S)):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.zeros(1, n_frames * n_patches, d))
        self.q = nn.Parameter(torch.randn(1, n_q, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(n_q * d, d), nn.SiLU(), nn.LayerNorm(d))
        self.heads = C.SpillHeads(d, n_h)

    def feat(self, zw):
        B = zw.shape[0]
        x = self.inp(zw.flatten(1, 2)) + self.pos
        o, _ = self.attn(self.q.expand(B, -1, -1), x, x, need_weights=False)
        return self.out(o.flatten(1))


# =========================================================================== world model
class DinoWMSpill(nn.Module):
    """Official DINO-WM predictor over frozen DINOv2 patch tokens (concat_dim=1) +
    spill head. Latent frame z_t = [dino patches_t (P,384) | proprio_emb_t | act_emb_t]
    tiled over patches -> (P, 404). The *state* is a window of num_hist frames."""

    def __init__(self, n_patches, d_low, d_act, num_hist=3, vis_dim=384, proprio_emb=10, action_emb=10,
                 depth=6, heads=16, mlp_dim=2048, dropout=0.1, head_d=256):
        super().__init__()
        self.P, self.H, self.vd, self.pe, self.ae = n_patches, num_hist, vis_dim, proprio_emb, action_emb
        self.proprio_encoder = nn.Linear(d_low, proprio_emb)      # == ProprioceptiveEmbedding(Conv1d k=1)
        self.action_encoder = nn.Linear(d_act, action_emb)
        self.dim = vis_dim + proprio_emb + action_emb
        self.predictor = ViTPredictor(num_patches=n_patches, num_frames=num_hist, dim=self.dim, depth=depth,
                                      heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        self.head = WindowSpillHead(vis_dim + proprio_emb, num_hist, n_patches, head_d)

    # ---- [visual_world_model.py encode / encode_act, concat_dim=1] -----------------
    def encode(self, vis, low, act):
        """vis (B,T,P,384) frozen tokens, low (B,T,d_low), act (B,T,d_act) normalised ->
        z (B,T,P,dim)."""
        P = vis.shape[2]
        pe = self.proprio_encoder(low)[:, :, None].expand(-1, -1, P, -1)
        ae = self.action_encoder(act)[:, :, None].expand(-1, -1, P, -1)
        return torch.cat([vis, pe, ae], -1)

    def replace_actions(self, z, act):
        """[replace_actions_from_z] z (B,t,P,dim), act (B,t,d_act)."""
        ae = self.action_encoder(act)[:, :, None].expand(-1, -1, z.shape[2], -1)
        return torch.cat([z[..., :-self.ae], ae], -1)

    def predict(self, z):
        """[predict] z (B,h<=H,P,dim) -> next-frame predictions for every frame (B,h,P,dim)."""
        B, T = z.shape[:2]
        return self.predictor(z.flatten(1, 2)).view(B, T, self.P, self.dim)

    def rollout(self, z_hist, actions):
        """[rollout] z_hist (B,H,P,dim) whose LAST frame's action dims = a_t (first planned
        action); actions (B,K,d_act) = a_{t+1..t+K} (the action applied after each
        imagined frame; pass zeros for the last if unknown). Returns imagined frames
        (B,K,P,dim): frame k = latent at t+1+k."""
        z = z_hist; out = []
        for k in range(actions.shape[1]):
            zn = self.predict(z[:, -self.H:])[:, -1:]
            zn = self.replace_actions(zn, actions[:, k:k + 1])
            out.append(zn); z = torch.cat([z[:, -self.H + 1:], zn], 1) if self.H > 1 else zn
        return torch.cat(out, 1)

    def state_part(self, z):
        return z[..., :self.vd + self.pe]

    def windows(self, z):
        """(B,T,P,dim) -> (B,T-H+1,H,P,dim) sliding windows (view)."""
        return z.unfold(1, self.H, 1).permute(0, 1, 4, 2, 3)

    def head_feat(self, zw):
        """zw (...,H,P,dim) window(s) -> (..., head_d)."""
        lead = zw.shape[:-3]
        f = self.head.feat(self.state_part(zw).reshape(-1, *zw.shape[-3:-1], self.vd + self.pe))
        return f.view(*lead, -1)

    def hazard(self, zw):
        return self.head.heads.hazard(self.head_feat(zw))

    def margin(self, zw):
        return self.head.heads.margin(self.head_feat(zw))[..., 0]

    def latent_loss(self, pred, tgt):
        """[forward: z_loss] MSE on visual + proprio dims (action dims excluded), tgt detached."""
        return F.mse_loss(self.state_part(pred), self.state_part(tgt).detach())


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
