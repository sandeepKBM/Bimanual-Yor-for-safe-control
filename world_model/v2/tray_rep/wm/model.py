"""wm/model.py -- d_obs-aware build / save / load of the kin_wm model (`latent_hj.kin_wm.build_model` takes d_obs but
`kin_wm.load_wm` does not pass it, so a 13-wide checkpoint cannot be reloaded there). SIMULATION / SAVED DATA ONLY."""
from __future__ import annotations


def build_wm(d_obs, dev="cpu"):
    from world_model.v2.latent_hj import kin_wm
    return kin_wm.build_model(d_obs=d_obs).to(dev)


def build_aux(wm, feat_dim=10):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(wm.d_h + wm.d_obs, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, feat_dim))


def load_wm(path, dev="cpu"):
    """frozen eval-mode WM from a wm.pt written by TrayTrainer.save_wm (d_obs stored in the checkpoint)."""
    import torch
    ck = torch.load(path, map_location=dev, weights_only=False)
    d_obs = int(ck["d_obs"])
    wm = build_wm(d_obs, dev)
    wm.load_state_dict(ck["state"]); wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm
