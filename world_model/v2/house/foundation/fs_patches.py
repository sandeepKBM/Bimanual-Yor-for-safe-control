"""Run-time (monkeypatch) fixes so the unmodified NVlabs/FoundationStereo model can be
captured in a CUDA graph.  Every fix removes a pageable host->device copy (or a per-call
CPU tensor build) from forward(); numerics are unchanged.  Upstream files are NOT edited.

Why: on this shared westeros box the CPU load average is ~80 on 72 cores, and the eager
model (~1000 tiny kernels per iteration) is launch-bound, so eager latency is dominated by
CPU contention.  A CUDA graph removes the CPU from the loop.

Runs inside the `foundation` conda env only (needs torch + timm + upstream repo on sys.path).
"""
from __future__ import annotations
import torch


def apply(model, device="cuda"):
    import core.foundation_stereo as cfs
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
    # (1) torchvision Normalize builds mean/std tensors on the host every call.
    cfs.normalize_image = lambda img: ((img - mean) / std).contiguous()
    # (2) model.dx lives on CPU and is copied per forward.
    model.dx = model.dx.to(device)
    # (3) timm edgenext PositionalEncodingFourier: ~torch.zeros(shape).to(device) (H2D copy).
    try:
        import math
        from timm.models import edgenext as ed

        def fwd(self, shape):
            dev = self.token_projection.weight.device
            dtype = self.token_projection.weight.dtype
            inv_mask = ~torch.zeros(shape, device=dev, dtype=torch.bool)
            y = inv_mask.cumsum(1, dtype=torch.float32)
            x = inv_mask.cumsum(2, dtype=torch.float32)
            eps = 1e-6
            y = y / (y[:, -1:, :] + eps) * self.scale
            x = x / (x[:, :, -1:] + eps) * self.scale
            dim_t = torch.arange(self.hidden_dim, dtype=torch.int64, device=dev).to(torch.float32)
            dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)
            px = x[:, :, :, None] / dim_t
            py = y[:, :, :, None] / dim_t
            px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=4).flatten(3)
            py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((py, px), dim=3).permute(0, 3, 1, 2)
            return self.token_projection(pos.to(dtype))

        ed.PositionalEncodingFourier.forward = fwd
    except Exception:  # pragma: no cover
        pass
    # (4) bilinear_sampler asserts torch.unique(ygrid).numel()==1 -> device sync, illegal in capture.
    import torch.nn.functional as F
    from torch.backends import cudnn
    import core.utils.utils as cu
    import core.geometry as cg

    def bilinear_sampler(img, coords, mode="bilinear", mask=False, low_memory=False):
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)  # stereo: H == 1, ygrid == 0
        with cudnn.flags(enabled=False):
            img = F.grid_sample(img.contiguous(), grid.contiguous(), align_corners=True)
        if mask:
            m = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, m.float()
        return img

    cu.bilinear_sampler = bilinear_sampler
    cg.bilinear_sampler = bilinear_sampler
    return model
