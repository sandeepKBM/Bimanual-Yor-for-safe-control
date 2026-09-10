"""experiments/residual_dynamics/models.py

Two parameterized torch models mapping input(36)=[q(12), qd(12), a(12)] to
residual(12) = qacc_realized - a (see perturbed_plant.py / README.md for
the exact definition):

  - MLP: plain feedforward net, configurable hidden dims/activation.
  - SIREN: sinusoidal-activation net (Sitzmann et al. 2020,
    "Implicit Neural Representations with Periodic Activation Functions"),
    first layer w0=30 (the paper's standard choice), hidden layers w0=1,
    with the paper's uniform initialization scheme so the pre-activation
    distribution stays ~U(-1,1) through depth. Final layer is a plain
    (non-sinusoidal) linear readout, standard for SIREN regression heads.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

INPUT_DIM = 36    # [q(12), qd(12), a(12)]
OUTPUT_DIM = 12   # residual, both arms

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = OUTPUT_DIM,
                 hidden_dims: Sequence[int] = (256, 256), activation: str = "relu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}, choose from {list(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]

        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SineLayer(nn.Module):
    """One SIREN layer: y = sin(w0 * (Wx + b)). `is_first` selects the
    paper's first-layer init (U(-1/in, 1/in)); otherwise the w0-scaled
    init U(-sqrt(6/in)/w0, sqrt(6/in)/w0) that keeps sin's argument in a
    good operating range through depth."""

    def __init__(self, in_dim: int, out_dim: int, w0: float, is_first: bool = False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_dim, out_dim)
        self._init_weights(in_dim, is_first)

    def _init_weights(self, in_dim: int, is_first: bool):
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = OUTPUT_DIM,
                 hidden_dims: Sequence[int] = (256, 256), w0_first: float = 30.0,
                 w0_hidden: float = 1.0):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(SineLayer(dims[i], dims[i + 1],
                                     w0=w0_first if i == 0 else w0_hidden,
                                     is_first=(i == 0)))
        self.hidden = nn.Sequential(*layers)

        self.readout = nn.Linear(dims[-1], out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / dims[-1]) / w0_hidden
            self.readout.weight.uniform_(-bound, bound)
            self.readout.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.hidden(x))


def build_model(arch: str, hidden_dims: Sequence[int] = (256, 256), **kwargs) -> nn.Module:
    if arch == "mlp":
        return MLP(hidden_dims=hidden_dims, **kwargs)
    if arch == "siren":
        return SIREN(hidden_dims=hidden_dims, **kwargs)
    raise ValueError(f"unknown arch {arch!r}, choose 'mlp' or 'siren'")
