"""SIREN network (Sitzmann et al. 2020, "Implicit Neural Representations with
Periodic Activation Functions") mapping the 12-D bimanual state
``[q_ego(6), q_int(6)]`` to a scalar safety value ``V(x)``.

Why SIREN here (not a ReLU/tanh MLP): the Fisac discounted-safety Bellman
backup (``fisac_safety.py``) needs a *smooth, well-behaved gradient*
``dV/dx`` everywhere (it is used to steer the min-max backup, not just for a
final zero-level-set query) -- ReLU MLPs have a piecewise-constant gradient
(zero almost everywhere between kinks) which starves that backup of signal,
and SIREN is the standard choice in the DeepReach line of work (see
``hj_reachability/README.md``'s note that ``train_config.yaml`` already uses
a "SIREN MLP" for the DeepReach bootstrap net) precisely because periodic
activations give a smooth, everywhere-nonzero gradient and sharp, accurate
level sets (the safe/unsafe boundary ``V(x)=0`` is what this value function
is ultimately used for downstream, e.g. in ``value_runtime.py``).

Standalone / torch-only (no project-specific imports), so it can be reused
verbatim if this ever needs to be lifted into another trainer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """``x -> sin(w0 * (Wx + b))``, with the SIREN paper's principled init.

    Sitzmann et al. 2020, Sec. 3.2 / supplement: the first layer's weights
    are drawn from ``U(-1/in_features, 1/in_features)`` (so the input to the
    first sine spans roughly one period regardless of ``w0``); every later
    layer's weights are drawn from
    ``U(-sqrt(6/in_features)/w0, sqrt(6/in_features)/w0)``, which keeps the
    pre-activation distribution (and hence the sine's operating point)
    stable across depth for a given ``w0``.
    """

    def __init__(self, in_features: int, out_features: int, w0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._siren_init(in_features)

    def _siren_init(self, in_features: int) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(6.0 / in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            # Bias init is not specified by the paper as being special;
            # zero-init keeps the pre-activation centered.
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN MLP: ``state(state_dim) -> scalar V(x)``.

    ``depth`` counts *hidden* SineLayers (so ``depth=3`` -> input SineLayer +
    2 hidden SineLayers + a final plain-linear output layer, all sharing
    ``width`` hidden units). The output layer is a plain (non-sinusoidal)
    ``nn.Linear`` -- standard SIREN practice for scalar-field regression (the
    value/SDF itself need not be periodic, only the features feeding it) --
    initialized with the same "later layer" SIREN bound so its input scale
    matches what the last SineLayer produces.

    ``w0_first=30`` (fixed per Sitzmann et al.'s recommendation for
    low-dimensional coordinate inputs); ``w0_hidden`` defaults to the same
    30, also per the paper's image/SDF experiments, but is exposed in case a
    lower-frequency value function turns out to train better empirically.
    """

    def __init__(self, in_features: int = 12, width: int = 128, depth: int = 3,
                 w0_first: float = 30.0, w0_hidden: float = 30.0,
                 out_features: int = 1):
        super().__init__()
        assert depth >= 1, "depth counts hidden SineLayers; need at least 1"
        layers = [SineLayer(in_features, width, w0=w0_first, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(width, width, w0=w0_hidden, is_first=False))
        self.net = nn.Sequential(*layers)

        self.output_linear = nn.Linear(width, out_features)
        with torch.no_grad():
            bound = np.sqrt(6.0 / width) / w0_hidden
            self.output_linear.weight.uniform_(-bound, bound)
            self.output_linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_linear(self.net(x))
