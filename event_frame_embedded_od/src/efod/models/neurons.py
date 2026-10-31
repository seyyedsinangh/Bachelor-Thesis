from __future__ import annotations

import math
import torch
from torch import nn


class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Smooth ATan-like surrogate derivative. Forward remains binary.
        grad = alpha / (1.0 + (math.pi * alpha * x / 2.0).pow(2)) / 2.0
        return grad_output * grad, None


class ParametricLIF(nn.Module):
    """Stateful PLIF neuron for direct event-stream training/inference.

    Dynamics follow the common Parametric-LIF form
        v <- v + sigmoid(w) * (x - (v - v_reset))
    with learnable 1/tau = sigmoid(w). A hard reset to ``v_reset`` is used.

    The module is intentionally small and framework-independent so the final project
    does not need to patch or depend on the original repository. It is not claimed to
    be byte-for-byte identical to any specific SpikingJelly release.
    """

    def __init__(
        self,
        init_tau: float = 2.0,
        threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        if init_tau <= 1.0:
            raise ValueError('init_tau must be > 1 for the sigmoid parameterization')
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.threshold = float(threshold)
        self.v_reset = float(v_reset)
        self.surrogate_alpha = float(surrogate_alpha)
        self.v: torch.Tensor | None = None
        self.last_spike_count: int = 0
        self.last_neuron_count: int = 0

    @property
    def reciprocal_tau(self) -> torch.Tensor:
        return torch.sigmoid(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device or self.v.dtype != x.dtype:
            self.v = torch.full_like(x, self.v_reset)
        self.v = self.v + self.reciprocal_tau * (x - (self.v - self.v_reset))
        spike = _ATanSpike.apply(self.v - self.threshold, self.surrogate_alpha)
        # Hard reset. Detaching the reset gate avoids adding an artificial reset-gradient path.
        gate = spike.detach()
        self.v = self.v * (1.0 - gate) + self.v_reset * gate
        self.last_spike_count = int(spike.detach().count_nonzero().item())
        self.last_neuron_count = spike.numel()
        return spike

    def reset_state(self) -> None:
        self.v = None

    def detach_state(self) -> None:
        if self.v is not None:
            self.v = self.v.detach()


def reset_snn_state(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, ParametricLIF):
            m.reset_state()


def detach_snn_state(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, ParametricLIF):
            m.detach_state()


def count_spiking_neurons(module: nn.Module) -> int:
    return sum(isinstance(m, ParametricLIF) for m in module.modules())
