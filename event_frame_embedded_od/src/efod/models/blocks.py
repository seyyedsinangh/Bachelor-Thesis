from __future__ import annotations

import torch
from torch import nn

from .neurons import ParametricLIF


class _TrackedBlock(nn.Module):
    is_spiking_block: bool = False

    def _record_input(self, x: torch.Tensor) -> None:
        self.last_input_shape = tuple(x.shape)
        self.last_theta_in = int((x.detach() != 0).count_nonzero().item())

    def _record_output(self, y: torch.Tensor) -> None:
        self.last_output_shape = tuple(y.shape)
        self.last_theta_out = int((y.detach() != 0).count_nonzero().item())


class PreActSpikingBlock(_TrackedBlock):
    """Cordone-style block: Padding -> BN(in) -> Conv -> PLIF."""
    is_spiking_block = True

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, init_tau=2.0, threshold=1.0):
        super().__init__()
        self.pad = nn.ConstantPad2d(padding, 0.0) if padding else nn.Identity()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=False)
        self.act = ParametricLIF(init_tau=init_tau, threshold=threshold)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding
        self.last_theta_in = 0
        self.last_theta_out = 0
        self.last_input_shape = None
        self.last_output_shape = None

    def forward(self, x):
        self._record_input(x)
        y = self.act(self.conv(self.bn(self.pad(x))))
        self._record_output(y)
        return y


class PreActANNBlock(_TrackedBlock):
    """Strict structural twin: Padding -> BN(in) -> Conv -> ReLU."""
    is_spiking_block = False

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, **_):
        super().__init__()
        self.pad = nn.ConstantPad2d(padding, 0.0) if padding else nn.Identity()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=False)
        self.act = nn.ReLU(inplace=False)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding

    def forward(self, x):
        self._record_input(x)
        y = self.act(self.conv(self.bn(self.pad(x))))
        self._record_output(y)
        return y


class NativeANNBlock(_TrackedBlock):
    """Conventional CNN block: Conv -> BN(out) -> ReLU."""
    is_spiking_block = False

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, **_):
        super().__init__()
        self.pad = nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding

    def forward(self, x):
        self._record_input(x)
        y = self.act(self.bn(self.conv(x)))
        self._record_output(y)
        return y


def make_block(pipeline: str, *args, **kwargs) -> nn.Module:
    if pipeline == 'event_snn':
        return PreActSpikingBlock(*args, **kwargs)
    if pipeline == 'strict_ann':
        return PreActANNBlock(*args, **kwargs)
    if pipeline == 'native_ann':
        return NativeANNBlock(*args, **kwargs)
    raise ValueError(f'Unknown pipeline: {pipeline}')


class ResCatStage(nn.Module):
    """Concatenation residual stage preserving Cordone's spike-friendly topology."""

    def __init__(
        self,
        pipeline: str,
        in_ch: int,
        hidden_ch: int,
        main_out_ch: int,
        residual_out_ch: int,
        init_tau: float = 2.0,
        threshold: float = 1.0,
    ):
        super().__init__()
        kw = dict(init_tau=init_tau, threshold=threshold)
        self.main1 = make_block(pipeline, in_ch, hidden_ch, 3, 1, 1, **kw)
        self.main2 = make_block(pipeline, hidden_ch, main_out_ch, 3, 2, 1, **kw)
        self.residual = make_block(pipeline, in_ch, residual_out_ch, 1, 2, 0, **kw)
        self.out_channels = main_out_ch + residual_out_ch

    def forward(self, x):
        return torch.cat([self.main2(self.main1(x)), self.residual(x)], dim=1)
