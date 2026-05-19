from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralDifferenceConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False, theta: float = 0.7) -> None:
        super().__init__()
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0,1], got {theta}")
        self.theta = float(theta)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.theta == 0.0:
            return out
        kernel_sum = self.conv.weight.sum(dim=(2, 3))  # (C_out, C_in)
        kernel_1x1 = kernel_sum.unsqueeze(-1).unsqueeze(-1)  # (C_out, C_in, 1, 1)
        diff = F.conv2d(x, kernel_1x1, bias=None, stride=self.stride, padding=0)
        return out - self.theta * diff

    def extra_repr(self) -> str:
        return f"theta={self.theta}"


def cdc_block(in_ch: int, out_ch: int, stride: int, theta: float) -> nn.Sequential:
    """CDC + BN + SiLU"""
    return nn.Sequential(
        CentralDifferenceConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, theta=theta),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True))


class CDCStem(nn.Module):
    """Two stacked CDC blocks"""
    def __init__(self, in_ch: int = 3, out_ch: int = 32, theta: float = 0.7) -> None:
        super().__init__()
        self.b1 = cdc_block(in_ch, out_ch, stride=1, theta=theta)
        self.b2 = cdc_block(out_ch, out_ch, stride=2, theta=theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b2(self.b1(x))