from __future__ import annotations


import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


def dsconv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                         nn.BatchNorm2d(in_ch), nn.SiLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1, bias=False),
                         nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))


class HighFreqGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate.view(1, -1, 1, 1)


def _haar_dwt_features(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


class WaveletStream(nn.Module):
    def __init__(self, in_ch_wav: int = 12, in_ch_def: int = 1, *, high_freq_gate: bool = True) -> None:
        super().__init__()
        self.use_defocus = in_ch_def > 0
        in_ch = in_ch_wav + (in_ch_def if self.use_defocus else 0)
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        self.b1 = dsconv(32, 32, stride=2)               # 128 -> 64
        self.gate = HighFreqGate(32) if high_freq_gate else nn.Identity()
        self.b2 = dsconv(32, 64, stride=2)               # 64 -> 32
        self.b3 = dsconv(64, 64)
        self.b4 = dsconv(64, 128, stride=2)              # 32 -> 16
        self.b5 = dsconv(128, 128, stride=2)             # 16 -> 8

    def forward(self, x_wav: torch.Tensor, x_def: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_defocus:
            assert x_def is not None, "wavelet stream configured with defocus but received None."
            x = torch.cat([x_wav, x_def], dim=1)
        else:
            x = x_wav
        x = self.stem(x)
        x = self.b1(x)
        x = self.gate(x)
        x = self.b2(x)
        x = self.b3(x)
        ll, lh, hl, hh = _haar_dwt_features(x)
        composed = ll + 0.25 * (lh + hl + hh)
        composed = F.interpolate(composed, scale_factor=2, mode="bilinear", align_corners=False)
        x = x + composed
        x = self.b4(x)
        x = self.b5(x)
        return x