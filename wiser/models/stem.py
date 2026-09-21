from __future__ import annotations

import torch
import torch.nn as nn
from wiser.models.bissm import BiSSM2D


def _dsconv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                         nn.BatchNorm2d(in_ch), nn.SiLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1, bias=False),
                         nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))


def _downsample(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False),
                         nn.BatchNorm2d(channels), nn.SiLU(inplace=True))


class RGBBackbone(nn.Module):
    def __init__(self, *, bissm_enabled: bool, bissm_kwargs: dict) -> None:
        super().__init__()
        # 128 -> 64
        self.s1 = nn.Sequential(_dsconv(32, 64), _dsconv(64, 64), _downsample(64))
        # 64 -> 32
        self.s2 = nn.Sequential(_dsconv(64, 128), _dsconv(128, 128), _downsample(128))
        # 32 -> 16, with BiSSM
        self.s3a = _dsconv(128, 128)
        self.s3_mix = BiSSM2D(channels=128, enabled=bissm_enabled, **bissm_kwargs)
        self.s3_down = _downsample(128)
        # 16 -> 8, with BiSSM and a final downsample
        self.s4a = _dsconv(128, 256)
        self.s4_mix = BiSSM2D(channels=256, enabled=bissm_enabled, **bissm_kwargs)
        self.s4_down = _downsample(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3a(x)
        x = self.s3_mix(x)
        x = self.s3_down(x)
        x = self.s4a(x)
        x = self.s4_mix(x)
        x = self.s4_down(x)
        return x