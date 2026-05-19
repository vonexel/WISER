from __future__ import annotations


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSCA(nn.Module):
    def __init__(self, dim_rgb: int = 256, dim_wav: int = 128, dim: int = 256, *, enabled: bool = True) -> None:
        super().__init__()
        if dim != dim_rgb:
            raise ValueError(f"SSCA requires dim_rgb == dim, got {dim_rgb} != {dim}")
        self.enabled = enabled
        self.wav_proj = nn.Conv2d(dim_wav, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        if enabled:
            self.gate = nn.Parameter(torch.zeros(dim))
        else:
            self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, rgb_feat: torch.Tensor, wav_feat: torch.Tensor) -> torch.Tensor:
        wav = self.bn(self.wav_proj(wav_feat))
        if not self.enabled:
            return self.fuse(torch.cat([rgb_feat, wav], dim=1))

        b, c, h, w = rgb_feat.shape
        scale = 1.0 / math.sqrt(c)
        rgb_tok = rgb_feat.flatten(2).transpose(1, 2)            # (B, N, C)
        wav_tok = wav.flatten(2).transpose(1, 2)
        # RGB-queries-Wavelet
        attn_rw = torch.softmax(rgb_tok @ wav_tok.transpose(1, 2) * scale, dim=-1)
        attn_r = attn_rw @ wav_tok                              # (B, N, C)
        # wavelet-queries-RGB
        attn_wr = torch.softmax(wav_tok @ rgb_tok.transpose(1, 2) * scale, dim=-1)
        attn_w = attn_wr @ rgb_tok
        attn_r = attn_r.transpose(1, 2).reshape(b, c, h, w)
        attn_w = attn_w.transpose(1, 2).reshape(b, c, h, w)
        gate = torch.sigmoid(self.gate).view(1, -1, 1, 1)
        return gate * (rgb_feat + attn_r) + (1.0 - gate) * (wav + attn_w)