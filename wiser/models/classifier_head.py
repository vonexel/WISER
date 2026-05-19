from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 256, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        bottleneck_dim = max(32, in_dim // 4)
        self.bottleneck = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.SiLU(inplace=True)
        self.embed = nn.Linear(bottleneck_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(12.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)             # (B, C)
        h = self.act(self.bottleneck(self.dropout(gap)))
        z = self.embed(h)
        z_norm = F.normalize(z, dim=-1, eps=1e-8)
        w_norm = F.normalize(self.classifier.weight, dim=-1, eps=1e-8)
        cos = F.linear(z_norm, w_norm)                           # (B, 1)
        scale = self.logit_scale.clamp(min=4.0, max=30.0)
        logits = scale * cos
        return logits, z_norm