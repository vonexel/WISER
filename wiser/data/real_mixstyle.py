from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class MixStyleParams:
    enabled: bool = True
    p_mix: float = 0.3
    eps: float = 1e-6


def real_mixstyle_(rgb: torch.Tensor, labels: torch.Tensor, params: MixStyleParams, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if not params.enabled or params.p_mix <= 0.0:
        return rgb
    if generator is not None:
        u = torch.rand((), generator=generator, device=rgb.device).item()
    else:
        u = torch.rand((), device=rgb.device).item()
    if u >= params.p_mix:
        return rgb
    real_idx = (labels == 0).nonzero(as_tuple=False).flatten()
    if real_idx.numel() < 2:
        return rgb
    perm = real_idx[torch.randperm(real_idx.numel(), generator=generator, device=real_idx.device)]
    a, b = real_idx[0].item(), perm[0].item()
    if a == b and real_idx.numel() > 1:
        b = perm[1].item()
    if a == b:
        return rgb
    mu_a = rgb[a].mean(dim=(-1, -2), keepdim=True)
    sd_a = rgb[a].std(dim=(-1, -2), keepdim=True).clamp(min=params.eps)
    mu_b = rgb[b].mean(dim=(-1, -2), keepdim=True)
    sd_b = rgb[b].std(dim=(-1, -2), keepdim=True).clamp(min=params.eps)
    rgb[a] = (rgb[a] - mu_a) / sd_a * sd_b + mu_b
    return rgb


def mixstyle_params_from_config(cfg) -> MixStyleParams:
    block = getattr(cfg, "real_mixstyle", None) or cfg
    return MixStyleParams(enabled=bool(getattr(block, "enabled", False)), p_mix=float(getattr(block, "p_mix", 0.3)))