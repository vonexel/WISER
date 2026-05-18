from __future__ import annotations

import torch
import numpy as np


def _dct_2d(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    device, dtype = x.device, x.dtype
    k = torch.arange(n, device=device, dtype=dtype)
    i = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(torch.pi / n * (i[:, None] + 0.5) * k[None, :])
    norm = torch.full((n,), (2.0 / n) ** 0.5, device=device, dtype=dtype)
    norm[0] = (1.0 / n) ** 0.5
    basis = basis * norm[None, :]
    return basis.t() @ x @ basis


def _idct_2d(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    device, dtype = x.device, x.dtype
    k = torch.arange(n, device=device, dtype=dtype)
    i = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(torch.pi / n * (i[:, None] + 0.5) * k[None, :])
    norm = torch.full((n,), (2.0 / n) ** 0.5, device=device, dtype=dtype)
    norm[0] = (1.0 / n) ** 0.5
    basis = basis * norm[None, :]
    return basis @ x @ basis.t()


def apply_freq_mask(x: torch.Tensor, *, p: float = 0.3, min_size_frac: float = 0.05, max_size_frac: float = 0.25,
                    high_freq_bias: float = 0.7, rng: np.random.Generator | None = None) -> torch.Tensor:
    rng = rng or np.random.default_rng()
    if rng.random() >= p:
        return x
    assert x.ndim == 3 and x.shape[-1] == x.shape[-2], "expected (C, H, H)"
    H = x.shape[-1]
    side = int(rng.uniform(min_size_frac, max_size_frac) * H)
    side = max(2, min(side, H - 1))
    if rng.random() < high_freq_bias:
        # high-freq quadrant: top-left of rectangle in [H/2, H-side]
        y0 = int(rng.integers(H // 2, max(H // 2 + 1, H - side)))
        x0 = int(rng.integers(H // 2, max(H // 2 + 1, H - side)))
    else:
        y0 = int(rng.integers(0, max(1, H - side)))
        x0 = int(rng.integers(0, max(1, H - side)))

    spec = _dct_2d(x.float())
    spec[:, y0 : y0 + side, x0 : x0 + side] = 0.0
    out = _idct_2d(spec)
    return out.clamp(0.0, 1.0).to(x.dtype)