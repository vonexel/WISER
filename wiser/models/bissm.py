from __future__ import annotations


import torch
import torch.nn as nn
from typing import Literal
import torch.nn.functional as F

_BACKEND: Literal["mamba", "linear", "eca"] = "linear"

try:
    import mamba_ssm
    from mamba_ssm import Mamba
    _BACKEND = "mamba"
except Exception:
    _BACKEND = "linear"


def selected_backend() -> str:
    return _BACKEND


class ECAGate(nn.Module):
    def __init__(self, channels: int, k_size: int = 5) -> None:
        super().__init__()
        self.k = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, 1, c)        # (B,1,C)
        y = torch.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y


class LinearAttention2D(nn.Module):
    """
    Single-head linear attention with feature map: O(N * d^2)
    """
    def __init__(self, channels: int, d_inner: int = 32) -> None:
        super().__init__()
        self.q = nn.Conv2d(channels, d_inner, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, d_inner, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, d_inner, kernel_size=1, bias=False)
        self.out = nn.Conv2d(d_inner, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm([channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2)                       # (B, d, N)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv = torch.einsum("bdn,ben->bde", k, v)        # (B, d, d)
        z = 1.0 / (q.transpose(1, 2) @ k.sum(-1).unsqueeze(-1) + 1e-6)
        ctx = torch.einsum("bdn,bde->ben", q, kv) * z.transpose(1, 2)
        ctx = ctx.view(b, -1, h, w)
        y = self.out(ctx)
        # residual + norm-on-channel
        y = y + x
        return self.norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class _CompactMamba1D(nn.Module):
    def __init__(self, channels: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.channels = channels
        try:
            self.inner = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
            self._mode = "mamba"
        except Exception:
            self.inner = LinearAttention2DAsToken(channels)
            self._mode = "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == "mamba":
            return self.inner(x)
        return self.inner(x)


class LinearAttention2DAsToken(nn.Module):
    def __init__(self, channels: int, d_inner: int = 32) -> None:
        super().__init__()
        self.q = nn.Linear(channels, d_inner, bias=False)
        self.k = nn.Linear(channels, d_inner, bias=False)
        self.v = nn.Linear(channels, d_inner, bias=False)
        self.out = nn.Linear(d_inner, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        q = F.elu(self.q(x)) + 1
        k = F.elu(self.k(x)) + 1
        v = self.v(x)
        kv = torch.einsum("bnd,bne->bde", k, v)
        z = 1.0 / (q @ k.sum(dim=1).unsqueeze(-1) + 1e-6)
        ctx = torch.einsum("bnd,bde->bne", q, kv) * z
        return self.norm(self.out(ctx) + x)


class BiSSM2D(nn.Module):
    """
    Bidirectional Selective State-Space Mixing
    """
    def __init__(self, channels: int, *, enabled: bool = True, d_state: int = 16, d_conv: int = 4,
                 expand: int = 1, bidirectional: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.enabled = enabled
        self.bidirectional = bidirectional
        if not enabled:
            self.block = ECAGate(channels)
            self._impl = "eca"
            return

        if _BACKEND == "mamba":
            self.block = _CompactMamba1D(channels, d_state=d_state, d_conv=d_conv, expand=expand)
            self._impl = "mamba"
        else:
            self.block = LinearAttention2DAsToken(channels)
            self._impl = "linear"

        if bidirectional:
            self.fuse_gate = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("fuse_gate", None)

    @property
    def implementation(self) -> str:
        return self._impl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.block(x)

        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)                       # (B, N, C)
        out_fwd = self.block(seq).transpose(1, 2).reshape(b, c, h, w)
        if not self.bidirectional:
            return out_fwd
        out_rev = self.block(seq.flip(dims=[1])).flip(dims=[1])
        out_rev = out_rev.transpose(1, 2).reshape(b, c, h, w)
        gate = torch.sigmoid(self.fuse_gate).view(1, -1, 1, 1)
        return gate * out_fwd + (1.0 - gate) * out_rev