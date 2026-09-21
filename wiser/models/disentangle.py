from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambda_)


def grl_lambda_schedule(epoch: int, total_epochs: int, lambda_max: float = 0.3) -> float:
    """sigmoidal warm-up"""
    if total_epochs <= 0:
        return float(lambda_max)
    t = max(0.0, min(1.0, epoch / total_epochs))
    return float(lambda_max) * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * t)).item()) - 1.0)


METHOD_INDEX: Dict[str, int] = {"real": 0,
                                "Deepfakes": 1,
                                "Face2Face": 2,
                                "FaceSwap": 3,
                                "NeuralTextures": 4,
                                "FaceShifter": 5}
NUM_METHODS = 6


def manip_to_method_label(manip: str) -> int:
    return METHOD_INDEX.get(manip, -1)


class ClassifierHeadCND(nn.Module):
    def __init__(self, in_dim: int = 256, zc_dim: int = 192, zn_dim: int = 64, dropout: float = 0.1,
                 num_methods: int = NUM_METHODS, bottleneck_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        bd = bottleneck_dim if bottleneck_dim is not None else max(32, in_dim // 4)
        self.bottleneck = nn.Linear(in_dim, bd)
        self.act = nn.SiLU(inplace=True)
        self.embed_zc = nn.Linear(bd, zc_dim)
        self.embed_zn = nn.Linear(bd, zn_dim)
        self.classifier = nn.Linear(zc_dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(12.0))
        self.aux_zn = nn.Linear(zn_dim, num_methods)
        self.aux_zc = nn.Linear(zc_dim, num_methods)
        self.register_buffer("lambda_grl", torch.tensor(0.0), persistent=False)
        self.zc_dim = int(zc_dim)
        self.zn_dim = int(zn_dim)

    def set_grl_lambda(self, value: float) -> None:
        self.lambda_grl.fill_(float(value))

    def forward(self, x: torch.Tensor, *, lambda_grl: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        h = self.act(self.bottleneck(self.dropout(gap)))
        z_c = self.embed_zc(h)
        z_n = self.embed_zn(h)
        z_c_norm = F.normalize(z_c, dim=-1, eps=1e-8)
        z_n_norm = F.normalize(z_n, dim=-1, eps=1e-8)
        w_norm = F.normalize(self.classifier.weight, dim=-1, eps=1e-8)
        cos = F.linear(z_c_norm, w_norm)
        scale = self.logit_scale.clamp(min=4.0, max=30.0)
        logits = scale * cos
        method_logits_zn = self.aux_zn(z_n_norm)
        lam = float(lambda_grl) if lambda_grl is not None else float(self.lambda_grl.item())
        z_c_reverse = grad_reverse(z_c_norm, lam)
        method_logits_zc = self.aux_zc(z_c_reverse)
        return logits, z_c_norm, z_n_norm, method_logits_zn, method_logits_zc


def compute_cnd_losses(*, method_logits_zn: torch.Tensor, method_logits_zc: Optional[torch.Tensor], method_label: torch.Tensor,
                       z_c: Optional[torch.Tensor] = None, z_n: Optional[torch.Tensor] = None, orth_enabled: bool = True) -> Dict[str, torch.Tensor]:
    valid = method_label >= 0
    out: Dict[str, torch.Tensor] = {}
    device = method_logits_zn.device
    if valid.any():
        labels = method_label[valid].long()
        out["aux_zn"] = F.cross_entropy(method_logits_zn[valid], labels)
        if method_logits_zc is not None:
            out["aux_zc_grl"] = F.cross_entropy(method_logits_zc[valid], labels)
        else:
            out["aux_zc_grl"] = torch.zeros((), device=device)
    else:
        out["aux_zn"] = torch.zeros((), device=device)
        out["aux_zc_grl"] = torch.zeros((), device=device)
    if orth_enabled and z_c is not None and z_n is not None:
        b = z_c.shape[0]
        gram = z_c.t() @ z_n  # (zc_dim, zn_dim)
        out["orth"] = (gram.pow(2).sum()) / max(1, b)
    return out