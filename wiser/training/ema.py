from __future__ import annotations


import math
import torch
import torch.nn as nn
from copy import deepcopy


def _is_bn_buffer(name: str) -> bool:
    return ("running_mean" in name) or ("running_var" in name) or ("num_batches_tracked" in name)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0) -> None:
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-float(self.updates) / self.tau))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point and not _is_bn_buffer(k):
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)