from __future__ import annotations

import math
import torch


def build_optimizer(model: torch.nn.Module, cfg, loss_fn: torch.nn.Module | None = None) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_classifier = ("head.classifier." in name) or ("head.logit_scale" in name)
        is_prototype = ".prototypes" in name
        if p.ndim <= 1 or name.endswith(".bias") or is_classifier or is_prototype:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if loss_fn is not None:
        # Prototypes belong to the loss, not to the detector's module tree.
        known = {id(p) for group in groups for p in group["params"]}
        extra = [p for p in loss_fn.parameters() if p.requires_grad and id(p) not in known]
        if extra:
            groups.append({"params": extra, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)


def build_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_frac: float, min_lr: float) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / base_lrs[0] if base_lrs[0] > 0 else 0.0
        return floor + (1.0 - floor) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
