from __future__ import annotations


import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass


class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t).clamp(min=1e-6).pow(self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class BrierLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        device_type = logits.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            p = torch.sigmoid(logits.float())
            t = targets.float()
            return F.mse_loss(p, t)


def fibonacci_spiral(n: int, d: int, *, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    base = torch.randn(n, d, generator=rng)
    base = F.normalize(base, dim=-1, eps=1e-8)
    return base


@dataclass(slots=True)
class MPHCOutput:
    loss: torch.Tensor
    pos_score: torch.Tensor
    neg_score: torch.Tensor


class MultiPrototypeHyperspherical(nn.Module):
    def __init__(self, embed_dim: int = 256, K: int = 4, margin: float = 0.35, temperature: float = 0.07,
                 asymmetric: bool = True, hard_negative_weight: float = 0.1, seed: int = 0, *, rp_enabled: bool = False,
                 lambda_real_margin: float = 0.0, real_delta: float = 0.40, hard_real_topk_frac: float = 0.25,
                 hard_real_weight: float = 2.0) -> None:
        super().__init__()
        self.K = int(K)
        self.embed_dim = int(embed_dim)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.asymmetric = asymmetric
        self.hard_negative_weight = float(hard_negative_weight)
        protos = fibonacci_spiral(2 * K, embed_dim, seed=seed)
        self.prototypes = nn.Parameter(protos)
        self.rp_enabled = bool(rp_enabled)
        self.lambda_real_margin = float(lambda_real_margin)
        self.real_delta = float(real_delta)
        self.hard_real_topk_frac = float(hard_real_topk_frac)
        self.hard_real_weight = float(hard_real_weight)

    def normalised_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-8)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, *, sample_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = F.normalize(embeddings, dim=-1, eps=1e-8)
        P = self.normalised_prototypes()                              # (2K, D)
        sims = z @ P.t()                                              # (B, 2K)
        K = self.K
        sims_real = sims[:, :K]
        sims_fake = sims[:, K:]
        s_real_max = sims_real.max(dim=-1).values
        s_fake_max = sims_fake.max(dim=-1).values
        if self.asymmetric:
            neg_real = sims_fake.mean(dim=-1)
            neg_fake = sims_real.mean(dim=-1)
        else:
            neg_real = s_fake_max
            neg_fake = s_real_max

        labels = labels.long()
        is_fake = labels.float()
        pos = s_fake_max * is_fake + s_real_max * (1 - is_fake)
        neg = neg_fake * is_fake + neg_real * (1 - is_fake)
        per_sample = -F.log_softmax(
            torch.stack([(pos - self.margin) / self.temperature, neg / self.temperature], dim=-1), dim=-1)[:, 0]
        if sample_mask is not None:
            sample_mask = sample_mask.to(per_sample.dtype)
            denom = sample_mask.sum().clamp(min=1.0)
            ce = (per_sample * sample_mask).sum() / denom
        else:
            ce = per_sample.mean()

        if self.hard_negative_weight > 0:
            sims_zz = z @ z.t() / self.temperature                    # (B, B)
            mask_other = labels.view(-1, 1) != labels.view(1, -1)
            mask_self = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
            sims_zz = sims_zz.masked_fill(~mask_other, float("-inf"))
            sims_zz = sims_zz.masked_fill(mask_self, float("-inf"))
            valid = mask_other.any(dim=-1)
            if sample_mask is not None:
                valid = valid & sample_mask.bool()
            if valid.any():
                hardest = sims_zz[valid].max(dim=-1).values
                pos_v = (pos[valid] - self.margin) / self.temperature
                ce_hard = -F.log_softmax(torch.stack([pos_v, hardest], dim=-1), dim=-1)[:, 0].mean()
                ce = ce + self.hard_negative_weight * ce_hard

        if self.rp_enabled and self.lambda_real_margin > 0.0:
            ce = ce + self.lambda_real_margin * self._real_margin_term(s_real_max, s_fake_max, is_fake, sample_mask)
        return ce

    def _real_margin_term(self, s_real_max: torch.Tensor, s_fake_max: torch.Tensor, is_fake: torch.Tensor,
                          sample_mask: Optional[torch.Tensor]) -> torch.Tensor:
        is_real = 1.0 - is_fake
        if sample_mask is not None:
            is_real = is_real * sample_mask.to(is_real.dtype)
        n_real = is_real.sum()
        if not bool(n_real > 0):
            return s_real_max.new_zeros(())
        hinge = (s_fake_max - s_real_max + self.real_delta).clamp_min(0.0)
        if self.hard_real_topk_frac > 0.0 and self.hard_real_weight > 1.0:
            real_idx = is_real.nonzero(as_tuple=False).flatten()
            if real_idx.numel() > 0:
                margins = (s_real_max - s_fake_max)[real_idx]
                k = max(1, int(round(self.hard_real_topk_frac * margins.numel())))
                k = min(k, margins.numel())
                hard_pos = real_idx[margins.topk(k, largest=False).indices]
                weight = torch.ones_like(hinge)
                weight[hard_pos] = self.hard_real_weight
            else:
                weight = torch.ones_like(hinge)
        else:
            weight = torch.ones_like(hinge)
        weighted = (hinge * is_real * weight).sum()
        norm = (is_real * weight).sum().clamp(min=1.0)
        return weighted / norm


RealnessPreservingMPHC = MultiPrototypeHyperspherical


class FrequencyConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, freq_projection: torch.Tensor, wavelet_feat: torch.Tensor) -> torch.Tensor:
        device_type = freq_projection.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            fp = freq_projection.float()
            wf = wavelet_feat.float()
            fft = torch.fft.rfft2(fp, norm="ortho")
            mag_rgb = fft.abs().mean(dim=1)                            # (B, h, w//2+1)
            h_target, w_target = mag_rgb.shape[-2:]
            wav_pool = F.adaptive_avg_pool2d(wf, (h_target, w_target)).mean(dim=1)

            def _z(x: torch.Tensor) -> torch.Tensor:
                mu = x.mean(dim=(-1, -2), keepdim=True)
                sd = x.std(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
                return (x - mu) / sd

            return F.mse_loss(_z(mag_rgb), _z(wav_pool))


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CombinedLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        focal_cfg = _cfg_get(cfg, "focal", None)
        mphc_cfg = _cfg_get(cfg, "mphc", None)
        freq_cfg = _cfg_get(cfg, "freq", None)
        brier_cfg = _cfg_get(cfg, "brier", None)
        cnd_cfg = _cfg_get(cfg, "cnd", None)

        self.focal = (FocalBCEWithLogits(alpha=focal_cfg.alpha, gamma=focal_cfg.gamma)
                      if focal_cfg is not None and focal_cfg.enabled else None)
        if mphc_cfg is not None and mphc_cfg.enabled:
            self.mphc = MultiPrototypeHyperspherical(embed_dim=mphc_cfg.embed_dim, K=mphc_cfg.K, margin=mphc_cfg.margin,
                                                     temperature=mphc_cfg.temperature, asymmetric=mphc_cfg.asymmetric,
                                                     hard_negative_weight=mphc_cfg.hard_negative_weight,
                                                     rp_enabled=bool(_cfg_get(mphc_cfg, "rp_enabled", False)),
                                                     lambda_real_margin=float(_cfg_get(mphc_cfg, "lambda_real_margin", 0.0)),
                                                     real_delta=float(_cfg_get(mphc_cfg, "real_delta", 0.40)),
                                                     hard_real_topk_frac=float(_cfg_get(mphc_cfg, "hard_real_topk_frac", 0.25)),
                                                     hard_real_weight=float(_cfg_get(mphc_cfg, "hard_real_weight", 2.0)))
        else:
            self.mphc = None
        self.freq = FrequencyConsistencyLoss() if freq_cfg is not None and freq_cfg.enabled else None
        self.brier = (BrierLoss()
            if brier_cfg is not None and bool(_cfg_get(brier_cfg, "enabled", False))
            else None)
        self.cnd_enabled = bool(_cfg_get(cnd_cfg, "enabled", False)) if cnd_cfg is not None else False
        self.cnd_aux_weight = float(_cfg_get(cnd_cfg, "aux_weight", 1.0)) if cnd_cfg is not None else 1.0
        self.cnd_grl_weight = float(_cfg_get(cnd_cfg, "grl_weight", 1.0)) if cnd_cfg is not None else 1.0
        self.orth_enabled = bool(_cfg_get(cnd_cfg, "orth_enabled", False)) if cnd_cfg is not None else False

    @staticmethod
    def _label_for_cls(labels: torch.Tensor, soft_target: Optional[torch.Tensor]) -> torch.Tensor:
        if soft_target is None:
            return labels.float()
        return soft_target.to(labels.device).float()

    def forward(self, outputs: dict, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=labels.device)

        is_soft = outputs.get("is_soft")
        target_soft = outputs.get("target_soft")
        cls_target = self._label_for_cls(labels, target_soft)

        if self.focal is not None:
            l = self.focal(outputs["logits"], cls_target)
            components["focal_bce"] = l.detach()
            total = total + self.cfg.focal.weight * l

        if self.brier is not None:
            l = self.brier(outputs["logits"], cls_target)
            components["brier"] = l.detach()
            brier_w = float(_cfg_get(self.cfg.brier, "weight", 0.05))
            total = total + brier_w * l

        if self.mphc is not None:
            mphc_mask = None
            if is_soft is not None:
                mphc_mask = (1.0 - is_soft.float()).to(outputs["embeddings"].device)
            l = self.mphc(outputs["embeddings"], labels, sample_mask=mphc_mask)
            components["mphc"] = l.detach()
            total = total + self.cfg.mphc.weight * l

        if self.freq is not None and "freq_projection" in outputs:
            l = self.freq(outputs["freq_projection"], outputs["wavelet_feat"])
            components["freq_consistency"] = l.detach()
            total = total + self.cfg.freq.weight * l

        if self.cnd_enabled and "method_logits_zn" in outputs and "method_label" in outputs:
            from wiser.models.disentangle import compute_cnd_losses

            cnd_losses = compute_cnd_losses(method_logits_zn=outputs["method_logits_zn"],
                                            method_logits_zc=outputs.get("method_logits_zc"),
                                            method_label=outputs["method_label"],
                                            z_c=outputs.get("z_c"),
                                            z_n=outputs.get("z_n"),
                                            orth_enabled=self.orth_enabled)
            cnd_w = float(_cfg_get(self.cfg.cnd, "weight", 0.3))
            orth_w = float(_cfg_get(self.cfg.cnd, "orth_weight", 0.05))
            cnd_total = (self.cnd_aux_weight * cnd_losses["aux_zn"] + self.cnd_grl_weight * cnd_losses["aux_zc_grl"])
            components["cnd_aux_zn"] = cnd_losses["aux_zn"].detach()
            components["cnd_aux_zc"] = cnd_losses["aux_zc_grl"].detach()
            total = total + cnd_w * cnd_total
            if self.orth_enabled and "orth" in cnd_losses:
                components["cnd_orth"] = cnd_losses["orth"].detach()
                total = total + orth_w * cnd_losses["orth"]

        components["total"] = total.detach()
        return total, components