from __future__ import annotations


import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass, field


LL_IDX = (0, 4, 8)
LH_IDX = (1, 5, 9)
HL_IDX = (2, 6, 10)
HH_IDX = (3, 7, 11)
HIGH_BAND_GROUPS = (LH_IDX, HL_IDX, HH_IDX)
HIGH_BAND_NAMES = ("LH", "HL", "HH")


def haar_dwt_batch(rgb01: torch.Tensor) -> torch.Tensor:
    if rgb01.dim() != 4 or rgb01.size(1) != 3:
        raise ValueError(f"expected (B,3,H,W), got {tuple(rgb01.shape)}")
    a = rgb01[:, :, 0::2, 0::2]
    b = rgb01[:, :, 0::2, 1::2]
    c = rgb01[:, :, 1::2, 0::2]
    d = rgb01[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    # Interleave channels in [LL_R, LH_R, HL_R, HH_R, LL_G, ...] order.
    out = torch.empty(rgb01.size(0), 12, rgb01.size(2) // 2, rgb01.size(3) // 2, dtype=rgb01.dtype, device=rgb01.device)
    for c_idx in range(3):
        out[:, c_idx * 4 + 0] = ll[:, c_idx]
        out[:, c_idx * 4 + 1] = lh[:, c_idx]
        out[:, c_idx * 4 + 2] = hl[:, c_idx]
        out[:, c_idx * 4 + 3] = hh[:, c_idx]
    return out


def haar_idwt_batch(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() != 4 or wav.size(1) != 12:
        raise ValueError(f"expected (B,12,h,w), got {tuple(wav.shape)}")
    b_, _, hs, ws = wav.shape
    out = torch.empty(b_, 3, hs * 2, ws * 2, dtype=wav.dtype, device=wav.device)
    for c_idx in range(3):
        ll = wav[:, c_idx * 4 + 0]
        lh = wav[:, c_idx * 4 + 1]
        hl = wav[:, c_idx * 4 + 2]
        hh = wav[:, c_idx * 4 + 3]
        out[:, c_idx, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        out[:, c_idx, 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
        out[:, c_idx, 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
        out[:, c_idx, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out.clamp(0.0, 1.0)


@dataclass(slots=True)
class AWBSBIv2Config:
    p_apply: float = 0.5
    band_temperature: float = 0.7
    min_strength: float = 0.05
    max_strength: float = 0.35
    strength_step: float = 0.02
    init_strength: float = 0.20

    hardness_enabled: bool = True
    ema_decay: float = 0.95
    hard_fake_threshold: float = 0.45
    false_real_threshold: float = 0.65

    soft_label_min: float = 0.55
    soft_label_max: float = 0.95

    real_guard_enabled: bool = True
    real_guard_cooldown_steps: int = 100
    real_guard_strength_multiplier: float = 0.75

    # Identity preservation
    perturb_ll: bool = False
    alpha_ll: float = 0.0


class AWBSBIHardnessController(nn.Module):
    def __init__(self, cfg: AWBSBIv2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("band_logits", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("strength", torch.tensor(float(cfg.init_strength), dtype=torch.float32))
        self.register_buffer("real_guard_steps_left", torch.tensor(0, dtype=torch.long))
        self.register_buffer("hard_pseudo_rate", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("real_guard_active", torch.tensor(0.0, dtype=torch.float32))

    def band_probs(self) -> torch.Tensor:
        return F.softmax(self.band_logits / max(self.cfg.band_temperature, 1e-3), dim=-1)

    def current_strength(self) -> torch.Tensor:
        s = self.strength
        if bool(self.real_guard_steps_left.item() > 0):
            s = s * self.cfg.real_guard_strength_multiplier
        return s.clamp(self.cfg.min_strength, self.cfg.max_strength)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, pseudo_mask: torch.Tensor, used_band_weights: torch.Tensor) -> dict[str, float]:
        cfg = self.cfg
        if not cfg.hardness_enabled:
            return {"band_prob_LH": float(self.band_probs()[0].item()),
                    "band_prob_HL": float(self.band_probs()[1].item()),
                    "band_prob_HH": float(self.band_probs()[2].item()),
                    "strength": float(self.current_strength().item()),
                    "hard_pseudo_rate": float(self.hard_pseudo_rate.item()),
                    "real_guard_active": float(self.real_guard_active.item())}

        if int(self.real_guard_steps_left.item()) > 0:
            self.real_guard_steps_left -= 1
        self.real_guard_active.fill_(1.0 if int(self.real_guard_steps_left.item()) > 0 else 0.0)

        prob_fake = torch.sigmoid(logits.float())
        pseudo_mask = pseudo_mask.bool()
        hard_pseudo_rate = 0.0
        if pseudo_mask.any():
            p = prob_fake[pseudo_mask]
            wb = used_band_weights[pseudo_mask].float()
            hard = p < cfg.hard_fake_threshold
            hard_pseudo_rate = float(hard.float().mean().item())
            if hard.any():
                bump = wb[hard].mean(dim=0)
                self.band_logits.mul_(cfg.ema_decay).add_((1.0 - cfg.ema_decay) * bump)
                self.strength.add_(cfg.strength_step)
            else:
                self.strength.sub_(cfg.strength_step * 0.5)
            self.strength.clamp_(cfg.min_strength, cfg.max_strength)
        self.hard_pseudo_rate.fill_(float(hard_pseudo_rate))

        if cfg.real_guard_enabled:
            is_real = (labels == 0) & (~pseudo_mask)
            if is_real.any():
                false_real = prob_fake[is_real] > cfg.false_real_threshold
                if false_real.any():
                    self.real_guard_steps_left.fill_(int(cfg.real_guard_cooldown_steps))
                    self.real_guard_active.fill_(1.0)
        return {"band_prob_LH": float(self.band_probs()[0].item()),
                "band_prob_HL": float(self.band_probs()[1].item()),
                "band_prob_HH": float(self.band_probs()[2].item()),
                "strength": float(self.current_strength().item()),
                "hard_pseudo_rate": float(self.hard_pseudo_rate.item()),
                "real_guard_active": float(self.real_guard_active.item())}


class AdaptiveWaveletBandSBI(nn.Module):
    def __init__(self, cfg: AWBSBIv2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.controller = AWBSBIHardnessController(cfg)
        for p in self.parameters():
            if p.requires_grad:
                raise RuntimeError("AdaptiveWaveletBandSBI must have zero trainable parameters.")

    def band_probs(self) -> torch.Tensor:
        return self.controller.band_probs()

    def current_strength(self) -> torch.Tensor:
        return self.controller.current_strength()

    def forward(self, batch: dict, *, generator: Optional[torch.Generator] = None) -> tuple[dict, dict]:
        meta = {"awb_v2_enabled": True, "band_weights": None,
                "perturb_strength": float(self.current_strength().item()), "soft_label": None,
                "hardness_score": None, "pseudo_mask": None}
        active = batch.get("awb_v2_active")
        wav_src = batch.get("awb_v2_wav_src")
        mask = batch.get("awb_v2_mask")
        if active is None or wav_src is None or mask is None:
            meta["awb_v2_enabled"] = False
            return batch, meta
        active = active.bool()
        if not active.any():
            B = batch["rgb"].size(0)
            meta["pseudo_mask"] = torch.zeros(B, dtype=torch.bool, device=batch["rgb"].device)
            meta["band_weights"] = torch.zeros(B, 3, device=batch["rgb"].device)
            meta["soft_label"] = torch.zeros(B, device=batch["rgb"].device)
            meta["hardness_score"] = 0.0
            return batch, meta

        device = batch["rgb"].device
        wav_tgt = batch["wavelet"].to(device)
        wav_src = wav_src.to(device)
        mask = mask.to(device)
        B = wav_tgt.size(0)
        band_probs = self.band_probs().to(device)
        if generator is not None:
            base = band_probs.unsqueeze(0).expand(B, -1)
            eps = torch.rand((B, 3), device=device, generator=generator) * 0.5
        else:
            base = band_probs.unsqueeze(0).expand(B, -1)
            eps = torch.rand((B, 3), device=device) * 0.5
        weights = base * (1.0 - 0.5) + eps
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        strength = self.current_strength().to(device)
        per_sample_strength = strength * active.float()
        wav_out = wav_tgt.clone()
        for b_idx, ch_idx in enumerate(HIGH_BAND_GROUPS):
            alpha = (per_sample_strength * weights[:, b_idx]).view(B, 1, 1, 1)
            mix = (alpha * mask).clamp(0.0, 1.0)
            for c_idx in ch_idx:
                wav_out[:, c_idx : c_idx + 1] = ((1.0 - mix) * wav_tgt[:, c_idx : c_idx + 1] + mix * wav_src[:, c_idx : c_idx + 1])

        pseudo_rgb01 = haar_idwt_batch(wav_out)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=pseudo_rgb01.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=pseudo_rgb01.dtype).view(1, 3, 1, 1)
        pseudo_rgb_norm = (pseudo_rgb01 - mean) / std

        denom = max(self.cfg.max_strength - self.cfg.min_strength, 1e-6)
        norm_s = ((per_sample_strength - self.cfg.min_strength) / denom).clamp(0.0, 1.0)
        soft = (self.cfg.soft_label_min + 0.40 * norm_s).clamp(self.cfg.soft_label_min, self.cfg.soft_label_max)
        active_f = active.float().view(B, 1, 1, 1).to(pseudo_rgb_norm.dtype)
        rgb_in = batch["rgb"].to(device).to(pseudo_rgb_norm.dtype)
        rgb_out = active_f * pseudo_rgb_norm + (1.0 - active_f) * rgb_in
        active_f_w = active.float().view(B, 1, 1, 1).to(wav_out.dtype)
        wav_in = wav_tgt
        wav_final = active_f_w * wav_out + (1.0 - active_f_w) * wav_in


        labels = batch["label"].to(device)
        new_labels = torch.where(active, torch.ones_like(labels), labels)
        old_soft = batch.get("target_soft")
        if old_soft is None:
            old_soft = labels.float()
        old_soft = old_soft.to(device).float()
        new_soft = torch.where(active, soft.to(old_soft.dtype), old_soft)
        old_is_soft = batch.get("is_soft")
        if old_is_soft is None:
            old_is_soft = torch.zeros_like(labels, dtype=torch.bool)
        old_is_soft = old_is_soft.to(device).bool()
        new_is_soft = old_is_soft | active

        batch_out = dict(batch)
        batch_out["rgb"] = rgb_out
        batch_out["wavelet"] = wav_final
        batch_out["label"] = new_labels
        batch_out["target_soft"] = new_soft
        batch_out["is_soft"] = new_is_soft

        meta["band_weights"] = weights.detach()
        meta["pseudo_mask"] = active.detach()
        meta["soft_label"] = soft.detach()
        meta["perturb_strength"] = float(strength.item())
        meta["hardness_score"] = float(self.controller.hard_pseudo_rate.item())
        return batch_out, meta

    @torch.no_grad()
    def update_controller(self, logits: torch.Tensor, labels: torch.Tensor, metadata: dict) -> dict[str, float]:
        if not metadata.get("awb_v2_enabled", False):
            return {}
        pseudo_mask = metadata.get("pseudo_mask")
        wb = metadata.get("band_weights")
        if pseudo_mask is None or wb is None:
            return {}
        return self.controller.update(logits=logits, labels=labels, pseudo_mask=pseudo_mask.to(logits.device), used_band_weights=wb.to(logits.device))


def awb_sbi_v2_config_from_omegaconf(cfg) -> AWBSBIv2Config:
    awb_v2 = getattr(cfg, "awb_v2", None) or getattr(cfg, "awb_sbi_v2", None) or cfg

    def _get(node, key: str, default):
        if node is None:
            return default
        if hasattr(node, "get"):
            return node.get(key, default)
        return getattr(node, key, default)

    bands = _get(awb_v2, "bands", None)
    strength = _get(awb_v2, "strength", None)
    hardness = _get(awb_v2, "hardness", None)
    soft = _get(awb_v2, "soft_labels", None)
    guard = _get(awb_v2, "real_preservation_guard", None)

    return AWBSBIv2Config(p_apply=float(_get(awb_v2, "p_apply", 0.5)),
                          band_temperature=float(_get(bands, "band_temperature", 0.7)),
                          min_strength=float(_get(strength, "min_strength", 0.05)),
                          max_strength=float(_get(strength, "max_strength", 0.35)),
                          strength_step=float(_get(strength, "strength_step", 0.02)),
                          init_strength=float(_get(strength, "init_strength", 0.20)),
                          hardness_enabled=bool(_get(hardness, "enabled", True)),
                          ema_decay=float(_get(hardness, "ema_decay", 0.95)),
                          hard_fake_threshold=float(_get(hardness, "hard_fake_threshold", 0.45)),
                          false_real_threshold=float(_get(hardness, "false_real_threshold", 0.65)),
                          soft_label_min=float(_get(soft, "min_label", 0.55)),
                          soft_label_max=float(_get(soft, "max_label", 0.95)),
                          real_guard_enabled=bool(_get(guard, "enabled", True)),
                          real_guard_cooldown_steps=int(_get(guard, "cooldown_steps", 100)),
                          real_guard_strength_multiplier=float(_get(guard, "strength_multiplier", 0.75)),
                          perturb_ll=bool(_get(bands, "perturb_ll", False)))


__all__ = ["AWBSBIv2Config", "AWBSBIHardnessController", "AdaptiveWaveletBandSBI",
           "HIGH_BAND_GROUPS", "HIGH_BAND_NAMES", "LL_IDX", "LH_IDX", "HL_IDX",
           "HH_IDX", "awb_sbi_v2_config_from_omegaconf", "haar_dwt_batch", "haar_idwt_batch"]