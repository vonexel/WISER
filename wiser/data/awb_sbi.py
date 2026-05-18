from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from wiser.data.sbi import (SBIParams,
                            _augment_source,
                            _convex_face_mask,
                            _feather,
                            _morphology,
                            _polygon_mask, _try_landmarks)


@dataclass(slots=True)
class AWBSBIParams:
    p: float = 0.5
    feather_sigma_min: float = 1.0
    feather_sigma_max: float = 5.0
    erode_dilate_max: int = 7
    alpha_ll: float = 0.15
    energy_boost: float = 1.5
    soft_label: float = 0.7


def _haar_dwt2d_chw(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    C, H, W = img.shape
    H2, W2 = H - (H % 2), W - (W % 2)
    img = img[:, :H2, :W2]
    a = img[:, 0::2, 0::2]
    b = img[:, 0::2, 1::2]
    c = img[:, 1::2, 0::2]
    d = img[:, 1::2, 1::2]
    LL = (a + b + c + d) * 0.5
    LH = (a + b - c - d) * 0.5
    HL = (a - b + c - d) * 0.5
    HH = (a - b - c + d) * 0.5
    return LL, LH, HL, HH


def _haar_idwt2d_chw(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_haar_dwt2d_chw`. Returns ``(C, 2*Hs, 2*Ws)``"""
    C, Hs, Ws = LL.shape
    out = np.empty((C, Hs * 2, Ws * 2), dtype=LL.dtype)
    out[:, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
    out[:, 0::2, 1::2] = (LL + LH - HL - HH) * 0.5
    out[:, 1::2, 0::2] = (LL - LH + HL - HH) * 0.5
    out[:, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5
    return out


def _gradient_magnitude(rgb_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sx * sx + sy * sy)


def awb_self_blend(image_rgb: np.ndarray, params: AWBSBIParams,
                   rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, bool, np.ndarray, float]:
    rng = rng or np.random.default_rng()
    if rng.random() >= params.p:
        zero_mask = np.zeros(image_rgb.shape[:2], dtype=np.float32)
        return image_rgb, False, zero_mask, 0.0

    h, w = image_rgb.shape[:2]
    landmarks = _try_landmarks(image_rgb)
    base_mask = _polygon_mask(h, w, landmarks) if landmarks is not None else _convex_face_mask(h, w)
    erode = int(rng.integers(0, params.erode_dilate_max + 1))
    dilate = int(rng.integers(0, params.erode_dilate_max + 1))
    base_mask = _morphology(base_mask, erode, dilate)
    sigma = float(rng.uniform(params.feather_sigma_min, params.feather_sigma_max))
    soft_mask = _feather(base_mask, sigma)
    grad = _gradient_magnitude(image_rgb)
    high_grad = grad > grad.mean()
    boost_full = np.where(high_grad, params.energy_boost, 1.0).astype(np.float32)
    soft_mask_boosted = np.clip(soft_mask * boost_full, 0.0, 1.0)
    pseudo_source = _augment_source(image_rgb, rng)
    if rng.random() < 0.3:
        pseudo_target = np.clip(
            image_rgb.astype(np.float32) * float(rng.uniform(0.9, 1.1)), 0, 255
        ).astype(np.uint8)
    else:
        pseudo_target = image_rgb
    Is = pseudo_source.astype(np.float32) / 255.0
    It = pseudo_target.astype(np.float32) / 255.0
    Is_chw = np.ascontiguousarray(Is.transpose(2, 0, 1))
    It_chw = np.ascontiguousarray(It.transpose(2, 0, 1))

    LL_s, LH_s, HL_s, HH_s = _haar_dwt2d_chw(Is_chw)
    LL_t, LH_t, HL_t, HH_t = _haar_dwt2d_chw(It_chw)

    Hs, Ws = LL_s.shape[1], LL_s.shape[2]
    M_high = cv2.resize(soft_mask_boosted, (Ws, Hs), interpolation=cv2.INTER_AREA)
    M_high = np.clip(M_high, 0.0, 1.0)[None, :, :]                  # (1, Hs, Ws)

    alpha = float(params.alpha_ll)
    LL_out = (1.0 - alpha) * LL_t + alpha * LL_s
    LH_out = M_high * LH_s + (1.0 - M_high) * LH_t
    HL_out = M_high * HL_s + (1.0 - M_high) * HL_t
    HH_out = M_high * HH_s + (1.0 - M_high) * HH_t

    blend_chw = _haar_idwt2d_chw(LL_out, LH_out, HL_out, HH_out)
    blend = np.clip(blend_chw.transpose(1, 2, 0), 0.0, 1.0)
    out = (blend * 255.0).astype(np.uint8)
    if out.shape != image_rgb.shape:
        full = pseudo_target.copy()
        full[: out.shape[0], : out.shape[1]] = out
        out = full
    return out, True, soft_mask, float(params.soft_label)


def awb_sbi_params_from_config(cfg) -> AWBSBIParams:
    sbi_cfg = getattr(cfg, "sbi", None) or cfg
    return AWBSBIParams(p=float(getattr(sbi_cfg, "p_sbi", 0.5)),
                        feather_sigma_min=float(getattr(sbi_cfg, "feather_sigma_min", 1.0)),
                        feather_sigma_max=float(getattr(sbi_cfg, "feather_sigma_max", 5.0)),
                        alpha_ll=float(getattr(sbi_cfg, "alpha_ll", 0.15)),
                        energy_boost=float(getattr(sbi_cfg, "energy_boost", 1.5)),
                        soft_label=float(getattr(sbi_cfg, "soft_label", 0.7)))


__all__ = ["AWBSBIParams",
           "awb_self_blend",
           "awb_sbi_params_from_config",
           "SBIParams"]