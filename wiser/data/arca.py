from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class ARCAParams:
    enabled: bool = True
    p_apply: float = 1.0
    jpeg_qf_min: int = 30
    jpeg_qf_max: int = 60
    noise_sigma_max: float = 8.0 / 255.0
    median_blur_kernel: int = 3
    gamma_min: float = 0.7
    gamma_max: float = 1.3
    p_jpeg: float = 0.7
    p_noise: float = 0.5
    p_blur: float = 0.3
    p_gamma: float = 0.5


def heavy_real_augment(image_rgb: np.ndarray, params: ARCAParams, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if not params.enabled:
        return image_rgb
    rng = rng or np.random.default_rng()
    if rng.random() >= params.p_apply:
        return image_rgb

    img = image_rgb.copy()
    if rng.random() < params.p_jpeg:
        qf = int(rng.integers(params.jpeg_qf_min, params.jpeg_qf_max + 1))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    if rng.random() < params.p_noise:
        sigma = float(rng.uniform(0.0, params.noise_sigma_max))
        if sigma > 0.0:
            noise = rng.normal(0.0, sigma * 255.0, size=img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < params.p_blur:
        k = int(params.median_blur_kernel)
        if k >= 3 and k % 2 == 1:
            img = cv2.medianBlur(img, k)

    if rng.random() < params.p_gamma:
        gamma = float(rng.uniform(params.gamma_min, params.gamma_max))
        lut = np.clip(255.0 * (np.linspace(0, 1, 256) ** (1.0 / max(1e-3, gamma))), 0, 255).astype(np.uint8)
        img = lut[img]
    return img


def arca_params_from_config(cfg) -> ARCAParams:
    arca_cfg = getattr(cfg, "arca", None) or cfg
    return ARCAParams(enabled=bool(getattr(arca_cfg, "enabled", True)),
                      p_apply=float(getattr(arca_cfg, "p_apply", 1.0)),
                      jpeg_qf_min=int(getattr(arca_cfg, "jpeg_qf_min", 30)),
                      jpeg_qf_max=int(getattr(arca_cfg, "jpeg_qf_max", 60)),
                      noise_sigma_max=float(getattr(arca_cfg, "noise_sigma_max", 8.0 / 255.0)),
                      median_blur_kernel=int(getattr(arca_cfg, "median_blur_kernel", 3)),
                      gamma_min=float(getattr(arca_cfg, "gamma_min", 0.7)),
                      gamma_max=float(getattr(arca_cfg, "gamma_max", 1.3)),
                      p_jpeg=float(getattr(arca_cfg, "p_jpeg", 0.7)),
                      p_noise=float(getattr(arca_cfg, "p_noise", 0.5)),
                      p_blur=float(getattr(arca_cfg, "p_blur", 0.3)),
                      p_gamma=float(getattr(arca_cfg, "p_gamma", 0.5)))