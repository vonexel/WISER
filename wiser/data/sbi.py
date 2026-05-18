from __future__ import annotations

import cv2
import torch
import numpy as np
import face_alignment
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SBIParams:
    p: float = 0.5
    feather_sigma_min: float = 1.0
    feather_sigma_max: float = 5.0
    erode_dilate_max: int = 7


_FACE_ALIGN_MODEL: object | None = None


def _try_landmarks(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    global _FACE_ALIGN_MODEL
    try:
        if _FACE_ALIGN_MODEL is None:
            _FACE_ALIGN_MODEL = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")
        preds = _FACE_ALIGN_MODEL.get_landmarks(image_rgb)
        if preds is None:
            return None
        return preds[0].astype(np.float32)
    except Exception:
        return None


def _convex_face_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h * 0.55, w * 0.5
    ry, rx = h * 0.42, w * 0.32
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    return mask.astype(np.float32)


def _polygon_mask(h: int, w: int, points: np.ndarray) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(np.float32) / 255.0


def _morphology(mask: np.ndarray, erode: int, dilate: int) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    if erode:
        mask = cv2.erode(mask, k, iterations=erode)
    if dilate:
        mask = cv2.dilate(mask, k, iterations=dilate)
    return mask


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    ksize = max(3, int(sigma * 3) | 1)
    return cv2.GaussianBlur(mask, (ksize, ksize), sigma)


def _augment_source(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.astype(np.float32)
    out = out * float(rng.uniform(0.85, 1.15))
    out = out + float(rng.uniform(-15.0, 15.0))
    if rng.random() < 0.5:
        ksize = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    h, w = out.shape[:2]
    tx, ty = float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8)


def self_blend(image_rgb: np.ndarray, params: SBIParams, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, bool, np.ndarray]:
    rng = rng or np.random.default_rng()
    if rng.random() >= params.p:
        return image_rgb, False, np.zeros(image_rgb.shape[:2], dtype=np.float32)

    h, w = image_rgb.shape[:2]
    landmarks = _try_landmarks(image_rgb)
    base_mask = _polygon_mask(h, w, landmarks) if landmarks is not None else _convex_face_mask(h, w)

    erode = int(rng.integers(0, params.erode_dilate_max + 1))
    dilate = int(rng.integers(0, params.erode_dilate_max + 1))
    base_mask = _morphology(base_mask, erode, dilate)
    sigma = float(rng.uniform(params.feather_sigma_min, params.feather_sigma_max))
    soft_mask = _feather(base_mask, sigma)

    pseudo_source = _augment_source(image_rgb, rng)
    if rng.random() < 0.3:
        pseudo_target = np.clip(image_rgb.astype(np.float32) * float(rng.uniform(0.9, 1.1)), 0, 255).astype(np.uint8)
    else:
        pseudo_target = image_rgb
    m = soft_mask[..., None]
    out = m * pseudo_source.astype(np.float32) + (1.0 - m) * pseudo_target.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), True, soft_mask


def self_blend_torch(image_rgb: torch.Tensor, params: SBIParams, rng: Optional[np.random.Generator] = None) -> tuple[torch.Tensor, bool, torch.Tensor]:
    arr = (image_rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    out, triggered, mask = self_blend(arr, params, rng=rng)
    out_t = torch.from_numpy(out).permute(2, 0, 1).float() / 255.0
    return out_t, triggered, torch.from_numpy(mask).float()