from __future__ import annotations

import math
import cv2
import torch
import numpy as np
from typing import Any
from wiser.configs.schemas import AugmentConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _albumentations_train(cfg: AugmentConfig, image_size: int) -> Any:
    return A.Compose([A.HorizontalFlip(p=cfg.horizontal_flip),
                      A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=cfg.color_jitter),
                      A.ImageCompression(quality_range=(70, 100), p=cfg.jpeg_compression),
                      # var_limit=(5,30) on 0-255 scale → std_range on [0,1] scale
                      A.GaussNoise(std_range=(math.sqrt(5) / 255.0, math.sqrt(30) / 255.0), p=cfg.gauss_noise),
                      A.GaussianBlur(blur_limit=(3, 5), p=cfg.gauss_blur),
                      A.RandomBrightnessContrast(p=cfg.brightness_contrast),
                      A.Affine(translate_percent=(-0.05, 0.05),
                               scale=(0.95, 1.05),
                               rotate=(-5, 5),
                               p=cfg.shift_scale_rotate),
                      A.Resize(image_size, image_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                      ToTensorV2()])


def _albumentations_eval(image_size: int) -> Any:
    return A.Compose([A.Resize(image_size, image_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                      ToTensorV2()])


def build_train_pipeline(cfg: AugmentConfig, image_size: int = 256):
    try:
        pipeline = _albumentations_train(cfg, image_size)
    except ImportError:
        pipeline = None

    if pipeline is None:
        return _numpy_train_fallback(image_size)

    def _apply(img: np.ndarray) -> torch.Tensor:
        return pipeline(image=img)["image"]
    return _apply


def build_eval_pipeline(image_size: int = 256):
    try:
        pipeline = _albumentations_eval(image_size)
    except ImportError:
        pipeline = None

    if pipeline is None:
        return _numpy_eval_fallback(image_size)

    def _apply(img: np.ndarray) -> torch.Tensor:
        return pipeline(image=img)["image"]
    return _apply


def _numpy_train_fallback(image_size: int):
    def _apply(img: np.ndarray) -> torch.Tensor:
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if np.random.random() < 0.5:
            img = img[:, ::-1]
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
    return _apply


def _numpy_eval_fallback(image_size: int):
    def _apply(img: np.ndarray) -> torch.Tensor:
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
    return _apply


def denormalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(x)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(x)
    return (x * std + mean).clamp(0.0, 1.0)