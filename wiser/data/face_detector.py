from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN
from typing import Iterable, Optional




class FaceDetector:
    def __init__(self, device: str = "cuda", image_size: int = 256, margin_frac: float = 0.30, select_largest: bool = True) -> None:
        self.image_size = image_size
        self.margin_frac = margin_frac
        self.device = device
        self._mtcnn = None
        self._haar = None
        self.select_largest = select_largest

    def _lazy_load_mtcnn(self):
        if self._mtcnn is None:
            self._mtcnn = MTCNN( image_size=self.image_size, margin=int(self.image_size * self.margin_frac), keep_all=False,
                                 select_largest=self.select_largest, post_process=False, device=self.device)
        return self._mtcnn

    def _lazy_load_haar(self):
        if self._haar is None:
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return self._haar

    def detect(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            mtcnn = self._lazy_load_mtcnn()
            pil = Image.fromarray(image_rgb)
            crop = mtcnn(pil)
            if crop is not None:
                arr = crop.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                return arr
        except Exception:
            pass
        return self._haar_fallback(image_rgb)

    def _haar_fallback(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            haar = self._lazy_load_haar()
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(64, 64))
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            margin = int(max(w, h) * self.margin_frac)
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(image_rgb.shape[1], x + w + margin)
            y1 = min(image_rgb.shape[0], y + h + margin)
            crop = image_rgb[y0:y1, x0:x1]
            return cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

    def detect_many(self, frames: Iterable[np.ndarray]) -> list[Optional[np.ndarray]]:
        return [self.detect(f) for f in frames]


def crop_path(cache_root: Path, dataset: str, label: str, video: str, idx: int) -> Path:
    return cache_root / dataset / "face_crops" / label / video / f"frame_{idx:06d}.jpg"