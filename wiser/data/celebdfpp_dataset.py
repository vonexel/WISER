from __future__ import annotations

import cv2
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from wiser.configs.schemas import DataConfig
from wiser.data.augmentation import build_eval_pipeline


def _category_of(label: str) -> str:
    name = label.lower()
    if name.startswith("fs") or name == "celeb-synthesis":
        return "FaceSwap"
    if name.startswith("fr"):
        return "FaceReenactment"
    if name.startswith("tf"):
        return "TalkingFace"
    return "FaceSwap"


@dataclass(slots=True)
class CelebRecord:
    label: str
    video: str
    frame_idx: int
    label_int: int
    category: str


class CelebDFPPDataset(Dataset):
    def __init__(self, cache_root: str | Path, data_cfg: DataConfig) -> None:
        self.cache_root = Path(cache_root) / "celebdfpp"
        self.data_cfg = data_cfg
        self.image_size = data_cfg.img_size
        self.transform = build_eval_pipeline(self.image_size)
        face_root = self.cache_root / "face_crops"
        if not face_root.is_dir():
            raise RuntimeError(f"No Celeb-DF++ cache at {face_root}. Run scripts/preprocess_celebdfpp.py first")
        self.records: list[CelebRecord] = []
        for label_dir in sorted(face_root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            label_int = 0 if label.lower().endswith("real") else 1
            category = _category_of(label)
            for vdir in sorted(label_dir.iterdir()):
                if not vdir.is_dir():
                    continue
                for f in sorted(vdir.glob("frame_*.jpg")):
                    idx = int(f.stem.split("_")[1])
                    self.records.append(CelebRecord(label, vdir.name, idx, label_int, category))
        if not self.records:
            raise RuntimeError(f"No Celeb-DF++ frames found under {face_root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        crop = cv2.imread(
            str(self.cache_root / "face_crops" / r.label / r.video / f"frame_{r.frame_idx:06d}.jpg"),
            cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb_t = self.transform(rgb).float()
        wav = np.load(self.cache_root / "wavelet_l1" / r.label / r.video / f"frame_{r.frame_idx:06d}.npy").astype(np.float32)
        defocus = np.load(self.cache_root / "defocus" / r.label / r.video / f"frame_{r.frame_idx:06d}.npy").astype(np.float32)
        if defocus.ndim == 2:
            defocus = defocus[None, ...]
        return {"rgb": rgb_t,
                "wavelet": torch.from_numpy(np.ascontiguousarray(wav)).float(),
                "defocus": torch.from_numpy(np.ascontiguousarray(defocus)).float(),
                "label": torch.tensor(r.label_int, dtype=torch.long),
                "manip": r.label,
                "sbi": False,
                "video_id": f"{r.label}/{r.video}",
                "category": r.category}