from __future__ import annotations


import cv2
import torch
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional
from torch.utils.data import Dataset, Sampler
from wiser.configs.schemas import AugmentConfig, DataConfig
from wiser.data.augmentation import build_eval_pipeline, build_train_pipeline
from wiser.data.awb_sbi import AWBSBIParams, awb_self_blend
from wiser.data.arca import ARCAParams, heavy_real_augment
from wiser.data.freq_mask import apply_freq_mask
from wiser.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from wiser.data.preprocessing import discover_frames, haar_dwt_stack
from wiser.data.sbi import (SBIParams, _augment_source, _convex_face_mask, _feather, _morphology,
                            _polygon_mask, _try_landmarks, self_blend)
from wiser.models.disentangle import manip_to_method_label


def _read_split(splits_path: Path) -> dict:
    with open(splits_path) as f:
        return json.load(f)


@dataclass(slots=True)
class FrameRecord:
    label: str
    video: str
    frame_idx: int
    label_int: int        # 0 real, 1 fake


class FFPPFrameDataset(Dataset):
    def __init__(self, cache_root: str | Path, splits_path: str | Path, split: str, data_cfg: DataConfig, augment_cfg: AugmentConfig,
                 train: bool) -> None:
        self.cache_root = Path(cache_root) / "ffpp"
        self.splits = _read_split(Path(splits_path))
        self.split = split
        self.train = train
        self.data_cfg = data_cfg
        self.augment_cfg = augment_cfg
        self.image_size = data_cfg.img_size
        self.wavelet_size = data_cfg.wavelet_size
        rows = discover_frames(Path(cache_root), "ffpp", self.splits, split)
        self.records: list[FrameRecord] = [FrameRecord(*r) for r in rows]
        if not self.records:
            raise RuntimeError(
                f"No frames discovered for split={split}. Run scripts/preprocess_ffpp.py first"
            )
        self.transform = (
            build_train_pipeline(augment_cfg, self.image_size)
            if train
            else build_eval_pipeline(self.image_size)
        )
        sbi_block = augment_cfg.sbi
        sbi_enabled = bool(getattr(sbi_block, "enabled", False))
        sbi_p = float(getattr(sbi_block, "p_sbi", 0.0))
        feather_min = float(getattr(sbi_block, "feather_sigma_min", 1.0))
        feather_max = float(getattr(sbi_block, "feather_sigma_max", 5.0))
        self.sbi_params = SBIParams(
            p=sbi_p if sbi_enabled and train else 0.0,
            feather_sigma_min=feather_min,
            feather_sigma_max=feather_max,
        )
        self.use_awb_sbi = bool(getattr(augment_cfg, "use_awb_sbi", False)) and train
        self.awb_params = AWBSBIParams(
            p=sbi_p if sbi_enabled and self.use_awb_sbi else 0.0,
            feather_sigma_min=feather_min,
            feather_sigma_max=feather_max,
            alpha_ll=float(getattr(sbi_block, "alpha_ll", 0.15)),
            energy_boost=float(getattr(sbi_block, "energy_boost", 1.5)),
            soft_label=float(getattr(sbi_block, "soft_label", 1.0)))
        arca_cfg = getattr(augment_cfg, "arca", None)
        self.arca_enabled = bool(getattr(arca_cfg, "enabled", False)) and train
        self.arca_p_real = float(getattr(arca_cfg, "p_apply_real", 0.5)) if arca_cfg is not None else 0.0
        self.arca_params = ARCAParams(
            enabled=self.arca_enabled,
            p_apply=1.0,
            jpeg_qf_min=int(getattr(arca_cfg, "jpeg_qf_min", 30)) if arca_cfg is not None else 30,
            jpeg_qf_max=int(getattr(arca_cfg, "jpeg_qf_max", 60)) if arca_cfg is not None else 60,
            noise_sigma_max=float(
                getattr(arca_cfg, "noise_sigma_max", 8.0 / 255.0)
            ) if arca_cfg is not None else 8.0 / 255.0,
            median_blur_kernel=int(getattr(arca_cfg, "median_blur_kernel", 3))
            if arca_cfg is not None
            else 3,
            gamma_min=float(getattr(arca_cfg, "gamma_min", 0.7)) if arca_cfg is not None else 0.7,
            gamma_max=float(getattr(arca_cfg, "gamma_max", 1.3)) if arca_cfg is not None else 1.3)
        self.freqmask_p = augment_cfg.freqmask.p_freqmask if augment_cfg.freqmask.enabled and train else 0.0

        awb_v2_cfg = getattr(augment_cfg, "awb_v2", None)
        self.awb_v2_enabled = bool(getattr(awb_v2_cfg, "enabled", False)) and train
        self.awb_v2_p_apply = float(getattr(awb_v2_cfg, "p_apply", 0.0)) if awb_v2_cfg is not None else 0.0
        self.wavelet_size = data_cfg.wavelet_size

    def __len__(self) -> int:
        return len(self.records)

    def _crop_path(self, r: FrameRecord) -> Path:
        return self.cache_root / "face_crops" / r.label / r.video / f"frame_{r.frame_idx:06d}.jpg"

    def _wav_path(self, r: FrameRecord) -> Path:
        return self.cache_root / "wavelet_l1" / r.label / r.video / f"frame_{r.frame_idx:06d}.npy"

    def _def_path(self, r: FrameRecord) -> Path:
        return self.cache_root / "defocus" / r.label / r.video / f"frame_{r.frame_idx:06d}.npy"

    def _load_rgb(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        rgb_uint8 = self._load_rgb(self._crop_path(r))
        sbi_triggered = False
        manip = r.label
        label_int = r.label_int
        target_soft = float(label_int)
        is_soft = False
        awb_v2_active = False
        awb_v2_wav_src: Optional[np.ndarray] = None
        awb_v2_mask: Optional[np.ndarray] = None

        if self.train and r.label == "real":
            if self.awb_v2_enabled and self.awb_v2_p_apply > 0.0 and np.random.random() < self.awb_v2_p_apply:
                awb_v2_active, awb_v2_wav_src, awb_v2_mask = self._prepare_awb_v2(rgb_uint8)
                if awb_v2_active:
                    manip = "sbi_pseudo"
            elif self.use_awb_sbi and self.awb_params.p > 0.0:
                rgb_uint8, sbi_triggered, _, soft_label = awb_self_blend(rgb_uint8, self.awb_params)
                if sbi_triggered:
                    manip = "sbi_pseudo"
                    label_int = 1
                    target_soft = float(soft_label)
                    is_soft = soft_label < 1.0
            elif self.sbi_params.p > 0.0:
                rgb_uint8, sbi_triggered, _ = self_blend(rgb_uint8, self.sbi_params)
                if sbi_triggered:
                    manip = "sbi_pseudo"
                    label_int = 1
                    target_soft = 1.0

            if (not sbi_triggered and not awb_v2_active and self.arca_enabled
                    and self.arca_p_real > 0.0 and np.random.random() < self.arca_p_real):
                rgb_uint8 = heavy_real_augment(rgb_uint8, self.arca_params)

        rgb_t = self.transform(rgb_uint8)

        if self.freqmask_p > 0.0:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            rgb01 = (rgb_t * std + mean).clamp(0, 1)
            rgb01 = apply_freq_mask(
                rgb01,
                p=self.freqmask_p,
                min_size_frac=self.augment_cfg.freqmask.min_size_frac,
                max_size_frac=self.augment_cfg.freqmask.max_size_frac,
                high_freq_bias=self.augment_cfg.freqmask.high_freq_bias)
            rgb_t = (rgb01 - mean) / std

        if sbi_triggered:
            wav = haar_dwt_stack(rgb_uint8).astype(np.float32)
        else:
            wav = np.load(self._wav_path(r)).astype(np.float32)
        defocus = np.load(self._def_path(r)).astype(np.float32)
        if defocus.ndim == 2:
            defocus = defocus[None, ...]

        method_label = manip_to_method_label(manip)

        wavelet_t = torch.from_numpy(np.ascontiguousarray(wav)).float()

        item = {"rgb": rgb_t.float(),
                "wavelet": wavelet_t,
                "defocus": torch.from_numpy(np.ascontiguousarray(defocus)).float(),
                "label": torch.tensor(label_int, dtype=torch.long),
                "target_soft": torch.tensor(target_soft, dtype=torch.float32),
                "is_soft": torch.tensor(is_soft, dtype=torch.bool),
                "method_label": torch.tensor(method_label, dtype=torch.long),
                "manip": manip,
                "sbi": sbi_triggered,
                "video_id": f"{r.label}/{r.video}"}

        if self.awb_v2_enabled:
            if awb_v2_active and awb_v2_wav_src is not None and awb_v2_mask is not None:
                wav_src_t = torch.from_numpy(np.ascontiguousarray(awb_v2_wav_src)).float()
                mask_t = torch.from_numpy(np.ascontiguousarray(awb_v2_mask)).float().unsqueeze(0)
            else:
                wav_src_t = torch.zeros_like(wavelet_t)
                mask_t = torch.zeros(1, wavelet_t.shape[1], wavelet_t.shape[2], dtype=torch.float32)
            item["awb_v2_active"] = torch.tensor(awb_v2_active, dtype=torch.bool)
            item["awb_v2_wav_src"] = wav_src_t
            item["awb_v2_mask"] = mask_t
        return item

    def _prepare_awb_v2(self, rgb_uint8: np.ndarray) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        h, w = rgb_uint8.shape[:2]
        rng = np.random.default_rng()
        landmarks = _try_landmarks(rgb_uint8)
        base_mask = _polygon_mask(h, w, landmarks) if landmarks is not None else _convex_face_mask(h, w)
        erode = int(rng.integers(0, 8))
        dilate = int(rng.integers(0, 8))
        base_mask = _morphology(base_mask, erode, dilate)
        sigma = float(rng.uniform(1.0, 5.0))
        soft_mask = _feather(base_mask, sigma).astype(np.float32)
        ws = self.wavelet_size
        mask_low = cv2.resize(soft_mask, (ws, ws), interpolation=cv2.INTER_AREA)
        mask_low = np.clip(mask_low, 0.0, 1.0).astype(np.float32)
        pseudo_source = _augment_source(rgb_uint8, rng)
        target_size = self.image_size
        if pseudo_source.shape[0] != target_size or pseudo_source.shape[1] != target_size:
            pseudo_source = cv2.resize(pseudo_source, (target_size, target_size), interpolation=cv2.INTER_AREA)
        wav_src = haar_dwt_stack(pseudo_source).astype(np.float32)
        return True, wav_src, mask_low


class PairedRealFakeSampler(Sampler[int]):
    def __init__(self, dataset: FFPPFrameDataset, generator: Optional[torch.Generator] = None, rounds_per_epoch: int = 1):
        self.dataset = dataset
        self.gen = generator or torch.Generator()
        self.rounds_per_epoch = max(1, int(rounds_per_epoch))
        self.real_index: dict[str, list[int]] = {}
        self.fake_index: dict[str, list[int]] = {}
        for i, r in enumerate(dataset.records):
            if r.label == "real":
                self.real_index.setdefault(r.video, []).append(i)
            else:
                target = r.video.split("_")[0]
                self.fake_index.setdefault(target, []).append(i)

    def _per_round(self) -> int:
        return sum(1 + (1 if k in self.fake_index else 0) for k in self.real_index)

    def __iter__(self) -> Iterator[int]:
        real_keys = list(self.real_index.keys())
        for _ in range(self.rounds_per_epoch):
            order = torch.randperm(len(real_keys), generator=self.gen).tolist()
            for k_idx in order:
                target = real_keys[k_idx]
                real_pool = self.real_index[target]
                fake_pool = self.fake_index.get(target, [])
                r = real_pool[int(torch.randint(0, len(real_pool), (1,), generator=self.gen).item())]
                yield r
                if fake_pool:
                    f = fake_pool[int(torch.randint(0, len(fake_pool), (1,), generator=self.gen).item())]
                    yield f

    def __len__(self) -> int:
        return self.rounds_per_epoch * self._per_round()