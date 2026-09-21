from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from wiser.utils import save_json
from wiser.utils.logging import get_logger

log = get_logger("wiser.data.preprocessing")


@dataclass(slots=True)
class PreprocessResult:
    video: str
    label: str
    frames_written: int
    frames_failed: int
    total_frames: int


def _open_video(path: Path):
    return cv2.VideoCapture(str(path))


def _video_iter(cap, stride: int, max_frames: int) -> Iterable[np.ndarray]:
    idx = 0
    yielded = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yielded += 1
            if yielded >= max_frames:
                break
        idx += 1


def haar_dwt_stack(image_rgb: np.ndarray) -> np.ndarray:
    img = image_rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    hs, ws = h // 2, w // 2
    out = np.empty((12, hs, ws), dtype=np.float32)

    for c in range(3):
        ch = img[..., c]
        # 2x2 Haar coefficients via averages and differences:
        a = ch[0::2, 0::2]
        b = ch[0::2, 1::2]
        cd = ch[1::2, 0::2]
        d = ch[1::2, 1::2]
        ll = (a + b + cd + d) * 0.5
        lh = (a + b - cd - d) * 0.5
        hl = (a - b + cd - d) * 0.5
        hh = (a - b - cd + d) * 0.5
        # Match shapes to (hs, ws)
        ll = ll[:hs, :ws]
        lh = lh[:hs, :ws]
        hl = hl[:hs, :ws]
        hh = hh[:hs, :ws]
        out[c * 4 + 0] = ll
        out[c * 4 + 1] = lh
        out[c * 4 + 2] = hl
        out[c * 4 + 3] = hh
    return out


def defocus_map(image_rgb: np.ndarray, target_size: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    sal = np.abs(lap)
    sal = cv2.boxFilter(sal, ddepth=cv2.CV_32F, ksize=(7, 7))
    if sal.max() > 0:
        sal = sal / (sal.max() + 1e-8)
    sal = cv2.resize(sal, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return sal.astype(np.float16)


def preprocess_video(video_path: Path, cache_root: Path, *, dataset: str, label: str, detector, image_size: int = 256,
                     wavelet_size: int = 128, frame_stride: int = 10, max_frames: int = 100, overwrite: bool = False) -> PreprocessResult:
    video = video_path.stem
    crop_root = cache_root / dataset / "face_crops" / label / video
    wav_root = cache_root / dataset / "wavelet_l1" / label / video
    def_root = cache_root / dataset / "defocus" / label / video
    crop_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)
    def_root.mkdir(parents=True, exist_ok=True)

    cap = _open_video(video_path)
    if not cap.isOpened():
        log.warning(f"could not open {video_path}")
        return PreprocessResult(video=video, label=label, frames_written=0, frames_failed=0, total_frames=0)

    written = 0
    failed = 0
    total = 0
    try:
        for idx, rgb in _video_iter(cap, frame_stride, max_frames):
            total += 1
            crop_path = crop_root / f"frame_{idx:06d}.jpg"
            wav_path = wav_root / f"frame_{idx:06d}.npy"
            def_path = def_root / f"frame_{idx:06d}.npy"
            if not overwrite and crop_path.exists() and wav_path.exists() and def_path.exists():
                continue

            crop = detector.detect(rgb)
            if crop is None:
                failed += 1
                continue

            cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            wav = haar_dwt_stack(crop).astype(np.float16)
            np.save(wav_path, wav)
            np.save(def_path, defocus_map(crop, target_size=wavelet_size))
            written += 1
    finally:
        cap.release()
    return PreprocessResult(video=video, label=label, frames_written=written, frames_failed=failed, total_frames=total)


def write_preprocess_log(cache_root: Path, dataset: str, results: list[PreprocessResult]) -> None:
    payload = {"videos": len(results), "frames_total": sum(r.total_frames for r in results),
               "frames_written": sum(r.frames_written for r in results), "frames_failed": sum(r.frames_failed for r in results),
               "bad_videos": sorted(r.video for r in results if r.total_frames > 0 and r.frames_failed / max(r.total_frames, 1) > 0.05)}
    save_json(cache_root / dataset / "preprocessing_log.json", payload)


def list_processed_frames(cache_root: Path, dataset: str, label: str, video: str) -> list[int]:
    folder = cache_root / dataset / "face_crops" / label / video
    if not folder.is_dir():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in folder.glob("frame_*.jpg"))


def existing_label_dirs(cache_root: Path, dataset: str) -> dict[str, list[str]]:
    base = cache_root / dataset / "face_crops"
    if not base.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for label_dir in base.iterdir():
        if not label_dir.is_dir():
            continue
        out[label_dir.name] = sorted(p.name for p in label_dir.iterdir() if p.is_dir())
    return out


def has_complete_video(cache_root: Path, dataset: str, label: str, video: str) -> bool:
    crops = list_processed_frames(cache_root, dataset, label, video)
    if not crops:
        return False
    wav = (cache_root / dataset / "wavelet_l1" / label / video)
    df = (cache_root / dataset / "defocus" / label / video)
    return wav.is_dir() and df.is_dir()


def discover_frames(cache_root: Path, dataset: str, splits: dict, split_name: str, manip: Optional[str] = None) -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    split_data = splits[split_name]
    for label, vids in split_data.items():
        if manip is not None and label != manip:
            continue
        label_int = 0 if label == "real" else 1
        for v in vids:
            for idx in list_processed_frames(cache_root, dataset, label, v):
                rows.append((label, v, idx, label_int))
    return rows