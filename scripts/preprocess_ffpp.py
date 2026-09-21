from __future__ import annotations

import sys
import argparse
from pathlib import Path
from wiser.data.face_detector import FaceDetector
from wiser.data.preprocessing import preprocess_video, write_preprocess_log
from wiser.data.splits import (FFPP_FAKE_TYPES,
                               build_ffpp_splits,
                               discover_videos,
                               save_ffpp_splits)
from wiser.utils import setup_logging
from wiser.utils.logging import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True, type=str)
    p.add_argument("--cache_root", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--frames_per_video", default=100, type=int)
    p.add_argument("--frame_stride", default=10, type=int)
    p.add_argument("--image_size", default=256, type=int)
    p.add_argument("--wavelet_size", default=128, type=int)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    setup_logging("INFO")
    log = get_logger("preprocess.ffpp")

    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    real_videos, by_manip = discover_videos(raw_root)
    log.info(f"FF++ found {len(real_videos)} real videos and {sum(len(v) for v in by_manip.values())} fakes.")

    detector = FaceDetector(device=args.device, image_size=args.image_size)

    results = []
    real_dir_candidates = [
        raw_root / "original_sequences" / "youtube" / "c23" / "videos",
        raw_root / "ff_c23" / "FaceForensics++_C23" / "original",
        raw_root / "FaceForensics++_C23" / "original",
        raw_root / "original",
    ]
    real_dir = next((p for p in real_dir_candidates if p.is_dir()), None)
    assert real_dir is not None

    for video in real_videos:
        results.append(
            preprocess_video(
                real_dir / f"{video}.mp4", cache_root, dataset="ffpp",
                label="real",
                detector=detector,
                image_size=args.image_size,
                wavelet_size=args.wavelet_size,
                frame_stride=args.frame_stride,
                max_frames=args.frames_per_video,
                overwrite=args.overwrite))

    for manip in FFPP_FAKE_TYPES:
        candidates = [
            raw_root / "manipulated_sequences" / manip / "c23" / "videos",
            raw_root / "ff_c23" / "FaceForensics++_C23" / manip,
            raw_root / "FaceForensics++_C23" / manip,
            raw_root / manip]
        manip_dir = next((p for p in candidates if p.is_dir()), None)
        if manip_dir is None:
            log.warning(f"{manip}: directory not found, skipping")
            continue
        for video in by_manip[manip]:
            results.append(
                preprocess_video(
                    manip_dir / f"{video}.mp4",
                    cache_root,
                    dataset="ffpp",
                    label=manip,
                    detector=detector,
                    image_size=args.image_size,
                    wavelet_size=args.wavelet_size,
                    frame_stride=args.frame_stride,
                    max_frames=args.frames_per_video,
                    overwrite=args.overwrite))

    write_preprocess_log(cache_root, "ffpp", results)
    splits = build_ffpp_splits(raw_root, val_fraction=0.14, test_fraction=0.14, seed=42)
    out = save_ffpp_splits(splits, cache_root)
    log.info(f"FF++ splits written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())