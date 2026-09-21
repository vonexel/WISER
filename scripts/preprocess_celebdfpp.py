from __future__ import annotations

import sys
import argparse
from pathlib import Path
from wiser.data.face_detector import FaceDetector
from wiser.data.preprocessing import preprocess_video, write_preprocess_log
from wiser.data.splits import discover_celebdfpp
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
    log = get_logger("preprocess.celebdfpp")
    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    by_label = discover_celebdfpp(raw_root)
    log.info(f"Celeb-DF++ found: {[(k, len(v)) for k, v in by_label.items()]}")
    detector = FaceDetector(device=args.device, image_size=args.image_size)

    results = []
    for label, video_paths in by_label.items():
        if not video_paths:
            continue
        for vp in video_paths:
            results.append(
                preprocess_video(vp, cache_root, dataset="celebdfpp",
                                 label=label,
                                 detector=detector,
                                 image_size=args.image_size,
                                 wavelet_size=args.wavelet_size,
                                 frame_stride=args.frame_stride,
                                 max_frames=args.frames_per_video,
                                 overwrite=args.overwrite))

    write_preprocess_log(cache_root, "celebdfpp", results)
    log.info(f"Celeb-DF++ preprocessing complete; {sum(r.frames_written for r in results)} frames written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())