from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from wiser.utils import save_json

FFPP_FAKE_TYPES = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter",)


@dataclass(slots=True)
class SplitSpec:
    real: list[str]
    by_manip: dict[str, list[str]]


def _real_id_of(fake_filename: str) -> str:
    stem = Path(fake_filename).stem
    return stem.split("_")[0]


def discover_videos(raw_root: Path) -> tuple[list[str], dict[str, list[str]]]:
    candidates_real = [
        raw_root / "original_sequences" / "youtube" / "c23" / "videos",
        raw_root / "ff_c23" / "FaceForensics++_C23" / "original",
        raw_root / "FaceForensics++_C23" / "original",
        raw_root / "original"]
    real_dir = next((p for p in candidates_real if p.is_dir()), None)
    if real_dir is None:
        raise FileNotFoundError(
            f"Could not locate FF++ real videos under {raw_root}. "
            f"Expected one of: {[str(p) for p in candidates_real]}")
    real_videos = sorted(p.stem for p in real_dir.glob("*.mp4"))

    by_manip: dict[str, list[str]] = {}
    for manip in FFPP_FAKE_TYPES:
        candidates = [
            raw_root / "manipulated_sequences" / manip / "c23" / "videos",
            raw_root / "ff_c23" / "FaceForensics++_C23" / manip,
            raw_root / "FaceForensics++_C23" / manip,
            raw_root / manip]
        manip_dir = next((p for p in candidates if p.is_dir()), None)
        by_manip[manip] = sorted(p.stem for p in manip_dir.glob("*.mp4")) if manip_dir else []
    return real_videos, by_manip


def build_ffpp_splits(raw_root: str | Path, *, val_fraction: float = 0.14, test_fraction: float = 0.14, seed: int = 42) -> dict[str, SplitSpec | int]:
    raw_root = Path(raw_root)
    real_videos, by_manip = discover_videos(raw_root)

    rng = random.Random(seed)
    shuffled = list(real_videos)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test_real = set(shuffled[:n_test])
    val_real = set(shuffled[n_test : n_test + n_val])
    train_real = set(shuffled[n_test + n_val :])

    def _bucket(video_ids: Iterable[str], real_set: set[str]) -> list[str]:
        return sorted(v for v in video_ids if _real_id_of(v) in real_set)

    out: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}
    for split_name, real_set in [("train", train_real), ("val", val_real), ("test", test_real)]:
        out[split_name]["real"] = sorted(real_set)
        for manip in FFPP_FAKE_TYPES:
            out[split_name][manip] = _bucket(by_manip[manip], real_set)
    out["seed"] = seed
    return out


def save_ffpp_splits(splits: dict, cache_root: str | Path) -> Path:
    cache_root = Path(cache_root)
    out = cache_root / "ffpp" / "splits.json"
    save_json(out, splits)
    return out


CELEBDFPP_CATEGORY_PREFIX = {"FaceReenact": "FR", "FaceSwap": "FS", "TalkingFace": "TF"}


def discover_celebdfpp(raw_root: Path) -> dict[str, list[Path]]:
    layouts = [raw_root, raw_root / "celebdfpp"]
    base = next((b for b in layouts if (b / "Celeb-real").is_dir() and (b / "Celeb-synthesis").is_dir()), None,)
    if base is None:
        raise FileNotFoundError(f"Celeb-DF++ raw layout not found under {raw_root}")

    out: dict[str, list[Path]] = {
        "Celeb-real": sorted((base / "Celeb-real").glob("*.mp4")),
        "YouTube-real": sorted((base / "YouTube-real").glob("*.mp4"))
        if (base / "YouTube-real").is_dir()
        else []}

    synth = base / "Celeb-synthesis"
    category_dirs = {p.name for p in synth.iterdir() if p.is_dir()}
    is_v2025 = bool(category_dirs & set(CELEBDFPP_CATEGORY_PREFIX))

    if is_v2025:
        for cat_name, prefix in CELEBDFPP_CATEGORY_PREFIX.items():
            cat_dir = synth / cat_name
            if not cat_dir.is_dir():
                continue
            for method_dir in sorted(cat_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                out[f"{prefix}-{method_dir.name}"] = sorted(method_dir.glob("*.mp4"))
    else:
        out["Celeb-synthesis"] = sorted(synth.glob("*.mp4"))
    return out