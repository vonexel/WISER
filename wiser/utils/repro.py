from __future__ import annotations

import os
import cv2
import torch
import random
import numpy as np
import platform
import subprocess
from typing import Optional
from contextlib import nullcontext


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    torch.set_num_threads(1)


def get_run_meta(seed: int) -> dict[str, str | int | bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        sha, dirty = "unknown", False

    cuda = torch.version.cuda or "cpu"
    return {"git_sha": sha,
            "git_dirty": dirty,
            "torch_version": torch.__version__,
            "cuda_runtime": cuda,
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "host": platform.node(),
            "seed": int(seed)}


def autocast_context() -> Optional[object]:
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()