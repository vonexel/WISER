from __future__ import annotations


import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from wiser.utils import save_json
from contextlib import nullcontext
from wiser.evaluation.metrics import compute_frame_metrics
from wiser.data.augmentation import IMAGENET_MEAN, IMAGENET_STD


@dataclass(slots=True)
class RobustnessAxis:
    name: str
    levels: list[float]
    fn_factory: Callable[[float], Callable[[torch.Tensor], torch.Tensor]]


def _jpeg(level: float) -> Callable[[torch.Tensor], torch.Tensor]:
    qf = int(level)

    def apply(rgb: torch.Tensor) -> torch.Tensor:
        out = []
        for x in rgb:
            arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), qf])
            if not ok:
                out.append(x)
                continue
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            out.append(torch.from_numpy(dec[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
        return torch.stack(out).to(rgb.device)
    return apply


def _gauss_noise(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(rgb: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(rgb) * (sigma / 255.0)
        return (rgb + noise).clamp(0, 1)
    return apply


def _gauss_blur(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    ksize = max(3, int(2 * round(2 * sigma) + 1))

    def apply(rgb: torch.Tensor) -> torch.Tensor:
        out = []
        for x in rgb:
            arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigma)
            out.append(torch.from_numpy(blurred).permute(2, 0, 1).float() / 255.0)
        return torch.stack(out).to(rgb.device)
    return apply


AXES = [RobustnessAxis("jpeg", [90, 70, 50, 30], _jpeg), RobustnessAxis("noise", [1.0, 5.0, 10.0], _gauss_noise),
        RobustnessAxis("blur", [0.5, 1.0, 2.0], _gauss_blur)]


@torch.inference_mode()
def run_robustness(model: torch.nn.Module, loader, *, device: torch.device, out_path: Path, autocast: bool = True) -> dict:
    out_path = Path(out_path)
    results: dict[str, dict[str, float]] = {axis.name: {} for axis in AXES}
    amp = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast and device.type == "cuda" else nullcontext())
    for axis in AXES:
        for level in axis.levels:
            apply = axis.fn_factory(level)
            scores: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            for batch in loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
                rgb01 = (rgb * std + mean).clamp(0, 1)
                rgb_pert = apply(rgb01)
                rgb_norm = (rgb_pert - mean) / std
                wav = batch["wavelet"].to(device, non_blocking=True)
                df = batch["defocus"].to(device, non_blocking=True)
                with amp:
                    out = model(rgb_norm, wav, df)
                scores.append(torch.sigmoid(out["logits"]).float().cpu().numpy())
                labels.append(batch["label"].numpy().astype(np.int8))
            s = np.concatenate(scores)
            l = np.concatenate(labels)
            m = compute_frame_metrics(s, l)
            results[axis.name][str(level)] = m.auc

    save_json(out_path, results)
    return results