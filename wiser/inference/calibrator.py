from __future__ import annotations


import math
import json
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.optimize import minimize_scalar
from dataclasses import asdict, dataclass


_MIN_VAL_PER_CLASS = 200


@dataclass(slots=True)
class CalibrationParams:
    temperature: float = 1.0
    prior_bias: float = 0.0
    threshold: float = 0.5
    val_subset_size: int = 0
    fitted: bool = False
    notes: str = ""

    def apply(self, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = max(1e-3, float(self.temperature))
        z = (logits.astype(np.float32) - float(self.prior_bias)) / T
        probs = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z))).astype(np.float32)
        preds = (probs >= float(self.threshold)).astype(np.int8)
        return probs, preds


def _balanced_subsample(labels: np.ndarray, *, target_per_class: int = 4000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n = min(len(pos), len(neg), int(target_per_class))
    if n <= 0:
        return np.array([], dtype=np.int64)
    pos_pick = rng.choice(pos, size=n, replace=False)
    neg_pick = rng.choice(neg, size=n, replace=False)
    return np.concatenate([pos_pick, neg_pick])


def _bce_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    def nll(T: float) -> float:
        T = max(1e-3, float(T))
        z = logits.astype(np.float64) / T
        return float(np.mean(np.logaddexp(0.0, -z) + (1.0 - labels) * z))

    res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded", options={"xatol": 1e-3})
    T = float(res.x)
    note = ""
    if T <= 0.5 + 1e-3 or T >= 5.0 - 1e-3:
        note = f"WARNING: temperature converged to bound T={T:.3f} (validation degenerate)"
    return T, note


def _balanced_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    pred = (scores >= threshold).astype(np.int8)
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    tpr = tp / max(1.0, tp + fn)
    tnr = tn / max(1.0, tn + fp)
    return 0.5 * (tpr + tnr)


def fit_calibrator(logits: np.ndarray, labels: np.ndarray, *, target_per_class: int = 4000, seed: int = 42) -> CalibrationParams:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: {logits.shape} vs {labels.shape}")

    idx = _balanced_subsample(labels, target_per_class=target_per_class, seed=seed)
    n_per = len(idx) // 2
    if n_per < _MIN_VAL_PER_CLASS:
        return CalibrationParams(temperature=1.0, prior_bias=0.0, threshold=0.5, val_subset_size=int(len(idx)),
                                 fitted=False, notes=f"validation subset too small: {n_per} per class < {_MIN_VAL_PER_CLASS}")

    sub_logits = logits[idx]
    sub_labels = labels[idx]
    T, note_T = _fit_temperature(sub_logits, sub_labels)
    if note_T:
        pass
    z_T = sub_logits / max(1e-3, T)
    bias = float(np.mean(z_T))
    z_calib = z_T - bias
    probs = 1.0 / (1.0 + np.exp(-z_calib))
    thresholds = np.linspace(0.05, 0.95, 200)
    bal_accs = np.array([_balanced_accuracy(probs, sub_labels, float(t)) for t in thresholds])
    th = float(thresholds[int(np.argmax(bal_accs))])
    return CalibrationParams(temperature=T, prior_bias=bias, threshold=th, val_subset_size=int(len(idx)), fitted=True, notes=note_T)


def save_calibration(path: str | Path, params: CalibrationParams) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)


def load_calibration(path: str | Path) -> CalibrationParams:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationParams(**data)


def apply_calibration(logits: np.ndarray, params: Optional[CalibrationParams]) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if params is None or not params.fitted:
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int8)
        return probs.astype(np.float32), preds
    return params.apply(logits)


Calibrator = CalibrationParams