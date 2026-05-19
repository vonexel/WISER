from __future__ import annotations


import numpy as np
from typing import Callable, Dict, Sequence


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def pool_mean(p: np.ndarray) -> float:
    return float(np.mean(p))


def pool_median(p: np.ndarray) -> float:
    return float(np.median(p))


def pool_trimmed_mean(p: np.ndarray, alpha: float = 0.1) -> float:
    n = len(p)
    if n == 0:
        return float("nan")
    k = int(np.floor(alpha * n))
    if 2 * k >= n:
        return float(np.median(p))
    sorted_p = np.sort(p)
    return float(np.mean(sorted_p[k : n - k]))


def pool_entropy_weighted(p: np.ndarray) -> float:
    if len(p) == 0:
        return float("nan")
    h = _bernoulli_entropy(p)
    w = 1.0 - h
    if float(np.sum(w)) < 1e-3:
        return float(np.mean(p))
    return float(np.sum(w * p) / np.sum(w))


def pool_top_k_mean(p: np.ndarray, k: int = 10) -> float:
    if len(p) == 0:
        return float("nan")
    centred = np.abs(p - 0.5)
    n = min(int(k), len(p))
    idx = np.argpartition(-centred, n - 1)[:n]
    return float(np.mean(p[idx]))


POOLING_FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {"mean": pool_mean, "median": pool_median,
                                                               "trimmed_mean": pool_trimmed_mean,
                                                               "entropy_weighted": pool_entropy_weighted,
                                                               "top_k_mean": pool_top_k_mean}


def pool_per_video(scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray, video_ids: Sequence[str],
                   *, method: str = "median") -> tuple[np.ndarray, np.ndarray, list[str]]:
    fn = POOLING_FUNCTIONS.get(method)
    if fn is None:
        raise ValueError(f"unknown pooling method: {method!r}; valid: {sorted(POOLING_FUNCTIONS)}")
    by_vid: dict[str, list[float]] = {}
    lab_by_vid: dict[str, int] = {}
    for s, l, v in zip(np.asarray(scores).tolist(), np.asarray(labels).tolist(), video_ids):
        by_vid.setdefault(v, []).append(float(s))
        lab_by_vid[v] = int(l)
    keys = sorted(by_vid.keys())
    out_s = np.array([fn(np.asarray(by_vid[k], dtype=np.float32)) for k in keys], dtype=np.float32)
    out_l = np.array([lab_by_vid[k] for k in keys], dtype=np.int8)
    return out_s, out_l, keys