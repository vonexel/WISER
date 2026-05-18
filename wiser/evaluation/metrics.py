from __future__ import annotations

import numpy as np
from typing import Sequence
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve)
from dataclasses import dataclass, field


def _safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, scores)
    bal = 0.5 * (tpr + (1 - fpr))
    return float(thr[int(np.argmax(bal))])


def brier(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(((scores - labels) ** 2).mean())


def ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(scores, bins[1:-1], right=False)
    ece_v = 0.0
    n = len(scores)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece_v += abs(conf - acc) * (mask.sum() / n)
    return float(ece_v)


def per_class_ece(scores: np.ndarray, labels: np.ndarray, target_class: int, n_bins: int = 15) -> float:
    mask = labels == target_class
    if not mask.any():
        return float("nan")
    s_sub = scores[mask]
    if target_class == 0:
        s_sub = 1.0 - s_sub
        l_sub = np.ones_like(labels[mask], dtype=np.int8)
    else:
        l_sub = labels[mask].astype(np.int8)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(s_sub, bins[1:-1], right=False)
    ece_v = 0.0
    n = len(s_sub)
    for b in range(n_bins):
        m = bin_ids == b
        if not m.any():
            continue
        conf = s_sub[m].mean()
        acc = l_sub[m].mean()
        ece_v += abs(conf - acc) * (m.sum() / n)
    return float(ece_v)


def real_fake_recalls(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> tuple[float, float, float]:
    pred = (scores >= threshold).astype(np.int8)
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    fake_recall = tp / max(1.0, tp + fn)
    real_recall = tn / max(1.0, tn + fp)
    return real_recall, fake_recall, 0.5 * (real_recall + fake_recall)


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr >= target_tpr
    if not eligible.any():
        return float("nan")
    return float(fpr[eligible].min())


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = fpr <= target_fpr
    if not eligible.any():
        return float("nan")
    return float(tpr[eligible].max())


@dataclass(slots=True)
class FrameMetrics:
    auc: float
    ap: float
    eer: float
    acc_at_05: float
    acc_at_optimal: float
    optimal_threshold: float
    f1: float
    brier: float
    ece: float
    real_recall: float = float("nan")
    fake_recall: float = float("nan")
    balanced_acc: float = float("nan")
    fpr_at_tpr95: float = float("nan")
    tpr_at_fpr05: float = float("nan")
    ece_real: float = float("nan")
    ece_fake: float = float("nan")

    def to_dict(self) -> dict:
        return {"auc": self.auc, "ap": self.ap, "eer": self.eer, "acc_at_05": self.acc_at_05,
                "acc_at_optimal": self.acc_at_optimal, "optimal_threshold": self.optimal_threshold,
                "f1": self.f1, "brier": self.brier, "ece": self.ece, "real_recall": self.real_recall,
                "fake_recall": self.fake_recall, "balanced_acc": self.balanced_acc, "fpr_at_tpr95": self.fpr_at_tpr95,
                "tpr_at_fpr05": self.tpr_at_fpr05, "ece_real": self.ece_real, "ece_fake": self.ece_fake}


def compute_frame_metrics(scores: np.ndarray, labels: np.ndarray, *, threshold: float = 0.5) -> FrameMetrics:
    scores = scores.astype(np.float32)
    labels = labels.astype(np.int8)
    auc = _safe_auc(scores, labels)
    ap = _safe_ap(scores, labels)
    e = eer(scores, labels)
    thr = optimal_threshold(scores, labels)
    acc_05 = float(((scores >= 0.5).astype(np.int8) == labels).mean())
    acc_opt = float(((scores >= thr).astype(np.int8) == labels).mean())
    f1 = (
        float(f1_score(labels, (scores >= thr).astype(np.int8), zero_division=0))
        if len(set(labels.tolist())) > 1
        else float("nan"))
    rr, fr, bal = real_fake_recalls(scores, labels, threshold=threshold)
    return FrameMetrics(
        auc=auc, ap=ap, eer=e, acc_at_05=acc_05, acc_at_optimal=acc_opt,
        optimal_threshold=thr, f1=f1, brier=brier(scores, labels), ece=ece(scores, labels),
        real_recall=rr, fake_recall=fr, balanced_acc=bal,
        fpr_at_tpr95=fpr_at_tpr(scores, labels, 0.95),
        tpr_at_fpr05=tpr_at_fpr(scores, labels, 0.05),
        ece_real=per_class_ece(scores, labels, target_class=0),
        ece_fake=per_class_ece(scores, labels, target_class=1))


def video_aggregate(scores: np.ndarray, labels: np.ndarray, video_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    by_vid: dict[str, list[float]] = {}
    lab_by_vid: dict[str, int] = {}
    for s, l, v in zip(scores.tolist(), labels.tolist(), video_ids):
        by_vid.setdefault(v, []).append(float(s))
        lab_by_vid[v] = int(l)
    keys = sorted(by_vid.keys())
    v_scores = np.array([np.mean(by_vid[k]) for k in keys], dtype=np.float32)
    v_labels = np.array([lab_by_vid[k] for k in keys], dtype=np.int8)
    return v_scores, v_labels


def compute_video_metrics(scores: np.ndarray, labels: np.ndarray, video_ids: Sequence[str]) -> dict:
    vs, vl = video_aggregate(scores, labels, video_ids)
    rr, fr, bal = real_fake_recalls(vs, vl)
    return {"auc": _safe_auc(vs, vl), "eer": eer(vs, vl),
            "real_recall": rr, "fake_recall": fr, "balanced_acc": bal}