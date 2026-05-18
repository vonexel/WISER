from __future__ import annotations


import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Iterable, Optional
from torch.utils.data import DataLoader
from wiser.evaluation.metrics import (compute_frame_metrics, compute_video_metrics, real_fake_recalls, video_aggregate)
from wiser.inference.calibrator import (CalibrationParams, apply_calibration, fit_calibrator, save_calibration)
from wiser.inference.video_pool import pool_per_video
from wiser.utils import save_json, save_npz


@dataclass(slots=True)
class EvalArtefacts:
    scores: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    video_ids: list[str]
    manip_or_category: list[str]
    embeddings: Optional[np.ndarray] = None


@torch.inference_mode()
def collect(model: nn.Module, loader: Iterable, *, device: torch.device, field_for_strata: str = "manip",
            capture_embeddings: bool = False, autocast: bool = True,) -> EvalArtefacts:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    video_ids: list[str] = []
    strata: list[str] = []
    emb_chunks: list[np.ndarray] = []

    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast and device.type == "cuda" else nullcontext())
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        wav = batch["wavelet"].to(device, non_blocking=True)
        df = batch["defocus"].to(device, non_blocking=True)
        with amp_ctx:
            out = model(rgb, wav, df)
        z = out["logits"].float().cpu().numpy()
        logits_chunks.append(z)
        labels.append(batch["label"].numpy().astype(np.int8))
        video_ids.extend(batch["video_id"])
        strata.extend(batch.get(field_for_strata, ["?"] * len(batch["label"])))
        if capture_embeddings:
            emb_chunks.append(out["embeddings"].float().cpu().numpy())

    logits = np.concatenate(logits_chunks)
    scores = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return EvalArtefacts(scores=scores, logits=logits.astype(np.float32), labels=np.concatenate(labels),
                         video_ids=video_ids, manip_or_category=strata, embeddings=np.concatenate(emb_chunks) if emb_chunks else None)


def per_stratum_auc(scores: np.ndarray, labels: np.ndarray, strata: list[str]) -> dict[str, dict[str, float]]:
    real_mask = labels == 0
    out: dict[str, dict[str, float]] = {}
    for s in sorted(set(strata)):
        if s == "real":
            continue
        mask = (np.array(strata) == s) | real_mask
        sub_scores = scores[mask]
        sub_labels = labels[mask]
        if len(set(sub_labels.tolist())) < 2:
            continue
        m = compute_frame_metrics(sub_scores, sub_labels)
        out[s] = {"auc": m.auc, "ap": m.ap, "eer": m.eer, "balanced_acc": m.balanced_acc}
    return out


def _evaluate_block(art: EvalArtefacts, *, calibration: Optional[CalibrationParams] = None,
                    video_pool: str = "mean", threshold: Optional[float] = None) -> tuple[dict, np.ndarray]:
    if calibration is not None:
        scores, _ = apply_calibration(art.logits, calibration)
        thr = float(calibration.threshold) if threshold is None else float(threshold)
    else:
        scores = art.scores
        thr = 0.5 if threshold is None else float(threshold)
    frame = compute_frame_metrics(scores, art.labels, threshold=thr).to_dict()
    vs, vl, _vid = pool_per_video(scores, art.labels, art.video_ids, method=video_pool)
    if len(vs) > 0 and len(set(vl.tolist())) > 1:
        vid_metrics = compute_frame_metrics(vs, vl, threshold=thr).to_dict()
    else:
        vid_metrics = {"auc": float("nan"), "balanced_acc": float("nan")}
    per = per_stratum_auc(scores, art.labels, art.manip_or_category)
    return {"frame": frame, "video": vid_metrics, "per_manipulation": per}, scores


def evaluate_full(model: nn.Module, in_loader: DataLoader, cross_loader: Optional[DataLoader], *, device: torch.device, out_dir: Path,
                  val_loader: Optional[DataLoader] = None, enable_calibration: bool = False, video_pool_calibrated: str = "median",
                  video_pool_uncalibrated: str = "mean", extras: Optional[dict] = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indomain = collect(model, in_loader, device=device, capture_embeddings=True)
    cross: Optional[EvalArtefacts] = None
    if cross_loader is not None:
        cross = collect(model, cross_loader, device=device, field_for_strata="category")
    cal: Optional[CalibrationParams] = None
    if enable_calibration and val_loader is not None:
        val = collect(model, val_loader, device=device, capture_embeddings=False)
        cal = fit_calibrator(val.logits, val.labels)
        save_calibration(out_dir / "calib.json", cal)

    if cal is None:
        in_block, _ = _evaluate_block(indomain, calibration=None, video_pool="mean")
        in_block_legacy = {"frame": in_block["frame"],
                           "video": compute_video_metrics(indomain.scores, indomain.labels, indomain.video_ids),
                           "per_manipulation": in_block["per_manipulation"]}
        crossdomain_payload: dict | None = None
        if cross is not None:
            cd_block, _ = _evaluate_block(cross, calibration=None, video_pool="mean")
            crossdomain_payload = {"frame": cd_block["frame"],
                                   "video": compute_video_metrics(cross.scores, cross.labels, cross.video_ids),
                                   "per_category": cd_block["per_manipulation"]}
        metrics = {"schema_version": "1.0", "indomain": in_block_legacy}
        if crossdomain_payload is not None:
            metrics["crossdomain"] = crossdomain_payload
        if extras:
            metrics.update(extras)
        save_json(out_dir / "metrics.json", metrics)
        _persist_predictions(out_dir, indomain, cross)
        if indomain.embeddings is not None:
            _persist_embeddings(out_dir, indomain)
        return metrics

    in_uncal, in_uncal_scores = _evaluate_block(indomain, calibration=None, video_pool=video_pool_uncalibrated)
    in_cal, in_cal_scores = _evaluate_block(indomain, calibration=cal, video_pool=video_pool_calibrated)
    crossdomain_uncal: dict | None = None
    crossdomain_cal: dict | None = None
    cd_uncal_scores = np.zeros(0, dtype=np.float32)
    cd_cal_scores = np.zeros(0, dtype=np.float32)
    if cross is not None:
        cd_uncal_block, cd_uncal_scores = _evaluate_block(
            cross, calibration=None, video_pool=video_pool_uncalibrated)
        cd_cal_block, cd_cal_scores = _evaluate_block(
            cross, calibration=cal, video_pool=video_pool_calibrated)
        crossdomain_uncal = {"frame": cd_uncal_block["frame"], "video": cd_uncal_block["video"], "per_category": cd_uncal_block["per_manipulation"]}
        crossdomain_cal = {"frame": cd_cal_block["frame"], "video": cd_cal_block["video"], "per_category": cd_cal_block["per_manipulation"]}

    metrics = {"schema_version": "2.0", "uncalibrated": {"indomain": {"frame": in_uncal["frame"],
                                                                      "video": in_uncal["video"],
                                                                      "per_manipulation": in_uncal["per_manipulation"]},
                                                         },
               "calibrated": {
                   "indomain": {
                       "frame": in_cal["frame"],
                       "video": in_cal["video"],
                       "per_manipulation": in_cal["per_manipulation"],
                   },
               },
               "calibration": {
                   "temperature": cal.temperature,
                   "prior_bias": cal.prior_bias,
                   "threshold": cal.threshold,
                   "val_subset_size": cal.val_subset_size,
                   "fitted": cal.fitted,
                   "notes": cal.notes,
               },
               }
    if crossdomain_uncal is not None:
        metrics["uncalibrated"]["crossdomain"] = crossdomain_uncal
        metrics["calibrated"]["crossdomain"] = crossdomain_cal
    if extras:
        metrics.update(extras)
    save_json(out_dir / "metrics.json", metrics)
    _persist_predictions(out_dir, indomain, cross, extra_scores={"indomain_scores_calibrated": in_cal_scores,
                                                                 "crossdomain_scores_calibrated": cd_cal_scores,})
    if indomain.embeddings is not None:
        _persist_embeddings(out_dir, indomain)
    return metrics


def _persist_predictions(out_dir: Path, indomain: EvalArtefacts, cross: Optional[EvalArtefacts], *,
                         extra_scores: Optional[dict[str, np.ndarray]] = None) -> None:
    payload = dict( indomain_scores=indomain.scores,
                    indomain_logits=indomain.logits,
                    indomain_labels=indomain.labels,
                    indomain_video_ids=np.array(indomain.video_ids),
                    indomain_manip=np.array(indomain.manip_or_category),
                    crossdomain_scores=cross.scores if cross is not None else np.zeros(0, dtype=np.float32),
                    crossdomain_logits=cross.logits if cross is not None else np.zeros(0, dtype=np.float32),
                    crossdomain_labels=cross.labels if cross is not None else np.zeros(0, dtype=np.int8),
                    crossdomain_video_ids=np.array(cross.video_ids) if cross is not None else np.array([]),
                    crossdomain_category=np.array(cross.manip_or_category) if cross is not None else np.array([]))
    if extra_scores:
        payload.update(extra_scores)
    save_npz(out_dir / "predictions.npz", **payload)


def _persist_embeddings(out_dir: Path, indomain: EvalArtefacts) -> None:
    sample_size = min(5000, len(indomain.embeddings))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(indomain.embeddings), size=sample_size, replace=False)
    save_npz(out_dir / "embeddings.npz",
             embeddings=indomain.embeddings[idx],
             labels=indomain.labels[idx],
             manip=np.array(indomain.manip_or_category)[idx],
             video_ids=np.array(indomain.video_ids)[idx],
             is_sbi=np.array([m == "sbi_pseudo" for m in indomain.manip_or_category])[idx])