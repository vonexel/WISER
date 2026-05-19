from __future__ import annotations


import torch
import json
import math
import time
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from torch.utils.tensorboard import SummaryWriter
from wiser.data.awb_sbi_v2 import (AdaptiveWaveletBandSBI, awb_sbi_v2_config_from_omegaconf)
from wiser.data.real_mixstyle import MixStyleParams, real_mixstyle_
from wiser.evaluation.metrics import compute_frame_metrics
from wiser.models.disentangle import grl_lambda_schedule
from wiser.training.callbacks import BestCheckpoint, EarlyStopping
from wiser.training.ema import ModelEMA
from wiser.training.optim import build_optimizer, build_scheduler
from wiser.utils import get_logger
from wiser.utils.repro import get_run_meta


log = get_logger("wiser.trainer")


@dataclass(slots=True)
class TrainerOutput:
    best_metric: float
    best_epoch: int
    epochs_run: int
    early_stopped: bool
    final_train_loss: float
    final_val_loss: float
    final_val_auc: float
    loss_components_at_best: dict
    wallclock_seconds: float
    lr_history: list[float] = field(default_factory=list)


class Trainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, cfg, train_loader: DataLoader, val_loader: DataLoader,
                 out_dir: Path, *, device: torch.device, seed: int = 0) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.optimizer = build_optimizer(self.model, cfg.training.optim)
        steps_per_epoch = max(1, len(train_loader))
        if cfg.training.sanity_steps > 0:
            steps_per_epoch = min(steps_per_epoch, cfg.training.sanity_steps)
        total_steps = max(1, cfg.training.epochs * steps_per_epoch)
        self.scheduler = build_scheduler(self.optimizer, total_steps=total_steps,
                                         warmup_frac=cfg.training.sched.warmup_frac,
                                         min_lr=cfg.training.sched.min_lr)
        self.ema = ModelEMA(self.model, decay=cfg.training.ema_decay)
        self.tb = SummaryWriter(log_dir=str(self.out_dir))
        self.ckpt = BestCheckpoint(self.out_dir / "ckpt", mode="max")
        self.es = EarlyStopping(cfg.training.early_stopping_patience, mode="max")
        self.global_step = 0
        self.seed = seed
        self.amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and cfg.training.amp_dtype == "bf16" else nullcontext())

        model_cfg = cfg.get("model", None) if hasattr(cfg, "get") else None
        cnd_block = getattr(model_cfg, "cnd", None) if model_cfg is not None else None
        self._cnd_enabled = bool(getattr(cnd_block, "enabled", False)) if cnd_block is not None else False
        self._grl_lambda_max = float(getattr(cnd_block, "grl_lambda_max", 0.3)) if cnd_block is not None else 0.3
        self._orth_warmup_epoch = int(getattr(cnd_block, "orth_warmup_epoch", 5)) if cnd_block is not None else 5
        loss_cnd = getattr(cfg.loss, "cnd", None)
        self._orth_target = bool(getattr(loss_cnd, "orth_enabled", False)) if loss_cnd is not None else False
        if self._orth_target:
            self.loss_fn.orth_enabled = False
        ms_block = getattr(cfg.augment, "real_mixstyle", None)
        self._mixstyle_params = MixStyleParams(enabled=bool(getattr(ms_block, "enabled", False)) if ms_block is not None else False,
                                               p_mix=float(getattr(ms_block, "p_mix", 0.3)) if ms_block is not None else 0.3)

        awb_v2_cfg = getattr(cfg.augment, "awb_v2", None)
        self._awb_v2_enabled = bool(getattr(awb_v2_cfg, "enabled", False))
        if self._awb_v2_enabled:
            self.awb_v2 = AdaptiveWaveletBandSBI(awb_sbi_v2_config_from_omegaconf(cfg.augment)).to(device)
            self._awb_v2_gen = torch.Generator(device=device).manual_seed(int(seed))
            self._awb_v2_log_band_probs = bool(getattr(getattr(awb_v2_cfg, "logging", None), "log_band_probs", True))
            self._awb_v2_log_strength = bool(getattr(getattr(awb_v2_cfg, "logging", None), "log_strength", True))
            self._awb_v2_log_soft = bool(getattr(getattr(awb_v2_cfg, "logging", None), "log_soft_label_mean", True))
        else:
            self.awb_v2 = None
            self._awb_v2_gen = None


    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        if self._cnd_enabled and hasattr(self.model.head, "set_grl_lambda"):
            lam = grl_lambda_schedule(epoch, self.cfg.training.epochs, self._grl_lambda_max)
            self.model.head.set_grl_lambda(lam)
        if self._orth_target and epoch >= self._orth_warmup_epoch:
            self.loss_fn.orth_enabled = True
        total = 0.0
        n = 0
        components_running: dict[str, list[float]] = {}
        scores_chunks: list[np.ndarray] = []
        labels_chunks: list[np.ndarray] = []
        t0 = time.perf_counter()
        for step, batch in enumerate(self.train_loader):
            if self.cfg.training.sanity_steps > 0 and step >= self.cfg.training.sanity_steps:
                break
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            wav = batch["wavelet"].to(self.device, non_blocking=True)
            df = batch["defocus"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            target_soft = batch.get("target_soft")
            if target_soft is not None:
                target_soft = target_soft.to(self.device, non_blocking=True)
            is_soft = batch.get("is_soft")
            if is_soft is not None:
                is_soft = is_soft.to(self.device, non_blocking=True)
            method_label = batch.get("method_label")
            if method_label is not None:
                method_label = method_label.to(self.device, non_blocking=True)

            awb_meta: dict = {"awb_v2_enabled": False}
            if self.awb_v2 is not None:
                aug_batch = {"rgb": rgb, "wavelet": wav, "label": labels,
                             "target_soft": target_soft, "is_soft": is_soft}
                v2_in = batch.get("awb_v2_active")
                if v2_in is not None:
                    aug_batch["awb_v2_active"] = v2_in.to(self.device, non_blocking=True)
                    aug_batch["awb_v2_wav_src"] = batch["awb_v2_wav_src"].to(self.device, non_blocking=True)
                    aug_batch["awb_v2_mask"] = batch["awb_v2_mask"].to(self.device, non_blocking=True)
                aug_batch, awb_meta = self.awb_v2(aug_batch, generator=self._awb_v2_gen)
                rgb = aug_batch["rgb"]
                wav = aug_batch["wavelet"]
                labels = aug_batch["label"]
                target_soft = aug_batch.get("target_soft")
                is_soft = aug_batch.get("is_soft")

            if self._mixstyle_params.enabled:
                real_mixstyle_(rgb, labels, self._mixstyle_params)
            self.optimizer.zero_grad(set_to_none=True)
            with self.amp_ctx:
                out = self.model(rgb, wav, df, return_features=True)
                if target_soft is not None:
                    out["target_soft"] = target_soft
                if is_soft is not None:
                    out["is_soft"] = is_soft
                if method_label is not None:
                    out["method_label"] = method_label
                loss, components = self.loss_fn(out, labels)
            if self.awb_v2 is not None and awb_meta.get("awb_v2_enabled", False):
                v2_scalars = self.awb_v2.update_controller(logits=out["logits"].detach(),
                                                           labels=labels.detach(),
                                                           metadata=awb_meta)
                self._log_awb_v2(v2_scalars, target_soft, awb_meta)
            if not torch.isfinite(loss):
                self._dump_bad_batch(epoch, step, batch, loss, components)
                raise RuntimeError(f"non-finite loss at epoch={epoch} step={step}")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.model)
            self.global_step += 1
            total += float(loss.detach().item()) * len(labels)
            n += int(len(labels))
            for k, v in components.items():
                components_running.setdefault(k, []).append(float(v))
            scores_chunks.append(torch.sigmoid(out["logits"].detach().float()).cpu().numpy())
            labels_chunks.append(labels.detach().cpu().numpy().astype(np.int8))
            if (step + 1) % self.cfg.training.log_every == 0:
                self._log_step(loss, grad_norm, components, epoch, step)
        train_loss = total / max(1, n)
        train_components = {k: float(np.mean(v)) for k, v in components_running.items()}
        if scores_chunks:
            s_all = np.concatenate(scores_chunks)
            l_all = np.concatenate(labels_chunks)
            train_metrics = compute_frame_metrics(s_all, l_all)
            train_auc = train_metrics.auc
        else:
            train_auc = float("nan")
            train_metrics = None
        self.tb.add_scalar("train/loss_total_epoch", train_loss, epoch)
        for k, v in train_components.items():
            self.tb.add_scalar(f"train/loss_{k}_epoch", v, epoch)
        if not math.isnan(train_auc):
            self.tb.add_scalar("train/auc", train_auc, epoch)
        return {"loss": train_loss,
                "auc": train_auc,
                "components": train_components,
                "metrics": train_metrics.to_dict() if train_metrics is not None else None,
                "secs": time.perf_counter() - t0}

    def _log_awb_v2(self, scalars: dict[str, float], target_soft, awb_meta: dict) -> None:
        if not scalars:
            return
        if self._awb_v2_log_band_probs:
            self.tb.add_scalar("train/awb_v2_band_prob_LH", scalars["band_prob_LH"], self.global_step)
            self.tb.add_scalar("train/awb_v2_band_prob_HL", scalars["band_prob_HL"], self.global_step)
            self.tb.add_scalar("train/awb_v2_band_prob_HH", scalars["band_prob_HH"], self.global_step)
        if self._awb_v2_log_strength:
            self.tb.add_scalar("train/awb_v2_strength_mean", scalars["strength"], self.global_step)
        self.tb.add_scalar("train/awb_v2_hard_pseudo_rate", scalars["hard_pseudo_rate"], self.global_step)
        self.tb.add_scalar("train/awb_v2_real_guard_active", scalars["real_guard_active"], self.global_step)
        if self._awb_v2_log_soft and target_soft is not None:
            pm = awb_meta.get("pseudo_mask")
            if pm is not None and bool(pm.any()):
                mean_soft = float(target_soft[pm.bool()].float().mean().item())
                self.tb.add_scalar("train/awb_v2_soft_label_mean", mean_soft, self.global_step)

    def _log_step(self, loss, grad_norm, components, epoch, step):
        self.tb.add_scalar("train/loss_total", float(loss), self.global_step)
        for k, v in components.items():
            self.tb.add_scalar(f"train/loss_{k}", float(v), self.global_step)
        self.tb.add_scalar("train/grad_norm", float(grad_norm), self.global_step)
        self.tb.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        log.info(
            f"epoch={epoch} step={step} loss={float(loss):.4f} "
            f"grad={float(grad_norm):.3f} lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )

    @torch.inference_mode()
    def _validate(self, epoch: int) -> dict:
        self.ema.module.eval()
        self.model.eval()
        scores, scores_live, labels = [], [], []
        components_running: dict[str, list[float]] = {}
        per_manip_scores: dict[str, list[float]] = {}
        per_manip_labels: dict[str, list[int]] = {}
        n = 0
        total_loss = 0.0
        for batch in self.val_loader:
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            wav = batch["wavelet"].to(self.device, non_blocking=True)
            df = batch["defocus"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            with self.amp_ctx:
                out = self.ema.module(rgb, wav, df, return_features=True)
                loss, components = self.loss_fn(out, y)
                out_live = self.model(rgb, wav, df, return_features=False)
            total_loss += float(loss.detach().item()) * len(y)
            n += int(len(y))
            for k, v in components.items():
                components_running.setdefault(k, []).append(float(v))
            s = torch.sigmoid(out["logits"].float()).cpu().numpy()
            s_live = torch.sigmoid(out_live["logits"].float()).cpu().numpy()
            l = y.cpu().numpy().astype(np.int8)
            scores.append(s)
            scores_live.append(s_live)
            labels.append(l)
            for m, sc, lab in zip(batch.get("manip", ["?"] * len(y)), s, l):
                per_manip_scores.setdefault(m, []).append(float(sc))
                per_manip_labels.setdefault(m, []).append(int(lab))
        s_all = np.concatenate(scores)
        s_live_all = np.concatenate(scores_live)
        l_all = np.concatenate(labels)
        m = compute_frame_metrics(s_all, l_all)
        m_live = compute_frame_metrics(s_live_all, l_all)
        self.tb.add_scalar("val/auc", m.auc, epoch)
        self.tb.add_scalar("val/auc_live", m_live.auc, epoch)
        self.tb.add_scalar("val/ap", m.ap, epoch)
        self.tb.add_scalar("val/eer", m.eer, epoch)
        self.tb.add_scalar("val/loss_total", total_loss / max(1, n), epoch)
        for manip in per_manip_scores:
            sc = np.array(per_manip_scores[manip])
            lab = np.array(per_manip_labels[manip])
            if len(set(lab.tolist())) < 2:
                continue
            self.tb.add_scalar(f"val/auc_per_manip/{manip}", compute_frame_metrics(sc, lab).auc, epoch)
        auc_for_stop = max(m.auc, m_live.auc)
        log.info(f"val epoch={epoch} auc_ema={m.auc:.4f} auc_live={m_live.auc:.4f} "
                 f"loss={total_loss / max(1, n):.4f}")
        return {"auc": auc_for_stop,
                "auc_ema": m.auc,
                "auc_live": m_live.auc,
                "loss": total_loss / max(1, n),
                "components": {k: float(np.mean(v)) for k, v in components_running.items()},
                "metrics": m.to_dict(),
                "metrics_live": m_live.to_dict()}

    def _dump_bad_batch(self, epoch, step, batch, loss, components):
        path = self.out_dir / "bad_batches" / f"epoch{epoch}_step{step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"epoch": epoch, "step": step, "loss": float(loss.detach().item()) if torch.is_tensor(loss) else None,
                   "components": {k: float(v) for k, v in components.items()}, "video_ids": list(batch.get("video_id", [])),
                   "manip": list(batch.get("manip", []))}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


    def fit(self) -> TrainerOutput:
        meta = get_run_meta(self.seed)
        with open(self.out_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        start = time.time()
        best_auc, best_epoch = -float("inf"), -1
        best_components: dict = {}
        epochs_run = 0
        early_stopped = False
        train_loss_last, val_loss_last, val_auc_last = float("nan"), float("nan"), float("nan")
        lr_history: list[float] = []

        for epoch in range(self.cfg.training.epochs):
            stats = self._train_one_epoch(epoch)
            train_loss_last = float(stats["loss"])
            train_auc = float(stats["auc"]) if stats["auc"] == stats["auc"] else float("nan")
            comp = stats["components"]
            comp_str = " ".join(f"{k}={v:.4f}" for k, v in comp.items())
            log.info(f"train epoch={epoch} auc={train_auc:.4f} loss={train_loss_last:.4f} "
                     f"components[{comp_str}] secs={stats['secs']:.1f}")
            lr_history.append(self.optimizer.param_groups[0]["lr"])
            if self.cfg.training.eval_during_training:
                vs = self._validate(epoch)
                val_loss_last = float(vs["loss"])
                val_auc_last = float(vs["auc"])
                payload = {"model": self.model.state_dict(),
                           "ema": self.ema.state_dict(),
                           "epoch": epoch,
                           "auc": val_auc_last,
                           "components": vs["components"]}
                if self.ckpt.update(val_auc_last, epoch, payload):
                    best_auc, best_epoch = val_auc_last, epoch
                    best_components = dict(vs["components"])
                self.ckpt.write_last(payload)
                if self.es.step(val_auc_last):
                    log.info(f"early stop at epoch={epoch} (best={best_auc:.4f} @ {best_epoch})")
                    early_stopped = True
                    epochs_run = epoch + 1
                    break
            epochs_run = epoch + 1

        self.tb.close()
        return TrainerOutput(best_metric=best_auc, best_epoch=best_epoch, epochs_run=epochs_run,
                             early_stopped=early_stopped, final_train_loss=train_loss_last, final_val_loss=val_loss_last,
                             final_val_auc=val_auc_last, loss_components_at_best=best_components,
                             wallclock_seconds=time.time() - start, lr_history=lr_history)