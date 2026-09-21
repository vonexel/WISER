from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from wiser.configs.schemas import register_configs
from wiser.data.celebdfpp_dataset import CelebDFPPDataset
from wiser.data.ffpp_dataset import FFPPFrameDataset, PairedRealFakeSampler
from wiser.evaluation.evaluator import evaluate_full
from wiser.models.losses import CombinedLoss
from wiser.models.wiser import build_from_cfg, count_parameters
from wiser.training.trainer import Trainer
from wiser.training.utils import build_loader, device_from_cfg, find_ffpp_splits
from wiser.utils import save_json, setup_logging
from wiser.utils.logging import get_logger
from wiser.utils.repro import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
register_configs()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    log = get_logger("wiser.train")
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(int(cfg.seed), deterministic=True)

    out_dir = Path(cfg.output_root) / cfg.experiment_id / str(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = device_from_cfg(cfg)
    log.info(f"device={device}")

    cache_root = Path(cfg.data.paths.cache_root)
    splits_path = find_ffpp_splits(cache_root)

    train_ds = FFPPFrameDataset(
        cache_root=cache_root, splits_path=splits_path, split="train",
        data_cfg=cfg.data, augment_cfg=cfg.augment, train=True)
    val_ds = FFPPFrameDataset(
        cache_root=cache_root, splits_path=splits_path, split="val",
        data_cfg=cfg.data, augment_cfg=cfg.augment, train=False)
    test_ds = FFPPFrameDataset(
        cache_root=cache_root, splits_path=splits_path, split="test",
        data_cfg=cfg.data, augment_cfg=cfg.augment, train=False)
    paired = PairedRealFakeSampler(
        train_ds, rounds_per_epoch=int(getattr(cfg.data, "rounds_per_epoch", 1)))
    train_loader = build_loader(
        train_ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        shuffle=False, sampler=paired, pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers, prefetch_factor=cfg.data.prefetch_factor,
        drop_last=True, seed=int(cfg.seed))
    val_loader = build_loader(
        val_ds, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers,
        shuffle=False, pin_memory=cfg.data.pin_memory, persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=2, drop_last=False, seed=int(cfg.seed))
    test_loader = build_loader(
        test_ds, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers,
        shuffle=False, pin_memory=cfg.data.pin_memory, persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=2, drop_last=False, seed=int(cfg.seed))
    cross_loader = None
    try:
        cross_ds = CelebDFPPDataset(cache_root=cache_root, data_cfg=cfg.data)
        cross_loader = build_loader(
            cross_ds, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers,
            shuffle=False, pin_memory=cfg.data.pin_memory, persistent_workers=cfg.data.persistent_workers,
            prefetch_factor=2, drop_last=False, seed=int(cfg.seed))
        log.info(f"cross-domain dataset size: {len(cross_ds)}")
    except RuntimeError as e:
        log.warning(f"cross-domain dataset unavailable: {e}")

    model = build_from_cfg(cfg.model)
    n_params = count_parameters(model)
    log.info(f"model params: {n_params:,}")

    loss_fn = CombinedLoss(cfg.loss)
    trainer = Trainer(
        model=model, loss_fn=loss_fn, cfg=cfg, train_loader=train_loader,
        val_loader=val_loader, out_dir=out_dir, device=device, seed=int(cfg.seed))
    out = trainer.fit()
    log.info(
        f"training done: best_auc={out.best_metric:.4f} epoch={out.best_epoch} "
        f"epochs_run={out.epochs_run} time={out.wallclock_seconds:.1f}s")


    best_path = out_dir / "ckpt" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        trainer.ema.load_state_dict(ckpt["ema"])
        trainer.ema.module.eval()
    cal_block = getattr(cfg.training, "calibration", None)
    enable_calibration = bool(getattr(cal_block, "enabled", False)) if cal_block is not None else False
    pool_cal = (str(getattr(cal_block, "video_pool_calibrated", "median")) if cal_block is not None else "median")
    pool_uncal = (str(getattr(cal_block, "video_pool_uncalibrated", "mean")) if cal_block is not None else "mean")
    metrics = evaluate_full(
        model=trainer.ema.module, in_loader=test_loader, cross_loader=cross_loader,
        device=device, out_dir=out_dir,
        val_loader=val_loader if enable_calibration else None,
        enable_calibration=enable_calibration,
        video_pool_calibrated=pool_cal,
        video_pool_uncalibrated=pool_uncal,
        extras={
            "experiment_id": cfg.experiment_id,
            "seed": int(cfg.seed),
            "params_total": n_params,
            "params_trainable": n_params,
            "best_epoch": out.best_epoch,
            "wallclock_seconds": out.wallclock_seconds,
            "training": {"epochs_run": out.epochs_run,
                         "early_stopped": out.early_stopped,
                         "final_train_loss": out.final_train_loss,
                         "final_val_loss": out.final_val_loss,
                         "final_val_auc": out.final_val_auc,
                         "lr_history": out.lr_history,
                         "loss_components_at_best": out.loss_components_at_best}})
    log.info("eval complete; metrics.json written to %s", out_dir)


if __name__ == "__main__":
    main()