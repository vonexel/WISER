from __future__ import annotations


import sys
import json
import torch
import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from wiser.utils.repro import seed_everything
from hydra import compose, initialize_config_dir
from wiser.configs.schemas import register_configs
from wiser.data.ffpp_dataset import FFPPFrameDataset
from wiser.evaluation.evaluator import evaluate_full
from wiser.utils.logging import get_logger, setup_logging
from wiser.data.celebdfpp_dataset import CelebDFPPDataset
from wiser.models.wiser import build_from_cfg, count_parameters
from wiser.training.utils import build_loader, device_from_cfg, find_ffpp_splits


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
register_configs()


def _load_e01_config(seed: int) -> DictConfig:
    conf_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config",
                      overrides=["experiment = E13_calibration_only",
                                 f"seed={seed}"])
    return cfg


def main() -> int:
    setup_logging("INFO")
    log = get_logger("calibrate_only")
    p = argparse.ArgumentParser()
    p.add_argument("--src-ckpt", required=True, help="Path to a trained best.pt (EMA weights).")
    p.add_argument("--out-dir", required=True, help="Output directory for E13 metrics.json + predictions.npz.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = _load_e01_config(args.seed)
    seed_everything(int(cfg.seed), deterministic=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = device_from_cfg(cfg)
    cache_root = Path(cfg.data.paths.cache_root)
    splits_path = find_ffpp_splits(cache_root)

    val_ds = FFPPFrameDataset(cache_root=cache_root, splits_path=splits_path, split="val",
                              data_cfg=cfg.data, augment_cfg=cfg.augment, train=False)
    test_ds = FFPPFrameDataset(cache_root=cache_root, splits_path=splits_path, split="test",
                               data_cfg=cfg.data, augment_cfg=cfg.augment, train=False)
    val_loader = build_loader(val_ds, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers,
                              shuffle=False, pin_memory=cfg.data.pin_memory,
                              persistent_workers=cfg.data.persistent_workers,
                              prefetch_factor=2, drop_last=False, seed=int(cfg.seed))
    test_loader = build_loader(test_ds, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers,
                               shuffle=False, pin_memory=cfg.data.pin_memory,
                               persistent_workers=cfg.data.persistent_workers,
                               prefetch_factor=2, drop_last=False, seed=int(cfg.seed))
    cross_loader = None
    try:
        cross_ds = CelebDFPPDataset(cache_root=cache_root, data_cfg=cfg.data)
        cross_loader = build_loader(cross_ds, batch_size=cfg.data.eval_batch_size,
                                    num_workers=cfg.data.num_workers, shuffle=False,
                                    pin_memory=cfg.data.pin_memory,
                                    persistent_workers=cfg.data.persistent_workers,
                                    prefetch_factor=2, drop_last=False, seed=int(cfg.seed))
    except RuntimeError as e:
        log.warning(f"cross-domain dataset unavailable: {e}")

    model = build_from_cfg(cfg.model)
    n_params = count_parameters(model)
    src_ckpt = Path(args.src_ckpt)
    if not src_ckpt.exists():
        log.error(f"source checkpoint not found: {src_ckpt}")
        return 2
    state = torch.load(src_ckpt, map_location=device, weights_only=True)
    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        model.load_state_dict(state["ema"]["module"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    log.info(f"loaded source checkpoint from {src_ckpt}")

    metrics = evaluate_full(
        model=model, in_loader=test_loader, cross_loader=cross_loader,
        device=device, out_dir=out_dir,
        val_loader=val_loader, enable_calibration=True,
        video_pool_calibrated="median", video_pool_uncalibrated="mean",
        extras={"experiment_id": "E03_calibration_only",
                "seed": int(cfg.seed),
                "params_total": n_params,
                "source_checkpoint": str(src_ckpt)})
    with open(out_dir / "metrics.json", "r") as f:
        log.info(f"metrics.json written to {out_dir} (schema {json.load(f)['schema_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())