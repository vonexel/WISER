from __future__ import annotations
import sys
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from wiser.configs.schemas import register_configs
from wiser.data.celebdfpp_dataset import CelebDFPPDataset
from wiser.data.ffpp_dataset import FFPPFrameDataset
from wiser.evaluation.evaluator import evaluate_full
from wiser.models.wiser import build_from_cfg, count_parameters
from wiser.training.utils import build_loader, device_from_cfg, find_ffpp_splits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    register_configs()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="Override config yaml")
    args = p.parse_args()

    config_yaml = args.config or str(Path(args.out) / "config_resolved.yaml")
    cfg = OmegaConf.load(config_yaml)

    device = device_from_cfg(cfg)
    cache_root = Path(cfg.data.paths.cache_root)
    splits_path = find_ffpp_splits(cache_root)

    test_ds = FFPPFrameDataset(cache_root=cache_root, splits_path=splits_path, split="test",
                               data_cfg=cfg.data, augment_cfg=cfg.augment, train=False)
    test_loader = build_loader(test_ds, batch_size=cfg.data.eval_batch_size,
                               num_workers=cfg.data.num_workers, shuffle=False,
                               drop_last=False, seed=args.seed, prefetch_factor=2)
    try:
        cross_ds = CelebDFPPDataset(cache_root=cache_root, data_cfg=cfg.data)
        cross_loader = build_loader(cross_ds, batch_size=cfg.data.eval_batch_size,
                                    num_workers=cfg.data.num_workers, shuffle=False,
                                    drop_last=False, seed=args.seed, prefetch_factor=2)
    except RuntimeError:
        cross_loader = None

    model = build_from_cfg(cfg.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()

    metrics = evaluate_full(model=model, in_loader=test_loader, cross_loader=cross_loader,
                            device=device, out_dir=Path(args.out),
                            extras={"experiment_id": args.experiment,
                                    "seed": int(args.seed),
                                    "params_total": count_parameters(model),
                                    "params_trainable": count_parameters(model)})
    print(f"wrote metrics to {Path(args.out) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())