from __future__ import annotations

import sys
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from wiser.data.celebdfpp_dataset import CelebDFPPDataset
from wiser.data.ffpp_dataset import FFPPFrameDataset
from wiser.evaluation.robustness import run_robustness
from wiser.models.wiser import build_from_cfg
from wiser.training.utils import build_loader, device_from_cfg, find_ffpp_splits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    config_yaml = args.config or str(Path(args.ckpt).resolve().parent.parent / "config_resolved.yaml")
    cfg = OmegaConf.load(config_yaml)
    device = device_from_cfg(cfg)
    cache_root = Path(cfg.data.paths.cache_root)

    splits_path = find_ffpp_splits(cache_root)
    in_ds = FFPPFrameDataset(
        cache_root=cache_root, splits_path=splits_path, split="test",
        data_cfg=cfg.data, augment_cfg=cfg.augment, train=False)
    in_loader = build_loader(in_ds, batch_size=cfg.data.eval_batch_size,
                             num_workers=cfg.data.num_workers, shuffle=False,
                             drop_last=False, seed=0, prefetch_factor=2)
    cross_loader = None
    try:
        cross_ds = CelebDFPPDataset(cache_root=cache_root, data_cfg=cfg.data)
        cross_loader = build_loader(cross_ds, batch_size=cfg.data.eval_batch_size,
                                    num_workers=cfg.data.num_workers, shuffle=False,
                                    drop_last=False, seed=0, prefetch_factor=2)
    except RuntimeError:
        pass

    model = build_from_cfg(cfg.model).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = state.get("ema", state.get("model", state))
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_robustness(model, in_loader, device=device, out_path=out_dir / "robustness_indomain.json")
    if cross_loader is not None:
        run_robustness(model, cross_loader, device=device, out_path=out_dir / "robustness_crossdomain.json")
    print("robustness JSONs written under", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())