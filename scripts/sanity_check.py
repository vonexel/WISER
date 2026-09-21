from __future__ import annotations

import sys
import time
import torch
from pathlib import Path
from omegaconf import OmegaConf
from wiser.models.losses import FocalBCEWithLogits
from wiser.models.wiser import build_from_cfg
from wiser.utils import setup_logging
from wiser.utils.logging import get_logger
from wiser.utils.repro import seed_everything


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> int:
    setup_logging("INFO")
    log = get_logger("sanity")
    seed_everything(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.create(
        {
            "embed_dim": 256, "num_classes": 1, "dropout": 0.3,
            "cdc": {"in_ch": 3, "out_ch": 32, "theta": 0.7},
            "wavelet": {"enabled": True, "use_defocus": True, "high_freq_gate": True},
            "bissm": {"enabled": True, "d_state": 16, "d_conv": 4, "expand": 1, "bidirectional": True},
            "ssca": {"enabled": True, "dim": 256, "heads": 1},
            "compile": False,
        }
    )
    model = build_from_cfg(cfg).to(device)
    loss_fn = FocalBCEWithLogits(alpha=0.25, gamma=2.0).to(device)

    rgb = torch.rand(8, 3, 256, 256, device=device)
    wav = torch.rand(8, 12, 128, 128, device=device)
    df = torch.rand(8, 1, 128, 128, device=device)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    target_loss = 0.05
    t0 = time.perf_counter()
    final_loss = float("inf")
    for step in range(100):
        out = model(rgb, wav, df)
        loss = loss_fn(out["logits"], labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        final_loss = float(loss.detach())
        if step % 10 == 0:
            log.info(f"step={step:3d} loss={final_loss:.4f}")
        if final_loss < target_loss:
            log.info(f"converged at step={step} loss={final_loss:.4f} ({time.perf_counter()-t0:.1f}s)")
            return 0
    log.warning(f"did not reach target_loss={target_loss}; final={final_loss:.4f}")
    return 1 if final_loss > 0.20 else 0


if __name__ == "__main__":
    sys.exit(main())