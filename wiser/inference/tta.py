from __future__ import annotations

import torch
from typing import Optional


@torch.inference_mode()
def flip_tta_logits(model: torch.nn.Module, rgb: torch.Tensor, wavelet: torch.Tensor, defocus: Optional[torch.Tensor] = None) -> torch.Tensor:
    out_a = model(rgb, wavelet, defocus)
    rgb_f = torch.flip(rgb, dims=(-1,))
    wav_f = torch.flip(wavelet, dims=(-1,))
    df_f = torch.flip(defocus, dims=(-1,)) if defocus is not None else None
    out_b = model(rgb_f, wav_f, df_f)
    return 0.5 * (out_a["logits"] + out_b["logits"])