from __future__ import annotations

import time
import torch
import argparse
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional
from wiser.models.cdc import CDCStem
from wiser.models.classifier_head import ClassifierHead
from wiser.models.disentangle import NUM_METHODS, ClassifierHeadCND
from wiser.models.ssca import SSCA
from wiser.models.stem import RGBBackbone
from wiser.models.wavelet_stream import WaveletStream


class WISER(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 1, dropout: float = 0.3, cdc: Optional[Dict[str, Any]] = None,
                 wavelet: Optional[Dict[str, Any]] = None, bissm: Optional[Dict[str, Any]] = None,
                 ssca: Optional[Dict[str, Any]] = None, cnd: Optional[Dict[str, Any]] = None, compile: bool = False) -> None:
        super().__init__()
        cdc = cdc or {}
        wavelet = wavelet or {}
        bissm = bissm or {}
        ssca = ssca or {}
        cnd = cnd or {}
        self.cfg = {"embed_dim": embed_dim, "num_classes": num_classes, "dropout": dropout, "cdc": cdc, "wavelet": wavelet,
                    "bissm": bissm, "ssca": ssca, "cnd": cnd, "compile": compile}
        # stem
        self.stem = CDCStem(in_ch=cdc.get("in_ch", 3), out_ch=cdc.get("out_ch", 32), theta=cdc.get("theta", 0.7))
        # RGB-backbone
        bissm_kwargs = {"d_state": bissm.get("d_state", 16), "d_conv": bissm.get("d_conv", 4),
                        "expand": bissm.get("expand", 1), "bidirectional": bissm.get("bidirectional", True)}
        self.backbone = RGBBackbone(bissm_enabled=bissm.get("enabled", True), bissm_kwargs=bissm_kwargs)
        # wavelet stream
        self.wavelet_enabled = wavelet.get("enabled", True)
        if self.wavelet_enabled:
            self.wavelet_stream = WaveletStream(in_ch_wav=12, in_ch_def=1 if wavelet.get("use_defocus", True) else 0,
                                                high_freq_gate=wavelet.get("high_freq_gate", True))
        else:
            self.wavelet_stream = _DefocusOnlyStream()
        # fusion
        self.ssca = SSCA(dim_rgb=256, dim_wav=128, dim=256, enabled=ssca.get("enabled", True))
        self.cnd_enabled = bool(cnd.get("enabled", False))
        if self.cnd_enabled:
            self.head = ClassifierHeadCND(in_dim=256, zc_dim=int(cnd.get("zc_dim", 192)), zn_dim=int(cnd.get("zn_dim", 64)),
                                          dropout=dropout, num_methods=int(cnd.get("num_methods", NUM_METHODS)))
        else:
            self.head = ClassifierHead(in_dim=256, embed_dim=embed_dim, dropout=dropout)
        self.freq_proj = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self._compile_requested = bool(compile)

    def forward(self, rgb: torch.Tensor, wavelet: torch.Tensor, defocus: Optional[torch.Tensor] = None, *,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        x = self.stem(rgb)                                # (B, 32, 128, 128)
        rgb_feat = self.backbone(x)                       # (B, 256, 8, 8)
        wav_feat = self.wavelet_stream(wavelet, defocus)  # (B, 128, 8, 8)
        fused = self.ssca(rgb_feat, wav_feat)             # (B, 256, 8, 8)
        if self.cnd_enabled:
            logits, z_c, z_n, method_logits_zn, method_logits_zc = self.head(fused)
            out: Dict[str, torch.Tensor] = {"logits": logits.squeeze(-1), "embeddings": z_c,
                                            "z_c": z_c, "z_n": z_n,
                                            "method_logits_zn": method_logits_zn, "method_logits_zc": method_logits_zc,
                                            "rgb_feat": rgb_feat,
                                            "wavelet_feat": wav_feat,
                                            "fused": fused}
        else:
            logits, embeddings = self.head(fused)
            out = {"logits": logits.squeeze(-1), "embeddings": embeddings,
                   "rgb_feat": rgb_feat, "wavelet_feat": wav_feat, "fused": fused}
        if return_features:
            out["freq_projection"] = self.freq_proj(rgb_feat)
        return out


class _DefocusOnlyStream(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True))

    def forward(self, _wavelet: torch.Tensor, defocus: torch.Tensor | None) -> torch.Tensor:
        if defocus is None:
            raise RuntimeError("WDJ-disabled (defocus-only) path requires the defocus tensor")
        return self.net(defocus)


def build_from_cfg(cfg: DictConfig) -> WISER:
    cnd_cfg = cfg.get("cnd", None) if hasattr(cfg, "get") else None
    cnd_dict = OmegaConf.to_container(cnd_cfg, resolve=True) if cnd_cfg is not None else None
    return WISER(embed_dim=cfg.embed_dim,
                 num_classes=cfg.num_classes,
                 dropout=cfg.dropout,
                 cdc=OmegaConf.to_container(cfg.cdc, resolve=True),
                 wavelet=OmegaConf.to_container(cfg.wavelet, resolve=True),
                 bissm=OmegaConf.to_container(cfg.bissm, resolve=True),
                 ssca=OmegaConf.to_container(cfg.ssca, resolve=True),
                 cnd=cnd_dict,
                 compile=cfg.get("compile", False))


def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not only_trainable))


def parameter_table(model: WISER) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for name, module in model.named_children():
        rows.append((name, count_parameters(module)))
    rows.append(("total", count_parameters(model)))
    return rows


def _print_summary() -> None:
    cfg = OmegaConf.create(
        {
            "embed_dim": 256,
            "num_classes": 1,
            "dropout": 0.3,
            "cdc": {"in_ch": 3, "out_ch": 32, "theta": 0.7},
            "wavelet": {"enabled": True, "use_defocus": True, "high_freq_gate": True},
            "bissm": {"enabled": True, "d_state": 16, "d_conv": 4, "expand": 1, "bidirectional": True},
            "ssca": {"enabled": True, "dim": 256, "heads": 1},
            "compile": False,
        }
    )
    model = build_from_cfg(cfg)
    print(model)
    for name, n in parameter_table(model):
        print(f"  {name:24s} {n:>10,}")


def _benchmark() -> None:
    cfg = OmegaConf.create(
        {
            "embed_dim": 256,
            "num_classes": 1,
            "dropout": 0.3,
            "cdc": {"in_ch": 3, "out_ch": 32, "theta": 0.7},
            "wavelet": {"enabled": True, "use_defocus": True, "high_freq_gate": True},
            "bissm": {"enabled": True, "d_state": 16, "d_conv": 4, "expand": 1, "bidirectional": True},
            "ssca": {"enabled": True, "dim": 256, "heads": 1},
            "compile": False,
        }
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_from_cfg(cfg).to(device).eval()
    rgb = torch.rand(64, 3, 256, 256, device=device)
    wav = torch.rand(64, 12, 128, 128, device=device)
    df = torch.rand(64, 1, 128, 128, device=device)
    with torch.no_grad():
        # warm-up
        for _ in range(3):
            model(rgb, wav, df)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(rgb, wav, df)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
    print(f"forward batch 64 on {device}: {dt*1000:.2f} ms ({dt*1000/64:.2f} ms/frame)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--print-summary", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    if args.print_summary:
        _print_summary()
    if args.benchmark:
        _benchmark()
    if not (args.print_summary or args.benchmark):
        _print_summary()