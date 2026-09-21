from __future__ import annotations


import torch
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from wiser.utils.repro import worker_init_fn


def collate_with_meta(batch: list[dict]) -> dict:
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        vs = [b[k] for b in batch]
        if torch.is_tensor(vs[0]):
            out[k] = torch.stack(vs)
        else:
            out[k] = vs
    return out


def build_loader(dataset, *, batch_size: int, num_workers: int, shuffle: bool, sampler=None, pin_memory: bool = True,
                 persistent_workers: bool = True, prefetch_factor: int = 4, drop_last: bool = False, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle if sampler is None else False,
                      sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers and num_workers > 0,
                      prefetch_factor=prefetch_factor if num_workers > 0 else None,
                      drop_last=drop_last,
                      worker_init_fn=worker_init_fn,
                      generator=g,
                      collate_fn=collate_with_meta)


def find_ffpp_splits(cache_root: Path) -> Path:
    p = Path(cache_root) / "ffpp" / "splits.json"
    if not p.exists():
        raise FileNotFoundError(
            f"FF++ splits not found at {p}. Run scripts/preprocess_ffpp.py first."
        )
    return p


def device_from_cfg(_cfg) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")