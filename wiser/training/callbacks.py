from __future__ import annotations

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class CheckpointState:
    best_metric: float = -float("inf")
    best_epoch: int = -1


class BestCheckpoint:
    def __init__(self, ckpt_dir: Path, *, mode: str = "max") -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.state = CheckpointState()

    def update(self, metric: float, epoch: int, payload: dict) -> bool:
        better = metric > self.state.best_metric if self.mode == "max" else metric < self.state.best_metric
        if better or self.state.best_epoch < 0:
            self.state.best_metric = metric
            self.state.best_epoch = epoch
            torch.save(payload, self.ckpt_dir / "best.pt")
            return True
        return False

    def write_last(self, payload: dict) -> None:
        torch.save(payload, self.ckpt_dir / "last.pt")


class EarlyStopping:
    def __init__(self, patience: int, *, mode: str = "max") -> None:
        self.patience = int(patience)
        self.mode = mode
        self.best: Optional[float] = None
        self.bad: int = 0

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience