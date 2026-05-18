from __future__ import annotations

import os
import csv
import json
import math
import tempfile
import numpy as np
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _atomic_write(path: Path, payload_writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False, suffix=".tmp", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        payload_writer(tmp)
    os.replace(tmp_path, path)


def _finite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_finite(x) for x in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(x) for x in obj]
    return obj


def save_json(path: str | os.PathLike[str], obj: Mapping[str, Any] | Sequence[Any], indent: int = 2) -> None:
    out = Path(path)
    sanitised = _finite(obj)
    _atomic_write(out, lambda f: json.dump(sanitised, f, indent=indent, default=str, allow_nan=False))


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: str | os.PathLike[str], header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    out = Path(path)

    def writer(f) -> None:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    _atomic_write(out, writer)


def save_npz(path: str | os.PathLike[str], **arrays: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, out)


def save_svg(path: str | os.PathLike[str], fig) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")