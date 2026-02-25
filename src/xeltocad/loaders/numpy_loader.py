"""NumPy .npy/.npz loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from .npy or .npz file."""
    if path.suffix == ".npz":
        data = np.load(path)
        if field_name is not None:
            if field_name not in data:
                raise KeyError(f"Field '{field_name}' not found in {path.name}. Available: {list(data.keys())}")
            return data[field_name]
        key = "density" if "density" in data else list(data.keys())[0]
        return data[key]
    else:
        return np.load(path)
