"""MATLAB .mat file loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

# Well-known MATLAB TO variable names, checked in priority order
_KNOWN_NAMES = ("xPhys", "densities", "x", "rho", "dc", "density")


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from a MATLAB .mat file.

    Auto-detects common TO variable names if field_name is not specified.
    """
    try:
        data = scipy.io.loadmat(path)
    except NotImplementedError:
        raise ValueError(
            f"Cannot read {path.name} — this appears to be a MATLAB v7.3+ file (HDF5 format).\n"
            "Save it as .h5 or resave in MATLAB with: save('file.mat', '-v7')"
        ) from None

    # Filter out MATLAB metadata keys
    user_keys = [k for k in data if not k.startswith("__")]

    if field_name is not None:
        if field_name not in user_keys:
            raise KeyError(f"Field '{field_name}' not found in {path.name}. Available: {user_keys}")
        return np.asarray(data[field_name], dtype=np.float64)

    # Auto-detect: try known names in priority order
    for name in _KNOWN_NAMES:
        if name in user_keys:
            return np.asarray(data[name], dtype=np.float64)

    # Fallback: single variable → use it; multiple → error
    if len(user_keys) == 1:
        return np.asarray(data[user_keys[0]], dtype=np.float64)

    raise ValueError(
        f"Multiple variables found in {path.name}: {user_keys}\n"
        "Specify which one with --field-name"
    )
