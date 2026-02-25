"""CSV/TXT loader for density fields."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from CSV or whitespace-delimited text file.

    If shape is provided, data is loaded as flat values and reshaped.
    Otherwise, 2D table structure is inferred from rows/columns.
    """
    # Try comma-delimited first, fall back to whitespace
    for delimiter in (",", None):
        try:
            data = np.loadtxt(path, delimiter=delimiter)
            break
        except ValueError:
            if delimiter is None:
                # Both delimiters failed — try skipping header
                try:
                    data = np.genfromtxt(path, delimiter=",", skip_header=1)
                    if np.isnan(data).all():
                        data = np.genfromtxt(path, skip_header=1)
                    break
                except ValueError:
                    raise
            continue
    else:
        # This shouldn't be reached, but just in case
        data = np.genfromtxt(path, delimiter=",", skip_header=1)

    if shape is not None:
        total = int(np.prod(shape))
        flat = data.ravel()
        if flat.size != total:
            raise ValueError(
                f"Cannot reshape {flat.size} values into shape {shape} "
                f"(requires {total} values)"
            )
        return flat.reshape(shape)

    return data
