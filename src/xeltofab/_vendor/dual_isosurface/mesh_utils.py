# Vendored from sdftoolbox (MIT License) — https://github.com/cheind/sdftoolbox
# Modifications: extracted only triangulate_quads.

from __future__ import annotations

import numpy as np


def triangulate_quads(quads: np.ndarray) -> np.ndarray:
    """Triangulate a quad mesh. Assumes CCW winding order.

    Params:
        quads: (M, 4) array of quad face indices

    Returns:
        tris: (M*2, 3) array of triangle face indices
    """
    tris = np.empty((quads.shape[0], 2, 3), dtype=quads.dtype)
    tris[:, 0, :] = quads[:, [0, 1, 2]]
    tris[:, 1, :] = quads[:, [0, 2, 3]]
    return tris.reshape(-1, 3)
