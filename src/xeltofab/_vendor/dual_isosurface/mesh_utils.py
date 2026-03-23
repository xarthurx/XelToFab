# Vendored from sdftoolbox (MIT License) — https://github.com/cheind/sdftoolbox
# Modifications: extracted only triangulate_quads and compute_face_normals.

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


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face normals for a mesh. Assumes CCW winding order.

    Params:
        verts: (N, 3) array of vertices
        faces: (M, F) array of face indices (F=3 for tris, F=4 for quads)

    Returns:
        normals: (M, 3) array of unit face normals
    """
    xyz = verts[faces]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, -1] - xyz[:, 0], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    normals /= norms
    return normals
