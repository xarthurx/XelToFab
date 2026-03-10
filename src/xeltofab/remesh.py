"""Isotropic remeshing via gpytoolbox (Botsch & Kobbelt algorithm)."""

from __future__ import annotations

import warnings

import numpy as np

from xeltofab.state import PipelineState


def remesh(state: PipelineState) -> PipelineState:
    """Apply isotropic explicit remeshing to produce uniform triangles.

    Uses gpytoolbox.remesh_botsch (edge split, collapse, flip, tangential smoothing).
    Auto-computes target edge length from average edge length if not specified.
    3D only. No-op for 2D contours or when remesh is disabled.
    Updates vertices and faces; clears smoothed_vertices.

    Note: boundary edges (where marching cubes clips at domain edges) are preserved
    by the algorithm and may retain lower-quality triangles. Interior triangles
    (typically 99%+ of the mesh) achieve FEA-ready quality.
    """
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state
    if not state.params.remesh:
        return state

    try:
        import gpytoolbox
    except ImportError:
        warnings.warn(
            "gpytoolbox not installed — skipping isotropic remeshing. Install with: uv sync --extra mesh-quality",
            stacklevel=2,
        )
        return state

    vertices = state.best_vertices.astype(np.float64, copy=False)
    faces = state.faces.astype(np.int32, copy=False)

    # Auto-compute target edge length from average if not specified
    if state.params.target_edge_length is not None:
        h = state.params.target_edge_length
    else:
        e0 = np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1)
        e1 = np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 1]], axis=1)
        e2 = np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=1)
        h = float(np.mean(np.concatenate([e0, e1, e2])))

    new_vertices, new_faces = gpytoolbox.remesh_botsch(
        vertices, faces, i=state.params.remesh_iterations, h=h, project=True
    )

    return state.model_copy(
        update={
            "vertices": new_vertices,
            "faces": new_faces,
            "smoothed_vertices": None,
        }
    )
