"""QEM mesh decimation via pyfqmr (Fast Quadric Mesh Reduction)."""

from __future__ import annotations

import warnings

import numpy as np

from xeltofab.state import PipelineState


def decimate(state: PipelineState) -> PipelineState:
    """Reduce mesh face count using Quadric Error Metrics edge collapse.

    Computes target face count from target_faces (if set) or decimate_ratio.
    Preserves boundary edges to protect marching cubes domain boundaries.
    3D only. No-op for 2D contours or when decimate is disabled.
    Updates vertices and faces; clears smoothed_vertices.
    """
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state
    if not state.params.decimate:
        return state

    try:
        import pyfqmr
    except ImportError:
        warnings.warn(
            "pyfqmr not installed — skipping QEM decimation. Install with: uv add pyfqmr",
            stacklevel=2,
        )
        return state

    vertices = np.asarray(state.best_vertices, dtype=np.float64)
    faces = np.asarray(state.faces, dtype=np.int32)

    # Compute target face count
    if state.params.target_faces is not None:
        target_count = state.params.target_faces
    else:
        target_count = max(4, int(len(faces) * state.params.decimate_ratio))

    # Skip if mesh is already at or below target
    if len(faces) <= target_count:
        return state

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(
        target_count=target_count,
        aggressiveness=state.params.decimate_aggressiveness,
        preserve_border=True,
        verbose=False,
    )
    new_vertices, new_faces, _ = simplifier.getMesh()

    return state.model_copy(
        update={
            "vertices": new_vertices,
            "faces": new_faces,
            "smoothed_vertices": None,
        }
    )
