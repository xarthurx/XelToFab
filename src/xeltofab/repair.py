"""Watertight mesh repair via pymeshlab."""

from __future__ import annotations

import warnings

from xeltofab.state import PipelineState


def repair(state: PipelineState) -> PipelineState:
    """Repair mesh topology: fix non-manifold edges/vertices, remove duplicates.

    3D only. No-op for 2D contours or when repair is disabled.
    Updates vertices and faces; clears smoothed_vertices.
    """
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state
    if not state.params.repair:
        return state

    try:
        import pymeshlab
    except ImportError:
        warnings.warn(
            "pymeshlab not installed — skipping mesh repair. "
            "Install with: uv sync --extra mesh-quality",
            stacklevel=2,
        )
        return state

    vertices = state.best_vertices
    faces = state.faces

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))

    # Fix non-manifold topology
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    # Remove degenerate geometry
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Recompute normals for consistency
    ms.compute_normal_per_face()
    ms.compute_normal_per_vertex()

    out = ms.current_mesh()
    if not out.is_compact():
        out.compact()

    return state.model_copy(update={
        "vertices": out.vertex_matrix(),
        "faces": out.face_matrix(),
        "smoothed_vertices": None,
    })
