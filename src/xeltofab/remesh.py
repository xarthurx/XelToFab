"""Isotropic remeshing via pymeshlab."""

from __future__ import annotations

import warnings

from xeltofab.state import PipelineState


def remesh(state: PipelineState) -> PipelineState:
    """Apply uniform remeshing to produce regular triangles.

    Uses generate_resampled_uniform_mesh with cellsize controlling triangle size.
    Auto-computes target edge length from average edge length if not specified.
    3D only. No-op for 2D contours or when remesh is disabled.
    Updates vertices and faces; clears smoothed_vertices.
    """
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state
    if not state.params.remesh:
        return state

    try:
        import pymeshlab
    except ImportError:
        warnings.warn(
            "pymeshlab not installed — skipping isotropic remeshing. "
            "Install with: uv sync --extra mesh-quality",
            stacklevel=2,
        )
        return state

    vertices = state.best_vertices
    faces = state.faces

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))

    # Determine target edge length (used as cellsize)
    if state.params.target_edge_length is not None:
        cellsize = pymeshlab.PureValue(state.params.target_edge_length)
    else:
        avg_len = ms.get_geometric_measures()["avg_edge_length"]
        cellsize = pymeshlab.PureValue(avg_len)

    # Ensure normals exist for the reconstruction
    ms.compute_normal_per_face()
    ms.compute_normal_per_vertex()

    ms.generate_resampled_uniform_mesh(cellsize=cellsize)

    # The resampled mesh is added as a new layer; switch to it
    ms.set_current_mesh(ms.mesh_number() - 1)

    out = ms.current_mesh()
    if not out.is_compact():
        out.compact()

    return state.model_copy(update={
        "vertices": out.vertex_matrix(),
        "faces": out.face_matrix(),
        "smoothed_vertices": None,
    })
