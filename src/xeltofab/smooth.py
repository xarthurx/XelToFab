"""Mesh smoothing operations."""

from __future__ import annotations

import trimesh

from xeltofab.state import PipelineState


def smooth(state: PipelineState) -> PipelineState:
    """Apply Taubin smoothing to 3D mesh. No-op for 2D contours."""
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state

    mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces, process=False)
    trimesh.smoothing.filter_taubin(
        mesh,
        iterations=state.params.taubin_iterations,
        lamb=state.params.taubin_lambda,
    )
    return state.model_copy(update={"smoothed_vertices": mesh.vertices})
