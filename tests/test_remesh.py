"""Tests for isotropic remeshing via gpytoolbox."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.state import PipelineParams, PipelineState

gpytoolbox = pytest.importorskip("gpytoolbox")

from xeltofab.remesh import remesh  # noqa: E402


def test_remesh_produces_valid_mesh(processed_3d: PipelineState):
    """Remeshing should produce a valid mesh and clear smoothed_vertices."""
    result = remesh(processed_3d)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[1] == 3
    assert result.faces.shape[1] == 3
    assert result.smoothed_vertices is None
    assert np.all(result.faces < result.vertices.shape[0])
    assert np.all(result.faces >= 0)


def test_remesh_improves_quality(sphere_density: np.ndarray):
    """Remeshing should improve min angle on the synthetic sphere."""
    try:
        import pyvista as pv
    except ImportError:
        pytest.skip("pyvista required for quality check")

    from xeltofab.pipeline import process

    # Build a state WITHOUT remeshing to get baseline quality
    no_remesh_params = PipelineParams(remesh=False)
    state_before = process(PipelineState(density=sphere_density, params=no_remesh_params))

    verts_before = state_before.best_vertices
    faces_before = state_before.faces
    fpv = np.column_stack([np.full(len(faces_before), 3), faces_before]).ravel()
    q_before = pv.PolyData(verts_before.astype(np.float64), fpv)
    q_before = q_before.compute_cell_quality(quality_measure="min_angle")
    min_before = float(np.min(q_before.cell_data["CellQuality"]))

    # Now remesh that state
    state_to_remesh = state_before.model_copy(
        update={"params": PipelineParams(remesh=True)}
    )
    result = remesh(state_to_remesh)
    fpv2 = np.column_stack([np.full(len(result.faces), 3), result.faces]).ravel()
    q_after = pv.PolyData(result.vertices.astype(np.float64), fpv2)
    q_after = q_after.compute_cell_quality(quality_measure="min_angle")
    min_after = float(np.min(q_after.cell_data["CellQuality"]))

    assert min_after > min_before


def test_remesh_custom_edge_length(processed_3d: PipelineState):
    """Custom target edge length should affect output mesh density."""
    params_coarse = PipelineParams(target_edge_length=5.0, remesh_iterations=3)
    state_coarse = processed_3d.model_copy(update={"params": params_coarse})
    result_coarse = remesh(state_coarse)

    params_fine = PipelineParams(target_edge_length=0.5, remesh_iterations=3)
    state_fine = processed_3d.model_copy(update={"params": params_fine})
    result_fine = remesh(state_fine)

    assert result_fine.faces.shape[0] > result_coarse.faces.shape[0]


def test_remesh_noop_2d(processed_2d: PipelineState):
    """Remesh is a no-op for 2D states."""
    result = remesh(processed_2d)
    assert result.contours is not None
    assert result.vertices is None


def test_remesh_disabled(processed_3d: PipelineState):
    """When remesh=False, state passes through unchanged."""
    state = processed_3d.model_copy(
        update={"params": PipelineParams(remesh=False)}
    )
    original_faces = state.faces.shape[0]
    result = remesh(state)
    assert result.faces.shape[0] == original_faces
