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


def _min_angle(vertices, faces):
    """Compute minimum triangle angle using PyVista."""
    import pyvista as pv

    fpv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    mesh = pv.PolyData(vertices.astype(np.float64), fpv)
    mesh = mesh.cell_quality(quality_measure="min_angle")
    return float(np.min(mesh.cell_data["min_angle"]))


def test_remesh_improves_quality(sphere_field: np.ndarray):
    """Remeshing should improve min angle on the synthetic sphere."""
    try:
        import pyvista as pv  # noqa: F401
    except ImportError:
        pytest.skip("pyvista required for quality check")

    from xeltofab.pipeline import process

    # Build a state WITHOUT remeshing to get baseline quality
    no_remesh_params = PipelineParams(remesh=False)
    state_before = process(PipelineState(field=sphere_field, params=no_remesh_params))
    min_before = _min_angle(state_before.best_vertices, state_before.faces)

    # Now remesh that state
    state_to_remesh = state_before.model_copy(update={"params": PipelineParams(remesh=True)})
    result = remesh(state_to_remesh)
    min_after = _min_angle(result.vertices, result.faces)

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
    state = processed_3d.model_copy(update={"params": PipelineParams(remesh=False)})
    original_faces = state.faces.shape[0]
    result = remesh(state)
    assert result.faces.shape[0] == original_faces
