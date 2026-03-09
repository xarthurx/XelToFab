"""Tests for isotropic remeshing."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.state import PipelineParams, PipelineState

pymeshlab = pytest.importorskip("pymeshlab")

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


def test_remesh_custom_edge_length(processed_3d: PipelineState):
    """Custom target edge length should affect output mesh density."""
    params_coarse = PipelineParams(target_edge_length=5.0)
    state_coarse = processed_3d.model_copy(update={"params": params_coarse})
    result_coarse = remesh(state_coarse)

    params_fine = PipelineParams(target_edge_length=0.5)
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
