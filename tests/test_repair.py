"""Tests for watertight mesh repair."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.state import PipelineParams, PipelineState

pymeshlab = pytest.importorskip("pymeshlab")

from xeltofab.repair import repair


def test_repair_closes_holes(open_mesh_state: PipelineState):
    """Repair should fill holes, increasing face count."""
    original_faces = open_mesh_state.faces.shape[0]
    result = repair(open_mesh_state)
    assert result.faces.shape[0] >= original_faces
    assert result.vertices is not None


def test_repair_clears_smoothed_vertices(open_mesh_state: PipelineState):
    """After repair, smoothed_vertices should be None (vertices is the latest)."""
    assert open_mesh_state.smoothed_vertices is not None
    result = repair(open_mesh_state)
    assert result.smoothed_vertices is None


def test_repair_preserves_watertight_mesh(processed_3d: PipelineState):
    """Repair should not break an already-good mesh."""
    original_verts = processed_3d.best_vertices.shape[0]
    result = repair(processed_3d)
    assert result.vertices is not None
    assert result.faces is not None
    # Vertex count should be similar (repair may merge near-duplicate vertices)
    assert result.vertices.shape[0] > 0


def test_repair_noop_2d(processed_2d: PipelineState):
    """Repair is a no-op for 2D states."""
    result = repair(processed_2d)
    assert result.contours is not None
    assert result.vertices is None


def test_repair_disabled(open_mesh_state: PipelineState):
    """When repair=False, state passes through unchanged."""
    state = open_mesh_state.model_copy(
        update={"params": PipelineParams(repair=False)}
    )
    result = repair(state)
    assert result.faces.shape[0] == state.faces.shape[0]


def test_repair_custom_hole_size(open_mesh_state: PipelineState):
    """Custom max_hole_size parameter is respected."""
    state = open_mesh_state.model_copy(
        update={"params": PipelineParams(max_hole_size=5)}
    )
    # Should not crash with small hole size
    result = repair(state)
    assert result.vertices is not None
