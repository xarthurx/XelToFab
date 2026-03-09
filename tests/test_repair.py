"""Tests for watertight mesh repair."""

from __future__ import annotations

import pytest

from xeltofab.state import PipelineParams, PipelineState

pymeshlab = pytest.importorskip("pymeshlab")

from xeltofab.repair import repair  # noqa: E402


def test_repair_closes_holes(open_mesh_state: PipelineState):
    """Repair should produce a valid mesh and clear smoothed_vertices."""
    assert open_mesh_state.smoothed_vertices is not None
    original_faces = open_mesh_state.faces.shape[0]
    result = repair(open_mesh_state)
    assert result.faces.shape[0] >= original_faces
    assert result.vertices is not None
    assert result.smoothed_vertices is None


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
