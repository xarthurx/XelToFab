# tests/test_pipeline.py
import numpy as np
import pytest

from xeltofab.pipeline import process
from xeltofab.state import PipelineParams, PipelineState

pymeshlab = pytest.importorskip("pymeshlab")


def test_process_2d_end_to_end(circle_density: np.ndarray):
    result = process(PipelineState(density=circle_density))
    assert result.binary is not None
    assert result.contours is not None
    assert result.volume_fraction is not None


def test_process_3d_end_to_end(sphere_density: np.ndarray):
    result = process(PipelineState(density=sphere_density))
    assert result.binary is not None
    assert result.vertices is not None
    assert result.faces is not None
    assert result.volume_fraction is not None
    # After repair+remesh, smoothed_vertices is cleared
    assert result.smoothed_vertices is None
    # All face indices within bounds
    assert np.all(result.faces < result.vertices.shape[0])


def test_process_3d_sdf_end_to_end(sphere_sdf: np.ndarray):
    """Full pipeline with SDF input skips preprocessing."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(density=sphere_sdf, params=params))
    assert result.binary is None
    assert result.volume_fraction is None
    assert result.vertices is not None
    assert result.faces is not None


def test_process_2d_sdf_end_to_end(circle_sdf: np.ndarray):
    """Full 2D pipeline with SDF input."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(density=circle_sdf, params=params))
    assert result.binary is None
    assert result.contours is not None


def test_process_3d_direct_density(sphere_density: np.ndarray):
    """Full pipeline with clean density, direct extraction."""
    params = PipelineParams(direct_extraction=True)
    result = process(PipelineState(density=sphere_density, params=params))
    assert result.binary is None
    assert result.vertices is not None


def test_process_3d_no_repair_remesh(sphere_density: np.ndarray):
    """Pipeline with repair/remesh disabled preserves smoothed_vertices."""
    params = PipelineParams(repair=False, remesh=False)
    result = process(PipelineState(density=sphere_density, params=params))
    assert result.smoothed_vertices is not None


def test_process_2d_unaffected_by_repair_remesh(circle_density: np.ndarray):
    """2D pipeline is unaffected by repair/remesh settings."""
    result = process(PipelineState(density=circle_density))
    assert result.contours is not None
    assert result.vertices is None
