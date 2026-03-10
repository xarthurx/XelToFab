# tests/test_pipeline.py
import numpy as np

from xeltofab.pipeline import process
from xeltofab.state import PipelineParams, PipelineState


def test_process_2d_end_to_end(circle_field: np.ndarray):
    result = process(PipelineState(field=circle_field))
    assert result.binary is not None
    assert result.contours is not None
    assert result.volume_fraction is not None


def test_process_3d_end_to_end(sphere_field: np.ndarray):
    result = process(PipelineState(field=sphere_field))
    assert result.binary is not None
    assert result.vertices is not None
    assert result.faces is not None
    assert result.volume_fraction is not None
    # After repair, smoothed_vertices is cleared (repair updates vertices directly)
    assert result.smoothed_vertices is None
    # All face indices within bounds
    assert np.all(result.faces < result.vertices.shape[0])


def test_process_3d_sdf_end_to_end(sphere_sdf: np.ndarray):
    """Full pipeline with SDF input skips preprocessing."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(field=sphere_sdf, params=params))
    assert result.binary is None
    assert result.volume_fraction is None
    assert result.vertices is not None
    assert result.faces is not None


def test_process_2d_sdf_end_to_end(circle_sdf: np.ndarray):
    """Full 2D pipeline with SDF input."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(field=circle_sdf, params=params))
    assert result.binary is None
    assert result.contours is not None


def test_process_3d_direct_density(sphere_field: np.ndarray):
    """Full pipeline with clean density, direct extraction."""
    params = PipelineParams(direct_extraction=True)
    result = process(PipelineState(field=sphere_field, params=params))
    assert result.binary is None
    assert result.vertices is not None


def test_process_3d_no_repair_remesh(sphere_field: np.ndarray):
    """Pipeline with repair/remesh/decimate disabled preserves smoothed_vertices."""
    params = PipelineParams(repair=False, remesh=False, decimate=False)
    result = process(PipelineState(field=sphere_field, params=params))
    assert result.smoothed_vertices is not None


def test_process_2d_unaffected_by_repair_remesh(circle_field: np.ndarray):
    """2D pipeline is unaffected by repair/remesh settings."""
    result = process(PipelineState(field=circle_field))
    assert result.contours is not None
    assert result.vertices is None
