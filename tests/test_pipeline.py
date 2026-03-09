# tests/test_pipeline.py
import numpy as np

from xeltofab.pipeline import process
from xeltofab.state import PipelineParams, PipelineState


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
    assert result.smoothed_vertices is not None
    assert result.volume_fraction is not None


def test_process_3d_sdf_end_to_end(sphere_sdf: np.ndarray):
    """Full pipeline with SDF input skips preprocessing."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(density=sphere_sdf, params=params))
    # No preprocessing → binary stays None
    assert result.binary is None
    assert result.volume_fraction is None
    # Extraction and smoothing still run
    assert result.vertices is not None
    assert result.faces is not None
    assert result.smoothed_vertices is not None


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
    assert result.smoothed_vertices is not None
