# tests/test_extract.py
import numpy as np
import pytest

from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.state import PipelineParams, PipelineState


def test_extract_2d_produces_contours(circle_density: np.ndarray):
    state = preprocess(PipelineState(density=circle_density))
    result = extract(state)
    assert result.contours is not None
    assert len(result.contours) > 0
    assert result.contours[0].shape[1] == 2


def test_extract_3d_produces_nonempty_mesh(sphere_density: np.ndarray):
    state = preprocess(PipelineState(density=sphere_density))
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape == (result.vertices.shape[0], 3)
    assert result.faces.shape == (result.faces.shape[0], 3)
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0


def test_extract_requires_preprocessing(circle_density: np.ndarray):
    state = PipelineState(density=circle_density)
    with pytest.raises(ValueError, match="binary field is None"):
        extract(state)


def test_extract_3d_direct_sdf(sphere_sdf: np.ndarray):
    """Direct extraction from continuous SDF at level=0."""
    params = PipelineParams(field_type="sdf")
    state = PipelineState(density=sphere_sdf, params=params)
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0


def test_extract_2d_direct_sdf(circle_sdf: np.ndarray):
    """Direct extraction of contours from continuous 2D SDF at level=0."""
    params = PipelineParams(field_type="sdf")
    state = PipelineState(density=circle_sdf, params=params)
    result = extract(state)
    assert result.contours is not None
    assert len(result.contours) > 0


def test_extract_3d_direct_density(sphere_density: np.ndarray):
    """Direct extraction from clean density field at level=0.5."""
    params = PipelineParams(field_type="density", direct_extraction=True)
    state = PipelineState(density=sphere_density, params=params)
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[0] > 0


def test_extract_3d_direct_custom_level(sphere_sdf: np.ndarray):
    """Direct extraction at a custom level (offset surface)."""
    params_offset = PipelineParams(field_type="sdf", extraction_level=0.1)
    state_offset = PipelineState(density=sphere_sdf, params=params_offset)
    result_offset = extract(state_offset)
    assert result_offset.vertices is not None

    params_zero = PipelineParams(field_type="sdf", extraction_level=0.0)
    state_zero = PipelineState(density=sphere_sdf, params=params_zero)
    result_zero = extract(state_zero)

    # Geometric assertion: SDF = sqrt(r)-0.5, so level=0.1 → r=0.6 (larger).
    # Marching cubes returns vertices in grid index coordinates.
    center = np.array([14.5, 14.5, 14.5])  # center of 30x30x30 grid
    mean_r_zero = float(np.mean(np.linalg.norm(result_zero.vertices - center, axis=1)))
    mean_r_offset = float(np.mean(np.linalg.norm(result_offset.vertices - center, axis=1)))
    assert mean_r_offset > mean_r_zero
