# tests/test_smooth.py
import numpy as np
import trimesh

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.smooth import smooth
from xeltocad.state import PipelineState


def test_smooth_3d_produces_smoothed_vertices(sphere_density: np.ndarray):
    state = extract(preprocess(PipelineState(density=sphere_density)))
    result = smooth(state)
    assert result.smoothed_vertices is not None
    assert result.smoothed_vertices.shape == result.vertices.shape


def test_smooth_3d_changes_vertices(sphere_density: np.ndarray):
    state = extract(preprocess(PipelineState(density=sphere_density)))
    result = smooth(state)
    assert not np.allclose(result.smoothed_vertices, result.vertices)


def test_smooth_3d_preserves_volume(sphere_density: np.ndarray):
    """Taubin smoothing should approximately preserve mesh volume."""
    state = extract(preprocess(PipelineState(density=sphere_density)))
    original_mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
    result = smooth(state)
    smoothed_mesh = trimesh.Trimesh(vertices=result.smoothed_vertices, faces=result.faces)
    ratio = smoothed_mesh.volume / original_mesh.volume
    assert ratio > 0.9, f"Volume ratio {ratio:.3f} is too low — smoothing destroyed the mesh"


def test_smooth_2d_is_noop(circle_density: np.ndarray):
    """Taubin smoothing only applies to 3D meshes; 2D contours pass through."""
    state = extract(preprocess(PipelineState(density=circle_density)))
    result = smooth(state)
    assert result.smoothed_vertices is None
    assert result.contours is not None
