# tests/test_smooth.py
import numpy as np

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.smooth import smooth
from xeltocad.state import PipelineState


def _make_3d_sphere():
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return (x**2 + y**2 + z**2 < 0.5**2).astype(float)


def _make_2d_circle():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return (x**2 + y**2 < 0.5**2).astype(float)


def test_smooth_3d_produces_smoothed_vertices():
    state = extract(preprocess(PipelineState(density=_make_3d_sphere())))
    result = smooth(state)
    assert result.smoothed_vertices is not None
    assert result.smoothed_vertices.shape == result.vertices.shape


def test_smooth_3d_changes_vertices():
    state = extract(preprocess(PipelineState(density=_make_3d_sphere())))
    result = smooth(state)
    # smoothed vertices should differ from original
    assert not np.allclose(result.smoothed_vertices, result.vertices)


def test_smooth_2d_is_noop():
    """Taubin smoothing only applies to 3D meshes; 2D contours pass through."""
    state = extract(preprocess(PipelineState(density=_make_2d_circle())))
    result = smooth(state)
    assert result.smoothed_vertices is None  # no 3D mesh to smooth
    assert result.contours is not None  # contours unchanged
