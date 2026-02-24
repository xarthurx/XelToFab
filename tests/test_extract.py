# tests/test_extract.py
import numpy as np

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.state import PipelineState


def _make_2d_circle():
    """Create a 2D density field with a filled circle."""
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return (x**2 + y**2 < 0.5**2).astype(float)


def _make_3d_sphere():
    """Create a 3D density field with a filled sphere."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return (x**2 + y**2 + z**2 < 0.5**2).astype(float)


def test_extract_2d_produces_contours():
    state = preprocess(PipelineState(density=_make_2d_circle()))
    result = extract(state)
    assert result.contours is not None
    assert len(result.contours) > 0
    assert result.contours[0].shape[1] == 2  # (K, 2) arrays


def test_extract_3d_produces_mesh():
    state = preprocess(PipelineState(density=_make_3d_sphere()))
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[1] == 3  # (N, 3)
    assert result.faces.shape[1] == 3  # (M, 3)


def test_extract_3d_mesh_is_nonempty():
    state = preprocess(PipelineState(density=_make_3d_sphere()))
    result = extract(state)
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0
