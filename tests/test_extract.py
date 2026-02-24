# tests/test_extract.py
import numpy as np
import pytest

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.state import PipelineState


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
