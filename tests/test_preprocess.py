# tests/test_preprocess.py
import numpy as np

from xeltocad.preprocess import preprocess
from xeltocad.state import PipelineState


def test_preprocess_2d_produces_binary():
    """Preprocessing a 2D density field should produce a binary array."""
    density = np.random.rand(50, 100)
    state = PipelineState(density=density)
    result = preprocess(state)
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 1})
    assert result.binary.shape == density.shape


def test_preprocess_3d_produces_binary():
    """Preprocessing a 3D density field should produce a binary array."""
    density = np.random.rand(10, 20, 30)
    state = PipelineState(density=density)
    result = preprocess(state)
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 1})
    assert result.binary.shape == density.shape


def test_preprocess_records_volume_fraction():
    """Volume fraction of original field should be recorded."""
    density = np.ones((10, 10)) * 0.7
    state = PipelineState(density=density)
    result = preprocess(state)
    assert result.volume_fraction is not None
    assert abs(result.volume_fraction - 0.7) < 0.01


def test_preprocess_removes_small_components():
    """Small disconnected blobs should be removed."""
    density = np.zeros((50, 50))
    density[10:40, 10:40] = 1.0  # large block
    density[2:4, 2:4] = 1.0  # tiny island (4 pixels)
    state = PipelineState(density=density)
    result = preprocess(state)
    # tiny island should be removed
    assert result.binary[3, 3] == 0
    # large block should remain
    assert result.binary[25, 25] == 1
