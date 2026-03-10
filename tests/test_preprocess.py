# tests/test_preprocess.py
import numpy as np

from xeltofab.preprocess import preprocess
from xeltofab.state import PipelineState


def test_preprocess_2d_produces_binary():
    field = np.random.rand(50, 100)
    state = PipelineState(field=field)
    result = preprocess(state)
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 1})
    assert result.binary.shape == field.shape


def test_preprocess_3d_produces_binary():
    field = np.random.rand(10, 20, 30)
    state = PipelineState(field=field)
    result = preprocess(state)
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 1})
    assert result.binary.shape == field.shape


def test_preprocess_records_volume_fraction():
    field = np.ones((10, 10)) * 0.7
    state = PipelineState(field=field)
    result = preprocess(state)
    assert result.volume_fraction is not None
    assert abs(result.volume_fraction - 0.7) < 0.01


def test_preprocess_removes_small_components():
    field = np.zeros((50, 50))
    field[10:40, 10:40] = 1.0  # large block
    field[2:4, 2:4] = 1.0  # tiny island (4 pixels)
    state = PipelineState(field=field)
    result = preprocess(state)
    assert result.binary[3, 3] == 0  # tiny island removed
    assert result.binary[25, 25] == 1  # large block remains


def test_preprocess_all_zero_density():
    """All-zero field should produce all-zero binary."""
    field = np.zeros((30, 30))
    state = PipelineState(field=field)
    result = preprocess(state)
    assert result.binary is not None
    assert np.all(result.binary == 0)


def test_preprocess_all_one_density():
    """All-one field should produce all-one binary."""
    field = np.ones((30, 30))
    state = PipelineState(field=field)
    result = preprocess(state)
    assert result.binary is not None
    assert np.all(result.binary == 1)
