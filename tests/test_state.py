# tests/test_state.py
import numpy as np
import pytest

from xeltocad.state import PipelineParams, PipelineState


def test_pipeline_params_defaults():
    params = PipelineParams()
    assert params.threshold == 0.5
    assert params.smooth_sigma == 1.0
    assert params.morph_radius == 1
    assert params.taubin_iterations == 20
    assert params.taubin_pass_band == 0.1


def test_pipeline_state_2d():
    density = np.random.rand(50, 100)
    state = PipelineState(density=density)
    assert state.ndim == 2
    assert state.params.threshold == 0.5
    assert state.binary is None
    assert state.volume_fraction is None


def test_pipeline_state_3d():
    density = np.random.rand(10, 20, 30)
    state = PipelineState(density=density)
    assert state.ndim == 3


def test_pipeline_state_rejects_1d():
    with pytest.raises(Exception):
        PipelineState(density=np.array([1.0, 2.0, 3.0]))


def test_pipeline_state_rejects_4d():
    with pytest.raises(Exception):
        PipelineState(density=np.random.rand(2, 3, 4, 5))


def test_pipeline_params_validates_threshold():
    with pytest.raises(Exception):
        PipelineParams(threshold=1.5)
    with pytest.raises(Exception):
        PipelineParams(threshold=-0.1)
