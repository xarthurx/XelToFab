# tests/test_state.py
import numpy as np
import pytest
from pydantic import ValidationError

from xeltofab.state import PipelineParams, PipelineState


def test_pipeline_params_defaults():
    params = PipelineParams()
    assert params.threshold == 0.5
    assert params.smooth_sigma == 1.0
    assert params.morph_radius == 1
    assert params.taubin_iterations == 20
    assert params.taubin_lambda == 0.5


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
    with pytest.raises(ValidationError):
        PipelineState(density=np.array([1.0, 2.0, 3.0]))


def test_pipeline_state_rejects_4d():
    with pytest.raises(ValidationError):
        PipelineState(density=np.random.rand(2, 3, 4, 5))


def test_pipeline_params_validates_threshold():
    with pytest.raises(ValidationError):
        PipelineParams(threshold=1.5)
    with pytest.raises(ValidationError):
        PipelineParams(threshold=-0.1)


def test_pipeline_params_field_type_defaults():
    params = PipelineParams()
    assert params.field_type == "density"
    assert params.direct_extraction is False
    assert params.extraction_level is None


def test_pipeline_params_sdf_smart_defaults():
    """SDF field type should auto-enable direct extraction and disable Gaussian preprocessing."""
    params = PipelineParams(field_type="sdf")
    assert params.direct_extraction is True
    assert params.smooth_sigma == 0.0


def test_pipeline_params_sdf_override_smooth():
    """User can re-enable smoothing for noisy SDF."""
    params = PipelineParams(field_type="sdf", smooth_sigma=2.0)
    assert params.smooth_sigma == 2.0
    assert params.direct_extraction is True


def test_pipeline_params_density_direct():
    """User can enable direct extraction for clean density fields."""
    params = PipelineParams(field_type="density", direct_extraction=True)
    assert params.direct_extraction is True
    assert params.threshold == 0.5  # threshold still available for density


def test_pipeline_params_effective_extraction_level():
    """extraction_level derives from field_type when not set explicitly."""
    assert PipelineParams(field_type="density").effective_extraction_level == 0.5
    assert PipelineParams(field_type="sdf").effective_extraction_level == 0.0
    assert PipelineParams(field_type="sdf", extraction_level=0.1).effective_extraction_level == 0.1


def test_pipeline_params_repair_defaults():
    params = PipelineParams()
    assert params.repair is True
    assert params.remesh is True
    assert params.target_edge_length is None


def test_pipeline_params_disable_repair_remesh():
    params = PipelineParams(repair=False, remesh=False)
    assert params.repair is False
    assert params.remesh is False


def test_pipeline_params_custom_remesh():
    params = PipelineParams(target_edge_length=0.5)
    assert params.target_edge_length == 0.5
