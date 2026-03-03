# tests/test_pipeline.py
import numpy as np

from xeltofab.pipeline import process
from xeltofab.state import PipelineState


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
