# tests/test_pipeline.py
import numpy as np

from xeltocad.pipeline import process
from xeltocad.state import PipelineState


def test_process_2d_end_to_end():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    density = (x**2 + y**2 < 0.5**2).astype(float)
    state = PipelineState(density=density)
    result = process(state)
    assert result.binary is not None
    assert result.contours is not None
    assert result.volume_fraction is not None


def test_process_3d_end_to_end():
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    state = PipelineState(density=density)
    result = process(state)
    assert result.binary is not None
    assert result.vertices is not None
    assert result.faces is not None
    assert result.smoothed_vertices is not None
    assert result.volume_fraction is not None
