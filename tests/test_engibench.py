# tests/test_engibench.py
import pytest

from xeltocad.io import load_engibench


@pytest.mark.network
def test_load_engibench_beams2d():
    """Load a Beams2D sample from EngiBench/IDEALLab."""
    state = load_engibench("IDEALLab/beams_2d_25_50_v0", index=0)
    assert state.ndim == 2
    assert state.density.shape == (25, 50)
