# tests/test_viz.py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

import numpy as np
from matplotlib.figure import Figure

from xeltocad.pipeline import process
from xeltocad.state import PipelineState
from xeltocad.viz import plot_density, plot_result, plot_comparison


def _process_2d():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    density = (x**2 + y**2 < 0.5**2).astype(float)
    return process(PipelineState(density=density))


def _process_3d():
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    return process(PipelineState(density=density))


def test_plot_density_2d():
    state = _process_2d()
    fig = plot_density(state)
    assert isinstance(fig, Figure)


def test_plot_density_3d():
    state = _process_3d()
    fig = plot_density(state)
    assert isinstance(fig, Figure)


def test_plot_result_2d():
    state = _process_2d()
    fig = plot_result(state)
    assert isinstance(fig, Figure)


def test_plot_result_3d():
    state = _process_3d()
    fig = plot_result(state)
    assert isinstance(fig, Figure)


def test_plot_comparison_2d():
    state = _process_2d()
    fig = plot_comparison(state)
    assert isinstance(fig, Figure)
