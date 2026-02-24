# tests/test_viz.py
from matplotlib.figure import Figure

from xeltocad.state import PipelineState
from xeltocad.viz import plot_comparison, plot_density, plot_result


def test_plot_density_2d(processed_2d: PipelineState):
    fig = plot_density(processed_2d)
    assert isinstance(fig, Figure)


def test_plot_density_3d(processed_3d: PipelineState):
    fig = plot_density(processed_3d)
    assert isinstance(fig, Figure)


def test_plot_result_2d(processed_2d: PipelineState):
    fig = plot_result(processed_2d)
    assert isinstance(fig, Figure)


def test_plot_result_3d(processed_3d: PipelineState):
    fig = plot_result(processed_3d)
    assert isinstance(fig, Figure)


def test_plot_comparison_2d(processed_2d: PipelineState):
    fig = plot_comparison(processed_2d)
    assert isinstance(fig, Figure)


def test_plot_comparison_3d(processed_3d: PipelineState):
    fig = plot_comparison(processed_3d)
    assert isinstance(fig, Figure)
