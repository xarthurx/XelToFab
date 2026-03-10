# tests/test_quality_plots.py
"""Tests for quality visualization functions."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from xeltofab.state import PipelineParams, PipelineState


@pytest.fixture
def mesh_state(sphere_field: np.ndarray) -> PipelineState:
    """A processed 3D state with vertices and faces."""
    from xeltofab.pipeline import process

    return process(PipelineState(field=sphere_field, params=PipelineParams()))


@pytest.fixture
def state_2d(processed_2d: PipelineState) -> PipelineState:
    return processed_2d


def test_heatmap_returns_plotter(mesh_state: PipelineState):
    from xeltofab.quality_plots import plot_quality_heatmap

    pl = plot_quality_heatmap(mesh_state, metric="min_angle")
    assert isinstance(pl, pv.Plotter)
    pl.close()


def test_heatmap_aspect_ratio(mesh_state: PipelineState):
    from xeltofab.quality_plots import plot_quality_heatmap

    pl = plot_quality_heatmap(mesh_state, metric="aspect_ratio")
    assert isinstance(pl, pv.Plotter)
    pl.close()


def test_heatmap_scaled_jacobian(mesh_state: PipelineState):
    from xeltofab.quality_plots import plot_quality_heatmap

    pl = plot_quality_heatmap(mesh_state, metric="scaled_jacobian")
    assert isinstance(pl, pv.Plotter)
    pl.close()


def test_heatmap_2d_raises(state_2d: PipelineState):
    from xeltofab.quality_plots import plot_quality_heatmap

    with pytest.raises(ValueError, match="3D"):
        plot_quality_heatmap(state_2d)


def test_heatmap_no_mesh_raises():
    from xeltofab.quality_plots import plot_quality_heatmap

    state = PipelineState(field=np.zeros((10, 10, 10)), params=PipelineParams())
    with pytest.raises(ValueError, match="mesh"):
        plot_quality_heatmap(state)


def test_heatmap_overview(mesh_state: PipelineState):
    from xeltofab.quality_plots import plot_quality_overview

    pl = plot_quality_overview(mesh_state)
    assert isinstance(pl, pv.Plotter)
    pl.close()


def test_histogram_returns_figure(mesh_state: PipelineState):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from xeltofab.quality_plots import plot_metric_histogram

    fig = plot_metric_histogram(mesh_state, metric="min_angle")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_histogram_custom_threshold(mesh_state: PipelineState):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from xeltofab.quality_plots import plot_metric_histogram

    fig = plot_metric_histogram(mesh_state, metric="min_angle", threshold=30.0)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_histogram_threshold_line(mesh_state: PipelineState):
    """Verify the threshold vertical line and pass-rate annotation are present."""
    import matplotlib.pyplot as plt

    from xeltofab.quality_plots import plot_metric_histogram

    fig = plot_metric_histogram(mesh_state, metric="min_angle")
    ax = fig.axes[0]
    # Check at least one vertical line exists (the threshold line)
    assert len(ax.lines) >= 1, "Expected threshold line"
    # Check annotation text exists (pass-rate box)
    assert len(ax.texts) >= 1, "Expected pass-rate annotation"
    plt.close(fig)


def test_histogram_no_mesh_raises():
    from xeltofab.quality_plots import plot_metric_histogram

    state = PipelineState(field=np.zeros((10, 10, 10)), params=PipelineParams())
    with pytest.raises(ValueError, match="mesh"):
        plot_metric_histogram(state)


def test_histogram_overview(mesh_state: PipelineState):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from xeltofab.quality_plots import plot_metric_overview

    fig = plot_metric_overview(mesh_state)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_histogram_2d_raises(state_2d: PipelineState):
    from xeltofab.quality_plots import plot_metric_histogram

    with pytest.raises(ValueError, match="3D"):
        plot_metric_histogram(state_2d)
