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
