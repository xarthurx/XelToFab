# tests/test_viz.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from xeltofab.pipeline import process
from xeltofab.state import PipelineParams, PipelineState
from xeltofab.field_plots import plot_comparison, plot_field, plot_result


def test_plot_field_2d(processed_2d: PipelineState):
    fig = plot_field(processed_2d)
    assert isinstance(fig, Figure)


def test_plot_field_3d(processed_3d: PipelineState):
    fig = plot_field(processed_3d)
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


def test_plot_result_2d_direct(circle_sdf: np.ndarray):
    """plot_result should work for 2D direct extraction (binary=None)."""
    params = PipelineParams(field_type="sdf")
    state = process(PipelineState(field=circle_sdf, params=params))
    assert state.binary is None
    fig = plot_result(state)
    assert fig is not None
    plt.close(fig)


def test_plot_comparison_2d_direct(circle_sdf: np.ndarray):
    """plot_comparison should work for 2D direct extraction (binary=None)."""
    params = PipelineParams(field_type="sdf")
    state = process(PipelineState(field=circle_sdf, params=params))
    assert state.binary is None
    fig = plot_comparison(state)
    assert fig is not None
    plt.close(fig)
