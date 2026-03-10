# src/xeltofab/quality_plots.py
"""Quality visualization: per-face heatmaps and metric histograms."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from xeltofab.state import PipelineState

# Ordered: aspect ratio, min angle, scaled Jacobian (matches spec overview panels)
_VALID_METRICS = ["aspect_ratio", "min_angle", "scaled_jacobian"]

# FEA quality thresholds (defaults)
_THRESHOLDS: dict[str, float] = {
    "aspect_ratio": 5.0,     # ratio, <= is passing
    "min_angle": 20.0,       # degrees, >= is passing
    "scaled_jacobian": 0.5,  # unitless, >= is passing
}

# Metrics where higher is better (True) vs lower is better (False)
_HIGHER_IS_BETTER: dict[str, bool] = {
    "aspect_ratio": False,
    "min_angle": True,
    "scaled_jacobian": True,
}

_METRIC_LABELS: dict[str, str] = {
    "aspect_ratio": "Aspect Ratio",
    "min_angle": "Min Angle (deg)",
    "scaled_jacobian": "Scaled Jacobian",
}


def _validate_3d_mesh(state: PipelineState) -> None:
    """Raise ValueError if state is not a 3D mesh."""
    if state.ndim != 3:
        raise ValueError(f"Quality plots require 3D mesh data, got {state.ndim}D")
    if state.best_vertices is None or state.faces is None:
        raise ValueError("Quality plots require mesh data (vertices and faces)")


def _build_pv_mesh(state: PipelineState) -> pv.PolyData:
    """Build a PyVista PolyData from pipeline state."""
    vertices = state.best_vertices
    faces_pv = np.column_stack([np.full(len(state.faces), 3), state.faces]).ravel()
    return pv.PolyData(vertices.astype(np.float64), faces_pv)


def _compute_cell_metric(pv_mesh: pv.PolyData, metric: str) -> np.ndarray:
    """Compute per-cell quality metric values."""
    qual = pv_mesh.cell_quality(quality_measure=metric)
    return np.asarray(qual.cell_data[metric], dtype=np.float64)


def plot_quality_heatmap(
    state: PipelineState,
    metric: str = "min_angle",
    cmap: str = "RdYlGn",
) -> pv.Plotter:
    """Render mesh with per-face coloring by quality metric.

    Parameters
    ----------
    state : PipelineState
        Processed pipeline state with 3D mesh data.
    metric : str
        Quality metric: "min_angle", "aspect_ratio", or "scaled_jacobian".
    cmap : str
        Colormap name. Default "RdYlGn" (red=bad, green=good).

    Returns
    -------
    pv.Plotter
        Off-screen plotter. Use pl.screenshot() to save, pl.close() to free.
    """
    _validate_3d_mesh(state)
    if metric not in _VALID_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {_VALID_METRICS}")

    pv_mesh = _build_pv_mesh(state)
    values = _compute_cell_metric(pv_mesh, metric)
    pv_mesh.cell_data[metric] = values

    # Reverse colormap for aspect_ratio (lower is better)
    effective_cmap = f"{cmap}_r" if not _HIGHER_IS_BETTER[metric] else cmap

    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    pl.add_mesh(
        pv_mesh,
        scalars=metric,
        cmap=effective_cmap,
        show_edges=True,
        edge_color="black",
        line_width=0.3,
        scalar_bar_args={"title": _METRIC_LABELS[metric]},
    )
    pl.camera_position = "iso"
    return pl
