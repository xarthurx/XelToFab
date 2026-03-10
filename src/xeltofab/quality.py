"""Mesh quality metrics via trimesh and pyvista."""

from __future__ import annotations

import numpy as np
import trimesh

from xeltofab.state import PipelineState


def compute_quality(state: PipelineState) -> dict:
    """Compute mesh quality metrics from a pipeline state.

    Returns a dict with metrics appropriate for the state's dimensionality.
    3D: vertex/face count, watertight, volume, surface area, aspect ratio,
        min angle, scaled Jacobian (pyvista required for last three).
    2D: contour count, total contour points.
    """
    metrics: dict = {"ndim": state.ndim}

    if state.ndim == 2:
        metrics["num_contours"] = len(state.contours) if state.contours else 0
        metrics["total_contour_points"] = sum(len(c) for c in state.contours) if state.contours else 0
        if state.volume_fraction is not None:
            metrics["volume_fraction"] = round(float(state.volume_fraction), 6)
        return metrics

    # 3D metrics
    vertices = state.best_vertices
    if vertices is None or state.faces is None:
        return metrics

    metrics["num_vertices"] = int(vertices.shape[0])
    metrics["num_faces"] = int(state.faces.shape[0])
    if state.volume_fraction is not None:
        metrics["volume_fraction"] = round(float(state.volume_fraction), 6)

    # Trimesh metrics
    mesh = trimesh.Trimesh(vertices=vertices, faces=state.faces, process=False)
    metrics["is_watertight"] = bool(mesh.is_watertight)
    metrics["surface_area"] = round(float(mesh.area), 6)
    if mesh.is_watertight:
        metrics["volume"] = round(float(mesh.volume), 6)

    # PyVista quality metrics (optional)
    try:
        import pyvista as pv

        faces_pv = np.column_stack([np.full(len(state.faces), 3), state.faces]).ravel()
        pv_mesh = pv.PolyData(vertices.astype(np.float64), faces_pv)

        quality_measures = ["aspect_ratio", "min_angle", "scaled_jacobian"]
        qual = pv_mesh.cell_quality(quality_measures)
        for metric_name in quality_measures:
            values = qual.cell_data[metric_name]
            metrics[metric_name] = {
                "min": round(float(np.min(values)), 4),
                "mean": round(float(np.mean(values)), 4),
                "max": round(float(np.max(values)), 4),
                "std": round(float(np.std(values)), 4),
            }
    except ImportError:
        pass

    return metrics
