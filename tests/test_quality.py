"""Tests for mesh quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.state import PipelineState

from xeltofab.quality import compute_quality


def test_quality_3d_basic(processed_3d: PipelineState):
    """Quality metrics should include vertex/face count, watertight, and surface area."""
    metrics = compute_quality(processed_3d)
    assert metrics["num_vertices"] > 0
    assert metrics["num_faces"] > 0
    assert isinstance(metrics["is_watertight"], bool)
    assert isinstance(metrics["surface_area"], float)
    assert metrics["surface_area"] > 0


def test_quality_3d_pyvista_metrics(processed_3d: PipelineState):
    """PyVista quality metrics should have min/mean/max/std."""
    pv = pytest.importorskip("pyvista")
    metrics = compute_quality(processed_3d)
    for metric in ("aspect_ratio", "min_angle", "scaled_jacobian"):
        assert metric in metrics, f"Missing metric: {metric}"
        assert "min" in metrics[metric]
        assert "mean" in metrics[metric]
        assert "max" in metrics[metric]
        assert "std" in metrics[metric]


def test_quality_2d(processed_2d: PipelineState):
    """2D quality metrics should include contour info."""
    metrics = compute_quality(processed_2d)
    assert metrics["num_contours"] > 0
    assert "total_contour_points" in metrics


def test_quality_no_mesh():
    """Quality on a state with no mesh should return minimal info."""
    state = PipelineState(density=np.zeros((10, 10, 10)))
    metrics = compute_quality(state)
    assert metrics["ndim"] == 3
    assert "num_vertices" not in metrics
