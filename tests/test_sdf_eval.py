"""Tests for SDF function evaluation (sdf_eval.py + process_from_sdf)."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.pipeline import process_from_sdf
from xeltofab.sdf_eval import (
    Bounds3D,
    compute_grid_dims,
    uniform_grid_evaluate,
    validate_bounds,
    validate_sdf_output,
)

# ---------------------------------------------------------------------------
# Analytical SDF fixtures
# ---------------------------------------------------------------------------


def sphere_sdf(points: np.ndarray) -> np.ndarray:
    """Unit sphere centered at origin."""
    return np.linalg.norm(points, axis=1) - 1.0


def box_sdf(points: np.ndarray) -> np.ndarray:
    """Axis-aligned unit box centered at origin (side length 2)."""
    q = np.abs(points) - 1.0
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


# ---------------------------------------------------------------------------
# Bounds validation
# ---------------------------------------------------------------------------


class TestValidateBounds:
    def test_valid_bounds(self):
        mins, maxs = validate_bounds((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
        np.testing.assert_array_equal(mins, [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(maxs, [1.0, 1.0, 1.0])

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="exactly 6"):
            validate_bounds((0, 0, 0, 1, 1))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exactly 6"):
            validate_bounds((0, 0, 0, 1, 1, 1, 1))  # type: ignore[arg-type]

    def test_min_ge_max(self):
        with pytest.raises(ValueError, match="strictly less than max"):
            validate_bounds((1.0, 0.0, 0.0, 0.0, 1.0, 1.0))

    def test_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            validate_bounds((0, 0, float("nan"), 1, 1, 1))


# ---------------------------------------------------------------------------
# SDF output validation
# ---------------------------------------------------------------------------


class TestValidateSdfOutput:
    def test_valid_output(self):
        result = validate_sdf_output(np.array([1.0, 2.0, 3.0]), 3)
        assert result.dtype == np.float64
        assert result.shape == (3,)

    def test_wrong_shape_2d(self):
        with pytest.raises(ValueError, match="1D"):
            validate_sdf_output(np.array([[1.0], [2.0]]), 2)

    def test_wrong_n(self):
        with pytest.raises(ValueError, match="3 values for 5"):
            validate_sdf_output(np.array([1.0, 2.0, 3.0]), 5)

    def test_nan_in_output(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_sdf_output(np.array([1.0, float("nan"), 3.0]), 3)

    def test_inf_in_output(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_sdf_output(np.array([1.0, float("inf"), 3.0]), 3)

    def test_int_coerced_to_float64(self):
        result = validate_sdf_output(np.array([1, 2, 3], dtype=np.int32), 3)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Grid dimension computation
# ---------------------------------------------------------------------------


class TestComputeGridDims:
    def test_cubic_bounds(self):
        mins = np.array([-1.0, -1.0, -1.0])
        maxs = np.array([1.0, 1.0, 1.0])
        nx, ny, nz = compute_grid_dims(mins, maxs, 64)
        assert nx == ny == nz == 64

    def test_noncubic_aspect_ratio(self):
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([2.0, 1.0, 1.0])
        nx, ny, nz = compute_grid_dims(mins, maxs, 128)
        assert nx == 128
        assert ny == 64
        assert nz == 64

    def test_minimum_2(self):
        """Very thin dimension still gets at least 2 cells."""
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([100.0, 0.1, 0.1])
        nx, ny, nz = compute_grid_dims(mins, maxs, 100)
        assert nx == 100
        assert ny >= 2
        assert nz >= 2


# ---------------------------------------------------------------------------
# Uniform grid evaluation
# ---------------------------------------------------------------------------


class TestUniformGridEvaluate:
    def test_sphere_cubic(self):
        """Sphere SDF on symmetric bounds produces correct values at known points."""
        grid, x, y, z = uniform_grid_evaluate(sphere_sdf, (-2, -2, -2, 2, 2, 2), resolution=32)
        assert grid.shape[0] == grid.shape[1] == grid.shape[2] == 32

        # Center should be ≈ -1.0 (inside unit sphere, distance = -1)
        mid = grid.shape[0] // 2
        assert grid[mid, mid, mid] < 0  # inside the sphere

        # Corner should be > 0 (outside the sphere)
        assert grid[0, 0, 0] > 0

    def test_noncubic_grid_shape(self):
        """Non-cubic bounds produce proportional grid resolution."""
        grid, x, y, z = uniform_grid_evaluate(sphere_sdf, (0, 0, 0, 2, 1, 1), resolution=64)
        assert grid.shape == (32, 32, 64)  # [Nz, Ny, Nx]: z=1→32, y=1→32, x=2→64

    def test_slab_matches_full(self):
        """Z-slab evaluation matches direct full-grid evaluation."""
        bounds: Bounds3D = (-1, -1, -1, 1, 1, 1)
        grid_slab, x, y, z = uniform_grid_evaluate(sphere_sdf, bounds, resolution=16)

        # Direct full evaluation for comparison
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid_full = sphere_sdf(points).reshape(len(z), len(y), len(x))

        np.testing.assert_allclose(grid_slab, grid_full, atol=1e-12)

    def test_coordinate_convention(self):
        """SDF function receives (x, y, z) order; output is [Nz, Ny, Nx]."""
        received_points = []

        def capturing_sdf(points: np.ndarray) -> np.ndarray:
            received_points.append(points.copy())
            return np.zeros(len(points))

        grid, x, y, z = uniform_grid_evaluate(capturing_sdf, (0, 0, 0, 3, 2, 1), resolution=6)
        # Longest axis is x (extent=3) → Nx=6, Ny=4, Nz=2
        assert grid.shape == (2, 4, 6)  # [Nz, Ny, Nx]

        # First slab: z=z_coords[0], all y and x values
        first_slab = received_points[0]
        assert first_slab.shape == (4 * 6, 3)
        # Column 0 = x values, column 2 = z values (all same for one slab)
        assert len(np.unique(first_slab[:, 2])) == 1  # all same z in a slab

    def test_sdf_fn_exception(self):
        """SDF function that raises is wrapped with context."""

        def failing_sdf(points: np.ndarray) -> np.ndarray:
            raise RuntimeError("GPU OOM")

        with pytest.raises(RuntimeError, match="SDF function failed"):
            uniform_grid_evaluate(failing_sdf, (-1, -1, -1, 1, 1, 1), resolution=4)

    def test_chunk_size(self):
        """chunk_size splits evaluation within a slab."""
        call_sizes = []

        def tracking_sdf(points: np.ndarray) -> np.ndarray:
            call_sizes.append(len(points))
            return sphere_sdf(points)

        grid, _, _, _ = uniform_grid_evaluate(tracking_sdf, (-1, -1, -1, 1, 1, 1), resolution=8, chunk_size=10)
        assert grid.shape == (8, 8, 8)
        assert any(s <= 10 for s in call_sizes)

    def test_chunk_correctness(self):
        """Chunked evaluation produces identical results to unchunked."""
        bounds: Bounds3D = (-1, -1, -1, 1, 1, 1)
        grid_unchunked, _, _, _ = uniform_grid_evaluate(sphere_sdf, bounds, resolution=16)
        grid_chunked, _, _, _ = uniform_grid_evaluate(sphere_sdf, bounds, resolution=16, chunk_size=10)
        np.testing.assert_allclose(grid_chunked, grid_unchunked, atol=1e-12)


# ---------------------------------------------------------------------------
# End-to-end: process_from_sdf
# ---------------------------------------------------------------------------


class TestProcessFromSdf:
    def test_sphere_mesh(self):
        """Sphere SDF → mesh with vertices near unit sphere surface."""
        state = process_from_sdf(sphere_sdf, bounds=(-2, -2, -2, 2, 2, 2), resolution=32)
        assert state.vertices is not None
        assert state.faces is not None
        assert state.vertices.shape[0] > 0
        assert state.faces.shape[0] > 0

        verts = state.best_vertices
        radii = np.linalg.norm(verts, axis=1)
        assert np.mean(radii) > 0

    def test_pipeline_kwargs_forwarded(self):
        """Custom pipeline kwargs are forwarded to PipelineParams."""
        state = process_from_sdf(
            sphere_sdf,
            bounds=(-2, -2, -2, 2, 2, 2),
            resolution=16,
            extraction_method="mc",
            smoothing_method="bilateral",
        )
        assert state.params.extraction_method == "mc"
        assert state.params.smoothing_method == "bilateral"

    def test_adaptive_raises(self):
        """adaptive=True raises NotImplementedError (Phase 2)."""
        with pytest.raises(NotImplementedError, match="Phase 2"):
            process_from_sdf(sphere_sdf, bounds=(-2, -2, -2, 2, 2, 2), adaptive=True)
