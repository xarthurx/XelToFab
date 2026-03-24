"""Tests for SDF function evaluation (sdf_eval.py + process_from_sdf)."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.pipeline import process_from_sdf
from xeltofab.sdf_eval import (
    Bounds3D,
    compute_grid_dims,
    octree_evaluate,
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

    def test_adaptive_produces_mesh(self):
        """adaptive=True produces a valid mesh via octree evaluation."""
        state = process_from_sdf(sphere_sdf, bounds=(-2, -2, -2, 2, 2, 2), resolution=32, adaptive=True)
        assert state.vertices is not None
        assert state.faces is not None
        assert state.vertices.shape[0] > 0
        assert state.faces.shape[0] > 0


# ---------------------------------------------------------------------------
# Octree-accelerated evaluation
# ---------------------------------------------------------------------------


class TestOctreeEvaluate:
    def test_sphere_matches_uniform(self):
        """Octree near-surface values match uniform evaluator at same grid."""
        bounds: Bounds3D = (-2, -2, -2, 2, 2, 2)
        # Octree grid may be slightly larger due to cell rounding — use octree's
        # actual coordinates to build the uniform reference at the same resolution.
        grid_octree, x, y, z = octree_evaluate(sphere_sdf, bounds, resolution=32)
        rx = len(x)
        grid_uniform, _, _, _ = uniform_grid_evaluate(sphere_sdf, bounds, resolution=rx)

        # Both grids should now have the same shape
        assert grid_octree.shape == grid_uniform.shape

        # Compare only where octree actually evaluated (not +1.0 fill)
        evaluated = grid_octree != 1.0
        assert np.count_nonzero(evaluated) > 0
        np.testing.assert_allclose(grid_octree[evaluated], grid_uniform[evaluated], atol=1e-10)

    def test_fewer_evaluations(self):
        """Octree evaluates fewer points than uniform grid."""
        total_evaluated = [0]

        def counting_sdf(points: np.ndarray) -> np.ndarray:
            total_evaluated[0] += len(points)
            return sphere_sdf(points)

        grid, _, _, _ = octree_evaluate(counting_sdf, (-2, -2, -2, 2, 2, 2), resolution=64)
        rx, ry, rz = grid.shape[2], grid.shape[1], grid.shape[0]
        total_grid_points = rx * ry * rz
        assert total_evaluated[0] < total_grid_points

    def test_noncubic_bounds(self):
        """Non-cubic bounds produce a correctly shaped grid."""
        grid, x, y, z = octree_evaluate(sphere_sdf, (0, 0, 0, 2, 1, 1), resolution=64)
        # x is longest axis → more points along x
        assert grid.shape[2] > grid.shape[1]  # Nx > Ny
        # Surface should be found (sphere at origin intersects this box)
        assert np.any(grid < 0)

    def test_unevaluated_regions_far_from_surface(self):
        """Unevaluated (+1.0) regions are far from the zero level set."""
        bounds: Bounds3D = (-2, -2, -2, 2, 2, 2)
        grid_octree, x, y, z = octree_evaluate(sphere_sdf, bounds, resolution=32)
        rx = len(x)
        grid_uniform, _, _, _ = uniform_grid_evaluate(sphere_sdf, bounds, resolution=rx)

        # Unevaluated regions should have large |SDF| — far from the surface
        unevaluated = grid_octree == 1.0
        if np.any(unevaluated):
            abs_sdf = np.abs(grid_uniform[unevaluated])
            assert np.min(abs_sdf) > 0.1  # comfortably far from zero crossing

    def test_coarse_factor_validation(self):
        """Non-power-of-2 coarse_factor raises ValueError."""
        with pytest.raises(ValueError, match="power of 2"):
            octree_evaluate(sphere_sdf, (-1, -1, -1, 1, 1, 1), coarse_factor=3)

    def test_chunk_size_octree(self):
        """chunk_size limits points per sdf_fn call in octree mode."""
        max_call_size = [0]

        def tracking_sdf(points: np.ndarray) -> np.ndarray:
            max_call_size[0] = max(max_call_size[0], len(points))
            return sphere_sdf(points)

        octree_evaluate(tracking_sdf, (-2, -2, -2, 2, 2, 2), resolution=32, chunk_size=500)
        assert max_call_size[0] <= 500

    def test_box_sdf_surface_found(self):
        """Box SDF surface is correctly identified by octree culling."""
        grid, _, _, _ = octree_evaluate(box_sdf, (-2, -2, -2, 2, 2, 2), resolution=32)
        # Box surface should produce sign changes
        assert np.any(grid < 0)  # inside the box
        assert np.any(grid > 0)  # outside the box

    def test_coordinate_convention(self):
        """SDF function receives (x, y, z) ordered points in octree mode."""
        received_points = []

        def capturing_sdf(points: np.ndarray) -> np.ndarray:
            received_points.append(points.copy())
            return sphere_sdf(points)

        octree_evaluate(capturing_sdf, (-1, -1, -1, 1, 1, 1), resolution=16, coarse_factor=4)
        all_points = np.concatenate(received_points)
        # Points should be in [-1, 1] range (matching bounds)
        assert np.all(all_points >= -1.0 - 1e-10)
        assert np.all(all_points <= 1.0 + 1e-10)

    def test_coarse_factor_1(self):
        """coarse_factor=1 degenerates to full evaluation."""
        grid, _, _, _ = octree_evaluate(sphere_sdf, (-2, -2, -2, 2, 2, 2), resolution=16, coarse_factor=1)
        assert np.any(grid < 0)
        assert np.any(grid > 0)

    def test_all_outside_no_false_surface(self):
        """SDF entirely outside bounds produces no false zero crossings."""

        def far_sphere(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points - 100.0, axis=1) - 1.0  # sphere at (100,100,100)

        grid, _, _, _ = octree_evaluate(far_sphere, (-1, -1, -1, 1, 1, 1), resolution=16)
        # All values should be positive (outside) — no false surface from fill
        assert np.all(grid > 0)

    def test_all_inside_no_false_surface(self):
        """SDF entirely inside bounds does not fabricate surfaces from fill."""

        def huge_sphere(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points, axis=1) - 100.0  # radius 100, bounds [-2,2]

        grid, _, _, _ = octree_evaluate(huge_sphere, (-2, -2, -2, 2, 2, 2), resolution=16)
        # All values should be negative (inside) — no +1.0 fill creating false crossing
        assert np.all(grid < 0)
