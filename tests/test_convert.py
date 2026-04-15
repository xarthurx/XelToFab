"""Tests for SDF→density conversion utilities (convert.py)."""

from __future__ import annotations

import numpy as np
import pytest

from xeltofab.convert import heaviside, linear_ramp, sdf_to_density, sigmoid

# ---------------------------------------------------------------------------
# Heaviside
# ---------------------------------------------------------------------------


class TestHeaviside:
    def test_binary_away_from_level(self):
        sdf = np.array([-2.0, -0.1, 0.1, 2.0])
        density = heaviside(sdf)
        np.testing.assert_array_equal(density, [1.0, 1.0, 0.0, 0.0])

    def test_at_level_is_half(self):
        assert heaviside(np.array([0.0]))[0] == 0.5

    def test_level_shift(self):
        sdf = np.array([-1.0, 0.5, 1.0, 2.0])
        density = heaviside(sdf, level=1.0)
        np.testing.assert_array_equal(density, [1.0, 1.0, 0.5, 0.0])

    def test_shape_preserved_3d(self):
        sdf = np.random.default_rng(0).standard_normal((4, 5, 6))
        density = heaviside(sdf)
        assert density.shape == sdf.shape
        assert density.dtype == np.float64

    def test_accepts_list(self):
        density = heaviside([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(density, [1.0, 0.5, 0.0])

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            heaviside(np.array([0.0, np.nan]))

    def test_rejects_non_finite_level(self):
        with pytest.raises(ValueError, match="level must be finite"):
            heaviside(np.array([0.0]), level=float("nan"))
        with pytest.raises(ValueError, match="level must be finite"):
            heaviside(np.array([0.0]), level=float("inf"))


# ---------------------------------------------------------------------------
# Linear ramp
# ---------------------------------------------------------------------------


class TestLinearRamp:
    def test_at_level_is_half(self):
        assert linear_ramp(np.array([0.0]), bandwidth=1.0)[0] == pytest.approx(0.5)

    def test_compact_support(self):
        sdf = np.array([-2.0, -1.0001, 1.0001, 2.0])
        density = linear_ramp(sdf, bandwidth=1.0)
        np.testing.assert_array_equal(density, [1.0, 1.0, 0.0, 0.0])

    def test_linear_in_band(self):
        sdf = np.array([-0.5, 0.0, 0.5])
        density = linear_ramp(sdf, bandwidth=1.0)
        np.testing.assert_allclose(density, [0.75, 0.5, 0.25])

    def test_bandwidth_scaling(self):
        sdf = np.array([1.0])
        d1 = linear_ramp(sdf, bandwidth=1.0)[0]
        d2 = linear_ramp(sdf, bandwidth=2.0)[0]
        assert d1 == pytest.approx(0.0)
        assert d2 == pytest.approx(0.25)

    def test_level_shift(self):
        sdf = np.array([1.0])
        density = linear_ramp(sdf, bandwidth=1.0, level=1.0)
        assert density[0] == pytest.approx(0.5)

    def test_rejects_non_positive_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be"):
            linear_ramp(np.array([0.0]), bandwidth=0.0)
        with pytest.raises(ValueError, match="bandwidth must be"):
            linear_ramp(np.array([0.0]), bandwidth=-1.0)

    def test_rejects_non_finite_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be"):
            linear_ramp(np.array([0.0]), bandwidth=float("nan"))
        with pytest.raises(ValueError, match="bandwidth must be"):
            linear_ramp(np.array([0.0]), bandwidth=float("inf"))

    def test_rejects_non_finite_level(self):
        with pytest.raises(ValueError, match="level must be finite"):
            linear_ramp(np.array([0.0]), bandwidth=1.0, level=float("nan"))
        with pytest.raises(ValueError, match="level must be finite"):
            linear_ramp(np.array([0.0]), bandwidth=1.0, level=float("inf"))

    def test_shape_preserved_3d(self):
        sdf = np.random.default_rng(0).standard_normal((3, 4, 5))
        density = linear_ramp(sdf, bandwidth=0.5)
        assert density.shape == sdf.shape
        assert np.all((density >= 0.0) & (density <= 1.0))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            linear_ramp(np.array([0.0, np.nan]), bandwidth=1.0)


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_at_level_is_half(self):
        assert sigmoid(np.array([0.0]), bandwidth=1.0)[0] == pytest.approx(0.5)

    def test_monotone_decreasing(self):
        sdf = np.linspace(-5.0, 5.0, 101)
        density = sigmoid(sdf, bandwidth=1.0)
        assert np.all(np.diff(density) <= 0)

    def test_strict_bounds_open_interval(self):
        sdf = np.array([-10.0, 0.0, 10.0])
        density = sigmoid(sdf, bandwidth=1.0)
        assert np.all(density > 0.0)
        assert np.all(density < 1.0)

    def test_symmetry_about_level(self):
        sdf = np.array([-2.0, -0.5, 0.5, 2.0])
        density = sigmoid(sdf, bandwidth=1.0)
        np.testing.assert_allclose(density[0] + density[3], 1.0)
        np.testing.assert_allclose(density[1] + density[2], 1.0)

    def test_numerical_stability_large_magnitude(self):
        # Must not produce NaN or Inf for |sdf|/bandwidth at extreme scales.
        sdf = np.array([-1e6, 1e6])
        density = sigmoid(sdf, bandwidth=1.0)
        assert np.all(np.isfinite(density))
        assert density[0] == pytest.approx(1.0)
        assert density[1] == pytest.approx(0.0)

    def test_bandwidth_controls_sharpness(self):
        sdf = np.array([1.0])
        d_sharp = sigmoid(sdf, bandwidth=0.1)[0]
        d_soft = sigmoid(sdf, bandwidth=10.0)[0]
        assert d_sharp < 0.01
        assert 0.4 < d_soft < 0.5

    def test_level_shift(self):
        assert sigmoid(np.array([1.0]), bandwidth=1.0, level=1.0)[0] == pytest.approx(0.5)

    def test_rejects_non_positive_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be"):
            sigmoid(np.array([0.0]), bandwidth=0.0)
        with pytest.raises(ValueError, match="bandwidth must be"):
            sigmoid(np.array([0.0]), bandwidth=-1.0)

    def test_rejects_non_finite_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be"):
            sigmoid(np.array([0.0]), bandwidth=float("nan"))
        with pytest.raises(ValueError, match="bandwidth must be"):
            sigmoid(np.array([0.0]), bandwidth=float("inf"))

    def test_rejects_non_finite_level(self):
        with pytest.raises(ValueError, match="level must be finite"):
            sigmoid(np.array([0.0]), bandwidth=1.0, level=float("nan"))
        with pytest.raises(ValueError, match="level must be finite"):
            sigmoid(np.array([0.0]), bandwidth=1.0, level=float("inf"))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            sigmoid(np.array([0.0, np.nan]), bandwidth=1.0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestSdfToDensity:
    def test_dispatches_heaviside(self):
        sdf = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(sdf_to_density(sdf, method="heaviside"), [1.0, 0.5, 0.0])

    def test_dispatches_linear(self):
        sdf = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(sdf_to_density(sdf, method="linear", bandwidth=1.0), [1.0, 0.5, 0.0])

    def test_dispatches_sigmoid(self):
        result = sdf_to_density(np.array([0.0]), method="sigmoid", bandwidth=1.0)
        assert result[0] == pytest.approx(0.5)

    def test_default_method_is_linear(self):
        sdf = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(
            sdf_to_density(sdf, bandwidth=1.0),
            sdf_to_density(sdf, method="linear", bandwidth=1.0),
        )

    def test_level_passthrough(self):
        sdf = np.array([1.0])
        assert sdf_to_density(sdf, method="heaviside", level=1.0)[0] == 0.5

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="unknown method"):
            sdf_to_density(np.array([0.0]), method="quadratic")  # type: ignore[arg-type]

    def test_heaviside_ignores_bandwidth(self):
        sdf = np.array([-1.0, 1.0])
        result = sdf_to_density(sdf, method="heaviside", bandwidth=-99.0)
        np.testing.assert_array_equal(result, [1.0, 0.0])

    def test_rejects_non_finite_scalars(self):
        sdf = np.array([0.0])
        with pytest.raises(ValueError, match="bandwidth must be"):
            sdf_to_density(sdf, method="linear", bandwidth=float("nan"))
        with pytest.raises(ValueError, match="bandwidth must be"):
            sdf_to_density(sdf, method="sigmoid", bandwidth=float("inf"))
        with pytest.raises(ValueError, match="level must be finite"):
            sdf_to_density(sdf, method="linear", bandwidth=1.0, level=float("nan"))
        with pytest.raises(ValueError, match="level must be finite"):
            sdf_to_density(sdf, method="heaviside", level=float("inf"))


# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    def test_top_level_imports(self):
        from xeltofab import heaviside as tl_h
        from xeltofab import linear_ramp as tl_l
        from xeltofab import sdf_to_density as tl_d
        from xeltofab import sigmoid as tl_s

        sdf = np.array([-1.0, 0.0, 1.0])
        assert tl_d(sdf).shape == (3,)
        assert tl_h(sdf).shape == (3,)
        assert tl_l(sdf).shape == (3,)
        assert tl_s(sdf).shape == (3,)


# ---------------------------------------------------------------------------
# Integration: converter in isolation (direct_extraction=True)
# ---------------------------------------------------------------------------


def _sphere_sdf(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points, axis=1) - 1.0


class TestIntegrationWithPipeline:
    def test_sdf_to_density_recovers_sphere(self):
        """SDF sphere → linear density → extract (no preprocess) yields a valid mesh.

        Uses direct_extraction=True and disables repair/remesh/decimate so the
        test measures the converter in isolation, not the full density pipeline.
        """
        from xeltofab import PipelineParams, PipelineState, process
        from xeltofab.sdf_eval import uniform_grid_evaluate

        bounds = (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5)
        sdf_grid, _, _, _ = uniform_grid_evaluate(_sphere_sdf, bounds, resolution=48)

        voxel = 3.0 / 47
        density = sdf_to_density(sdf_grid, method="linear", bandwidth=2 * voxel)

        assert density.shape == sdf_grid.shape
        assert np.all((density >= 0.0) & (density <= 1.0))

        params = PipelineParams(
            field_type="density",
            direct_extraction=True,
            extraction_level=0.5,
            repair=False,
            remesh=False,
            decimate=False,
        )
        state = PipelineState(field=density, params=params)
        result = process(state)
        assert result.vertices is not None
        assert result.faces is not None
        assert result.vertices.shape[0] > 100

    def test_all_three_methods_extract_valid_mesh(self):
        from xeltofab import PipelineParams, PipelineState, process
        from xeltofab.sdf_eval import uniform_grid_evaluate

        bounds = (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5)
        sdf_grid, _, _, _ = uniform_grid_evaluate(_sphere_sdf, bounds, resolution=32)
        voxel = 3.0 / 31
        for method in ("heaviside", "linear", "sigmoid"):
            density = sdf_to_density(sdf_grid, method=method, bandwidth=2 * voxel)  # type: ignore[arg-type]
            assert density.shape == sdf_grid.shape
            assert np.all((density >= 0.0) & (density <= 1.0))
            params = PipelineParams(
                field_type="density",
                direct_extraction=True,
                extraction_level=0.5,
                repair=False,
                remesh=False,
                decimate=False,
            )
            result = process(PipelineState(field=density, params=params))
            assert result.vertices is not None, f"method={method} produced no vertices"
            assert result.faces is not None, f"method={method} produced no faces"
            assert result.vertices.shape[0] > 50, f"method={method} mesh too small"
