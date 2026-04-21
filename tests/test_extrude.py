"""Tests for xeltofab.extrude."""

from __future__ import annotations

import numpy as np
import pytest

import xeltofab as xtf
from xeltofab.extrude import _build_binary, _trace_contours


def test_extrude_module_importable():
    """The extrude_2d symbol is reachable via the package root."""
    assert callable(xtf.extrude_2d)


def test_rejects_1d_field():
    with pytest.raises(ValueError, match="2D"):
        xtf.extrude_2d(np.ones(10), thickness=5.0)


def test_rejects_3d_field():
    with pytest.raises(ValueError, match="2D"):
        xtf.extrude_2d(np.ones((4, 4, 4)), thickness=5.0)


def test_rejects_zero_thickness():
    with pytest.raises(ValueError, match="thickness"):
        xtf.extrude_2d(np.ones((4, 4)), thickness=0.0)


def test_rejects_negative_thickness():
    with pytest.raises(ValueError, match="thickness"):
        xtf.extrude_2d(np.ones((4, 4)), thickness=-1.0)


def test_binary_density_default_level():
    """Density field thresholds at 0.5 by default."""
    field = np.array([[0.2, 0.8], [0.6, 0.4]])
    binary = _build_binary(
        field,
        field_type="density",
        level=None,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    np.testing.assert_array_equal(binary, [[False, True], [True, False]])


def test_binary_density_custom_level():
    """Density field honors explicit level."""
    field = np.array([[0.2, 0.8], [0.6, 0.4]])
    binary = _build_binary(
        field,
        field_type="density",
        level=0.7,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    np.testing.assert_array_equal(binary, [[False, True], [False, False]])


def test_binary_sdf_inside_is_negative():
    """SDF threshold: material where value <= level (default 0.0)."""
    field = np.array([[-0.5, 0.5], [-0.1, 0.1]])
    binary = _build_binary(
        field,
        field_type="sdf",
        level=None,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    np.testing.assert_array_equal(binary, [[True, False], [True, False]])


def test_binary_empty_raises():
    """All-below-threshold input raises ValueError."""
    field = np.zeros((4, 4))
    with pytest.raises(ValueError, match="no material"):
        _build_binary(
            field,
            field_type="density",
            level=None,
            smooth_sigma=0.0,
            fill_holes=False,
            min_component_area=0,
        )


def test_binary_gaussian_smooth_bridges_gap():
    """Smoothing a single-pixel gap between two blocks bridges it before threshold."""
    field = np.zeros((5, 9), dtype=float)
    field[:, :4] = 1.0
    field[:, 5:] = 1.0
    b0 = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    assert not b0[:, 4].any()
    b1 = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=2.0,
        fill_holes=False,
        min_component_area=0,
    )
    assert b1[:, 4].any()


def test_binary_fill_holes_closes_pinhole():
    """fill_holes=True morphologically closes a 1-pixel pinhole."""
    field = np.ones((5, 5), dtype=float)
    field[2, 2] = 0.0
    b_off = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    assert not b_off[2, 2]
    b_on = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=True,
        min_component_area=0,
    )
    assert b_on[2, 2]


def test_binary_min_component_area_drops_orphan():
    """min_component_area removes small disconnected islands."""
    field = np.zeros((10, 10), dtype=float)
    field[1:4, 1:4] = 1.0
    field[7:9, 7:9] = 1.0
    b = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=5,
    )
    assert b[1:4, 1:4].all()
    assert not b[7:9, 7:9].any()


def test_trace_contours_single_blob():
    """A single filled rectangle in the interior produces one closed contour."""
    binary = np.zeros((10, 10), dtype=bool)
    binary[3:7, 3:7] = True
    contours = _trace_contours(binary)
    assert len(contours) == 1
    c = contours[0]
    np.testing.assert_allclose(c[0], c[-1])
    assert c[:, 0].min() >= 2.4
    assert c[:, 0].max() <= 6.6


def test_trace_contours_two_disjoint_blobs():
    """Two disjoint filled regions produce two contours."""
    binary = np.zeros((10, 20), dtype=bool)
    binary[2:5, 2:5] = True
    binary[2:5, 12:15] = True
    contours = _trace_contours(binary)
    assert len(contours) == 2


def test_trace_contours_blob_on_boundary_is_closed():
    """Material touching the image edge still yields a closed contour (via the zero pad)."""
    binary = np.zeros((10, 10), dtype=bool)
    binary[0:5, 0:5] = True
    contours = _trace_contours(binary)
    assert len(contours) == 1
    c = contours[0]
    np.testing.assert_allclose(c[0], c[-1])
