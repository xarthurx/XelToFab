"""Tests for xeltofab.extrude."""

from __future__ import annotations

import numpy as np
import pytest

import xeltofab as xtf


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
