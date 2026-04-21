"""Tests for xeltofab.extrude."""

from __future__ import annotations

import numpy as np
import pytest

import xeltofab as xtf


def test_extrude_module_importable():
    """The extrude_2d symbol is reachable via the package root."""
    assert callable(xtf.extrude_2d)


def test_extrude_stub_raises_not_implemented():
    """Skeleton stub raises NotImplementedError until later tasks fill it in."""
    field = np.ones((4, 4), dtype=float)
    with pytest.raises(NotImplementedError):
        xtf.extrude_2d(field, thickness=1.0)
