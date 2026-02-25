"""Tests for MATLAB .mat loader."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from xeltocad.loaders.matlab_loader import load

# Well-known MATLAB TO variable names the auto-detect should find
_AUTO_DETECT_NAMES = ["xPhys", "densities", "x", "rho", "dc", "density"]


def test_load_mat_auto_detect_xPhys(tmp_path: Path):
    arr = np.random.rand(50, 100)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"xPhys": arr})
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_mat_auto_detect_density(tmp_path: Path):
    arr = np.random.rand(20, 40)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"density": arr})
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_mat_explicit_field_name(tmp_path: Path):
    arr1 = np.random.rand(10, 20)
    arr2 = np.random.rand(10, 20)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"custom_field": arr1, "other": arr2})
    result = load(path, field_name="custom_field", shape=None)
    np.testing.assert_array_almost_equal(result, arr1)


def test_load_mat_single_variable_fallback(tmp_path: Path):
    """When no known name matches but only one variable, use it."""
    arr = np.random.rand(30, 60)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"my_weird_name": arr})
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_mat_multiple_unknown_variables_raises(tmp_path: Path):
    """Multiple unknown variables without field_name should raise."""
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"foo": np.zeros((5, 5)), "bar": np.ones((5, 5))})
    with pytest.raises(ValueError, match="Multiple variables found"):
        load(path, field_name=None, shape=None)


def test_load_mat_missing_field_name_raises(tmp_path: Path):
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"real_field": np.zeros((5, 5))})
    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)


def test_load_mat_returns_2d_or_3d(tmp_path: Path):
    for shape in [(10, 20), (5, 10, 15)]:
        arr = np.random.rand(*shape)
        path = tmp_path / "test.mat"
        scipy.io.savemat(path, {"xPhys": arr})
        result = load(path, field_name=None, shape=None)
        assert result.ndim in (2, 3)
        assert result.shape == shape
