"""Tests for NumPy loader."""

from pathlib import Path

import numpy as np
import pytest

from xeltocad.loaders.numpy_loader import load


def test_load_npy(tmp_path: Path):
    arr = np.random.rand(50, 100)
    path = tmp_path / "test.npy"
    np.save(path, arr)
    result = load(path, field_name=None, shape=None)
    assert np.array_equal(result, arr)


def test_load_npz_density_key(tmp_path: Path):
    arr = np.random.rand(10, 20, 30)
    path = tmp_path / "test.npz"
    np.savez(path, density=arr)
    result = load(path, field_name=None, shape=None)
    assert np.array_equal(result, arr)


def test_load_npz_first_array_fallback(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.npz"
    np.savez(path, my_data=arr)
    result = load(path, field_name=None, shape=None)
    assert np.array_equal(result, arr)


def test_load_npz_explicit_field_name(tmp_path: Path):
    arr1 = np.random.rand(10, 20)
    arr2 = np.random.rand(10, 20)
    path = tmp_path / "test.npz"
    np.savez(path, first=arr1, second=arr2)
    result = load(path, field_name="second", shape=None)
    assert np.array_equal(result, arr2)


def test_load_npz_missing_field_name(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.npz"
    np.savez(path, my_data=arr)
    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)


def test_load_returns_ndarray(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.npy"
    np.save(path, arr)
    result = load(path, field_name=None, shape=None)
    assert isinstance(result, np.ndarray)
    assert result.ndim in (2, 3)
