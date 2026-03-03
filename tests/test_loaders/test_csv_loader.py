"""Tests for CSV/TXT loader."""

from pathlib import Path

import numpy as np
import pytest

from xeltofab.loaders.csv_loader import load


def test_load_csv_2d_table(tmp_path: Path):
    arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_csv_flat_with_shape_2d(tmp_path: Path):
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    result = load(path, field_name=None, shape=(2, 3))
    expected = arr.reshape(2, 3)
    np.testing.assert_array_almost_equal(result, expected)


def test_load_csv_flat_with_shape_3d(tmp_path: Path):
    arr = np.random.rand(24)
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    result = load(path, field_name=None, shape=(2, 3, 4))
    np.testing.assert_array_almost_equal(result, arr.reshape(2, 3, 4))


def test_load_txt_whitespace_delimited(tmp_path: Path):
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    path = tmp_path / "test.txt"
    np.savetxt(path, arr, delimiter=" ")
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_csv_with_header(tmp_path: Path):
    path = tmp_path / "test.csv"
    path.write_text("col1,col2,col3\n0.1,0.2,0.3\n0.4,0.5,0.6\n")
    result = load(path, field_name=None, shape=None)
    expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_array_almost_equal(result, expected)


def test_load_csv_shape_mismatch_raises(tmp_path: Path):
    arr = np.array([0.1, 0.2, 0.3])
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    with pytest.raises(ValueError, match="Cannot reshape"):
        load(path, field_name=None, shape=(2, 3))
