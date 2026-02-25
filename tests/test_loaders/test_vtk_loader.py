"""Tests for VTK loader."""
from pathlib import Path

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from xeltocad.loaders.vtk_loader import load


def test_load_vtr_cell_data(tmp_path: Path):
    """Load density from a rectilinear grid with cell data."""
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 11),  # 10 cells in x
        np.linspace(0, 1, 21),  # 20 cells in y
    )
    density = np.random.rand(grid.n_cells)
    grid.cell_data["density"] = density
    path = tmp_path / "test.vtr"
    grid.save(path)

    result = load(path, field_name=None, shape=None)
    assert result.ndim == 2
    assert result.shape == (10, 20)
    # VTK uses Fortran (column-major) ordering for cell data
    np.testing.assert_array_almost_equal(result.ravel(order="F"), density, decimal=5)


def test_load_vtk_explicit_field_name(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["custom_name"] = np.random.rand(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    result = load(path, field_name="custom_name", shape=None)
    assert result.ndim == 2


def test_load_vtr_3d(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 8),
        np.linspace(0, 1, 4),
    )
    density = np.random.rand(grid.n_cells)
    grid.cell_data["rho"] = density
    path = tmp_path / "test.vtr"
    grid.save(path)

    result = load(path, field_name=None, shape=None)
    assert result.ndim == 3
    assert result.shape == (5, 7, 3)


def test_load_vtk_auto_detect_xphys(tmp_path: Path):
    """Auto-detect should find 'xPhys' field."""
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["xPhys"] = np.random.rand(grid.n_cells)
    grid.cell_data["something_else"] = np.zeros(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    result = load(path, field_name=None, shape=None)
    assert result.shape == (5, 5)


def test_load_vtk_no_matching_field_raises(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["temperature"] = np.random.rand(grid.n_cells)
    grid.cell_data["pressure"] = np.random.rand(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    with pytest.raises(ValueError, match="Could not auto-detect"):
        load(path, field_name=None, shape=None)


def test_load_vtk_missing_field_name_raises(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["density"] = np.random.rand(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)
