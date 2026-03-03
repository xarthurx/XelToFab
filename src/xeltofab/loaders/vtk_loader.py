"""VTK structured grid loader (.vtk, .vtr, .vti)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista

# Field names to auto-detect (case-insensitive check)
_KNOWN_PATTERNS = ("density", "rho", "xphys", "x")


def _find_density_field(mesh: pyvista.DataSet, field_name: str | None) -> tuple[np.ndarray, str, bool]:
    """Find and return (array, name, is_cell_data) for the density field."""
    # Build lookup: name -> (array, is_cell_data)
    all_fields: dict[str, tuple[np.ndarray, bool]] = {}
    for name in mesh.cell_data:
        all_fields[name] = (np.asarray(mesh.cell_data[name]), True)
    for name in mesh.point_data:
        if name not in all_fields:
            all_fields[name] = (np.asarray(mesh.point_data[name]), False)

    if field_name is not None:
        if field_name not in all_fields:
            raise KeyError(f"Field '{field_name}' not found. Available: {list(all_fields.keys())}")
        arr, is_cell = all_fields[field_name]
        return arr, field_name, is_cell

    # Auto-detect: check cell_data first (TO densities are per-element)
    for source_data, is_cell in ((mesh.cell_data, True), (mesh.point_data, False)):
        for name in source_data:
            if any(pat in name.lower() for pat in _KNOWN_PATTERNS):
                return np.asarray(source_data[name]), name, is_cell

    # Single field fallback
    if len(all_fields) == 1:
        name = next(iter(all_fields))
        arr, is_cell = all_fields[name]
        return arr, name, is_cell

    raise ValueError(
        f"Could not auto-detect density field. Available: {list(all_fields.keys())}\nSpecify with --field-name"
    )


def _grid_dimensions(mesh: pyvista.DataSet, *, cell: bool) -> tuple[int, ...]:
    """Extract grid dimensions from a structured VTK dataset.

    If cell=True, returns cell counts (node_count - 1 per axis).
    If cell=False, returns node counts directly (for point_data).
    """
    if hasattr(mesh, "dimensions"):
        if cell:
            dims = tuple(int(d) - 1 for d in mesh.dimensions if d > 1)
        else:
            dims = tuple(int(d) for d in mesh.dimensions if d > 1)
        return dims
    raise ValueError(f"Cannot determine grid dimensions for {type(mesh).__name__}")


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from a VTK structured grid file."""
    mesh = pyvista.read(str(path))
    values, _, is_cell_data = _find_density_field(mesh, field_name)

    dims = _grid_dimensions(mesh, cell=is_cell_data)
    # VTK uses Fortran-like ordering internally; reshape and transpose to C order
    return values.reshape(dims[::-1]).transpose().astype(np.float64)
