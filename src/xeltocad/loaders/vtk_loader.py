"""VTK structured grid loader (.vtk, .vtr, .vti)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista

# Field names to auto-detect (case-insensitive check)
_KNOWN_PATTERNS = ("density", "rho", "xphys", "x")


def _find_density_field(mesh: pyvista.DataSet, field_name: str | None) -> tuple[np.ndarray, str]:
    """Find and return the density field array and its source (cell_data or point_data)."""
    # Collect all available fields
    all_fields: dict[str, np.ndarray] = {}
    for name in mesh.cell_data:
        all_fields[name] = np.asarray(mesh.cell_data[name])
    for name in mesh.point_data:
        if name not in all_fields:
            all_fields[name] = np.asarray(mesh.point_data[name])

    if field_name is not None:
        if field_name not in all_fields:
            raise KeyError(
                f"Field '{field_name}' not found. Available: {list(all_fields.keys())}"
            )
        return all_fields[field_name], field_name

    # Auto-detect: check cell_data first (TO densities are per-element)
    for source_data in (mesh.cell_data, mesh.point_data):
        for name in source_data:
            if any(pat in name.lower() for pat in _KNOWN_PATTERNS):
                return np.asarray(source_data[name]), name

    # Single field fallback
    if len(all_fields) == 1:
        name = next(iter(all_fields))
        return all_fields[name], name

    raise ValueError(
        f"Could not auto-detect density field. Available: {list(all_fields.keys())}\n"
        "Specify with --field-name"
    )


def _grid_dimensions(mesh: pyvista.DataSet) -> tuple[int, ...]:
    """Extract cell dimensions from a structured VTK dataset."""
    if hasattr(mesh, "dimensions"):
        # RectilinearGrid, ImageData, StructuredGrid — dimensions are node counts
        dims = tuple(int(d) - 1 for d in mesh.dimensions if d > 1)
        return dims
    raise ValueError(f"Cannot determine grid dimensions for {type(mesh).__name__}")


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from a VTK structured grid file."""
    mesh = pyvista.read(str(path))
    values, _ = _find_density_field(mesh, field_name)

    dims = _grid_dimensions(mesh)
    # VTK uses Fortran-like ordering internally; reshape and transpose to C order
    return values.reshape(dims[::-1]).transpose().astype(np.float64)
