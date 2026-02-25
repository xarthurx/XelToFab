"""Density field loading and mesh export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from xeltocad.loaders import resolve_loader
from xeltocad.state import PipelineParams, PipelineState


def load_density(
    path: str | Path,
    field_name: str | None = None,
    shape: tuple[int, ...] | None = None,
    params: PipelineParams | None = None,
) -> PipelineState:
    """Load a density field from a supported file format.

    Supported formats: .npy, .npz, .mat, .vtk, .vtr, .vti, .csv, .txt, .h5, .hdf5, .xdmf
    """
    path = Path(path)
    if params is None:
        params = PipelineParams()

    loader = resolve_loader(path)
    density = loader(path, field_name, shape)

    return PipelineState(density=density, params=params)


def save_mesh(state: PipelineState, path: str | Path) -> None:
    """Save extracted mesh to file (STL, OBJ, PLY)."""
    path = Path(path)

    if state.ndim == 3:
        vertices = state.smoothed_vertices if state.smoothed_vertices is not None else state.vertices
        if vertices is None or state.faces is None:
            raise ValueError("No mesh to save — run extract() first")
        mesh = trimesh.Trimesh(vertices=vertices, faces=state.faces)
        mesh.export(path)
    else:
        raise ValueError(f"2D contour export to {path.suffix} not supported — use viz for 2D output")
