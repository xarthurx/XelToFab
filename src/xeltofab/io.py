"""Scalar field loading and mesh export."""

from __future__ import annotations

from pathlib import Path

import trimesh

from xeltofab.loaders import resolve_loader
from xeltofab.state import PipelineParams, PipelineState


def load_field(
    path: str | Path,
    field_name: str | None = None,
    shape: tuple[int, ...] | None = None,
    params: PipelineParams | None = None,
) -> PipelineState:
    """Load a scalar field from a supported file format.

    Supported formats: .npy, .npz, .mat, .vtk, .vtr, .vti, .csv, .txt, .h5, .hdf5, .xdmf
    """
    path = Path(path)
    if params is None:
        params = PipelineParams()

    loader = resolve_loader(path)
    field = loader(path, field_name, shape)

    return PipelineState(field=field, params=params)


# Deprecated alias — use load_field() instead
load_density = load_field


def save_mesh(state: PipelineState, path: str | Path) -> None:
    """Save extracted mesh to file (STL, OBJ, PLY)."""
    path = Path(path)

    if state.ndim == 3:
        vertices = state.best_vertices
        if vertices is None or state.faces is None:
            raise ValueError("No mesh to save — run extract() first")
        mesh = trimesh.Trimesh(vertices=vertices, faces=state.faces)
        mesh.export(path)
    else:
        raise ValueError(f"2D contour export to {path.suffix} not supported — use viz for 2D output")
