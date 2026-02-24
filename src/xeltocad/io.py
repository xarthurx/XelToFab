"""Density field loading and mesh export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from xeltocad.state import PipelineParams, PipelineState


def load_density(
    path: str | Path,
    params: PipelineParams | None = None,
) -> PipelineState:
    """Load a density field from .npy or .npz file."""
    path = Path(path)
    if params is None:
        params = PipelineParams()

    if path.suffix == ".npz":
        data = np.load(path)
        # Use first array, or 'density' key if present
        key = "density" if "density" in data else list(data.keys())[0]
        density = data[key]
    else:
        density = np.load(path)

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


def load_engibench(
    dataset_id: str,
    index: int = 0,
    design_key: str = "optimal_design",
    split: str = "train",
    params: PipelineParams | None = None,
) -> PipelineState:
    """Load a density field from EngiBench/IDEALLab HuggingFace datasets.

    Args:
        dataset_id: HuggingFace dataset ID, e.g. "IDEALLab/beams_2d_25_50_v0".
        index: Sample index within the split.
        design_key: Column name containing the density array.
        split: Dataset split to load from.
        params: Pipeline parameters.
    """
    from datasets import load_dataset

    if params is None:
        params = PipelineParams()

    dataset = load_dataset(dataset_id, split=split)
    sample = dataset[index]
    density = np.array(sample[design_key])

    return PipelineState(density=density, params=params)
