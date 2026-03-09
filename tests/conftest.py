"""Shared test fixtures and configuration."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before any pyplot import

import numpy as np
import pytest

from xeltofab.extract import extract
from xeltofab.pipeline import process
from xeltofab.preprocess import preprocess
from xeltofab.state import PipelineState


@pytest.fixture
def circle_density() -> np.ndarray:
    """2D density field with a filled circle."""
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return (x**2 + y**2 < 0.5**2).astype(float)


@pytest.fixture
def sphere_density() -> np.ndarray:
    """3D density field with a filled sphere."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return (x**2 + y**2 + z**2 < 0.5**2).astype(float)


@pytest.fixture
def small_sphere_density() -> np.ndarray:
    """Small 3D sphere for fast CLI tests."""
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    return (x**2 + y**2 + z**2 < 0.5**2).astype(float)


@pytest.fixture
def small_sphere_sdf() -> np.ndarray:
    """Small 3D SDF sphere for fast CLI tests."""
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    return np.sqrt(x**2 + y**2 + z**2) - 0.5


@pytest.fixture
def sphere_sdf() -> np.ndarray:
    """3D signed distance field for a sphere (negative inside, positive outside)."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return np.sqrt(x**2 + y**2 + z**2) - 0.5


@pytest.fixture
def circle_sdf() -> np.ndarray:
    """2D signed distance field for a circle."""
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return np.sqrt(x**2 + y**2) - 0.5


@pytest.fixture
def open_mesh_state(sphere_density: np.ndarray) -> PipelineState:
    """Pipeline state with an open mesh (faces removed to create holes)."""
    state = PipelineState(density=sphere_density)
    state = preprocess(state)
    state = extract(state)
    # Remove faces to create holes in the mesh
    open_faces = state.faces[:-20]
    return state.model_copy(update={"faces": open_faces, "smoothed_vertices": state.vertices.copy()})


@pytest.fixture
def processed_2d(circle_density: np.ndarray) -> PipelineState:
    """Fully processed 2D pipeline state."""
    return process(PipelineState(density=circle_density))


@pytest.fixture
def processed_3d(sphere_density: np.ndarray) -> PipelineState:
    """Fully processed 3D pipeline state."""
    return process(PipelineState(density=sphere_density))
