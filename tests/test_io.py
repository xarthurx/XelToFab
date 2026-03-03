# tests/test_io.py
from pathlib import Path

import numpy as np
import pytest
import trimesh

from xeltofab.io import load_density, save_mesh
from xeltofab.state import PipelineState


def test_load_density_npy(tmp_path: Path):
    arr = np.random.rand(50, 100)
    path = tmp_path / "test.npy"
    np.save(path, arr)
    state = load_density(path)
    assert state.ndim == 2
    assert np.array_equal(state.density, arr)


def test_load_density_npz(tmp_path: Path):
    arr = np.random.rand(10, 20, 30)
    path = tmp_path / "test.npz"
    np.savez(path, density=arr)
    state = load_density(path)
    assert state.ndim == 3
    assert np.array_equal(state.density, arr)


def test_save_mesh_stl(tmp_path: Path, processed_3d: PipelineState):
    out = tmp_path / "output.stl"
    save_mesh(processed_3d, out)
    assert out.exists()
    loaded = trimesh.load(out)
    assert len(loaded.vertices) > 0


def test_save_mesh_obj(tmp_path: Path, processed_3d: PipelineState):
    out = tmp_path / "output.obj"
    save_mesh(processed_3d, out)
    assert out.exists()


def test_save_mesh_2d_raises(processed_2d: PipelineState, tmp_path: Path):
    with pytest.raises(ValueError, match="2D contour export"):
        save_mesh(processed_2d, tmp_path / "output.stl")


def test_save_mesh_before_extract_raises(sphere_density: np.ndarray, tmp_path: Path):
    state = PipelineState(density=sphere_density)
    with pytest.raises(ValueError, match="No mesh to save"):
        save_mesh(state, tmp_path / "output.stl")
