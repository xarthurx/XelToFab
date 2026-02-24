# tests/test_io.py
from pathlib import Path

import numpy as np
import trimesh

from xeltocad.io import load_density, save_mesh
from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.smooth import smooth
from xeltocad.state import PipelineState


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


def test_save_mesh_stl(tmp_path: Path):
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    state = smooth(extract(preprocess(PipelineState(density=density))))
    out = tmp_path / "output.stl"
    save_mesh(state, out)
    assert out.exists()
    loaded = trimesh.load(out)
    assert len(loaded.vertices) > 0


def test_save_mesh_obj(tmp_path: Path):
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    state = smooth(extract(preprocess(PipelineState(density=density))))
    out = tmp_path / "output.obj"
    save_mesh(state, out)
    assert out.exists()
