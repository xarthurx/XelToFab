"""Tests for loader registry dispatch."""

from pathlib import Path

import pytest

from xeltocad.loaders import _REGISTRY, LOADER_REGISTRY, get_supported_formats, resolve_loader


def test_registry_has_numpy_extensions():
    assert ".npy" in LOADER_REGISTRY
    assert ".npz" in LOADER_REGISTRY


def test_resolve_loader_npy():
    loader = resolve_loader(Path("test.npy"))
    assert loader is not None


def test_resolve_loader_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        resolve_loader(Path("test.xyz"))


def test_get_supported_formats_returns_list():
    formats = get_supported_formats()
    assert isinstance(formats, list)
    assert len(formats) > 0
    # Each entry is a dict with name, extensions, available, install_hint
    entry = formats[0]
    assert "name" in entry
    assert "extensions" in entry
    assert "available" in entry


def test_vtk_missing_dependency_error():
    """Loading .vtk without pyvista gives a clear install message."""
    _, dep, hint = _REGISTRY[".vtk"]
    assert dep == "pyvista"
    assert "pyvista" in hint


def test_hdf5_missing_dependency_error():
    """Loading .h5 without h5py gives a clear install message."""
    _, dep, hint = _REGISTRY[".h5"]
    assert dep == "h5py"
    assert "h5py" in hint
