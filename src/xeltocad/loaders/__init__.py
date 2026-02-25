"""Loader registry — dispatches file loading by extension."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

# Type alias for loader functions
LoaderFunc = Callable[[Path, str | None, tuple[int, ...] | None], np.ndarray]

# Registry: extension -> (module_path, dependency_name, install_hint)
# Loaders are imported lazily to avoid requiring optional dependencies at import time.
_REGISTRY: dict[str, tuple[str, str | None, str | None]] = {
    ".npy": ("xeltocad.loaders.numpy_loader", None, None),
    ".npz": ("xeltocad.loaders.numpy_loader", None, None),
    ".mat": ("xeltocad.loaders.matlab_loader", None, None),
    ".csv": ("xeltocad.loaders.csv_loader", None, None),
    ".txt": ("xeltocad.loaders.csv_loader", None, None),
    ".vtk": ("xeltocad.loaders.vtk_loader", "pyvista", "uv add --optional vtk pyvista"),
    ".vtr": ("xeltocad.loaders.vtk_loader", "pyvista", "uv add --optional vtk pyvista"),
    ".vti": ("xeltocad.loaders.vtk_loader", "pyvista", "uv add --optional vtk pyvista"),
    ".h5": ("xeltocad.loaders.hdf5_loader", "h5py", "uv add --optional hdf5 h5py"),
    ".hdf5": ("xeltocad.loaders.hdf5_loader", "h5py", "uv add --optional hdf5 h5py"),
    ".xdmf": ("xeltocad.loaders.hdf5_loader", "h5py", "uv add --optional hdf5 h5py"),
}

# Public view of supported extensions
LOADER_REGISTRY: dict[str, str] = {ext: info[0] for ext, info in _REGISTRY.items()}


def resolve_loader(path: Path) -> LoaderFunc:
    """Return the load() function for the given file's extension."""
    ext = path.suffix.lower()
    if ext not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {supported}")

    module_path, dep_name, install_hint = _REGISTRY[ext]

    # Check optional dependency before importing loader
    if dep_name is not None:
        try:
            __import__(dep_name)
        except ImportError:
            raise ImportError(f"Loading {ext} files requires {dep_name}.\nInstall it with: {install_hint}") from None

    import importlib

    module = importlib.import_module(module_path)
    return module.load


# Format metadata for CLI listing
_FORMAT_INFO = [
    {"name": "numpy", "extensions": [".npy", ".npz"], "dep": None, "install_hint": "(built-in)"},
    {"name": "matlab", "extensions": [".mat"], "dep": None, "install_hint": "(built-in, via scipy)"},
    {"name": "csv", "extensions": [".csv", ".txt"], "dep": None, "install_hint": "(built-in)"},
    {
        "name": "vtk",
        "extensions": [".vtk", ".vtr", ".vti"],
        "dep": "pyvista",
        "install_hint": "uv add --optional vtk pyvista",
    },
    {
        "name": "hdf5",
        "extensions": [".h5", ".hdf5", ".xdmf"],
        "dep": "h5py",
        "install_hint": "uv add --optional hdf5 h5py",
    },
]


def get_supported_formats() -> list[dict]:
    """Return list of format info dicts with availability status."""
    result = []
    for info in _FORMAT_INFO:
        available = True
        if info["dep"] is not None:
            try:
                __import__(info["dep"])
            except ImportError:
                available = False
        result.append(
            {
                "name": info["name"],
                "extensions": info["extensions"],
                "available": available,
                "install_hint": info["install_hint"],
            }
        )
    return result
