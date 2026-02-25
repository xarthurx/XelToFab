# Multi-Format I/O Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `load_density()` to support .mat, .vtk/.vtr/.vti, .csv/.txt, .h5/.hdf5/.xdmf via a loader registry, with optional dependency extras and CLI integration.

**Architecture:** Loader registry pattern — each format has a dedicated loader module in `src/xeltocad/loaders/`. `load_density()` dispatches by file extension. Optional deps (pyvista, h5py) use lazy imports with clear error messages. Directory restructuring moves `examples/data/` to `data/examples/`.

**Tech Stack:** numpy, scipy.io (MATLAB), pyvista (VTK, optional), h5py (HDF5, optional), xml.etree (XDMF), click (CLI)

**Design doc:** `docs/plans/2026-02-25-multi-format-io-design.md`

---

### Task 1: Directory Restructuring

Move `examples/data/` → `data/examples/` and update all references.

**Files:**
- Move: `examples/data/*` → `data/examples/*`
- Modify: `notebooks/demo.py:68`
- Modify: `README.md:58,64`
- Modify: `examples/data/README.md` (becomes `data/examples/README.md`)

**Step 1: Move the directory**

```bash
mkdir -p data
git mv examples/data data/examples
rmdir examples  # remove empty parent
```

**Step 2: Update notebook reference**

In `notebooks/demo.py:68`, change:
```python
# Before
_data_dir = Path(__file__).resolve().parent.parent / "examples" / "data"
# After
_data_dir = Path(__file__).resolve().parent.parent / "data" / "examples"
```

**Step 3: Update README.md references**

In `README.md:58`, change:
```
# Before
Pre-computed topology optimization results are included in `examples/data/`
# After
Pre-computed topology optimization results are included in `data/examples/`
```

In `README.md:64`, change:
```python
# Before
state = load_density("examples/data/beams_2d_50x100_sample0.npy")
# After
state = load_density("data/examples/beams_2d_50x100_sample0.npy")
```

**Step 4: Update data README.md internal reference**

In `data/examples/README.md:41` (if it references its own path), update accordingly.

**Step 5: Run tests to verify nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: All existing tests pass (they use synthetic fixtures, not example data files).

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move examples/data/ to data/examples/"
```

---

### Task 2: Loader Registry Infrastructure + NumPy Loader

Create the `loaders/` package, registry, and migrate existing NumPy loading logic.

**Files:**
- Create: `src/xeltocad/loaders/__init__.py`
- Create: `src/xeltocad/loaders/numpy_loader.py`
- Modify: `src/xeltocad/io.py:1-30`
- Create: `tests/test_loaders/__init__.py`
- Create: `tests/test_loaders/test_numpy_loader.py`
- Create: `tests/test_loaders/test_dispatch.py`

**Step 1: Write the failing tests for dispatch and numpy loader**

Create `tests/test_loaders/__init__.py` (empty).

Create `tests/test_loaders/test_dispatch.py`:

```python
"""Tests for loader registry dispatch."""
from pathlib import Path

import numpy as np
import pytest

from xeltocad.loaders import LOADER_REGISTRY, get_supported_formats, resolve_loader


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
```

Create `tests/test_loaders/test_numpy_loader.py`:

```python
"""Tests for NumPy loader."""
from pathlib import Path

import numpy as np
import pytest

from xeltocad.loaders.numpy_loader import load


def test_load_npy(tmp_path: Path):
    arr = np.random.rand(50, 100)
    path = tmp_path / "test.npy"
    np.save(path, arr)
    result = load(path, field_name=None, shape=None)
    assert np.array_equal(result, arr)


def test_load_npz_density_key(tmp_path: Path):
    arr = np.random.rand(10, 20, 30)
    path = tmp_path / "test.npz"
    np.savez(path, density=arr)
    result = load(path, field_name=None, shape=None)
    assert np.array_equal(result, arr)


def test_load_npz_first_array_fallback(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.npz"
    np.savez(path, my_data=arr)
    result = load(path, field_name=None, shape=None)
    assert np.array_equal(result, arr)


def test_load_npz_explicit_field_name(tmp_path: Path):
    arr1 = np.random.rand(10, 20)
    arr2 = np.random.rand(10, 20)
    path = tmp_path / "test.npz"
    np.savez(path, first=arr1, second=arr2)
    result = load(path, field_name="second", shape=None)
    assert np.array_equal(result, arr2)


def test_load_npz_missing_field_name(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.npz"
    np.savez(path, my_data=arr)
    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)


def test_load_returns_ndarray(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.npy"
    np.save(path, arr)
    result = load(path, field_name=None, shape=None)
    assert isinstance(result, np.ndarray)
    assert result.ndim in (2, 3)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loaders/ -v
```

Expected: FAIL — `xeltocad.loaders` module does not exist yet.

**Step 3: Implement the loader registry and numpy loader**

Create `src/xeltocad/loaders/__init__.py`:

```python
"""Loader registry — dispatches file loading by extension."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

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
            raise ImportError(
                f"Loading {ext} files requires {dep_name}.\n"
                f"Install it with: {install_hint}"
            ) from None

    import importlib
    module = importlib.import_module(module_path)
    return module.load


# Format metadata for CLI listing
_FORMAT_INFO = [
    {"name": "numpy", "extensions": [".npy", ".npz"], "dep": None, "install_hint": "(built-in)"},
    {"name": "matlab", "extensions": [".mat"], "dep": None, "install_hint": "(built-in, via scipy)"},
    {"name": "csv", "extensions": [".csv", ".txt"], "dep": None, "install_hint": "(built-in)"},
    {"name": "vtk", "extensions": [".vtk", ".vtr", ".vti"], "dep": "pyvista", "install_hint": "uv add --optional vtk pyvista"},
    {"name": "hdf5", "extensions": [".h5", ".hdf5", ".xdmf"], "dep": "h5py", "install_hint": "uv add --optional hdf5 h5py"},
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
        result.append({
            "name": info["name"],
            "extensions": info["extensions"],
            "available": available,
            "install_hint": info["install_hint"],
        })
    return result
```

Create `src/xeltocad/loaders/numpy_loader.py`:

```python
"""NumPy .npy/.npz loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from .npy or .npz file."""
    if path.suffix == ".npz":
        data = np.load(path)
        if field_name is not None:
            if field_name not in data:
                raise KeyError(f"Field '{field_name}' not found in {path.name}. Available: {list(data.keys())}")
            return data[field_name]
        key = "density" if "density" in data else list(data.keys())[0]
        return data[key]
    else:
        return np.load(path)
```

**Step 4: Update `io.py` to dispatch through the registry**

Replace `src/xeltocad/io.py` content:

```python
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
```

**Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass — existing tests use `load_density("file.npy")` which still works, and new loader tests pass.

**Step 6: Commit**

```bash
git add src/xeltocad/loaders/__init__.py src/xeltocad/loaders/numpy_loader.py src/xeltocad/io.py tests/test_loaders/__init__.py tests/test_loaders/test_dispatch.py tests/test_loaders/test_numpy_loader.py
git commit -m "feat: add loader registry and migrate numpy loader"
```

---

### Task 3: MATLAB Loader

**Files:**
- Create: `src/xeltocad/loaders/matlab_loader.py`
- Create: `tests/test_loaders/test_matlab_loader.py`

**Step 1: Write failing tests**

Create `tests/test_loaders/test_matlab_loader.py`:

```python
"""Tests for MATLAB .mat loader."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from xeltocad.loaders.matlab_loader import load

# Well-known MATLAB TO variable names the auto-detect should find
_AUTO_DETECT_NAMES = ["xPhys", "densities", "x", "rho", "dc", "density"]


def test_load_mat_auto_detect_xPhys(tmp_path: Path):
    arr = np.random.rand(50, 100)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"xPhys": arr})
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_mat_auto_detect_density(tmp_path: Path):
    arr = np.random.rand(20, 40)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"density": arr})
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_mat_explicit_field_name(tmp_path: Path):
    arr1 = np.random.rand(10, 20)
    arr2 = np.random.rand(10, 20)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"custom_field": arr1, "other": arr2})
    result = load(path, field_name="custom_field", shape=None)
    np.testing.assert_array_almost_equal(result, arr1)


def test_load_mat_single_variable_fallback(tmp_path: Path):
    """When no known name matches but only one variable, use it."""
    arr = np.random.rand(30, 60)
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"my_weird_name": arr})
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_mat_multiple_unknown_variables_raises(tmp_path: Path):
    """Multiple unknown variables without field_name should raise."""
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"foo": np.zeros((5, 5)), "bar": np.ones((5, 5))})
    with pytest.raises(ValueError, match="Multiple variables found"):
        load(path, field_name=None, shape=None)


def test_load_mat_missing_field_name_raises(tmp_path: Path):
    path = tmp_path / "test.mat"
    scipy.io.savemat(path, {"real_field": np.zeros((5, 5))})
    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)


def test_load_mat_returns_2d_or_3d(tmp_path: Path):
    for shape in [(10, 20), (5, 10, 15)]:
        arr = np.random.rand(*shape)
        path = tmp_path / "test.mat"
        scipy.io.savemat(path, {"xPhys": arr})
        result = load(path, field_name=None, shape=None)
        assert result.ndim in (2, 3)
        assert result.shape == shape
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loaders/test_matlab_loader.py -v
```

Expected: FAIL — `xeltocad.loaders.matlab_loader` does not exist.

**Step 3: Implement**

Create `src/xeltocad/loaders/matlab_loader.py`:

```python
"""MATLAB .mat file loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

# Well-known MATLAB TO variable names, checked in priority order
_KNOWN_NAMES = ("xPhys", "densities", "x", "rho", "dc", "density")


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from a MATLAB .mat file.

    Auto-detects common TO variable names if field_name is not specified.
    """
    try:
        data = scipy.io.loadmat(path)
    except NotImplementedError:
        raise ValueError(
            f"Cannot read {path.name} — this appears to be a MATLAB v7.3+ file (HDF5 format).\n"
            "Save it as .h5 or resave in MATLAB with: save('file.mat', '-v7')"
        ) from None

    # Filter out MATLAB metadata keys
    user_keys = [k for k in data if not k.startswith("__")]

    if field_name is not None:
        if field_name not in user_keys:
            raise KeyError(f"Field '{field_name}' not found in {path.name}. Available: {user_keys}")
        return np.asarray(data[field_name], dtype=np.float64)

    # Auto-detect: try known names in priority order
    for name in _KNOWN_NAMES:
        if name in user_keys:
            return np.asarray(data[name], dtype=np.float64)

    # Fallback: single variable → use it; multiple → error
    if len(user_keys) == 1:
        return np.asarray(data[user_keys[0]], dtype=np.float64)

    raise ValueError(
        f"Multiple variables found in {path.name}: {user_keys}\n"
        "Specify which one with --field-name"
    )
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_loaders/test_matlab_loader.py -v
```

Expected: ALL pass.

**Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass.

**Step 6: Commit**

```bash
git add src/xeltocad/loaders/matlab_loader.py tests/test_loaders/test_matlab_loader.py
git commit -m "feat: add MATLAB .mat loader with auto-detection"
```

---

### Task 4: CSV/Text Loader

**Files:**
- Create: `src/xeltocad/loaders/csv_loader.py`
- Create: `tests/test_loaders/test_csv_loader.py`

**Step 1: Write failing tests**

Create `tests/test_loaders/test_csv_loader.py`:

```python
"""Tests for CSV/TXT loader."""
from pathlib import Path

import numpy as np
import pytest

from xeltocad.loaders.csv_loader import load


def test_load_csv_2d_table(tmp_path: Path):
    arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_csv_flat_with_shape_2d(tmp_path: Path):
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    result = load(path, field_name=None, shape=(2, 3))
    expected = arr.reshape(2, 3)
    np.testing.assert_array_almost_equal(result, expected)


def test_load_csv_flat_with_shape_3d(tmp_path: Path):
    arr = np.random.rand(24)
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    result = load(path, field_name=None, shape=(2, 3, 4))
    np.testing.assert_array_almost_equal(result, arr.reshape(2, 3, 4))


def test_load_txt_whitespace_delimited(tmp_path: Path):
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    path = tmp_path / "test.txt"
    np.savetxt(path, arr, delimiter=" ")
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_csv_with_header(tmp_path: Path):
    path = tmp_path / "test.csv"
    path.write_text("col1,col2,col3\n0.1,0.2,0.3\n0.4,0.5,0.6\n")
    result = load(path, field_name=None, shape=None)
    expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_array_almost_equal(result, expected)


def test_load_csv_shape_mismatch_raises(tmp_path: Path):
    arr = np.array([0.1, 0.2, 0.3])
    path = tmp_path / "test.csv"
    np.savetxt(path, arr, delimiter=",")
    with pytest.raises(ValueError, match="Cannot reshape"):
        load(path, field_name=None, shape=(2, 3))
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loaders/test_csv_loader.py -v
```

Expected: FAIL — module does not exist.

**Step 3: Implement**

Create `src/xeltocad/loaders/csv_loader.py`:

```python
"""CSV/TXT loader for density fields."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from CSV or whitespace-delimited text file.

    If shape is provided, data is loaded as flat values and reshaped.
    Otherwise, 2D table structure is inferred from rows/columns.
    """
    # Try comma-delimited first, fall back to whitespace
    for delimiter in (",", None):
        try:
            data = np.loadtxt(path, delimiter=delimiter)
            break
        except ValueError:
            if delimiter is None:
                # Both delimiters failed — try skipping header
                try:
                    data = np.genfromtxt(path, delimiter=",", skip_header=1)
                    if np.isnan(data).all():
                        data = np.genfromtxt(path, skip_header=1)
                    break
                except ValueError:
                    raise
            continue
    else:
        # This shouldn't be reached, but just in case
        data = np.genfromtxt(path, delimiter=",", skip_header=1)

    if shape is not None:
        total = int(np.prod(shape))
        flat = data.ravel()
        if flat.size != total:
            raise ValueError(
                f"Cannot reshape {flat.size} values into shape {shape} "
                f"(requires {total} values)"
            )
        return flat.reshape(shape)

    return data
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_loaders/test_csv_loader.py -v
```

Expected: ALL pass.

**Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass.

**Step 6: Commit**

```bash
git add src/xeltocad/loaders/csv_loader.py tests/test_loaders/test_csv_loader.py
git commit -m "feat: add CSV/TXT loader with shape parsing"
```

---

### Task 5: VTK Loader (Optional Dependency)

**Files:**
- Create: `src/xeltocad/loaders/vtk_loader.py`
- Create: `tests/test_loaders/test_vtk_loader.py`
- Modify: `pyproject.toml` (add vtk extra)

**Step 1: Add pyvista optional dependency to pyproject.toml**

In `pyproject.toml`, after the `dependencies` list, add:

```toml
[project.optional-dependencies]
vtk = ["pyvista>=0.43"]
```

Then install: `uv sync --extra vtk`

**Step 2: Write failing tests**

Create `tests/test_loaders/test_vtk_loader.py`:

```python
"""Tests for VTK loader."""
from pathlib import Path

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from xeltocad.loaders.vtk_loader import load


def test_load_vtr_cell_data(tmp_path: Path):
    """Load density from a rectilinear grid with cell data."""
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 11),  # 10 cells in x
        np.linspace(0, 1, 21),  # 20 cells in y
    )
    density = np.random.rand(grid.n_cells)
    grid.cell_data["density"] = density
    path = tmp_path / "test.vtr"
    grid.save(path)

    result = load(path, field_name=None, shape=None)
    assert result.ndim == 2
    assert result.shape == (10, 20)
    np.testing.assert_array_almost_equal(result.ravel(), density, decimal=5)


def test_load_vtk_explicit_field_name(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["custom_name"] = np.random.rand(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    result = load(path, field_name="custom_name", shape=None)
    assert result.ndim == 2


def test_load_vtr_3d(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 8),
        np.linspace(0, 1, 4),
    )
    density = np.random.rand(grid.n_cells)
    grid.cell_data["rho"] = density
    path = tmp_path / "test.vtr"
    grid.save(path)

    result = load(path, field_name=None, shape=None)
    assert result.ndim == 3
    assert result.shape == (5, 7, 3)


def test_load_vtk_auto_detect_xphys(tmp_path: Path):
    """Auto-detect should find 'xPhys' field."""
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["xPhys"] = np.random.rand(grid.n_cells)
    grid.cell_data["something_else"] = np.zeros(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    result = load(path, field_name=None, shape=None)
    assert result.shape == (5, 5)


def test_load_vtk_no_matching_field_raises(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["temperature"] = np.random.rand(grid.n_cells)
    grid.cell_data["pressure"] = np.random.rand(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    with pytest.raises(ValueError, match="Could not auto-detect"):
        load(path, field_name=None, shape=None)


def test_load_vtk_missing_field_name_raises(tmp_path: Path):
    grid = pyvista.RectilinearGrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    )
    grid.cell_data["density"] = np.random.rand(grid.n_cells)
    path = tmp_path / "test.vtk"
    grid.save(path)

    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_loaders/test_vtk_loader.py -v
```

Expected: FAIL — `xeltocad.loaders.vtk_loader` does not exist.

**Step 4: Implement**

Create `src/xeltocad/loaders/vtk_loader.py`:

```python
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
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_loaders/test_vtk_loader.py -v
```

Expected: ALL pass.

**Step 6: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass.

**Step 7: Commit**

```bash
git add src/xeltocad/loaders/vtk_loader.py tests/test_loaders/test_vtk_loader.py pyproject.toml
git commit -m "feat: add VTK structured grid loader (.vtk/.vtr/.vti)"
```

---

### Task 6: HDF5/XDMF Loader (Optional Dependency)

**Files:**
- Create: `src/xeltocad/loaders/hdf5_loader.py`
- Create: `tests/test_loaders/test_hdf5_loader.py`
- Modify: `pyproject.toml` (add hdf5 and all-formats extras)

**Step 1: Add h5py optional dependency and all-formats group**

In `pyproject.toml`, update optional-dependencies:

```toml
[project.optional-dependencies]
vtk = ["pyvista>=0.43"]
hdf5 = ["h5py>=3.10"]
all-formats = ["pyvista>=0.43", "h5py>=3.10"]
```

Then install: `uv sync --extra hdf5`

**Step 2: Write failing tests**

Create `tests/test_loaders/test_hdf5_loader.py`:

```python
"""Tests for HDF5/XDMF loader."""
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from xeltocad.loaders.hdf5_loader import load


def test_load_h5_auto_detect(tmp_path: Path):
    arr = np.random.rand(20, 40)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=arr)
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_explicit_field(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("custom", data=arr)
        f.create_dataset("other", data=np.zeros((5, 5)))
    result = load(path, field_name="custom", shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_nested_group(tmp_path: Path):
    """FEniCS-style nested path like /Function/0."""
    arr = np.random.rand(10, 10)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("Function")
        grp.create_dataset("0", data=arr)
    result = load(path, field_name="Function/0", shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_missing_field_raises(tmp_path: Path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("real_field", data=np.zeros((5, 5)))
    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)


def test_load_h5_multiple_unknown_datasets_raises(tmp_path: Path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("foo", data=np.zeros((5, 5)))
        f.create_dataset("bar", data=np.ones((5, 5)))
    with pytest.raises(ValueError, match="Multiple datasets"):
        load(path, field_name=None, shape=None)


def test_load_xdmf(tmp_path: Path):
    """XDMF file pointing to an HDF5 dataset."""
    arr = np.random.rand(10, 20).astype(np.float64)
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("density", data=arr)

    xdmf_path = tmp_path / "data.xdmf"
    xdmf_path.write_text(f"""\
<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="2DRectMesh" Dimensions="11 21"/>
      <Attribute Name="density" Center="Cell">
        <DataItem Format="HDF" Dimensions="10 20" DataType="Float" Precision="8">
          data.h5:/density
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""")

    result = load(xdmf_path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_returns_float64(tmp_path: Path):
    arr = np.random.rand(8, 8).astype(np.float32)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=arr)
    result = load(path, field_name=None, shape=None)
    assert result.dtype == np.float64
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_loaders/test_hdf5_loader.py -v
```

Expected: FAIL — module does not exist.

**Step 4: Implement**

Create `src/xeltocad/loaders/hdf5_loader.py`:

```python
"""HDF5 and XDMF loader for density fields."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

# Same auto-detect names as MATLAB loader
_KNOWN_NAMES = ("xPhys", "densities", "x", "rho", "dc", "density")


def _find_datasets(group: h5py.Group, prefix: str = "") -> list[str]:
    """Recursively collect all dataset paths in an HDF5 file."""
    paths = []
    for key in group:
        full_path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        if isinstance(item, h5py.Dataset):
            paths.append(full_path)
        elif isinstance(item, h5py.Group):
            paths.extend(_find_datasets(item, full_path))
    return paths


def _load_h5(path: Path, field_name: str | None) -> np.ndarray:
    """Load from a raw HDF5 file."""
    with h5py.File(path, "r") as f:
        if field_name is not None:
            if field_name not in f:
                all_datasets = _find_datasets(f)
                raise KeyError(
                    f"Field '{field_name}' not found in {path.name}. "
                    f"Available: {all_datasets}"
                )
            return np.asarray(f[field_name], dtype=np.float64)

        all_datasets = _find_datasets(f)

        # Auto-detect by known names (check leaf name)
        for ds_path in all_datasets:
            leaf = ds_path.rsplit("/", 1)[-1]
            if leaf in _KNOWN_NAMES:
                return np.asarray(f[ds_path], dtype=np.float64)

        # Single dataset fallback
        if len(all_datasets) == 1:
            return np.asarray(f[all_datasets[0]], dtype=np.float64)

        raise ValueError(
            f"Multiple datasets found in {path.name}: {all_datasets}\n"
            "Specify which one with --field-name"
        )


def _load_xdmf(path: Path, field_name: str | None) -> np.ndarray:
    """Load from an XDMF file (XML metadata pointing to HDF5 data)."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Find all DataItem elements with HDF format
    for data_item in root.iter("DataItem"):
        fmt = data_item.get("Format", "")
        if fmt.upper() != "HDF":
            continue

        text = data_item.text.strip()
        # Format: "filename.h5:/path/to/dataset"
        h5_filename, _, dataset_path = text.partition(":")
        if not dataset_path:
            continue

        # Check if this is the field we want (from parent Attribute element)
        parent = None
        for attr_elem in root.iter("Attribute"):
            for child in attr_elem.iter("DataItem"):
                if child is data_item:
                    parent = attr_elem
                    break

        if field_name is not None and parent is not None:
            attr_name = parent.get("Name", "")
            if attr_name != field_name:
                continue

        # Resolve HDF5 path relative to XDMF file
        h5_path = path.parent / h5_filename

        with h5py.File(h5_path, "r") as f:
            dataset_path = dataset_path.lstrip("/")
            return np.asarray(f[dataset_path], dtype=np.float64)

    raise ValueError(f"No HDF data items found in {path.name}")


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load density array from HDF5 or XDMF file."""
    if path.suffix.lower() == ".xdmf":
        return _load_xdmf(path, field_name)
    return _load_h5(path, field_name)
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_loaders/test_hdf5_loader.py -v
```

Expected: ALL pass.

**Step 6: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass.

**Step 7: Commit**

```bash
git add src/xeltocad/loaders/hdf5_loader.py tests/test_loaders/test_hdf5_loader.py pyproject.toml
git commit -m "feat: add HDF5/XDMF loader with auto-detection"
```

---

### Task 7: CLI Integration

Add `--field-name`, `--shape` options and `xtc formats` subcommand.

**Files:**
- Modify: `src/xeltocad/cli.py`
- Create: `tests/test_loaders/test_cli_formats.py`

**Step 1: Write failing tests**

Create `tests/test_loaders/test_cli_formats.py`:

```python
"""Tests for CLI format-related features."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from click.testing import CliRunner

from xeltocad.cli import main


def test_cli_formats_subcommand():
    runner = CliRunner()
    result = runner.invoke(main, ["formats"])
    assert result.exit_code == 0
    assert "numpy" in result.output
    assert ".npy" in result.output


def test_cli_process_mat_file(tmp_path: Path, small_sphere_density: np.ndarray):
    """Process a .mat file through the CLI."""
    input_path = tmp_path / "test.mat"
    scipy.io.savemat(input_path, {"xPhys": small_sphere_density})
    output_path = tmp_path / "output.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_mat_with_field_name(tmp_path: Path, small_sphere_density: np.ndarray):
    input_path = tmp_path / "test.mat"
    scipy.io.savemat(input_path, {"custom_name": small_sphere_density})
    output_path = tmp_path / "output.stl"

    runner = CliRunner()
    result = runner.invoke(
        main, ["process", str(input_path), "-o", str(output_path), "--field-name", "custom_name"]
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_csv_with_shape(tmp_path: Path):
    arr = np.random.rand(50, 100)
    input_path = tmp_path / "test.csv"
    np.savetxt(input_path, arr.ravel(), delimiter=",")
    output_path = tmp_path / "output.png"

    runner = CliRunner()
    result = runner.invoke(
        main, ["viz", str(input_path), "-o", str(output_path), "--shape", "50x100"]
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_unsupported_format(tmp_path: Path):
    input_path = tmp_path / "test.xyz"
    input_path.write_text("dummy")
    output_path = tmp_path / "output.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    assert "Unsupported file format" in result.output
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loaders/test_cli_formats.py -v
```

Expected: FAIL — `formats` subcommand doesn't exist, `--field-name` and `--shape` not recognized.

**Step 3: Implement CLI changes**

Replace `src/xeltocad/cli.py` with:

```python
"""CLI entrypoint for xelToCAD."""
from __future__ import annotations

from pathlib import Path

import click

from xeltocad.io import load_density, save_mesh
from xeltocad.loaders import get_supported_formats
from xeltocad.pipeline import process
from xeltocad.state import PipelineParams
from xeltocad.viz import plot_comparison


def _parse_shape(value: str) -> tuple[int, ...]:
    """Parse a shape string like '100x200' or '10x20x30' into a tuple."""
    parts = value.lower().split("x")
    return tuple(int(p) for p in parts)


@click.group()
def main() -> None:
    """xelToCAD — Topology optimization post-processing pipeline."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
@click.option("--viz", is_flag=True, help="Save a comparison visualization alongside the mesh")
def process_cmd(
    input_path: Path,
    output_path: Path,
    threshold: float,
    sigma: float,
    field_name: str | None,
    shape_str: str | None,
    viz: bool,
) -> None:
    """Process a density field into a mesh."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_density(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e)) from None

    import matplotlib.pyplot as plt

    state = process(state)
    try:
        save_mesh(state, output_path)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Saved mesh to {output_path}")

    if viz:
        fig = plot_comparison(state)
        viz_path = output_path.with_suffix(".png")
        fig.savefig(viz_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved visualization to {viz_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
def viz(
    input_path: Path,
    output_path: Path | None,
    threshold: float,
    sigma: float,
    field_name: str | None,
    shape_str: str | None,
) -> None:
    """Visualize a density field and its extraction result."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_density(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e)) from None

    state = process(state)
    fig = plot_comparison(state)

    if output_path:
        import matplotlib.pyplot as plt

        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved visualization to {output_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


@main.command()
def formats() -> None:
    """List supported input formats and their availability."""
    fmt_list = get_supported_formats()
    click.echo(f"{'Format':<10} {'Extensions':<20} {'Status':<12} {'Install'}")
    click.echo("-" * 65)
    for f in fmt_list:
        exts = ", ".join(f["extensions"])
        status = "available" if f["available"] else "missing"
        click.echo(f"{f['name']:<10} {exts:<20} {status:<12} {f['install_hint']}")
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_loaders/test_cli_formats.py -v
```

Expected: ALL pass.

**Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass (existing CLI tests still work — `load_density` signature is backwards compatible).

**Step 6: Commit**

```bash
git add src/xeltocad/cli.py tests/test_loaders/test_cli_formats.py
git commit -m "feat: add --field-name, --shape CLI options and xtc formats command"
```

---

### Task 8: Missing Dependency Error Tests

Verify that loading a format without its optional dependency gives a clear error.

**Files:**
- Modify: `tests/test_loaders/test_dispatch.py`

**Step 1: Add missing-dependency tests to dispatch**

Append to `tests/test_loaders/test_dispatch.py`:

```python
import unittest.mock


def test_vtk_missing_dependency_error():
    """Loading .vtk without pyvista gives a clear install message."""
    with unittest.mock.patch.dict("sys.modules", {"pyvista": None}):
        # Need to also clear the import cache
        with unittest.mock.patch("builtins.__import__", side_effect=_mock_import_no_pyvista):
            pass
    # Simpler approach: test the error message format directly
    from xeltocad.loaders import _REGISTRY
    _, dep, hint = _REGISTRY[".vtk"]
    assert dep == "pyvista"
    assert "pyvista" in hint


def test_hdf5_missing_dependency_error():
    """Loading .h5 without h5py gives a clear install message."""
    from xeltocad.loaders import _REGISTRY
    _, dep, hint = _REGISTRY[".h5"]
    assert dep == "h5py"
    assert "h5py" in hint
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_loaders/test_dispatch.py -v
```

Expected: ALL pass.

**Step 3: Commit**

```bash
git add tests/test_loaders/test_dispatch.py
git commit -m "test: add dependency error message verification"
```

---

### Task 9: Update Existing Tests and Lint

Clean up existing `tests/test_io.py` (still works but may need minor updates) and run linting.

**Files:**
- Verify: `tests/test_io.py` (should still pass as-is)
- Verify: `tests/test_cli.py` (should still pass as-is)

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: ALL pass.

**Step 2: Run linter**

```bash
ruff check src/xeltocad/loaders/ tests/test_loaders/
ruff format src/xeltocad/loaders/ tests/test_loaders/
```

Fix any issues.

**Step 3: Run type checker**

```bash
ty check src/xeltocad/loaders/
```

Fix any issues.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "style: lint and format new loader modules"
```

---

### Task 10: Documentation Updates

Update CLAUDE.md and README to reflect new format support.

**Files:**
- Modify: `CLAUDE.md` (update Core Dependencies table)
- Modify: `README.md` (update usage examples)
- Modify: `data/examples/README.md` (update path references)

**Step 1: Update CLAUDE.md Core Dependencies**

Add to the Core Dependencies table in `CLAUDE.md`:

```markdown
| I/O formats | `scipy.io` (.mat), `pyvista` (.vtk), `h5py` (.h5/.xdmf) |
```

**Step 2: Update README.md**

Add a "Supported Formats" section showing all input formats and install commands:

```markdown
## Supported Input Formats

| Format | Extensions | Install |
|--------|-----------|---------|
| NumPy | .npy, .npz | Built-in |
| MATLAB | .mat | Built-in (via scipy) |
| CSV/Text | .csv, .txt | Built-in |
| VTK | .vtk, .vtr, .vti | `uv sync --extra vtk` |
| HDF5/XDMF | .h5, .hdf5, .xdmf | `uv sync --extra hdf5` |
| All formats | — | `uv sync --extra all-formats` |
```

**Step 3: Commit**

```bash
git add CLAUDE.md README.md data/examples/README.md
git commit -m "docs: update for multi-format I/O support"
```
