# MVP Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a density-to-mesh pipeline (2D + 3D) with preprocessing, extraction, Taubin smoothing, file I/O, matplotlib visualization, and CLI.

**Architecture:** Pydantic `PipelineState` threaded through pure stage functions (one module per stage). CLI via click, entrypoint `xtc`.

**Tech Stack:** numpy, scipy, scikit-image, trimesh, pydantic, matplotlib, click

---

### Task 1: Add dependencies (pydantic, matplotlib, click)

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add deps**

Run: `uv add pydantic matplotlib click`

**Step 2: Verify**

Run: `uv run python -c "import pydantic, matplotlib, click; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pydantic, matplotlib, click"
```

---

### Task 2: State models (`state.py`)

**Files:**
- Create: `src/xeltocad/state.py`
- Test: `tests/test_state.py`

**Step 1: Write failing tests**

```python
# tests/test_state.py
import numpy as np
import pytest

from xeltocad.state import PipelineParams, PipelineState


def test_pipeline_params_defaults():
    params = PipelineParams()
    assert params.threshold == 0.5
    assert params.smooth_sigma == 1.0
    assert params.morph_radius == 1
    assert params.taubin_iterations == 20
    assert params.taubin_pass_band == 0.1


def test_pipeline_state_2d():
    density = np.random.rand(50, 100)
    state = PipelineState(density=density)
    assert state.ndim == 2
    assert state.params.threshold == 0.5
    assert state.binary is None
    assert state.volume_fraction is None


def test_pipeline_state_3d():
    density = np.random.rand(10, 20, 30)
    state = PipelineState(density=density)
    assert state.ndim == 3


def test_pipeline_state_rejects_1d():
    with pytest.raises(Exception):
        PipelineState(density=np.array([1.0, 2.0, 3.0]))


def test_pipeline_state_rejects_4d():
    with pytest.raises(Exception):
        PipelineState(density=np.random.rand(2, 3, 4, 5))


def test_pipeline_params_validates_threshold():
    with pytest.raises(Exception):
        PipelineParams(threshold=1.5)
    with pytest.raises(Exception):
        PipelineParams(threshold=-0.1)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xeltocad.state'`

**Step 3: Write implementation**

```python
# src/xeltocad/state.py
"""Pipeline state and parameter models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PipelineParams(BaseModel):
    """Configurable parameters for the pipeline."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    smooth_sigma: float = Field(default=1.0, ge=0.0)
    morph_radius: int = Field(default=1, ge=0)
    taubin_iterations: int = Field(default=20, ge=0)
    taubin_pass_band: float = Field(default=0.1, gt=0.0, le=1.0)


class PipelineState(BaseModel):
    """Immutable pipeline state threaded through stage functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    density: np.ndarray
    ndim: int = 0  # computed from density
    params: PipelineParams = Field(default_factory=PipelineParams)

    binary: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    contours: list[np.ndarray] | None = None
    smoothed_vertices: np.ndarray | None = None
    volume_fraction: float | None = None

    @field_validator("density")
    @classmethod
    def validate_density(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim not in (2, 3):
            raise ValueError(f"density must be 2D or 3D, got {v.ndim}D")
        return v

    @model_validator(mode="after")
    def set_ndim(self) -> PipelineState:
        self.ndim = self.density.ndim
        return self
```

**Step 4: Add pytest as dev dep and run tests**

Run: `uv add --dev pytest && uv run pytest tests/test_state.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add src/xeltocad/state.py tests/test_state.py pyproject.toml uv.lock
git commit -m "feat: add PipelineState and PipelineParams models"
```

---

### Task 3: Preprocessing (`preprocess.py`)

**Files:**
- Create: `src/xeltocad/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write failing tests**

```python
# tests/test_preprocess.py
import numpy as np

from xeltocad.preprocess import preprocess
from xeltocad.state import PipelineState


def test_preprocess_2d_produces_binary():
    """Preprocessing a 2D density field should produce a binary array."""
    density = np.random.rand(50, 100)
    state = PipelineState(density=density)
    result = preprocess(state)
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 1})
    assert result.binary.shape == density.shape


def test_preprocess_3d_produces_binary():
    """Preprocessing a 3D density field should produce a binary array."""
    density = np.random.rand(10, 20, 30)
    state = PipelineState(density=density)
    result = preprocess(state)
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 1})
    assert result.binary.shape == density.shape


def test_preprocess_records_volume_fraction():
    """Volume fraction of original field should be recorded."""
    density = np.ones((10, 10)) * 0.7
    state = PipelineState(density=density)
    result = preprocess(state)
    assert result.volume_fraction is not None
    assert abs(result.volume_fraction - 0.7) < 0.01


def test_preprocess_removes_small_components():
    """Small disconnected blobs should be removed."""
    density = np.zeros((50, 50))
    density[10:40, 10:40] = 1.0  # large block
    density[2:4, 2:4] = 1.0      # tiny island (4 pixels)
    state = PipelineState(density=density)
    result = preprocess(state)
    # tiny island should be removed
    assert result.binary[3, 3] == 0
    # large block should remain
    assert result.binary[25, 25] == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/preprocess.py
"""Density field preprocessing: threshold, smooth, cleanup."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.morphology import binary_closing, binary_opening, remove_small_objects

from xeltocad.state import PipelineState


def preprocess(state: PipelineState) -> PipelineState:
    """Preprocess density field: smooth → threshold → morphology → keep largest component."""
    params = state.params
    density = state.density

    # Record original volume fraction
    volume_fraction = float(np.mean(density))

    # Gaussian smooth
    smoothed = gaussian_filter(density, sigma=params.smooth_sigma)

    # Threshold to binary
    binary = (smoothed >= params.threshold).astype(np.uint8)

    # Morphological cleanup
    if params.morph_radius > 0:
        if state.ndim == 2:
            from skimage.morphology import disk

            selem = disk(params.morph_radius)
        else:
            from skimage.morphology import ball

            selem = ball(params.morph_radius)
        binary = binary_opening(binary, selem).astype(np.uint8)
        binary = binary_closing(binary, selem).astype(np.uint8)

    # Remove small disconnected components
    binary_bool = binary.astype(bool)
    min_size = max(binary.size // 200, 8)  # at least 0.5% of total or 8 pixels
    cleaned = remove_small_objects(binary_bool, min_size=min_size)
    binary = cleaned.astype(np.uint8)

    return state.model_copy(update={"binary": binary, "volume_fraction": volume_fraction})
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/xeltocad/preprocess.py tests/test_preprocess.py
git commit -m "feat: add density field preprocessing"
```

---

### Task 4: Mesh extraction (`extract.py`)

**Files:**
- Create: `src/xeltocad/extract.py`
- Test: `tests/test_extract.py`

**Step 1: Write failing tests**

```python
# tests/test_extract.py
import numpy as np

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.state import PipelineState


def _make_2d_circle():
    """Create a 2D density field with a filled circle."""
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return (x**2 + y**2 < 0.5**2).astype(float)


def _make_3d_sphere():
    """Create a 3D density field with a filled sphere."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return (x**2 + y**2 + z**2 < 0.5**2).astype(float)


def test_extract_2d_produces_contours():
    state = preprocess(PipelineState(density=_make_2d_circle()))
    result = extract(state)
    assert result.contours is not None
    assert len(result.contours) > 0
    assert result.contours[0].shape[1] == 2  # (K, 2) arrays


def test_extract_3d_produces_mesh():
    state = preprocess(PipelineState(density=_make_3d_sphere()))
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[1] == 3  # (N, 3)
    assert result.faces.shape[1] == 3     # (M, 3)


def test_extract_3d_mesh_is_nonempty():
    state = preprocess(PipelineState(density=_make_3d_sphere()))
    result = extract(state)
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/extract.py
"""Mesh/contour extraction from preprocessed binary fields."""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours, marching_cubes

from xeltocad.state import PipelineState


def extract(state: PipelineState) -> PipelineState:
    """Extract mesh (3D) or contours (2D) from the binary field."""
    if state.binary is None:
        raise ValueError("binary field is None — run preprocess() first")

    if state.ndim == 2:
        return _extract_2d(state)
    return _extract_3d(state)


def _extract_2d(state: PipelineState) -> PipelineState:
    """Extract contours from 2D binary field using marching squares."""
    contours = find_contours(state.binary.astype(float), level=0.5)
    return state.model_copy(update={"contours": contours})


def _extract_3d(state: PipelineState) -> PipelineState:
    """Extract triangle mesh from 3D binary field using marching cubes."""
    vertices, faces, _, _ = marching_cubes(state.binary.astype(float), level=0.5)
    return state.model_copy(update={"vertices": vertices, "faces": faces})
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_extract.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/xeltocad/extract.py tests/test_extract.py
git commit -m "feat: add 2D contour and 3D mesh extraction"
```

---

### Task 5: Taubin smoothing (`smooth.py`)

**Files:**
- Create: `src/xeltocad/smooth.py`
- Test: `tests/test_smooth.py`

**Step 1: Write failing tests**

```python
# tests/test_smooth.py
import numpy as np

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.smooth import smooth
from xeltocad.state import PipelineState


def _make_3d_sphere():
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return (x**2 + y**2 + z**2 < 0.5**2).astype(float)


def _make_2d_circle():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return (x**2 + y**2 < 0.5**2).astype(float)


def test_smooth_3d_produces_smoothed_vertices():
    state = extract(preprocess(PipelineState(density=_make_3d_sphere())))
    result = smooth(state)
    assert result.smoothed_vertices is not None
    assert result.smoothed_vertices.shape == result.vertices.shape


def test_smooth_3d_changes_vertices():
    state = extract(preprocess(PipelineState(density=_make_3d_sphere())))
    result = smooth(state)
    # smoothed vertices should differ from original
    assert not np.allclose(result.smoothed_vertices, result.vertices)


def test_smooth_2d_is_noop():
    """Taubin smoothing only applies to 3D meshes; 2D contours pass through."""
    state = extract(preprocess(PipelineState(density=_make_2d_circle())))
    result = smooth(state)
    assert result.smoothed_vertices is None  # no 3D mesh to smooth
    assert result.contours is not None       # contours unchanged
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_smooth.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/smooth.py
"""Mesh smoothing operations."""

from __future__ import annotations

import trimesh

from xeltocad.state import PipelineState


def smooth(state: PipelineState) -> PipelineState:
    """Apply Taubin smoothing to 3D mesh. No-op for 2D contours."""
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state

    mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
    trimesh.smoothing.filter_taubin(
        mesh,
        iterations=state.params.taubin_iterations,
        lamb=state.params.taubin_pass_band,
    )
    return state.model_copy(update={"smoothed_vertices": mesh.vertices})
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_smooth.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/xeltocad/smooth.py tests/test_smooth.py
git commit -m "feat: add Taubin mesh smoothing"
```

---

### Task 6: I/O (`io.py`)

**Files:**
- Create: `src/xeltocad/io.py`
- Test: `tests/test_io.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/io.py
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_io.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/xeltocad/io.py tests/test_io.py
git commit -m "feat: add density loading and mesh export"
```

---

### Task 7: Pipeline orchestrator (`pipeline.py`)

**Files:**
- Create: `src/xeltocad/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import numpy as np

from xeltocad.pipeline import process
from xeltocad.state import PipelineState


def test_process_2d_end_to_end():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    density = (x**2 + y**2 < 0.5**2).astype(float)
    state = PipelineState(density=density)
    result = process(state)
    assert result.binary is not None
    assert result.contours is not None
    assert result.volume_fraction is not None


def test_process_3d_end_to_end():
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    state = PipelineState(density=density)
    result = process(state)
    assert result.binary is not None
    assert result.vertices is not None
    assert result.faces is not None
    assert result.smoothed_vertices is not None
    assert result.volume_fraction is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/pipeline.py
"""Pipeline orchestrator."""

from __future__ import annotations

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.smooth import smooth
from xeltocad.state import PipelineState


def process(state: PipelineState) -> PipelineState:
    """Run the full preprocessing → extraction → smoothing pipeline."""
    state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    return state
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/xeltocad/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator"
```

---

### Task 8: Visualization (`viz.py`)

**Files:**
- Create: `src/xeltocad/viz.py`
- Test: `tests/test_viz.py`

**Step 1: Write failing tests**

```python
# tests/test_viz.py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

import numpy as np
from matplotlib.figure import Figure

from xeltocad.pipeline import process
from xeltocad.state import PipelineState
from xeltocad.viz import plot_density, plot_result, plot_comparison


def _process_2d():
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    density = (x**2 + y**2 < 0.5**2).astype(float)
    return process(PipelineState(density=density))


def _process_3d():
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    return process(PipelineState(density=density))


def test_plot_density_2d():
    state = _process_2d()
    fig = plot_density(state)
    assert isinstance(fig, Figure)


def test_plot_density_3d():
    state = _process_3d()
    fig = plot_density(state)
    assert isinstance(fig, Figure)


def test_plot_result_2d():
    state = _process_2d()
    fig = plot_result(state)
    assert isinstance(fig, Figure)


def test_plot_result_3d():
    state = _process_3d()
    fig = plot_result(state)
    assert isinstance(fig, Figure)


def test_plot_comparison_2d():
    state = _process_2d()
    fig = plot_comparison(state)
    assert isinstance(fig, Figure)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/viz.py
"""Visualization functions for density fields and meshes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from xeltocad.state import PipelineState


def plot_density(state: PipelineState) -> Figure:
    """Plot the raw density field. 2D: heatmap. 3D: mid-plane slices."""
    if state.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(state.density, cmap="viridis", origin="lower", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Density")
        ax.set_title("Density Field")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        d = state.density
        slices = [
            (d[d.shape[0] // 2, :, :], "XY (mid-Z)"),
            (d[:, d.shape[1] // 2, :], "XZ (mid-Y)"),
            (d[:, :, d.shape[2] // 2], "YZ (mid-X)"),
        ]
        for ax, (sl, title) in zip(axes, slices):
            im = ax.imshow(sl, cmap="viridis", origin="lower", vmin=0, vmax=1)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle("Density Field (mid-plane slices)")
    fig.tight_layout()
    return fig


def plot_result(state: PipelineState) -> Figure:
    """Plot extraction result. 2D: contours on binary. 3D: trisurf wireframe."""
    if state.ndim == 2:
        fig, ax = plt.subplots()
        ax.imshow(state.binary, cmap="gray", origin="lower")
        if state.contours is not None:
            for contour in state.contours:
                ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1.5)
        ax.set_title("Extracted Contours")
    else:
        vertices = state.smoothed_vertices if state.smoothed_vertices is not None else state.vertices
        if vertices is None or state.faces is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No mesh data", ha="center", va="center")
            return fig
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=state.faces,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.1,
        )
        ax.set_title("Extracted Mesh")
    fig.tight_layout()
    return fig


def plot_comparison(state: PipelineState) -> Figure:
    """Side-by-side: density field vs extraction result."""
    if state.ndim == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(state.density, cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax1.set_title("Density Field")
        ax2.imshow(state.binary, cmap="gray", origin="lower")
        if state.contours is not None:
            for contour in state.contours:
                ax2.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1.5)
        ax2.set_title("Extracted Contours")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        d = state.density
        ax1.imshow(d[d.shape[0] // 2, :, :], cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax1.set_title("Density (mid-Z slice)")
        vertices = state.smoothed_vertices if state.smoothed_vertices is not None else state.vertices
        if vertices is not None and state.faces is not None:
            ax2.remove()
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=state.faces,
                alpha=0.7,
                edgecolor="k",
                linewidth=0.1,
            )
        ax2.set_title("Extracted Mesh")
    fig.suptitle(f"Volume fraction: {state.volume_fraction:.3f}" if state.volume_fraction else "")
    fig.tight_layout()
    return fig
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_viz.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/xeltocad/viz.py tests/test_viz.py
git commit -m "feat: add matplotlib visualization"
```

---

### Task 9: CLI (`cli.py`) + entrypoint

**Files:**
- Create: `src/xeltocad/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from xeltocad.cli import main


def test_cli_process_3d(tmp_path: Path):
    # Create test input
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, density)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_with_params(tmp_path: Path):
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, density)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(
        main, ["process", str(input_path), "-o", str(output_path), "--threshold", "0.4", "--sigma", "1.5"]
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_viz_2d(tmp_path: Path):
    y, x = np.mgrid[-1:1:50j, -1:1:50j]
    density = (x**2 + y**2 < 0.5**2).astype(float)
    input_path = tmp_path / "circle.npy"
    np.save(input_path, density)
    output_path = tmp_path / "circle.png"

    runner = CliRunner()
    result = runner.invoke(main, ["viz", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/xeltocad/cli.py
"""CLI entrypoint for xelToCAD."""

from __future__ import annotations

from pathlib import Path

import click

from xeltocad.io import load_density, save_mesh
from xeltocad.pipeline import process
from xeltocad.state import PipelineParams
from xeltocad.viz import plot_comparison


@click.group()
def main() -> None:
    """xelToCAD — Topology optimization post-processing pipeline."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("--viz", is_flag=True, help="Save a comparison visualization alongside the mesh")
def process_cmd(input_path: Path, output_path: Path, threshold: float, sigma: float, viz: bool) -> None:
    """Process a density field into a mesh."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    state = load_density(input_path, params=params)
    state = process(state)
    save_mesh(state, output_path)
    click.echo(f"Saved mesh to {output_path}")

    if viz:
        fig = plot_comparison(state)
        viz_path = output_path.with_suffix(".png")
        fig.savefig(viz_path, dpi=150)
        click.echo(f"Saved visualization to {viz_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
def viz(input_path: Path, output_path: Path | None, threshold: float, sigma: float) -> None:
    """Visualize a density field and its extraction result."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    state = load_density(input_path, params=params)
    state = process(state)
    fig = plot_comparison(state)

    if output_path:
        fig.savefig(output_path, dpi=150)
        click.echo(f"Saved visualization to {output_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
```

**Step 4: Add `[project.scripts]` to pyproject.toml**

Add to `pyproject.toml` after `[project]` section:

```toml
[project.scripts]
xtc = "xeltocad.cli:main"
```

**Step 5: Run tests**

Run: `uv sync && uv run pytest tests/test_cli.py -v`
Expected: 3 passed

**Step 6: Verify CLI works**

Run: `uv run xtc --help`
Expected: Shows help with `process` and `viz` subcommands

**Step 7: Commit**

```bash
git add src/xeltocad/cli.py tests/test_cli.py pyproject.toml uv.lock
git commit -m "feat: add CLI entrypoint (xtc)"
```

---

### Task 10: Run full test suite + lint

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (approximately 22 tests)

**Step 2: Run ruff**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: All checks passed

**Step 3: Fix any issues, then commit**

```bash
git add -A
git commit -m "chore: lint and format cleanup"
```

---

### Task 11: EngiBench integration (`io.py` extension)

**Files:**
- Modify: `src/xeltocad/io.py`
- Test: `tests/test_engibench.py`

**Step 1: Add engibench dep**

Run: `uv add engibench`

Note: If `engibench` is not pip-installable, use `datasets` (HuggingFace) directly instead:
Run: `uv add datasets`

**Step 2: Write failing test**

```python
# tests/test_engibench.py
import pytest

from xeltocad.io import load_engibench


@pytest.mark.network
def test_load_engibench_beams2d():
    """Load a Beams2D sample from EngiBench."""
    state = load_engibench("Beams2D", index=0)
    assert state.ndim == 2
    assert state.density.shape[0] > 0
    assert state.density.shape[1] > 0
```

**Step 3: Implement `load_engibench` in `io.py`**

Add to `src/xeltocad/io.py`:

```python
def load_engibench(
    problem_name: str,
    index: int = 0,
    params: PipelineParams | None = None,
) -> PipelineState:
    """Load a density field from EngiBench HuggingFace datasets."""
    from datasets import load_dataset

    if params is None:
        params = PipelineParams()

    dataset = load_dataset(f"EngiBench/{problem_name}", split="train")
    sample = dataset[index]

    # EngiBench stores density as numpy arrays under 'design' key
    density = np.array(sample["design"])

    return PipelineState(density=density, params=params)
```

**Step 4: Run test**

Run: `uv run pytest tests/test_engibench.py -v -m network`
Expected: 1 passed (requires network)

**Step 5: Commit**

```bash
git add src/xeltocad/io.py tests/test_engibench.py pyproject.toml uv.lock
git commit -m "feat: add EngiBench dataset loading"
```

---

### Task 12: End-to-end demo + final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_engibench.py`
Expected: All pass

**Step 2: Run lint**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: Clean

**Step 3: Test CLI end-to-end**

```bash
# Generate a test density field
uv run python -c "
import numpy as np
z, y, x = np.mgrid[-1:1:40j, -1:1:40j, -1:1:40j]
density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
np.save('test_sphere.npy', density)
print('Created test_sphere.npy')
"

# Process it
uv run xtc process test_sphere.npy -o test_sphere.stl --viz

# Verify outputs exist
ls -la test_sphere.stl test_sphere.png
```

**Step 4: Cleanup test files and commit**

```bash
rm test_sphere.npy test_sphere.stl test_sphere.png
git add -A
git commit -m "chore: final verification and cleanup"
```
