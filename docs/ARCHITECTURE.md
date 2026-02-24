# Architecture

## Overview

XelToCAD is a topology optimization post-processing pipeline that converts density fields (numpy arrays) into clean triangle meshes and 2D contour representations. The pipeline is implemented as a sequence of pure functions that thread an immutable `PipelineState` object through each stage.

## Pipeline Stages

```
density array (numpy, [0,1])
        │
        ▼
┌──────────────┐
│  Preprocess   │  Gaussian smooth → threshold → morphological cleanup
│               │  → remove small components
└──────┬───────┘
       │  binary array (0/1)
       ▼
┌──────────────┐
│   Extract     │  2D: marching squares → contours
│               │  3D: marching cubes → triangle mesh
└──────┬───────┘
       │  vertices + faces (3D) or contour arrays (2D)
       ▼
┌──────────────┐
│   Smooth      │  3D: Taubin smoothing (via trimesh)
│               │  2D: no-op (contours pass through)
└──────┬───────┘
       │  smoothed mesh / contours
       ▼
    Output: STL/OBJ/PLY (3D) or PNG visualization (2D)
```

## Module Map

```
src/xeltocad/
├── state.py        PipelineState + PipelineParams (Pydantic models)
├── preprocess.py   Density field preprocessing (smooth, threshold, morphology)
├── extract.py      Mesh/contour extraction (marching cubes/squares)
├── smooth.py       Taubin mesh smoothing
├── pipeline.py     Orchestrator: process() chains preprocess → extract → smooth
├── io.py           File I/O: load .npy/.npz, save STL/OBJ, load EngiBench datasets
├── viz.py          Matplotlib visualization (density, result, comparison plots)
├── cli.py          Click CLI (xtc process, xtc viz)
└── __init__.py
```

## Data Flow

All pipeline functions follow the same signature:

```python
def stage(state: PipelineState) -> PipelineState:
```

`PipelineState` is a Pydantic model with `arbitrary_types_allowed` for numpy arrays. Functions return a new state via `model_copy(update={...})` rather than mutating in place. Key fields:

| Field | Type | Set by |
|-------|------|--------|
| `density` | `ndarray` | user input |
| `ndim` | `int` | auto-computed (2 or 3) |
| `params` | `PipelineParams` | user input |
| `binary` | `ndarray` | `preprocess()` |
| `volume_fraction` | `float` | `preprocess()` |
| `contours` | `list[ndarray]` | `extract()` (2D only) |
| `vertices` | `ndarray` | `extract()` (3D only) |
| `faces` | `ndarray` | `extract()` (3D only) |
| `smoothed_vertices` | `ndarray` | `smooth()` (3D only) |

## CLI

The `xtc` command (installed via `[project.scripts]`) exposes two subcommands:

- **`xtc process <input> -o <output>`** — Run the full pipeline and export a mesh file
- **`xtc viz <input> [-o <output>]`** — Run the pipeline and display/save a comparison plot

Both accept `--threshold` and `--sigma` to configure `PipelineParams`.

## Dependencies

| Purpose | Library |
|---------|---------|
| Density smoothing | `scipy.ndimage` (Gaussian filter) |
| Morphological ops | `scikit-image` (opening, closing, remove_small_objects) |
| Contour extraction | `scikit-image` (find_contours) |
| Mesh extraction | `scikit-image` (marching_cubes) |
| Mesh smoothing | `trimesh` (Taubin filter) |
| Mesh I/O | `trimesh` (STL, OBJ, PLY export) |
| State models | `pydantic` |
| Visualization | `matplotlib` |
| CLI | `click` |
| EngiBench datasets | `engibench`, `datasets` (HuggingFace) |

## Testing

Tests mirror the module structure in `tests/`:

```
tests/
├── test_state.py       Model validation (6 tests)
├── test_preprocess.py  Preprocessing behavior (4 tests)
├── test_extract.py     Extraction output shapes (3 tests)
├── test_smooth.py      Smoothing effects + 2D no-op (3 tests)
├── test_io.py          File round-trip (4 tests)
├── test_pipeline.py    End-to-end 2D + 3D (2 tests)
├── test_viz.py         Plot generation (5 tests)
├── test_cli.py         CLI invocation (3 tests)
└── test_engibench.py   EngiBench loading (1 test, @network)
```

Run with `uv run pytest tests/ -v` (exclude network tests: `--ignore=tests/test_engibench.py`).

## Future Stages (Not Yet Implemented)

See [TODO.md](TODO.md) for the full backlog. The pipeline is designed to extend with additional stages:

- **Decimation** — QEM edge collapse after smoothing
- **Remeshing** — Isotropic remeshing for FEA-quality elements
- **Mesh-to-CAD** — Patch decomposition + NURBS fitting + B-Rep assembly
