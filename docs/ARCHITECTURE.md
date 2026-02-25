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
├── io.py           File I/O: multi-format load (via loaders/), save STL/OBJ
├── loaders/        Format-specific loaders (dispatched by extension)
│   ├── __init__.py     Loader registry, resolve_loader(), get_supported_formats()
│   ├── numpy_loader.py .npy/.npz
│   ├── matlab_loader.py .mat (auto-detects TO variable names)
│   ├── csv_loader.py   .csv/.txt (with shape parsing)
│   ├── vtk_loader.py   .vtk/.vtr/.vti (optional: pyvista)
│   └── hdf5_loader.py  .h5/.hdf5/.xdmf (optional: h5py)
├── viz.py          Matplotlib visualization (density, result, comparison plots)
├── cli.py          Click CLI (xtc process, xtc viz, xtc formats)
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

The `xtc` command (installed via `[project.scripts]`) exposes three subcommands:

- **`xtc process <input> -o <output>`** — Run the full pipeline and export a mesh file
- **`xtc viz <input> [-o <output>]`** — Run the pipeline and display/save a comparison plot
- **`xtc formats`** — List supported input formats and their availability

`process` and `viz` accept `--threshold`, `--sigma`, `--field-name` (for multi-variable files), and `--shape` (for flat CSV/TXT data, e.g. `50x100`).

## Dependencies

| Purpose | Library |
|---------|---------|
| Density smoothing | `scipy.ndimage` (Gaussian filter) |
| Morphological ops | `scikit-image` (opening, closing, remove_small_objects) |
| Contour extraction | `scikit-image` (find_contours) |
| Mesh extraction | `scikit-image` (marching_cubes) |
| Mesh smoothing | `trimesh` (Taubin filter) |
| Mesh I/O | `trimesh` (STL, OBJ, PLY export) |
| MATLAB loading | `scipy.io` (loadmat) |
| VTK loading | `pyvista` (optional — `uv sync --extra vtk`) |
| HDF5/XDMF loading | `h5py` (optional — `uv sync --extra hdf5`) |
| State models | `pydantic` |
| Visualization | `matplotlib` |
| CLI | `click` |

## Testing

Tests mirror the module structure in `tests/`:

```
tests/
├── conftest.py             Shared fixtures + Agg backend
├── test_state.py           Model validation (6 tests)
├── test_preprocess.py      Preprocessing behavior (6 tests)
├── test_extract.py         Extraction output shapes (3 tests)
├── test_smooth.py          Smoothing effects + volume preservation (4 tests)
├── test_io.py              File round-trip (6 tests)
├── test_pipeline.py        End-to-end 2D + 3D (2 tests)
├── test_viz.py             Plot generation (6 tests)
├── test_cli.py             CLI invocation (4 tests)
└── test_loaders/
    ├── test_dispatch.py        Registry + format resolution (6 tests)
    ├── test_numpy_loader.py    NumPy .npy/.npz (6 tests)
    ├── test_matlab_loader.py   MATLAB .mat (7 tests)
    ├── test_csv_loader.py      CSV/TXT (6 tests)
    ├── test_vtk_loader.py      VTK .vtk/.vtr/.vti (6 tests)
    ├── test_hdf5_loader.py     HDF5/XDMF (7 tests)
    └── test_cli_formats.py     CLI format features (5 tests)
```

Run with `uv run pytest tests/ -v`.

## Future Stages (Not Yet Implemented)

See [TODO.md](TODO.md) for the full backlog. The pipeline is designed to extend with additional stages:

- **Decimation** — QEM edge collapse after smoothing
- **Remeshing** — Isotropic remeshing for FEA-quality elements
- **Mesh-to-CAD** — Patch decomposition + NURBS fitting + B-Rep assembly
