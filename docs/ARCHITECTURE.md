# Architecture

## Overview

XelToFab is a design field post-processing pipeline that converts scalar design fields (numpy arrays) into clean triangle meshes and 2D contour representations. The pipeline is implemented as a sequence of pure functions that thread an immutable `PipelineState` object through each stage.

## Pipeline Stages

```
scalar field (numpy)
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
│   Smooth      │  3D: Taubin or bilateral filtering (via trimesh/numpy)
│               │  2D: no-op (contours pass through)
└──────┬───────┘
       │  smoothed mesh / contours
       ▼
┌──────────────┐
│   Repair      │  3D: fix non-manifold edges/vertices (via pymeshlab)
│               │  2D: no-op
└──────┬───────┘
       │  repaired mesh
       ▼
┌──────────────┐
│   Remesh      │  3D: isotropic remeshing (via gpytoolbox, Botsch & Kobbelt)
│               │  2D: no-op
└──────┬───────┘
       │  uniform triangle mesh
       ▼
    Output: STL/OBJ/PLY (3D) or PNG visualization (2D)
```

## Module Map

```
scripts/
└── benchmark_baseline.py  Quality baseline capture (metrics, renders, summary)

benchmarks/
└── baseline/              Output from benchmark_baseline.py (STL, PNG, metrics.json, summary.md)

website/                       Documentation site (Fumadocs + Next.js 16)
├── app/                       Next.js app router (layouts, pages)
├── content/docs/              MDX documentation content
├── components/                Custom React components (MeshViewer)
├── lib/                       Fumadocs source config
└── public/models/             Sample STL files for interactive viewer

src/xeltofab/
├── state.py        PipelineState + PipelineParams (Pydantic models)
├── preprocess.py   Field preprocessing (smooth, threshold, morphology)
├── extract.py      Mesh/contour extraction (marching cubes/squares)
├── smooth.py       Mesh smoothing (Taubin λ-μ or bilateral normal-similarity)
├── repair.py       Watertight mesh repair (pymeshlab)
├── remesh.py       Isotropic remeshing (gpytoolbox, Botsch & Kobbelt)
├── quality.py      Mesh quality metrics (pyvista + trimesh)
├── pipeline.py     Orchestrator: process() chains preprocess → extract → smooth → repair → remesh
├── io.py           File I/O: multi-format load (via loaders/), save STL/OBJ
├── loaders/        Format-specific loaders (dispatched by extension)
│   ├── __init__.py     Loader registry, resolve_loader(), get_supported_formats()
│   ├── numpy_loader.py .npy/.npz
│   ├── matlab_loader.py .mat (auto-detects TO variable names)
│   ├── csv_loader.py   .csv/.txt (with shape parsing)
│   ├── vtk_loader.py   .vtk/.vtr/.vti (optional: pyvista)
│   └── hdf5_loader.py  .h5/.hdf5/.xdmf (optional: h5py)
├── viz.py          Matplotlib visualization (field, result, comparison plots)
├── cli.py          Click CLI (xtf process, xtf viz, xtf formats)
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
| `field` | `ndarray` | user input |
| `ndim` | `int` | auto-computed (2 or 3) |
| `params` | `PipelineParams` | user input |
| `binary` | `ndarray` | `preprocess()` |
| `volume_fraction` | `float` | `preprocess()` |
| `contours` | `list[ndarray]` | `extract()` (2D only) |
| `vertices` | `ndarray` | `extract()` (3D only) |
| `faces` | `ndarray` | `extract()` (3D only) |
| `smoothed_vertices` | `ndarray` | `smooth()` (3D only); cleared by `repair()`/`remesh()` |

The `best_vertices` property returns `smoothed_vertices` if available, otherwise `vertices`. Use this instead of the manual fallback pattern. After repair/remesh, `smoothed_vertices` is `None` because `vertices` itself contains the latest geometry.

## Field Types and Extraction Modes

The pipeline supports two field types and two extraction modes:

| Field type | Default level | Use case |
|------------|--------------|----------|
| `density` | 0.5 | Classical TO solvers, occupancy networks |
| `sdf` | 0.0 | Neural SDF models (NITO, NTopo, DeepSDF) |

| Extraction mode | Preprocessing | Use case |
|----------------|---------------|----------|
| Preprocessed (default for density) | Gaussian smooth, threshold, morphology | Noisy TO density fields |
| Direct (`direct_extraction=True`, default for SDF) | Skipped | Clean neural field outputs, converged solvers |

## CLI

The `xtf` command (installed via `[project.scripts]`) exposes three subcommands:

- **`xtf process <input> -o <output>`** — Run the full pipeline and export a mesh file
- **`xtf viz <input> [-o <output>]`** — Run the pipeline and display/save a comparison plot
- **`xtf formats`** — List supported input formats and their availability

`process` and `viz` accept `--threshold`, `--sigma`, `--field-name` (for multi-variable files), `--shape` (for flat CSV/TXT data, e.g. `50x100`), `--field-type` (`density` or `sdf`), `--direct` (skip preprocessing), `--no-repair`, and `--no-remesh`.

## Dependencies

| Purpose | Library |
|---------|---------|
| Field smoothing | `scipy.ndimage` (Gaussian filter) |
| Morphological ops | `scikit-image` (opening, closing, remove_small_objects) |
| Contour extraction | `scikit-image` (find_contours) |
| Mesh extraction | `scikit-image` (marching_cubes) |
| Mesh smoothing | `trimesh` (Taubin filter), `numpy` (bilateral filter) |
| Mesh repair | `pymeshlab` (optional — `uv sync --extra mesh-quality`) |
| Isotropic remeshing | `gpytoolbox` (optional — `uv sync --extra mesh-quality`) |
| Quality metrics | `pyvista` + `trimesh` |
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
├── test_state.py           Model validation (12 tests)
├── test_preprocess.py      Preprocessing behavior (6 tests)
├── test_extract.py         Extraction output shapes (7 tests)
├── test_smooth.py          Taubin + bilateral smoothing (9 tests)
├── test_repair.py          Watertight mesh repair (3 tests)
├── test_remesh.py          Isotropic remeshing (5 tests)
├── test_quality.py         Mesh quality metrics (4 tests)
├── test_io.py              File round-trip (6 tests)
├── test_pipeline.py        End-to-end 2D + 3D (7 tests)
├── test_viz.py             Plot generation (8 tests)
├── test_cli.py             CLI invocation (8 tests)
└── test_loaders/
    ├── test_dispatch.py        Registry + format resolution (6 tests)
    ├── test_numpy_loader.py    NumPy .npy/.npz (7 tests)
    ├── test_matlab_loader.py   MATLAB .mat (7 tests)
    ├── test_csv_loader.py      CSV/TXT (6 tests)
    ├── test_vtk_loader.py      VTK .vtk/.vtr/.vti (7 tests)
    ├── test_hdf5_loader.py     HDF5/XDMF (9 tests)
    └── test_cli_formats.py     CLI format features (6 tests)
```

Run with `uv run pytest tests/ -v`.

## Future Stages (Not Yet Implemented)

See [TODO.md](TODO.md) for the full backlog. The pipeline is designed to extend with additional stages:

- **Decimation** — QEM edge collapse for mesh simplification
- **Mesh-to-CAD** — Patch decomposition + NURBS fitting + B-Rep assembly
