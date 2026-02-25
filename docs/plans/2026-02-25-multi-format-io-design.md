# Multi-Format I/O Support Design

**Date:** 2026-02-25
**Status:** Approved
**Scope:** Expand `load_density()` to support .mat, .vtk/.vtr/.vti, .csv/.txt, .h5/.hdf5/.xdmf

## Motivation

The pipeline currently accepts only `.npy`/`.npz` files (from Python/ML ecosystems). Real TO workflows produce data in many formats: MATLAB codes output `.mat`, simulation tools output VTK, FEniCS uses XDMF+HDF5, and engineers often export `.csv`. Supporting these formats makes the tool usable beyond the EngiBench dataset origin.

## Formats to Support

| Format | Extensions | Dependency | Priority | Install |
|--------|-----------|------------|----------|---------|
| NumPy | .npy, .npz | numpy | Already done | Built-in |
| MATLAB | .mat | scipy.io | High | Built-in (scipy is core dep) |
| VTK structured | .vtk, .vtr, .vti | pyvista | High | `xeltocad[vtk]` |
| CSV/Text | .csv, .txt | numpy | Medium | Built-in |
| HDF5 | .h5, .hdf5 | h5py | Medium | `xeltocad[hdf5]` |
| XDMF | .xdmf | h5py + xml.etree | Medium | `xeltocad[hdf5]` |

## Architecture: Loader Registry

### Module Structure

```
src/xeltocad/
├── io.py                    # save_mesh() stays; load_density() dispatches to loaders
├── loaders/
│   ├── __init__.py          # LOADER_REGISTRY dict, resolve_loader(), format listing
│   ├── numpy_loader.py      # .npy, .npz (migrated from current io.py)
│   ├── matlab_loader.py     # .mat
│   ├── vtk_loader.py        # .vtk, .vtr, .vti
│   ├── csv_loader.py        # .csv, .txt
│   └── hdf5_loader.py       # .h5, .hdf5, .xdmf
```

### Dispatch Logic

`load_density(path, field_name=None, shape=None, params=None)` reads file extension, looks up the loader in `LOADER_REGISTRY`, calls it. Each loader returns a raw `np.ndarray`. `load_density()` wraps it in `PipelineState`.

### Loader Contract

Every loader module exposes:

```python
def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray
```

- `field_name`: Which variable/field to extract (.mat, .vtk, .h5). Ignored by formats that don't need it.
- `shape`: Grid dimensions for flat data (.csv/.txt). Format: `(nx, ny)` or `(nx, ny, nz)`.
- Returns: numpy array, 2D or 3D, values in [0, 1] continuous range.

## Format-Specific Loader Details

### `.mat` (MATLAB) -- `matlab_loader.py`

- **Library:** `scipy.io.loadmat` (already a core dependency)
- **Auto-detect:** Try variable names in order: `xPhys`, `densities`, `x`, `rho`, `dc`, `density`. Skip MATLAB metadata keys (starting with `__`).
- **Fallback:** If no known name found and only one non-metadata variable exists, use it. If multiple, raise error listing available variables so user can pass `--field-name`.
- **MATLAB v7.3+:** These are HDF5 files. `scipy.io.loadmat` raises `NotImplementedError`. Detect this and give clear error: "This is a MATLAB v7.3+ file. Use .h5 extension or install h5py."
- **Array ordering:** MATLAB stores column-major (Fortran order). Transpose to row-major (C order) for numpy consistency.

### `.vtk` / `.vtr` / `.vti` -- `vtk_loader.py`

- **Library:** `pyvista` (optional extra `[vtk]`)
- **Loading:** `pyvista.read(path)` handles all three formats.
- **Field extraction:** Check `cell_data` first (TO densities are per-element), then `point_data`. Auto-detect: field names containing "density", "rho", "x", "xphys" (case-insensitive). Fall back to `field_name` parameter.
- **Array reshaping:** For `RectilinearGrid` and `ImageData`, pyvista provides grid dimensions. For legacy `.vtk`, read structured grid dimensions from header. Flat scalar array reshaped to `(nz, ny, nx)` (VTK ordering) then transposed to `(nx, ny, nz)`.

### `.csv` / `.txt` -- `csv_loader.py`

- **Library:** numpy only (`np.loadtxt` / `np.genfromtxt`)
- **2D table mode (default for .csv):** `np.loadtxt(path, delimiter=',')` -- shape inferred from rows x columns.
- **Flat list mode:** If `shape` provided via `--shape`, load all values and reshape. Accepts `NxM` (2D) or `NxMxK` (3D).
- **Auto-detect delimiter:** Try comma first, fall back to whitespace (covers .txt tab/space-separated).
- **Header skip:** If first line is non-numeric, skip automatically.

### `.h5` / `.hdf5` / `.xdmf` -- `hdf5_loader.py`

- **Library:** `h5py` (optional extra `[hdf5]`)
- **Raw HDF5 (.h5/.hdf5):** Open file, auto-detect density dataset using same name heuristics as .mat. Support nested groups (e.g., `"/Function/0"` for FEniCS). `field_name` overrides auto-detect.
- **XDMF (.xdmf):** Parse XML (xml.etree.ElementTree, stdlib) to find HDF5 file path and dataset path, then load via h5py.
- **Grid dimensions:** XDMF contains topology info for reshaping. Raw HDF5 uses dataset's native shape.

## CLI Integration

### New Options

Add to `process` and `viz` commands:

- `--field-name` / `-f`: Specify which variable/field to extract (for .mat, .vtk, .h5)
- `--shape`: Specify grid dimensions for flat data (for .csv/.txt), format: `NxM` or `NxMxK`

### New Subcommand

`xtc formats` -- lists all supported formats and their install status:

```
$ xtc formats
Format    Extensions        Status      Install
numpy     .npy, .npz        available   (built-in)
matlab    .mat              available   (built-in, via scipy)
csv       .csv, .txt        available   (built-in)
vtk       .vtk, .vtr, .vti  missing     uv add xeltocad[vtk]
hdf5      .h5, .hdf5, .xdmf missing     uv add xeltocad[hdf5]
```

### Usage Examples

```bash
xtc process input.mat -o output.stl --field-name xPhys
xtc process density.csv -o output.stl --shape 100x200
xtc process results.vtk -o output.stl --field-name density
xtc process results.h5 -o output.stl --field-name "/Function/0"
```

## Dependency Management

### pyproject.toml Extras

```toml
[project.optional-dependencies]
vtk = ["pyvista>=0.43"]
hdf5 = ["h5py>=3.10"]
all-formats = ["pyvista>=0.43", "h5py>=3.10"]
```

Note: .mat uses scipy (core dep), .csv uses numpy (core dep). Only VTK and HDF5 need extras.

### Missing Dependency Errors

When a user tries to load a format without its optional dependency:

```
Error: Loading .vtk files requires pyvista.
Install it with: uv add xeltocad[vtk]
```

## Directory Restructuring

Move example data from `examples/data/` to `data/examples/`:

```
data/
└── examples/
    ├── README.md
    ├── beams_2d_*.npy
    ├── heat_conduction_3d_*.npy
    └── thermoelastic_3d_*.npy
```

Update all references in tests, CLI, docs, and the marimo notebook.

## Testing Strategy

### Test Structure

```
tests/
├── test_io.py                    # Updated for new load_density signature
├── test_loaders/
│   ├── test_numpy_loader.py      # Migrated from test_io.py
│   ├── test_matlab_loader.py
│   ├── test_vtk_loader.py
│   ├── test_csv_loader.py
│   ├── test_hdf5_loader.py
│   └── test_dispatch.py          # Registry dispatch, unknown format errors
```

### Test Data

No large bundled test files. All test data created synthetically in fixtures:

- `.mat`: `scipy.io.savemat()` with small arrays
- `.vtk`: `pyvista.RectilinearGrid()` (skip if pyvista not installed)
- `.csv`: Write small strings to temp files
- `.h5`: `h5py.File()` (skip if h5py not installed)
- `.xdmf`: Minimal XML + HDF5 pairs in temp files

Optional dependency tests use `pytest.importorskip()`.

### Test Cases Per Loader (6 each)

1. Happy path -- load well-formed file
2. Auto-detect -- correct field found without `field_name`
3. Explicit field -- `field_name` parameter overrides auto-detect
4. Missing field -- clear error when field not found
5. Missing dependency -- clear error message with install instructions
6. Shape validation -- result is 2D or 3D numpy array with values in [0, 1]

## Public API Changes

### `load_density()` Signature Change

```python
# Before
def load_density(path: str | Path, params: PipelineParams | None = None) -> PipelineState

# After
def load_density(
    path: str | Path,
    field_name: str | None = None,
    shape: tuple[int, ...] | None = None,
    params: PipelineParams | None = None,
) -> PipelineState
```

Backwards compatible -- existing calls with just `path` and `params` continue to work.
