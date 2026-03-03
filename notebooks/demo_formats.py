# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xeltofab",
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XelToFab — Supported Data Formats

    Interactive demo of all file formats supported by the `load_density` / `save_mesh`
    I/O layer. Each section creates a temporary file in the chosen format, loads it back
    through the loader registry, and visualises the round-tripped density field.

    **Input formats:** `.npy` · `.npz` · `.mat` · `.csv` · `.txt` · `.vtk` · `.vtr` · `.vti` · `.h5` · `.hdf5` · `.xdmf`
    **Output formats (3-D mesh):** `.stl` · `.obj` · `.ply`
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, np, plt, tempfile


@app.cell
def _():
    from xeltofab.io import load_density, save_mesh
    from xeltofab.loaders import get_supported_formats
    from xeltofab.state import PipelineParams
    from xeltofab.viz import plot_density

    return PipelineParams, get_supported_formats, load_density, plot_density, save_mesh


# ── Format availability ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Format Availability

    Which loaders are available in the current environment?
    Install optional extras with `uv sync --extra vtk` or `uv sync --extra hdf5`
    (or `uv sync --extra all-formats` for everything).
    """)
    return


@app.cell
def _(get_supported_formats, mo):
    _rows = []
    for fmt in get_supported_formats():
        _exts = ", ".join(f"`{e}`" for e in fmt["extensions"])
        _status = "available" if fmt["available"] else "**not installed**"
        _hint = fmt["install_hint"] if not fmt["available"] else "—"
        _rows.append(f"| {fmt['name']} | {_exts} | {_status} | `{_hint}` |")

    _table = (
        "| Format | Extensions | Status | Install |\n"
        "|--------|-----------|--------|---------|\n"
        + "\n".join(_rows)
    )
    mo.md(_table)
    return


# ── Synthetic test field ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synthetic Density Field

    A simple 2-D half-annulus used for all round-trip tests below.
    Values are continuous in [0, 1] — exactly what the pipeline expects.
    """)
    return


@app.cell
def _(np):
    def make_test_field(res: int = 100) -> np.ndarray:
        """Half-annulus with smooth density gradient."""
        y, x = np.mgrid[-1:1:complex(res), -1:1:complex(res)]
        r = np.sqrt(x**2 + y**2)
        # Annulus: inner=0.25, outer=0.7, upper half only
        mask = (r > 0.25) & (r < 0.7) & (y > 0)
        field = np.where(mask, 1.0 - (r - 0.25) / 0.45, 0.0)
        return np.clip(field, 0.0, 1.0)

    test_field = make_test_field()
    return (test_field,)


@app.cell
def _(mo, plot_density, plt, test_field):
    from xeltofab.state import PipelineParams as _PP, PipelineState as _PS

    _state = _PS(density=test_field, params=_PP())
    _fig = plot_density(_state)
    _out = mo.as_html(_fig)
    plt.close(_fig)
    _out


# ── Helper: round-trip visualiser ─────────────────────────────────────


@app.cell
def _(mo, plot_density, plt):
    def show_round_trip(state, label: str):
        """Plot a round-tripped density field and print stats."""
        d = state.density
        _fig = plot_density(state)
        _html = mo.as_html(_fig)
        plt.close(_fig)
        info = mo.md(
            f"**{label}**  \n"
            f"Shape: `{d.shape}` · dtype: `{d.dtype}` · "
            f"min: `{d.min():.4f}` · max: `{d.max():.4f}` · "
            f"mean: `{d.mean():.4f}`"
        )
        return mo.vstack([info, _html])

    return (show_round_trip,)


# ── 1. NumPy .npy ────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 — NumPy `.npy`

    The simplest format: a single numpy array stored in binary.
    No extra dependencies required.

    ```python
    np.save("density.npy", array)
    state = load_density("density.npy")
    ```
    """)
    return


@app.cell
def _(Path, load_density, np, show_round_trip, tempfile, test_field):
    with tempfile.TemporaryDirectory() as _td:
        _p = Path(_td) / "density.npy"
        np.save(_p, test_field)
        _state = load_density(_p)
    npy_out = show_round_trip(_state, ".npy round-trip")
    npy_out


# ── 2. NumPy .npz ────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 — NumPy `.npz`

    Compressed archive of named arrays. The loader auto-detects a key
    named `"density"`, or falls back to the first key in the archive.
    Use `field_name=` to pick a specific key.

    ```python
    np.savez_compressed("data.npz", density=array, metadata=info)
    state = load_density("data.npz")                     # auto-detects "density"
    state = load_density("data.npz", field_name="density")  # explicit
    ```
    """)
    return


@app.cell
def _(Path, load_density, np, show_round_trip, tempfile, test_field):
    with tempfile.TemporaryDirectory() as _td:
        _p = Path(_td) / "data.npz"
        np.savez_compressed(_p, density=test_field, extra=np.zeros(5))
        _state = load_density(_p)  # auto-detects "density" key
    npz_out = show_round_trip(_state, ".npz round-trip (auto-detected 'density' key)")
    npz_out


# ── 3. MATLAB .mat ───────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 — MATLAB `.mat`

    Loads `.mat` files saved with MATLAB ≤ v7 (`scipy.io.loadmat`).
    Auto-detects common topology-optimisation variable names:
    `xPhys`, `densities`, `x`, `rho`, `dc`, `density`.

    ```python
    scipy.io.savemat("result.mat", {"xPhys": array})
    state = load_density("result.mat")           # auto-detects "xPhys"
    state = load_density("result.mat", field_name="xPhys")
    ```

    > **Note:** MATLAB v7.3+ files are HDF5-based — load them as `.h5` instead.
    """)
    return


@app.cell
def _(Path, load_density, show_round_trip, tempfile, test_field):
    import scipy.io as _sio

    with tempfile.TemporaryDirectory() as _td:
        _p = Path(_td) / "result.mat"
        _sio.savemat(_p, {"xPhys": test_field})
        _state = load_density(_p)  # auto-detects "xPhys"
    mat_out = show_round_trip(_state, ".mat round-trip (auto-detected 'xPhys')")
    mat_out


# ── 4. CSV / TXT ─────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 — CSV / TXT

    Plain-text density grids — comma or whitespace delimited.
    2-D tables are loaded directly. For flattened 1-D data, pass
    `shape=` to reshape.

    ```python
    np.savetxt("field.csv", array, delimiter=",")
    state = load_density("field.csv")

    # Flat CSV with explicit reshape
    np.savetxt("flat.txt", array.ravel())
    state = load_density("flat.txt", shape=(80, 160))
    ```
    """)
    return


@app.cell
def _(Path, load_density, np, show_round_trip, tempfile, test_field):
    with tempfile.TemporaryDirectory() as _td:
        # Comma-delimited 2-D table
        _p = Path(_td) / "field.csv"
        np.savetxt(_p, test_field, delimiter=",")
        _state_csv = load_density(_p)

    csv_out = show_round_trip(_state_csv, ".csv round-trip (comma-delimited 2-D table)")
    csv_out


@app.cell
def _(Path, load_density, np, show_round_trip, tempfile, test_field):
    with tempfile.TemporaryDirectory() as _td:
        # Whitespace-delimited flat data with explicit reshape
        _p = Path(_td) / "flat.txt"
        np.savetxt(_p, test_field.ravel())
        _state_txt = load_density(_p, shape=test_field.shape)

    txt_out = show_round_trip(
        _state_txt,
        f".txt round-trip (flat → reshaped to {test_field.shape})",
    )
    txt_out


# ── 5. VTK (.vtk / .vtr / .vti) ──────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 — VTK `.vtk` / `.vtr` / `.vti`

    Structured VTK grids — common output from topology optimisation solvers.
    Requires **pyvista** (`uv sync --extra vtk`).

    ```python
    import pyvista as pv
    grid = pv.ImageData(dimensions=(nx+1, ny+1, 1))
    grid.cell_data["density"] = array.ravel(order="F")
    grid.save("field.vti")

    state = load_density("field.vti")
    ```
    """)
    return


@app.cell
def _(Path, PipelineParams, load_density, mo, np, show_round_trip, tempfile, test_field):
    try:
        import pyvista as _pv
        from xeltofab.state import PipelineState as _PS

        _ny, _nx = test_field.shape
        with tempfile.TemporaryDirectory() as _td:
            # ImageData (.vti) — 2-D structured grid
            # VTK uses x-fastest ordering; ravel with C order so x (cols) varies fastest
            _grid = _pv.ImageData(dimensions=(_nx + 1, _ny + 1, 1))
            _grid.cell_data["density"] = test_field.ravel(order="C")
            _p = Path(_td) / "field.vti"
            _grid.save(str(_p))
            _state_vti = load_density(_p)

        # VTK loader outputs (x, y) axis order — transpose back to numpy (rows, cols)
        _density_fixed = _state_vti.density.T
        _state_fixed = _PS(density=_density_fixed, params=PipelineParams())

        vtk_out = show_round_trip(
            _state_fixed,
            ".vti round-trip (transposed from VTK (x,y) back to numpy (rows,cols))",
        )
    except ImportError:
        vtk_out = mo.md(
            "> **VTK loader not available.** Install with: `uv sync --extra vtk`"
        )
    vtk_out


# ── 6. HDF5 (.h5 / .hdf5) ────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 — HDF5 `.h5` / `.hdf5`

    Hierarchical Data Format — widely used for large-scale simulation
    output and MATLAB v7.3+ files. Requires **h5py** (`uv sync --extra hdf5`).

    The loader auto-detects dataset names matching common topology-optimisation
    conventions (`xPhys`, `densities`, `density`, `rho`, etc.).

    ```python
    import h5py
    with h5py.File("result.h5", "w") as f:
        f.create_dataset("density", data=array)

    state = load_density("result.h5")
    state = load_density("result.h5", field_name="density")
    ```
    """)
    return


@app.cell
def _(Path, load_density, mo, show_round_trip, tempfile, test_field):
    try:
        import h5py as _h5py

        with tempfile.TemporaryDirectory() as _td:
            _p = Path(_td) / "result.h5"
            with _h5py.File(_p, "w") as _f:
                _f.create_dataset("density", data=test_field)
            _state_h5 = load_density(_p)

        h5_out = show_round_trip(_state_h5, ".h5 round-trip (h5py)")
    except ImportError:
        h5_out = mo.md(
            "> **HDF5 loader not available.** Install with: `uv sync --extra hdf5`"
        )
    h5_out


# ── 7. XDMF (.xdmf) ─────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 — XDMF `.xdmf`

    XML metadata paired with an HDF5 data file — used by FEniCS, ParaView,
    and other simulation frameworks. The XDMF file describes the grid layout
    and points to datasets inside a companion `.h5` file.

    Requires **h5py** (`uv sync --extra hdf5`).

    ```python
    state = load_density("simulation.xdmf")
    state = load_density("simulation.xdmf", field_name="density")
    ```
    """)
    return


@app.cell
def _(Path, load_density, mo, show_round_trip, tempfile, test_field):
    try:
        import h5py as _h5py

        _ny, _nx = test_field.shape
        with tempfile.TemporaryDirectory() as _td:
            _td = Path(_td)
            # Write companion HDF5 data
            _h5_path = _td / "data.h5"
            with _h5py.File(_h5_path, "w") as _f:
                _f.create_dataset("density", data=test_field)

            # Write XDMF descriptor
            _xdmf_path = _td / "simulation.xdmf"
            _xdmf_path.write_text(f"""\
<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="2DRectMesh" NumberOfElements="{_ny} {_nx}"/>
      <Geometry GeometryType="ORIGIN_DXDY">
        <DataItem Dimensions="2" NumberType="Float" Format="XML">0.0 0.0</DataItem>
        <DataItem Dimensions="2" NumberType="Float" Format="XML">1.0 1.0</DataItem>
      </Geometry>
      <Attribute Name="density" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{_ny} {_nx}" NumberType="Float" Format="HDF">data.h5:/density</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""")
            _state_xdmf = load_density(_xdmf_path)

        xdmf_out = show_round_trip(_state_xdmf, ".xdmf round-trip (XML + HDF5)")
    except ImportError:
        xdmf_out = mo.md(
            "> **XDMF loader not available.** Install with: `uv sync --extra hdf5`"
        )
    xdmf_out


# ── 8. Mesh export (STL / OBJ / PLY) ─────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 — Mesh Export: `.stl` / `.obj` / `.ply`

    After running the 3-D pipeline, `save_mesh()` exports the extracted
    surface mesh via **trimesh**. Only 3-D results can be exported as meshes.

    ```python
    from xeltofab.io import save_mesh
    from xeltofab.pipeline import process

    state_3d = load_density("sphere.npy")
    result = process(state_3d)
    save_mesh(result, "output.stl")
    save_mesh(result, "output.obj")
    save_mesh(result, "output.ply")
    ```

    Below we create a small 3-D sphere, run the pipeline, and export to all
    three mesh formats — reporting file sizes.
    """)
    return


@app.cell
def _(Path, mo, np, plt, save_mesh, tempfile):
    from xeltofab.pipeline import process as _process
    from xeltofab.state import PipelineParams as _PP, PipelineState as _PS
    from xeltofab.viz import plot_result as _plot_result

    # Synthetic 3-D sphere
    _res = 60
    _z, _y, _x = np.mgrid[-1:1:complex(_res), -1:1:complex(_res), -1:1:complex(_res)]
    _sphere = (_x**2 + _y**2 + _z**2 < 0.5**2).astype(float)

    _state3d = _PS(density=_sphere, params=_PP(threshold=0.5, smooth_sigma=0.5, taubin_iterations=10))
    _result3d = _process(_state3d)

    # 3-D mesh visualization
    _fig = _plot_result(_result3d)
    _mesh_plot = mo.as_html(_fig)
    plt.close(_fig)

    _lines = [
        f"**3-D sphere:** shape `{_sphere.shape}` · "
        f"vertices `{_result3d.vertices.shape[0]:,}` · "
        f"faces `{_result3d.faces.shape[0]:,}`",
        "",
        "| Format | File size |",
        "|--------|-----------|",
    ]

    with tempfile.TemporaryDirectory() as _td:
        for _ext in (".stl", ".obj", ".ply"):
            _p = Path(_td) / f"sphere{_ext}"
            save_mesh(_result3d, _p)
            _lines.append(f"| `{_ext}` | {_p.stat().st_size:,} bytes |")

    mo.vstack([_mesh_plot, mo.md("\n".join(_lines))])


# ── Summary ───────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Category | Extensions | Dependency | Notes |
    |----------|-----------|------------|-------|
    | NumPy | `.npy`, `.npz` | built-in | Simplest; `.npz` supports named arrays |
    | MATLAB | `.mat` | built-in (scipy) | Auto-detects `xPhys`, `rho`, etc.; v7.3+ → use `.h5` |
    | CSV/TXT | `.csv`, `.txt` | built-in | Comma or whitespace; use `shape=` for flat data |
    | VTK | `.vtk`, `.vtr`, `.vti` | `pyvista` | Structured grids; auto-corrects Fortran ordering |
    | HDF5 | `.h5`, `.hdf5` | `h5py` | Hierarchical; auto-detects known variable names |
    | XDMF | `.xdmf` | `h5py` | XML descriptor + HDF5 data; FEniCS/ParaView compatible |
    | Mesh out | `.stl`, `.obj`, `.ply` | `trimesh` | 3-D only; uses Taubin-smoothed vertices if available |

    All loaders return `np.float64` arrays shaped `(nx, ny)` or `(nx, ny, nz)`.
    The unified entry point is:

    ```python
    from xeltofab.io import load_density, save_mesh
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
