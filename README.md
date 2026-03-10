<h1 align="center">XelToFab</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13%2B-blue.svg" alt="Python 3.13+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

A design field post-processing pipeline that transforms continuous scalar fields from optimization solvers and neural networks into clean, fabrication-ready triangle meshes (3D) and contour representations (2D). The pipeline handles thresholding, smoothing, mesh extraction, quality improvement, visualization, and multi-format export.

## Quick Start

```bash
# Install
uv sync

# Process a 3D density field into an STL mesh
uv run xtf process density.npy -o output.stl

# Process with custom parameters and save a comparison plot
uv run xtf process density.npy -o output.stl --threshold 0.4 --sigma 1.5 --viz

# Visualize a 2D density field
uv run xtf viz density_2d.npy -o comparison.png
```

## Pipeline

```
Scalar Field → Preprocess → Extract → Smooth → Mesh/Contours
     (.npy)      threshold     marching    Taubin     (.stl/.obj)
                 + morphology  cubes/sq.   filter
```

1. **Preprocess** — Gaussian smoothing, Heaviside thresholding, morphological cleanup, small component removal
2. **Extract** — Marching cubes (3D) or marching squares (2D) via scikit-image
3. **Smooth** — Taubin smoothing via trimesh (3D meshes only)

## Python API

```python
import numpy as np
from xeltofab.state import PipelineState, PipelineParams
from xeltofab.pipeline import process
from xeltofab.io import save_mesh

# Create a scalar field (e.g., a sphere)
z, y, x = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]
field = (x**2 + y**2 + z**2 < 0.5**2).astype(float)

# Run the pipeline
params = PipelineParams(threshold=0.5, smooth_sigma=1.0)
state = PipelineState(field=field, params=params)
result = process(state)

# Export
save_mesh(result, "sphere.stl")
```

### Using Example Data

Pre-computed topology optimization results are included in `data/examples/` (sourced from [IDEALLab EngiBench](https://huggingface.co/IDEALLab)):

```python
from xeltofab.io import load_field
from xeltofab.pipeline import process

state = load_field("data/examples/beams_2d_50x100_sample0.npy")
result = process(state)
```

## Supported Input Formats

| Format | Extensions | Install |
|--------|-----------|---------|
| NumPy | .npy, .npz | Built-in |
| MATLAB | .mat | Built-in (via scipy) |
| CSV/Text | .csv, .txt | Built-in |
| VTK | .vtk, .vtr, .vti | `uv sync --extra vtk` |
| HDF5/XDMF | .h5, .hdf5, .xdmf | `uv sync --extra hdf5` |
| All formats | — | `uv sync --extra all-formats` |

List available formats: `uv run xtf formats`

## Development

```bash
uv sync                          # Install deps
uv run pytest tests/ -v          # Run tests
uv run ruff check src/ tests/    # Lint
uv run ruff format src/ tests/   # Format
uv run marimo edit notebooks/demo.py  # Interactive demo
```

## Project Structure

```
src/xeltofab/
├── state.py        Pipeline state + parameter models
├── preprocess.py   Field preprocessing
├── extract.py      Mesh/contour extraction
├── smooth.py       Taubin smoothing
├── pipeline.py     Orchestrator
├── io.py           File I/O (multi-format load, mesh export)
├── loaders/        Format-specific loaders (numpy, matlab, csv, vtk, hdf5)
├── field_plots.py  Matplotlib visualization
└── cli.py          CLI (xtf)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for project management
