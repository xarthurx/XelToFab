# XelToCAD

Topology optimization post-processing pipeline: density fields to meshes to CAD.

Converts continuous density arrays from topology optimization solvers into clean triangle meshes (3D) or contour representations (2D), with preprocessing, smoothing, visualization, and file export.

## Quick Start

```bash
# Install
uv sync

# Process a 3D density field into an STL mesh
uv run xtc process density.npy -o output.stl

# Process with custom parameters and save a comparison plot
uv run xtc process density.npy -o output.stl --threshold 0.4 --sigma 1.5 --viz

# Visualize a 2D density field
uv run xtc viz density_2d.npy -o comparison.png
```

## Pipeline

```
Density Array → Preprocess → Extract → Smooth → Mesh/Contours
     (.npy)      threshold     marching    Taubin     (.stl/.obj)
                 + morphology  cubes/sq.   filter
```

1. **Preprocess** — Gaussian smoothing, Heaviside thresholding, morphological cleanup, small component removal
2. **Extract** — Marching cubes (3D) or marching squares (2D) via scikit-image
3. **Smooth** — Taubin smoothing via trimesh (3D meshes only)

## Python API

```python
import numpy as np
from xeltocad.state import PipelineState, PipelineParams
from xeltocad.pipeline import process
from xeltocad.io import save_mesh

# Create a density field (e.g., a sphere)
z, y, x = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]
density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)

# Run the pipeline
params = PipelineParams(threshold=0.5, smooth_sigma=1.0)
state = PipelineState(density=density, params=params)
result = process(state)

# Export
save_mesh(result, "sphere.stl")
```

### Using Example Data

Pre-computed topology optimization results are included in `data/examples/` (sourced from [IDEALLab EngiBench](https://huggingface.co/IDEALLab)):

```python
from xeltocad.io import load_density
from xeltocad.pipeline import process

state = load_density("data/examples/beams_2d_50x100_sample0.npy")
result = process(state)
```

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
src/xeltocad/
├── state.py        Pipeline state + parameter models
├── preprocess.py   Density preprocessing
├── extract.py      Mesh/contour extraction
├── smooth.py       Taubin smoothing
├── pipeline.py     Orchestrator
├── io.py           File I/O (load .npy/.npz, export mesh)
├── viz.py          Matplotlib visualization
└── cli.py          CLI (xtc)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for project management
