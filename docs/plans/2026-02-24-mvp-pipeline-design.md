# MVP Pipeline Design — 2026-02-24

## Goal

Build a density-to-mesh pipeline supporting both 2D and 3D topology optimization fields. MVP scope: preprocessing + mesh extraction + Taubin smoothing + file export + matplotlib visualization. CLI (`xtc`) + Python library API.

## Data Flow

Pydantic `PipelineState` dataclass threaded through pure stage functions. Each function takes state in, returns new state out.

```
load_density("input.npy")
  → PipelineState(density=..., ndim=2|3)
    → preprocess(state)       # threshold + gaussian + morphology + components
      → extract(state)        # marching_cubes (3D) or find_contours (2D)
        → smooth(state)       # taubin (3D mesh only)
          → save_mesh(state)  # STL/OBJ/PLY (3D), PNG/SVG (2D contours)
```

## Data Model

```python
class PipelineParams(BaseModel):
    threshold: float = 0.5           # density threshold
    smooth_sigma: float = 1.0        # gaussian filter sigma
    morph_radius: int = 1            # morphological cleanup radius
    taubin_iterations: int = 20      # Taubin smoothing passes
    taubin_pass_band: float = 0.1    # Taubin pass-band parameter

class PipelineState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    density: np.ndarray              # raw field, (H,W) or (D,H,W)
    ndim: int                        # 2 or 3
    params: PipelineParams

    binary: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    contours: list[np.ndarray] | None = None
    smoothed_vertices: np.ndarray | None = None
    volume_fraction: float | None = None
```

## Module Layout

```
src/xeltocad/
├── __init__.py
├── cli.py           # click CLI, "xtc" entrypoint
├── state.py         # PipelineParams, PipelineState (Pydantic)
├── pipeline.py      # process(state) → state orchestrator
├── preprocess.py    # threshold, gaussian smooth, morphology, connected components
├── extract.py       # marching_cubes (3D), find_contours (2D)
├── smooth.py        # taubin_smooth (3D meshes)
├── io.py            # load_density, load_engibench, save_mesh
└── viz.py           # plot_density, plot_mesh, plot_comparison
```

## CLI

```
uv run xtc process input.npy -o output.stl
uv run xtc process input.npy -o output.obj --threshold 0.4 --sigma 1.5
uv run xtc viz input.npy
uv run xtc process input.npy -o output.stl --viz
```

Entrypoint: `[project.scripts] xtc = "xeltocad.cli:main"`

## I/O

- **Input**: `.npy`, `.npz` (numpy density arrays)
- **Output 3D**: `.stl`, `.obj`, `.ply` via trimesh
- **Output 2D**: `.png`, `.svg` contour plots
- **EngiBench**: `load_engibench(problem_name, index)` pulls from HuggingFace, caches under `.cache/engibench/`

## Visualization (MVP)

Matplotlib-based:
- 2D: `imshow` heatmap + contour overlay
- 3D: Slice views (XY/XZ/YZ mid-planes), `plot_trisurf` wireframe
- Side-by-side: density vs extracted mesh
- All functions take `PipelineState`, return `matplotlib.Figure`
- CLI `--viz` flag saves to PNG

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Gaussian filter, morphology, connected components |
| scikit-image | Marching cubes/squares, contour extraction |
| trimesh | Mesh I/O, repair, Taubin smoothing |
| pydantic | State/params validation and serialization |
| matplotlib | Visualization |
| click | CLI framework |

## Key Decisions

- **Pydantic over dataclass**: Input validation, JSON serialization, consistent with aiShapGrammar
- **Pure functions over class methods**: Each stage is independently testable, explicit I/O, AI-friendly
- **Both 2D and 3D from start**: Unified pipeline dispatches on `ndim`, avoids refactoring later
- **EngiBench as primary test data**: Real TO fields, no synthetic fixtures needed
- **Taubin smoothing only for MVP**: Volume-preserving, one-line via trimesh. QEM decimation, remeshing, repair deferred
