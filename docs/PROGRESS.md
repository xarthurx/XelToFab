# PROGRESS.md

Session log of learnings, failures, solutions discovered, and context gathered during work on the XelToFab project.

---

## Accumulated Project Wisdom

### 2026-02-24 — MVP Pipeline Implementation

**Problem:** scikit-image `binary_opening`, `binary_closing`, and `remove_small_objects(min_size=)` are deprecated (v0.26+, removed in v0.28).

**Root cause:** Plan was written against older scikit-image API.

**Resolution:** Used `opening`/`closing` from `skimage.morphology` and `max_size=` parameter for `remove_small_objects`. Fix: `3ee48f5`

**Prevention:** Always check for deprecation warnings during initial test runs (`-W error::FutureWarning`).

---

### 2026-02-24 — EngiBench Dataset Integration

**Problem:** Plan assumed dataset at `EngiBench/Beams2D` on HuggingFace, which doesn't exist.

**Root cause:** The actual EngiBench datasets are published under the `IDEALLab` organization (e.g., `IDEALLab/beams_2d_25_50_v0`) and use `optimal_design` as the column key (not `design`).

**Resolution:** Explored HuggingFace API to find actual dataset IDs and column names. Updated `load_engibench()` to accept full `dataset_id` and configurable `design_key`. Fix: `bd65188`

**Prevention:** When integrating with external data sources, always probe the actual API/schema before implementing — don't trust plan assumptions about third-party data formats.

---

### 2026-02-24 — Code Review: Taubin Lambda Default Destroys Meshes

**Problem:** `taubin_pass_band=0.1` was passed as `lamb` to `trimesh.smoothing.filter_taubin()`. This is the shrinkage factor, not a pass-band frequency. With `lamb=0.1` and trimesh's default `nu=0.5`, the Taubin constraint `0 < 1/λ - 1/ν < 0.1` evaluates to 8.0 — massively violated. A sphere's volume went from 1555 to -1159 (inverted faces).

**Root cause:** The plan conflated "pass band" (a frequency-domain concept) with the `lamb` shrinkage parameter. The test only checked that vertices changed, not that the result was geometrically valid.

**Resolution:** Renamed to `taubin_lambda`, changed default to `0.5` (trimesh's own default). Added `test_smooth_3d_preserves_volume` asserting volume ratio > 0.9. Fix: `22ac85e`

**Prevention:** When wrapping a library's smoothing/filtering function, verify parameter semantics against the library's own defaults — don't invent parameter names that imply different semantics. Always test geometric validity (volume, orientation), not just "something changed."

---

### 2026-02-24 — Code Review: Test Infrastructure Issues

**Problem:** (1) Test fixtures (`_make_2d_circle`, `_make_3d_sphere`) duplicated across 4 test files. (2) Matplotlib `Agg` backend set via module-level `matplotlib.use("Agg")` in `test_viz.py` — fragile if import order changes. (3) No tests for error paths (extract without preprocess, save_mesh on 2D, CLI 2D process). (4) `plot_comparison_3d` code path with `ax2.remove()` was untested. (5) CLI leaked matplotlib figures (no `plt.close`).

**Root cause:** Parallel session followed the plan's test code verbatim without consolidating shared helpers or adding negative tests.

**Resolution:** Created `tests/conftest.py` with shared fixtures and Agg backend. Added 7 edge-case tests. Added `plt.close(fig)` in CLI. Wrapped `save_mesh` errors with `click.ClickException`. Fix: `22ac85e`

**Prevention:** When writing tests via TDD: (1) create `conftest.py` with shared fixtures from the start, (2) always test error/guard paths, not just happy paths, (3) set matplotlib backend in conftest, not per-file, (4) close figures after saving in CLI code.

---

### 2026-02-25 — Multi-Format I/O: TYPE_CHECKING Guard on Runtime Type Alias

**Problem:** `LoaderFunc` type alias in `loaders/__init__.py` used `Callable` under `TYPE_CHECKING` guard, but the alias was evaluated at runtime (used in function signatures).

**Root cause:** Plan placed `Callable` import inside `if TYPE_CHECKING:` block — correct for annotations-only usage but not for runtime type aliases.

**Resolution:** Moved `from collections.abc import Callable` to unconditional import. Fix: `004366a`

**Prevention:** Type aliases used at runtime (not just in annotations) must import their components unconditionally, not under `TYPE_CHECKING`.

---

### 2026-02-25 — Multi-Format I/O: VTK Cell Data Ordering

**Problem:** VTK loader test compared `result.ravel()` against original flat density array, but VTK stores cell data in Fortran (column-major) order while numpy's `ravel()` default is C (row-major).

**Root cause:** Test assumed flat data ordering would be preserved through reshape+transpose. The loader correctly restructures data to C-order spatial layout, but the flat representations differ.

**Resolution:** Changed test assertion to use `result.ravel(order="F")` for comparison against the original VTK-ordered density. Fix: `e5f9858`

**Prevention:** When testing VTK data roundtrips, always account for VTK's Fortran-like cell ordering. Use `order="F"` for ravel comparisons or compare spatial indexing directly.

---

### 2026-02-25 — Code Review: 6 Loader Bugs (Codex Review)

**Problem:** External code review (Codex) identified 6 bugs across 4 loader files:
1. **(High)** XDMF loader iterated all `DataItem` elements globally; geometry DataItems (under `Geometry`) appear before density Attributes, silently returning coordinates.
2. **(High)** VTK loader always reshaped using cell dimensions, but `_find_density_field` could return point_data arrays (n_points ≠ n_cells), causing a crash.
3. **(Medium)** CLI `process`/`viz` only caught `ValueError`/`ImportError`, letting `KeyError` from `--field-name` miss bubble out as a raw traceback.
4. **(Medium)** `numpy_loader.py` checked `path.suffix == ".npz"` case-sensitively. Since `resolve_loader()` lowercases extensions, `.NPZ` dispatched correctly but the npz branch was skipped, returning an `NpzFile` object instead of `ndarray`.
5. **(Medium)** XDMF parser used literal tag names (`"DataItem"`, `"Attribute"`); XML namespaces prefix these, causing no matches.
6. **(Medium)** XDMF `partition(":")` split Windows drive-letter paths (`C:/dir/file.h5:/density`) at the wrong colon.

**Root cause:** Each bug was a missed edge case in the original implementation plan — no tests covered these paths.

**Resolution:**
- Issue 1: Rewrote `_load_xdmf` to iterate only `Attribute` elements, then find child DataItems. Added `_strip_ns()` helper (also fixes issue 5).
- Issue 2: `_find_density_field` now returns `is_cell_data` flag; `_grid_dimensions` takes `cell` kwarg to return node or cell dims appropriately.
- Issue 3: Added `KeyError` to CLI except clauses.
- Issue 4: Changed to `path.suffix.lower() == ".npz"`.
- Issue 5: `_strip_ns()` strips `{namespace}` prefix from XML tags before comparison.
- Issue 6: Changed `partition(":")` to `rpartition(":")` via `_parse_hdf_ref()` helper.
- Added 4 regression tests covering all 6 issues.

**Prevention:** For file format loaders: (1) always test with realistic multi-element files (geometry + density), not just minimal single-field files; (2) normalize case at the point of comparison, not just at dispatch; (3) catch all exception types loaders can raise in CLI wrappers; (4) use `rpartition` for path-like splits that may contain the delimiter in the prefix; (5) never match XML tag names literally — always handle namespaces.

---

### 2026-03-09 — Trimesh Vertex Merging Breaks Face Indices in smooth()

**Problem:** `smooth()` created a `trimesh.Trimesh` with default `process=True`, which calls `merge_vertices()` and can reduce vertex count. The smoothed vertices were saved but faces were not updated, leaving stale indices. When `save_mesh()` then created a new Trimesh from smoothed_vertices + original faces, face indices were out of bounds → `IndexError`.

**Root cause:** `trimesh.Trimesh()` with default `process=True` merges duplicate/close vertices and reindexes faces. The `smooth()` function only saved `mesh.vertices` but not `mesh.faces`, so `state.faces` still had pre-merge indices.

**Resolution:** Pass `process=False` to `trimesh.Trimesh()` in `smooth()`. Taubin smoothing only needs topology, not a processed mesh. Fix: `c2e43c6`

**Prevention:** When constructing trimesh meshes for intermediate operations (smoothing, analysis), always use `process=False` unless you explicitly need vertex merging. If `process=True` is used, save both updated vertices AND updated faces.

---

### 2026-03-09 — CLI Defeats PipelineParams Smart Defaults via Explicit Params

**Problem:** CLI commands always passed `smooth_sigma=sigma` and `direct_extraction=direct` to `PipelineParams()`, even when using Click defaults. Pydantic's `model_fields_set` saw these as "explicitly set" and the `apply_field_type_defaults` validator skipped them. Result: `--field-type sdf` ran with `smooth_sigma=1.0` and `direct_extraction=False` instead of the intended SDF defaults.

**Root cause:** Click always provides default values for all options. Passing them directly to PipelineParams made defaults indistinguishable from user-specified values.

**Resolution:** Used `ctx.get_parameter_source()` to detect whether `--sigma` and `--direct` were explicitly provided on the command line. Only pass explicitly-set values to PipelineParams, allowing the model_validator's smart defaults to apply. Extracted `_build_params()` helper shared by both `process_cmd` and `viz`. Fix: `c2e43c6`

**Prevention:** When CLI options map to model fields that have smart defaults based on `model_fields_set`, use Click's `ctx.get_parameter_source(param_name) == ParameterSource.COMMANDLINE` to distinguish user input from Click defaults. Never blindly forward all Click defaults to Pydantic models that use `model_fields_set` for conditional logic.

---

### 2026-03-09 — pymeshlab Remeshing Degrades Mesh Quality

**Problem:** Plan specified `meshing_isotropic_explicit_remeshing` and `meshing_close_holes` for Tier 2 quality improvement. Neither filter is available in pymeshlab 2025.7. The alternative `generate_resampled_uniform_mesh` (Poisson-based surface reconstruction) produced degenerate triangles: synthetic sphere min angle dropped from 30.4° to 0.3°, aspect ratio mean jumped from 1.20 to 4.50.

**Root cause:** pymeshlab 2025.7 ships a reduced filter set. GPU-dependent plugins fail to load in WSL2 (missing `libOpenGL.so.0`), and some filters (including the critical remeshing and hole-closing ones) are simply absent from the build.

**Resolution:** Disabled remeshing by default (`remesh=False` in `PipelineParams`). Repair is enabled (harmless no-op on clean meshes). The remesh module remains available for opt-in experimentation.

**Prevention:** When depending on specific library filters, verify availability at implementation time (not just planning time) by introspecting the actual installed API. Plan should have included a `pytest.importorskip` + filter availability check as a prerequisite step.

---

### 2026-03-09 — Benchmark Baseline and best_vertices Refactoring

**What:** Created `scripts/benchmark_baseline.py` to capture mesh quality metrics (aspect ratio, min angle, scaled Jacobian) and visualizations across 8 models (EngiBench 3D, Corner-Based TO, EngiBench 2D, synthetic). Added Corner-Based TO dataset (2 files from Bielecki et al.). Baseline results in `benchmarks/baseline/`.

**Key baseline findings:** Real TO models have poor minimum angles (0.5°–3.2°) and none are watertight — clear targets for Tier 2 quality improvements.

**Refactoring:** Added `best_vertices` property to `PipelineState` to eliminate the duplicated `smoothed_vertices if ... else vertices` pattern across 5 locations (io.py, viz.py, benchmark). Extracted `_to_pyvista()` helper to avoid double PolyData construction. Used modern `cell_quality()` API instead of deprecated `compute_cell_quality()`.

**Fix:** `3be9dc4`

---

### 2026-03-10 — pymeshlab Plugins Fail on WSL2 Due to Missing libOpenGL

**Problem:** pymeshlab 2025.7's `meshing_isotropic_explicit_remeshing` and `meshing_close_holes`
filters were unavailable. The fallback `generate_resampled_uniform_mesh` produced degenerate
triangles (min angle dropped from 30° to 0.3° on sphere).

**Root cause:** These filters live in `libfilter_meshing.so`, which requires `libOpenGL.so.0`.
On WSL2 without `libopengl0` installed, the plugin silently fails to load. pymeshlab does not
error — it just omits the filters from its registry.

**Resolution:** Replaced pymeshlab remeshing with `gpytoolbox.remesh_botsch` (Botsch & Kobbelt
algorithm). gpytoolbox is a pure C++/Python library with no OpenGL dependency. Kept pymeshlab
for repair operations (non-manifold fix, duplicate removal) which use core plugins unaffected
by the OpenGL issue.

**Prevention:** When depending on pymeshlab filters, verify they exist at runtime with
`hasattr(ms, 'filter_name')` or try/except. Document which filters require which plugins.
For headless/WSL environments, prefer libraries without GUI framework dependencies.

---

### 2026-03-10 — Boundary Triangles Limit Worst-Case Mesh Quality Metrics

**Problem:** After remeshing, FEA worst-case metrics (min angle, max aspect ratio) showed no
improvement on TO models. Mean/median metrics improved dramatically.

**Root cause:** All degenerate triangles are at domain boundary edges — where marching cubes
clips the isosurface at grid boundaries. The Botsch & Kobbelt algorithm correctly preserves
boundary vertices and edges, so these triangles remain unchanged.

**Resolution:** Accepted as inherent limitation of voxel-based extraction. After remeshing,
99.6% of faces meet FEA quality targets (min angle >20°). Boundary faces are typically
constrained in FEA anyway. Documented in quality log.

**Prevention:** Report median/percentile quality metrics alongside worst-case. When interpreting
mesh quality, distinguish boundary vs interior faces.

---

### 2026-03-10 — Bilateral Mesh Filter: Fleishman Normal-Displacement Fails on Coarse Meshes

**Problem:** Initial bilateral filter implementation (Fleishman et al. 2003) displaced vertices only along normals based on neighbor heights above the tangent plane. This failed: (1) on convex surfaces all heights are negative → systematic inward drift → 20% volume loss; (2) tangential noise is not corrected at all; (3) on coarse meshes with few neighbors per vertex, the filter offered no advantage over Taubin.

**Root cause:** Fleishman's method assumes dense tessellation and primarily addresses normal-direction noise. Marching cubes meshes have isotropic staircase noise (both normal and tangential), and our test meshes have relatively few vertices per local neighborhood.

**Resolution:** Switched to normal-similarity bilateral filtering — standard Laplacian-style displacement weighted by spatial distance × normal similarity. Added per-iteration volume correction (uniform scaling to match original volume) to counter inherent shrinkage. This correctly reduces displacement at feature edges (where neighbor normals diverge) while smoothing flat regions normally.

**Prevention:** When implementing a mesh processing algorithm from a specific paper, verify that its assumptions (mesh density, noise model, neighborhood size) match the actual input data. Test on representative meshes, not just synthetic primitives.

---

### 2026-03-10 — Website Documentation Illustrations (Tier 1)

**Problem:** 16 of 17 website doc pages were text-only, making conceptual content (pipeline architecture, field types, parameters, quality metrics) hard to understand without visuals.

**Resolution:** Created `scripts/generate_doc_images.py` — a reproducible script generating 7 PNG images via matplotlib/pyvista for the 4 Tier 1 guide pages:
- Pipeline flow diagram (monochromatic blue palette)
- Pipeline stages progression (corner 3D model)
- Field types comparison (density vs SDF, 1×2)
- 3 parameter sensitivity strips (threshold, sigma, taubin — separate images per section)
- Quality metrics composite (scaled Jacobian heatmap + histogram)

Images embedded in MDX pages via `<img>` tags. Supports `--only NAME` for selective regeneration.

**Prevention:** For matplotlib figure layout, always use `fig.text()` with explicit coordinates for multi-panel titles instead of `ax.set_title()` — the latter positions relative to each subplot's bounding box, which varies between image types (imshow vs screenshot), causing misalignment.
