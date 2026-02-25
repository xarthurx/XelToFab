# PROGRESS.md

Session log of learnings, failures, solutions discovered, and context gathered during work on the pixelToCAD project.

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
