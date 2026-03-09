# Field-Type-Aware Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Support SDF and clean density field inputs with direct extraction (bypassing binarization) for higher-quality meshes from neural field outputs and clean TO solvers.

**Architecture:** Add `field_type` and `direct_extraction` to `PipelineParams` with smart defaults. Branch in `extract()` to operate on continuous field when direct, skip `preprocess()` in `pipeline.py`. Fully backward compatible — all 85 existing tests must pass unchanged.

**Tech Stack:** pydantic (model validators), scikit-image (marching cubes/squares), pytest, click

---

### Task 1: Add `field_type`, `direct_extraction`, `extraction_level` to PipelineParams

**Files:**
- Modify: `src/xeltofab/state.py:9-16`
- Test: `tests/test_state.py`

**Step 1: Write the failing tests**

Add to `tests/test_state.py`:

```python
def test_pipeline_params_field_type_defaults():
    params = PipelineParams()
    assert params.field_type == "density"
    assert params.direct_extraction is False
    assert params.extraction_level is None


def test_pipeline_params_sdf_smart_defaults():
    """SDF field type should auto-enable direct extraction and disable smoothing."""
    params = PipelineParams(field_type="sdf")
    assert params.direct_extraction is True
    assert params.smooth_sigma == 0.0


def test_pipeline_params_sdf_override_smooth():
    """User can re-enable smoothing for noisy SDF."""
    params = PipelineParams(field_type="sdf", smooth_sigma=2.0)
    assert params.smooth_sigma == 2.0
    assert params.direct_extraction is True


def test_pipeline_params_density_direct():
    """User can enable direct extraction for clean density fields."""
    params = PipelineParams(field_type="density", direct_extraction=True)
    assert params.direct_extraction is True
    assert params.threshold == 0.5  # threshold still available for density


def test_pipeline_params_effective_extraction_level():
    """extraction_level derives from field_type when not set explicitly."""
    assert PipelineParams(field_type="density").effective_extraction_level == 0.5
    assert PipelineParams(field_type="sdf").effective_extraction_level == 0.0
    assert PipelineParams(field_type="sdf", extraction_level=0.1).effective_extraction_level == 0.1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL — `field_type`, `direct_extraction`, `extraction_level`, `effective_extraction_level` do not exist.

**Step 3: Implement PipelineParams changes**

In `src/xeltofab/state.py`, update `PipelineParams`:

```python
from typing import Literal

class PipelineParams(BaseModel):
    """Configurable parameters for the pipeline."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    smooth_sigma: float = Field(default=1.0, ge=0.0)
    morph_radius: int = Field(default=1, ge=0)
    taubin_iterations: int = Field(default=20, ge=0)
    taubin_lambda: float = Field(default=0.5, gt=0.0, le=1.0)

    field_type: Literal["density", "sdf"] = "density"
    direct_extraction: bool = False
    extraction_level: float | None = None

    @model_validator(mode="after")
    def apply_field_type_defaults(self) -> PipelineParams:
        """Apply smart defaults based on field_type.

        Only override values that were not explicitly set by the user.
        SDF defaults: direct_extraction=True, smooth_sigma=0.0.
        """
        if self.field_type == "sdf":
            # Check which fields were explicitly provided vs defaulted
            explicitly_set = self.model_fields_set
            if "direct_extraction" not in explicitly_set:
                self.direct_extraction = True
            if "smooth_sigma" not in explicitly_set:
                self.smooth_sigma = 0.0
        return self

    @property
    def effective_extraction_level(self) -> float:
        """Extraction level: explicit value, or 0.0 for SDF, threshold for density."""
        if self.extraction_level is not None:
            return self.extraction_level
        return 0.0 if self.field_type == "sdf" else self.threshold
```

Note: `PipelineParams` needs the `Literal` import added at the top of the file: `from typing import Literal`.

**Step 4: Run all state tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: All tests PASS (existing + new).

**Step 5: Run full suite to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All 85 existing tests + 5 new tests PASS.

**Step 6: Commit**

```bash
git add src/xeltofab/state.py tests/test_state.py
git commit -m "feat: add field_type, direct_extraction, extraction_level to PipelineParams"
```

---

### Task 2: Update `extract()` for direct extraction

**Files:**
- Modify: `src/xeltofab/extract.py`
- Modify: `tests/conftest.py` (add SDF fixtures)
- Test: `tests/test_extract.py`

**Step 1: Add SDF fixtures to conftest**

Add to `tests/conftest.py`:

```python
@pytest.fixture
def sphere_sdf() -> np.ndarray:
    """3D signed distance field for a sphere (negative inside, positive outside)."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return np.sqrt(x**2 + y**2 + z**2) - 0.5


@pytest.fixture
def circle_sdf() -> np.ndarray:
    """2D signed distance field for a circle."""
    y, x = np.mgrid[-1:1:100j, -1:1:100j]
    return np.sqrt(x**2 + y**2) - 0.5
```

**Step 2: Write failing tests**

Add to `tests/test_extract.py`:

```python
from xeltofab.state import PipelineParams


def test_extract_3d_direct_sdf(sphere_sdf: np.ndarray):
    """Direct extraction from continuous SDF at level=0."""
    params = PipelineParams(field_type="sdf")
    state = PipelineState(density=sphere_sdf, params=params)
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0


def test_extract_2d_direct_sdf(circle_sdf: np.ndarray):
    """Direct extraction of contours from continuous 2D SDF at level=0."""
    params = PipelineParams(field_type="sdf")
    state = PipelineState(density=circle_sdf, params=params)
    result = extract(state)
    assert result.contours is not None
    assert len(result.contours) > 0


def test_extract_3d_direct_density(sphere_density: np.ndarray):
    """Direct extraction from clean density field at level=0.5."""
    params = PipelineParams(field_type="density", direct_extraction=True)
    state = PipelineState(density=sphere_density, params=params)
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[0] > 0


def test_extract_3d_direct_custom_level(sphere_sdf: np.ndarray):
    """Direct extraction at a custom level (offset surface)."""
    params = PipelineParams(field_type="sdf", extraction_level=0.1)
    state = PipelineState(density=sphere_sdf, params=params)
    result = extract(state)
    assert result.vertices is not None
    # Offset surface should have smaller radius → fewer vertices than level=0
    params_zero = PipelineParams(field_type="sdf", extraction_level=0.0)
    state_zero = PipelineState(density=sphere_sdf, params=params_zero)
    result_zero = extract(state_zero)
    assert result.vertices.shape[0] < result_zero.vertices.shape[0]
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -v`
Expected: FAIL — `extract()` raises ValueError "binary field is None" for direct extraction states.

**Step 4: Implement extract() changes**

Replace `src/xeltofab/extract.py`:

```python
"""Mesh/contour extraction from density/SDF fields."""

from __future__ import annotations

from skimage.measure import find_contours, marching_cubes

from xeltofab.state import PipelineState


def extract(state: PipelineState) -> PipelineState:
    """Extract mesh (3D) or contours (2D).

    In direct mode, extracts from the continuous input field at the configured level.
    Otherwise, extracts from the preprocessed binary field at level 0.5.
    """
    if state.params.direct_extraction:
        field = state.density.astype(float)
        level = state.params.effective_extraction_level
    else:
        if state.binary is None:
            raise ValueError("binary field is None — run preprocess() first")
        field = state.binary.astype(float)
        level = 0.5

    if state.ndim == 2:
        return _extract_2d(state, field, level)
    return _extract_3d(state, field, level)


def _extract_2d(state: PipelineState, field, level: float) -> PipelineState:
    """Extract contours from 2D field using marching squares."""
    contours = find_contours(field, level=level)
    return state.model_copy(update={"contours": contours})


def _extract_3d(state: PipelineState, field, level: float) -> PipelineState:
    """Extract triangle mesh from 3D field using marching cubes."""
    vertices, faces, _, _ = marching_cubes(field, level=level)
    return state.model_copy(update={"vertices": vertices, "faces": faces})
```

**Step 5: Run all extract tests**

Run: `uv run pytest tests/test_extract.py -v`
Expected: All tests PASS (existing + new).

**Step 6: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 7: Commit**

```bash
git add src/xeltofab/extract.py tests/conftest.py tests/test_extract.py
git commit -m "feat: support direct extraction from continuous SDF and density fields"
```

---

### Task 3: Update `pipeline.py` to skip preprocess for direct extraction

**Files:**
- Modify: `src/xeltofab/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing tests**

Add to `tests/test_pipeline.py`:

```python
from xeltofab.state import PipelineParams


def test_process_3d_sdf_end_to_end(sphere_sdf: np.ndarray):
    """Full pipeline with SDF input skips preprocessing."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(density=sphere_sdf, params=params))
    # No preprocessing → binary stays None
    assert result.binary is None
    assert result.volume_fraction is None
    # Extraction and smoothing still run
    assert result.vertices is not None
    assert result.faces is not None
    assert result.smoothed_vertices is not None


def test_process_2d_sdf_end_to_end(circle_sdf: np.ndarray):
    """Full 2D pipeline with SDF input."""
    params = PipelineParams(field_type="sdf")
    result = process(PipelineState(density=circle_sdf, params=params))
    assert result.binary is None
    assert result.contours is not None


def test_process_3d_direct_density(sphere_density: np.ndarray):
    """Full pipeline with clean density, direct extraction."""
    params = PipelineParams(direct_extraction=True)
    result = process(PipelineState(density=sphere_density, params=params))
    assert result.binary is None
    assert result.vertices is not None
    assert result.smoothed_vertices is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `process()` always calls `preprocess()`, which fails or produces wrong results for SDF.

**Step 3: Implement pipeline.py changes**

Update `src/xeltofab/pipeline.py`:

```python
"""Pipeline orchestrator."""

from __future__ import annotations

from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.smooth import smooth
from xeltofab.state import PipelineState


def process(state: PipelineState) -> PipelineState:
    """Run the pipeline: [preprocess] -> extraction -> smoothing.

    When direct_extraction is enabled, preprocessing is skipped and extraction
    operates on the continuous input field directly.
    """
    if not state.params.direct_extraction:
        state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    return state
```

**Step 4: Run all pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All tests PASS (existing + new).

**Step 5: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add src/xeltofab/pipeline.py tests/test_pipeline.py
git commit -m "feat: skip preprocessing in pipeline when direct_extraction is enabled"
```

---

### Task 4: Add CLI flags `--field-type` and `--direct`

**Files:**
- Modify: `src/xeltofab/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
def test_cli_process_sdf(tmp_path: Path, small_sphere_density: np.ndarray):
    """CLI process with --field-type sdf."""
    # Create a small SDF field
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    sdf = np.sqrt(x**2 + y**2 + z**2) - 0.5
    input_path = tmp_path / "sphere_sdf.npy"
    np.save(input_path, sdf)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(
        main, ["process", str(input_path), "-o", str(output_path), "--field-type", "sdf"]
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_process_direct(tmp_path: Path, small_sphere_density: np.ndarray):
    """CLI process with --direct flag for clean density."""
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, small_sphere_density)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(
        main, ["process", str(input_path), "-o", str(output_path), "--direct"]
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `--field-type` and `--direct` are unknown options.

**Step 3: Implement CLI changes**

In `src/xeltofab/cli.py`, add options to both `process_cmd` and `viz`:

For `process_cmd` (add after line 33):
```python
@click.option("--field-type", type=click.Choice(["density", "sdf"]), default="density", help="Input field type")
@click.option("--direct", is_flag=True, help="Direct extraction from continuous field (skip preprocessing)")
```

Update `process_cmd` function signature to accept `field_type: str` and `direct: bool`.

Update params construction:
```python
params = PipelineParams(
    threshold=threshold,
    smooth_sigma=sigma,
    field_type=field_type,
    direct_extraction=direct,
)
```

Apply the same two options and params construction to the `viz` command.

**Step 4: Run all CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS (existing + new).

**Step 5: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add src/xeltofab/cli.py tests/test_cli.py
git commit -m "feat: add --field-type and --direct CLI options"
```

---

### Task 5: Final verification and documentation update

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/TODO.md`

**Step 1: Run full test suite with lint**

Run: `uv run pytest tests/ -v && ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: All tests PASS, no lint errors, formatting clean.

**Step 2: Update ARCHITECTURE.md**

Add a section after "Data Flow" documenting the two extraction paths:

```markdown
## Field Types and Extraction Modes

The pipeline supports two field types and two extraction modes:

| Field type | Default level | Use case |
|------------|--------------|----------|
| `density` | 0.5 | Classical TO solvers, occupancy networks |
| `sdf` | 0.0 | Neural SDF models (NITO, NTopo, DeepSDF) |

| Extraction mode | Preprocessing | Use case |
|----------------|---------------|----------|
| Preprocessed (default for density) | Gaussian smooth → threshold → morphology | Noisy TO density fields |
| Direct (`direct_extraction=True`, default for SDF) | Skipped | Clean neural field outputs, converged solvers |
```

**Step 3: Update TODO.md**

Mark the relevant Tier 4 item as done. The neural SDF support line becomes:
```markdown
- [x] Neural SDF representations (NITO / NTopo style) — field-type-aware extraction with `field_type='sdf'`
```

**Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md docs/TODO.md
git commit -m "docs: update architecture and backlog for field-type-aware extraction"
```
