# Field-Type-Aware Extraction Design

**Date:** 2026-03-09
**Status:** Approved

## Motivation

XelToFab's pipeline currently assumes all inputs are noisy density fields in [0, 1] from classical topology optimization solvers. Every input goes through Gaussian smoothing → Heaviside thresholding → morphological cleanup → binarization before marching cubes extraction.

ML-based models (neural SDFs like NITO/NTopo, occupancy networks, well-converged TO solvers) produce clean, continuous fields where this preprocessing is unnecessary or harmful. Specifically:

- **Neural SDF outputs** are smooth by construction — Gaussian smoothing adds nothing. Binarizing an SDF before marching cubes destroys the continuous distance information that produces smooth meshes. The correct extraction level is 0.0, not 0.5.
- **Clean density fields** from converged solvers or neural density models also benefit from direct extraction at level 0.5, skipping the binarization step.

The design philosophy is that XelToFab is a library that ML projects call (`import xeltofab`), not the other way around. ML frameworks handle model evaluation and grid sampling; XelToFab handles array → mesh → fabrication-ready geometry.

## Design

### Two Orthogonal Axes

| Axis | Options | Controls |
|------|---------|----------|
| **Field type** | `density`, `sdf` | Extraction level (0.5 vs 0.0) and value interpretation |
| **Field quality** | noisy (preprocessed) vs clean (direct) | Whether to binarize before extraction |

These are independent: a noisy SDF needs preprocessing; a clean density field doesn't.

### Parameter Changes (`PipelineParams`)

New fields:

```python
field_type: Literal['density', 'sdf'] = 'density'
direct_extraction: bool = False
extraction_level: float | None = None  # None = derived from field_type
```

Smart defaults via model validator:
- `field_type='sdf'` → `direct_extraction=True`, `smooth_sigma=0.0` (unless explicitly overridden)
- `field_type='density'` → current defaults preserved
- `extraction_level=None` → 0.5 for density, 0.0 for SDF

The `threshold` constraint (`ge=0.0, le=1.0`) remains as-is — it only applies to the preprocessing binarization step, not to extraction level. `extraction_level` is unconstrained.

### Pipeline Flow

**Preprocessed path** (`direct_extraction=False`, default for density):

```
field → preprocess(Gaussian smooth → threshold → morphology → cleanup) → extract(binary, level=0.5) → smooth
```

Unchanged from current behavior.

**Direct path** (`direct_extraction=True`, default for SDF):

```
field → extract(continuous field, level=extraction_level) → smooth
```

`preprocess()` is skipped. `extract()` reads from `state.density` (the input field) instead of `state.binary`.

### File Changes

| File | Change |
|------|--------|
| `state.py` | Add `field_type`, `direct_extraction`, `extraction_level` to `PipelineParams` with smart defaults via model validator |
| `preprocess.py` | No change — simply skipped on the direct path |
| `extract.py` | Branch on `direct_extraction`: use `state.density` + `extraction_level` instead of `state.binary` + 0.5 |
| `pipeline.py` | Skip `preprocess()` when `direct_extraction=True` |
| `cli.py` | Add `--field-type` and `--direct` options to `process` and `viz` subcommands |
| `tests/` | ~6 new test functions |

### State Model

- `density` field name retained (backward compatible). Documents that it holds the input field regardless of type.
- No value range validation change needed — currently only checks ndim, which works for SDF values.
- `binary` stays `None` on the direct path.

### Extract Changes

```python
def extract(state: PipelineState) -> PipelineState:
    if state.params.direct_extraction:
        field = state.density.astype(float)
        level = state.params.effective_extraction_level  # 0.0 for SDF, 0.5 for density
    else:
        if state.binary is None:
            raise ValueError("binary field is None — run preprocess() first")
        field = state.binary.astype(float)
        level = 0.5
    # ... marching cubes / find_contours with field and level
```

### CLI

```bash
# Classical TO density (current behavior, unchanged)
xtf process beam.npy -o beam.stl

# Neural SDF output
xtf process neural_sdf.npy --field-type sdf -o result.stl

# Clean density field, skip preprocessing
xtf process clean_density.npy --direct -o result.stl
```

### Testing Plan

1. SDF direct extraction — synthetic signed distance to sphere, verify mesh at zero level set
2. Density direct extraction — clean density, verify extraction at 0.5 without preprocessing
3. SDF smart defaults — `field_type='sdf'` auto-sets `direct_extraction=True`, `smooth_sigma=0`
4. Parameter override — SDF with Gaussian smoothing re-enabled
5. CLI flags — `--field-type sdf` and `--direct`
6. Backward compatibility — all 85 existing tests pass unchanged

## Out of Scope

- **Differentiable extraction** (FlexiCubes/DMTet) — XelToFab is post-processing; by the time someone calls us, training is done.
- **Grid generation / model evaluation** — ML frameworks handle this. XelToFab takes numpy arrays.
- **PyTorch dependency** — not introduced. Input is always numpy arrays.
