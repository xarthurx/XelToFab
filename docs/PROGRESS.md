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
