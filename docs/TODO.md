# TODO

## Quality-Enhanced Post-Processing (Tier 2)

> Baseline captured: `benchmarks/baseline/` (metrics + visualizations from current pipeline)

- [x] QEM decimation (quadric edge collapse) — `pyfqmr` (`decimate.py`)
- [x] Feature-preserving smoothing — bilateral mesh filtering (`smoothing_method='bilateral'`)
- [x] Watertight repair — non-manifold fixing via pymeshlab (`repair.py`)
- [x] Isotropic remeshing — `gpytoolbox.remesh_botsch` (Botsch & Kobbelt); 99%+ faces FEA-ready
- [x] Mesh quality metrics — aspect ratio, min angle, scaled Jacobian via `quality.py`

## Housekeeping

- [x] Fix PyVista deprecation warnings (`compute_cell_quality` → `cell_quality`)
- [x] Update `ARCHITECTURE.md` — add decimate stage, update module map, CLI section
- [x] Update website docs for decimation feature (pipeline overview, parameters, CLI pages)
- [ ] Commit pending `decimate.py`/`test_decimate.py` cleanup

## Benchmark Enhancements

- [x] Quality heatmaps — color triangles by aspect ratio / min angle / Jacobian (pyvista)
- [x] Metric histograms — distribution plots of mesh quality metrics per model

## Visualization Upgrades

- [ ] Polyscope integration — interactive 3D mesh/volume inspection (`polyscope` Python package)
- [ ] Volume rendering of density fields
- [ ] Interactive parameter tuning with live preview

## Mesh-to-CAD Pipeline (Tier 3 — Research Frontier, on hold)

- [ ] Patch decomposition — quadrangulation via Instant Meshes, motorcycle graph layout
- [ ] NURBS surface fitting — B-spline fitting per patch, oscillation control
- [ ] B-Rep assembly — PythonOCC / CadQuery for STEP export
- [ ] Skeleton extraction + lofting for beam-like TO results
- [ ] Investigate AMRTO pipeline integration

## Neural-Enhanced Pipeline (Tier 4)

- [ ] FlexiCubes / Kaolin for differentiable mesh extraction (requires CUDA)
- [ ] Neural SDF representations (NITO / NTopo style)
  - [x] SDF field-type support for extraction (`field_type='sdf'`, `direct_extraction`)
- [ ] Point2CAD concepts for learned B-Rep reconstruction
