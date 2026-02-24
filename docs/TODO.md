# TODO

## Quality-Enhanced Post-Processing (Tier 2)

- [ ] QEM decimation (quadric edge collapse) — `pymeshlab` or `pyfqmr`
- [ ] Isotropic remeshing — `pymeshlab` `meshing_isotropic_explicit_remeshing()`
- [ ] Feature-preserving smoothing — bilateral mesh filtering, two-step normal smoothing
- [ ] Watertight repair — hole filling, non-manifold fixing via trimesh
- [ ] Mesh quality metrics — aspect ratio, min angle, scaled Jacobian via `pyvista.mesh.quality()`

## Visualization Upgrades

- [ ] Polyscope integration — interactive 3D mesh/volume inspection (`polyscope` Python package)
- [ ] Volume rendering of density fields
- [ ] Interactive parameter tuning with live preview

## Mesh-to-CAD Pipeline (Tier 3 — Research Frontier)

- [ ] Patch decomposition — quadrangulation via Instant Meshes, motorcycle graph layout
- [ ] NURBS surface fitting — B-spline fitting per patch, oscillation control
- [ ] B-Rep assembly — PythonOCC / CadQuery for STEP export
- [ ] Skeleton extraction + lofting for beam-like TO results
- [ ] Investigate AMRTO pipeline integration

## Neural-Enhanced Pipeline (Tier 4)

- [ ] FlexiCubes / Kaolin for differentiable mesh extraction (requires CUDA)
- [ ] Neural SDF representations (NITO / NTopo style)
- [ ] Point2CAD concepts for learned B-Rep reconstruction
