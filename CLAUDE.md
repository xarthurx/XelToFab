# CLAUDE.md

## Project Overview

**XelToFab** — Design field post-processing pipeline: scalar fields → meshes → fabrication-ready geometry. Python 3.13+, `uv` project management. Package: `xeltofab` (`src/xeltofab/`).

Research notes: `~/winHome/repo/xKnowledgeSync/99_academic.work/MAX_Mark/proj_XelToFab/`
Background material: `~/winHome/repo/xKnowledgeSync/99_academic.work/MAX_Mark/` (brainstorms, strategy, glossary)
Key reference: [`{Review} The Topology Optimization Post-Processing Pipeline.md`](file:///home/xarthurx/winHome/repo/xKnowledgeSync/99_academic.work/MAX_Mark/proj_XelToFab/%7BReview%7D%20The%20Topology%20Optimization%20Post-Processing%20Pipeline.md) — full literature review covering all 4 pipeline stages, toolchain, and feasibility analysis.
Collaboration: M4X–IDEAL Lab with Prof. Mark Fuge.

## Pipeline Architecture

1. **Preprocessing** — Threshold (Heaviside), smooth (Gaussian), clean (morphology + connected components)
2. **Mesh Extraction** — Marching cubes/squares (scikit-image); differentiable via FlexiCubes/Kaolin
3. **Mesh Post-processing** — Taubin smoothing, QEM decimation, isotropic remeshing, repair
4. **Mesh-to-CAD** — Patch decomposition, NURBS fitting, B-Rep assembly (research frontier)

## Project-Specific Commands

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Run tests | `uv run pytest tests/ -v` |
| Lint | `ruff check` |
| Format | `ruff format` |
| Type check | `ty check` |
| CLI help | `uv run xtf --help` |
| Dev docs site | `cd website && bun run dev` |
| Build docs site | `cd website && bun run build` |

## Core Dependencies

| Stage | Libraries |
|-------|-----------|
| Preprocessing | `scipy.ndimage`, `scikit-image` |
| Mesh extraction | `scikit-image` (marching cubes), `pyvista`/VTK |
| Smoothing / decimation | `pyvista`, `pymeshlab`, `open3d` |
| Mesh I/O / repair | `trimesh` |
| Quality metrics | `pyvista` (`mesh.quality()`) |
| NURBS / CAD export | `pythonocc-core`, `cadquery` |
| Format conversion | `meshio` |
| I/O formats | `scipy.io` (.mat), `pyvista` (.vtk), `h5py` (.h5/.xdmf) |

## Version Management

The project version is the single source of truth in `pyproject.toml`. When bumping the version, **always update both files**:
- `pyproject.toml` — `version = "X.Y.Z"`
- `website/package.json` — `"version": "X.Y.Z"`

## Conventions

- Input: numpy arrays — density fields in [0, 1] or SDF fields (unbounded)
- Mesh quality targets (FEA): aspect ratio < 5, min angle > 20°, scaled Jacobian > 0.5
- Always validate volume preservation after smoothing
- Re-verify connectivity after decimation (thin TO features can collapse)
- B-spline fitting: keep < 100 points per parametric direction to avoid oscillation

## Documentation Layout

| File | Location | Purpose |
|------|----------|---------|
| `CLAUDE.md` | repo root | Agent instructions (this file) |
| `README.md` | repo root | Project overview, quick start, API examples |
| `ARCHITECTURE.md` | `docs/` | Module map, data flow, dependency table |
| `TODO.md` | `docs/` | Feature backlog (tiered) |
| `PROGRESS.md` | `docs/` | Incident log & project memory |

## Confidential Plans

Design docs and implementation plans are stored externally (not synced with this repo):
`~/repo/repo_plans/xelToFab/`

## docs/TODO.md — Feature Backlog

`docs/TODO.md` tracks deferred features and future work tiers. Update it when scoping new work or deferring functionality. Current tiers: quality-enhanced post-processing, Polyscope visualization, mesh-to-CAD, neural-enhanced pipeline.

## docs/PROGRESS.md — Incident Log & Project Memory

`docs/PROGRESS.md` is a living record of problems, solutions, and hard-won knowledge. Update it **immediately** when:
- A bug is found and fixed
- A failed approach is abandoned
- A non-obvious design decision is made
- A configuration or environment issue is resolved
- A test failure reveals a misunderstanding

Each entry **must** include:
1. **Problem** — What went wrong or what changed
2. **Root cause** — Why it happened
3. **Resolution** — How it was fixed (include the **Git commit ID**, e.g. `fix: 8e2ab7e`)
4. **Prevention** — How to avoid this class of problem in the future

**Critical rule: Do NOT repeat the same type of mistake.** Before starting any work, review `docs/PROGRESS.md` for past issues in the same area. If a similar problem has already been documented, follow the prevention guidance — repeating a documented mistake is unacceptable.

## Keeping Documentation in Sync

Before finishing a session:
1. Check changes made and update any stale docs
2. Update `docs/PROGRESS.md` with any issues encountered
3. Re-read this file for any instructions updated during the session
