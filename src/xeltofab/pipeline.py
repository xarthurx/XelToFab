"""Pipeline orchestrator."""

from __future__ import annotations

from typing import Any, Literal

from xeltofab.decimate import decimate
from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.remesh import remesh
from xeltofab.repair import repair
from xeltofab.sdf_eval import Bounds3D, SDFFunction, octree_evaluate, uniform_grid_evaluate
from xeltofab.smooth import smooth
from xeltofab.state import PipelineParams, PipelineState


def process_from_sdf(
    sdf_fn: SDFFunction,
    bounds: Bounds3D,
    resolution: int = 128,
    adaptive: bool = False,
    extraction_method: Literal["mc", "dc", "surfnets", "manifold"] = "dc",
    chunk_size: int | None = None,
    **pipeline_kwargs: Any,
) -> PipelineState:
    """Evaluate an SDF function and run the full pipeline.

    This is the entry point for neural/analytical SDF functions.
    Evaluates the SDF on a uniform grid, then feeds into the standard
    extract → smooth → repair → remesh → decimate pipeline.

    Parameters
    ----------
    sdf_fn : SDFFunction
        Callable: [N, 3] float64 → [N] float64 signed distances.
    bounds : Bounds3D
        (xmin, ymin, zmin, xmax, ymax, zmax) spatial region.
    resolution : int
        Cells along the longest axis; shorter axes proportional.
    adaptive : bool
        Use octree-accelerated evaluation. Reduces evaluations from O(N³) to
        ~O(N²) by culling cells far from the surface at coarse resolution.
    extraction_method : str
        Extraction backend: 'mc', 'dc', 'surfnets', 'manifold'.
    chunk_size : int | None
        Max points per sdf_fn call. None = entire Z-slab at once.
    **pipeline_kwargs
        Forwarded to PipelineParams (smoothing_method, decimate_ratio, etc.).

    Returns
    -------
    PipelineState
        Pipeline result with extracted and post-processed mesh.
    """
    # Coordinates discarded: extraction operates in grid-index space, not world space.
    if adaptive:
        grid, _, _, _ = octree_evaluate(sdf_fn, bounds, resolution, chunk_size=chunk_size)
    else:
        grid, _, _, _ = uniform_grid_evaluate(sdf_fn, bounds, resolution, chunk_size)

    params = PipelineParams(
        field_type="sdf",
        extraction_method=extraction_method,
        **pipeline_kwargs,
    )
    state = PipelineState(field=grid, params=params)
    return process(state)


def process(state: PipelineState) -> PipelineState:
    """Run the pipeline: [preprocess] -> extract -> smooth -> [repair] -> [remesh] -> [decimate].

    When direct_extraction is enabled, preprocessing is skipped and extraction
    operates on the continuous input field directly.
    Repair, remesh, and decimate are enabled by default.
    """
    if not state.params.direct_extraction:
        state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    if state.params.needs_repair:
        state = repair(state)
    state = remesh(state)
    state = decimate(state)
    return state
