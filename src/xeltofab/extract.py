"""Mesh/contour extraction from scalar fields.

All extraction backends return vertices in grid index coordinates
(matching scikit-image marching_cubes behavior).
"""

from __future__ import annotations

import warnings

import numpy as np
from skimage.measure import find_contours, marching_cubes

from xeltofab.state import PipelineState


def extract(state: PipelineState) -> PipelineState:
    """Extract mesh (3D) or contours (2D).

    Dispatches to the configured extraction_method:
    - 'mc': Marching Cubes (scikit-image)
    - 'dc': Dual Contouring with QEF (vendored sdftoolbox, or isoext GPU)
    - 'surfnets': Naive Surface Nets (vendored sdftoolbox)
    - 'manifold': Marching Tetrahedra via manifold3d (guaranteed manifold output)

    2D fields always use marching squares regardless of extraction_method.
    """
    if state.params.direct_extraction:
        field = np.asarray(state.field, dtype=np.float64)
    else:
        if state.binary is None:
            raise ValueError("binary field is None — run preprocess() first")
        field = np.asarray(state.binary, dtype=np.float64)
    level = state.params.effective_extraction_level

    if state.ndim == 2:
        return _extract_2d(state, field, level)

    method = state.params.extraction_method
    if method == "mc":
        result = _extract_3d_mc(state, field, level)
    elif method == "dc":
        result = _extract_3d_dc(state, field, level)
    elif method == "surfnets":
        result = _extract_3d_surfnets(state, field, level)
    elif method == "manifold":
        result = _extract_3d_manifold(state, field, level)
    else:
        raise ValueError(f"Unknown extraction_method: {method!r}")

    # Empty mesh guard: catch all backends producing no geometry
    if result.vertices is not None and result.vertices.shape[0] == 0:
        raise ValueError(
            "Extraction produced no geometry — check field values and extraction level"
        )
    return result


def _require_repair_for_non_manifold(state: PipelineState) -> None:
    """Require pymeshlab for DC/surfnets (can produce non-manifold output)."""
    try:
        import pymeshlab  # noqa: F401
    except ImportError:
        raise ImportError(
            f"pymeshlab is required for extraction_method='{state.params.extraction_method}' "
            "(DC/surfnets can produce non-manifold output that needs repair). "
            "Install with: uv sync --extra mesh-quality"
        ) from None


def _extract_2d(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract contours from 2D field using marching squares."""
    contours = find_contours(field, level=level)
    return state.model_copy(update={"contours": contours})


def _extract_3d_mc(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract triangle mesh from 3D field using marching cubes."""
    vertices, faces, _, _ = marching_cubes(field, level=level)
    return state.model_copy(update={"vertices": vertices, "faces": faces})


def _extract_3d_dc(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract triangle mesh using Dual Contouring (QEF vertex placement).

    Tries GPU-accelerated isoext if available, falls back to vendored CPU implementation.
    Requires pymeshlab for repair of non-manifold output.
    """
    _require_repair_for_non_manifold(state)
    shifted = field - level  # shift so isosurface is at zero

    try:
        return _extract_3d_dc_gpu(state, shifted)
    except (ImportError, RuntimeError) as e:
        warnings.warn(
            f"GPU DC unavailable ({e!r}), falling back to CPU",
            stacklevel=2,
        )

    from xeltofab._vendor.dual_isosurface import dual_isosurface

    vertices, faces = dual_isosurface(shifted, vertex_strategy="dc")
    return state.model_copy(update={"vertices": vertices, "faces": faces})


def _extract_3d_dc_gpu(state: PipelineState, shifted_field: np.ndarray) -> PipelineState:
    """GPU-accelerated Dual Contouring via isoext (requires PyTorch + CUDA)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    import isoext

    tensor = torch.from_numpy(shifted_field).float().cuda()
    grid = isoext.UniformGrid(list(shifted_field.shape))
    verts_t, faces_t = isoext.dual_contouring(grid, tensor)
    vertices = verts_t.cpu().numpy()
    faces = faces_t.cpu().numpy()
    return state.model_copy(update={"vertices": vertices, "faces": faces})


def _extract_3d_surfnets(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract triangle mesh using Naive Surface Nets (centroid vertex placement)."""
    _require_repair_for_non_manifold(state)
    shifted = field - level

    from xeltofab._vendor.dual_isosurface import dual_isosurface

    vertices, faces = dual_isosurface(shifted, vertex_strategy="surfnets")
    return state.model_copy(update={"vertices": vertices, "faces": faces})


def _extract_3d_manifold(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract triangle mesh using manifold3d (guaranteed manifold, watertight output).

    Uses marching tetrahedra on a BCC grid. Requires manifold3d optional dependency.
    Wraps the numpy grid array with RegularGridInterpolator as a callable SDF.
    """
    try:
        import manifold3d
    except ImportError:
        raise ImportError(
            "manifold3d not installed — required for extraction_method='manifold'. "
            "Install with: uv sync --extra manifold"
        ) from None

    from scipy.interpolate import RegularGridInterpolator

    nz, ny, nx = field.shape
    z = np.linspace(0, nz - 1, nz)
    y = np.linspace(0, ny - 1, ny)
    x = np.linspace(0, nx - 1, nx)
    interp = RegularGridInterpolator(
        (z, y, x), field, method="linear", bounds_error=False, fill_value=level + 1.0
    )

    # manifold3d: positive = inside, our SDF: negative = inside → negate
    # For density fields (values 0-1, level=0.5): inside = field > level → (field - level) > 0
    # So for density: no negation needed. For SDF (negative inside): negate.
    is_sdf = state.params.field_type == "sdf"
    _pt = np.zeros((1, 3))

    def sdf_func(x_val: float, y_val: float, z_val: float) -> float:
        _pt[0] = (z_val, y_val, x_val)  # manifold3d passes (x,y,z), our grid is (z,y,x)
        val = float(interp(_pt)[0]) - level
        return -val if is_sdf else val

    bounds = [0.0, 0.0, 0.0, float(nx - 1), float(ny - 1), float(nz - 1)]
    edge_length = 1.0

    m = manifold3d.Manifold.level_set(sdf_func, bounds, edge_length)
    mesh = m.to_mesh()
    vertices = np.asarray(mesh.vert_properties[:, :3], dtype=np.float64)
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)

    # Swap x,z back to match our (z,y,x) grid coordinate convention
    vertices = vertices[:, [2, 1, 0]]

    return state.model_copy(update={"vertices": vertices, "faces": faces})
