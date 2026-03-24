"""SDF function evaluation on structured grids.

Evaluates arbitrary SDF callables on uniform or octree-adaptive grids,
producing dense numpy arrays compatible with the existing extraction pipeline.

Coordinate convention:
    User-facing bounds are (x, y, z) order.
    SDF functions receive points as [N, 3] in (x, y, z) order.
    Output arrays are [Nz, Ny, Nx] — matching scikit-image / existing pipeline.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class SDFFunction(Protocol):
    """Universal SDF evaluation interface.

    Any callable accepting [N, 3] float64 points and returning [N] float64
    signed distances satisfies this protocol.
    """

    def __call__(self, points: np.ndarray) -> np.ndarray: ...


# Grouped format: (xmin, ymin, zmin, xmax, ymax, zmax)
Bounds3D = tuple[float, float, float, float, float, float]


def validate_bounds(bounds: Bounds3D) -> tuple[np.ndarray, np.ndarray]:
    """Validate bounds and return (min_corner, max_corner) arrays.

    Raises ValueError if bounds are invalid.
    """
    if len(bounds) != 6:
        raise ValueError(f"bounds must have exactly 6 elements, got {len(bounds)}")

    mins = np.array(bounds[:3], dtype=np.float64)
    maxs = np.array(bounds[3:], dtype=np.float64)

    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        raise ValueError("bounds must be finite numeric values")

    if not np.all(mins < maxs):
        bad = [(("x", "y", "z")[i], mins[i], maxs[i]) for i in range(3) if mins[i] >= maxs[i]]
        raise ValueError(f"each min must be strictly less than max, violations: {bad}")

    return mins, maxs


def compute_grid_dims(bounds_min: np.ndarray, bounds_max: np.ndarray, resolution: int) -> tuple[int, int, int]:
    """Compute grid point counts preserving aspect ratio.

    Resolution specifies points along the longest bounding box axis.
    Shorter axes get proportionally fewer points (minimum 2).
    """
    extents = bounds_max - bounds_min
    longest = extents.max()
    dims = np.maximum(np.round(resolution * extents / longest).astype(int), 2)
    # dims order: (x, y, z) matching bounds order
    return int(dims[0]), int(dims[1]), int(dims[2])


def validate_sdf_output(result: np.ndarray, n_points: int) -> np.ndarray:
    """Validate and coerce SDF function output.

    Checks shape, NaN/inf, and coerces to float64.
    """
    result = np.asarray(result)

    if result.ndim != 1:
        raise ValueError(f"SDF function must return 1D array, got {result.ndim}D with shape {result.shape}")

    if result.shape[0] != n_points:
        raise ValueError(f"SDF function returned {result.shape[0]} values for {n_points} input points")

    result = result.astype(np.float64, copy=False)

    n_bad = np.count_nonzero(~np.isfinite(result))
    if n_bad > 0:
        raise ValueError(f"SDF function returned {n_bad}/{n_points} non-finite values (NaN or Inf)")

    return result


def uniform_grid_evaluate(
    sdf_fn: SDFFunction,
    bounds: Bounds3D,
    resolution: int = 128,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate an SDF function on a uniform grid.

    Returns (grid, x_coords, y_coords, z_coords) where grid has shape [Nz, Ny, Nx].
    Evaluates one Z-slab at a time to bound memory usage.

    Parameters
    ----------
    sdf_fn : SDFFunction
        Callable: [N, 3] float64 → [N] float64 signed distances.
    bounds : Bounds3D
        (xmin, ymin, zmin, xmax, ymax, zmax) in grouped format.
    resolution : int
        Cells along the longest bounding box axis. Shorter axes proportional.
    chunk_size : int | None
        Max points per sdf_fn call within a slab. None = entire slab at once.

    Returns
    -------
    grid : np.ndarray
        SDF values with shape [Nz, Ny, Nx].
    x_coords, y_coords, z_coords : np.ndarray
        1D coordinate arrays for each axis.
    """
    bounds_min, bounds_max = validate_bounds(bounds)
    nx, ny, nz = compute_grid_dims(bounds_min, bounds_max, resolution)

    x_coords = np.linspace(bounds_min[0], bounds_max[0], nx)
    y_coords = np.linspace(bounds_min[1], bounds_max[1], ny)
    z_coords = np.linspace(bounds_min[2], bounds_max[2], nz)

    grid = np.empty((nz, ny, nx), dtype=np.float64)
    n_per_slab = nx * ny

    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    points = np.empty((n_per_slab, 3), dtype=np.float64)
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    slab_result = np.empty(n_per_slab, dtype=np.float64) if chunk_size else None
    first_slab = True

    for k in range(nz):
        points[:, 2] = z_coords[k]

        try:
            if chunk_size is not None and n_per_slab > chunk_size:
                if slab_result is None:
                    slab_result = np.empty(n_per_slab, dtype=np.float64)
                for i in range(0, n_per_slab, chunk_size):
                    end = min(i + chunk_size, n_per_slab)
                    slab_result[i:end] = sdf_fn(points[i:end])
                result = slab_result
            else:
                result = sdf_fn(points)
        except Exception as e:
            raise RuntimeError(
                f"SDF function failed at {n_per_slab} points in Z-slab {k}/{nz} "
                f"(z={z_coords[k]:.4f}), region [{bounds_min} → {bounds_max}]: {e}"
            ) from e

        if first_slab:
            result = validate_sdf_output(result, n_per_slab)
            first_slab = False
        else:
            result = np.asarray(result, dtype=np.float64)

        grid[k] = result.reshape(ny, nx)

    return grid, x_coords, y_coords, z_coords


# ---------------------------------------------------------------------------
# Octree-accelerated evaluation (Phase 2)
# ---------------------------------------------------------------------------

# 8 corner offsets of a unit cell: (dx, dy, dz) for dx,dy,dz in {0,1}
_CORNER_OFFSETS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    dtype=np.int64,
)


def _compute_coarse_dims(nx: int, ny: int, nz: int, coarse_factor: int) -> tuple[int, int, int]:
    """Compute coarse cell counts from target point counts and coarse factor.

    Target has (nx-1, ny-1, nz-1) cells. Coarse divides each by coarse_factor.
    """
    cx = max(1, math.ceil((nx - 1) / coarse_factor))
    cy = max(1, math.ceil((ny - 1) / coarse_factor))
    cz = max(1, math.ceil((nz - 1) / coarse_factor))
    return cx, cy, cz


def _deduplicate_coords(coords: np.ndarray, rx: int, ry: int, rz: int) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate 3D integer coordinates via 1D key hashing.

    Parameters
    ----------
    coords : [N, 3] int64 — (ix, iy, iz) coordinates
    rx, ry, rz : max coordinate + 1 per axis (key space dimensions)

    Returns
    -------
    unique_coords : [M, 3] int64
    inverse : [N] int64 — maps each input coord to its unique index
    """
    keys = (
        coords[:, 0].astype(np.int64) * (ry * rz) + coords[:, 1].astype(np.int64) * rz + coords[:, 2].astype(np.int64)
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    unique_coords = np.column_stack(
        [
            unique_keys // (ry * rz),
            (unique_keys // rz) % ry,
            unique_keys % rz,
        ]
    ).astype(np.int64)
    return unique_coords, inverse


def _evaluate_at_coords(
    sdf_fn: SDFFunction,
    grid_coords: np.ndarray,
    bounds_min: np.ndarray,
    extent: np.ndarray,
    node_counts: np.ndarray,
    chunk_size: int | None,
    validate: bool = False,
) -> np.ndarray:
    """Evaluate SDF at world-space positions corresponding to grid coordinates.

    Parameters
    ----------
    grid_coords : [M, 3] int64 — (ix, iy, iz) in final grid node space
    node_counts : [3] — (Rx, Ry, Rz) total nodes per axis
    """
    world_pts = bounds_min + (grid_coords.astype(np.float64) / (node_counts - 1)) * extent

    if chunk_size is not None and len(world_pts) > chunk_size:
        result = np.empty(len(world_pts), dtype=np.float64)
        for i in range(0, len(world_pts), chunk_size):
            end = min(i + chunk_size, len(world_pts))
            result[i:end] = sdf_fn(world_pts[i:end])
    else:
        result = sdf_fn(world_pts)

    result = np.asarray(result, dtype=np.float64)
    if validate:
        result = validate_sdf_output(result, len(world_pts))
    return result


def octree_evaluate(
    sdf_fn: SDFFunction,
    bounds: Bounds3D,
    resolution: int = 128,
    coarse_factor: int = 8,
    lipschitz: float = 1.1,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate an SDF function using octree-accelerated coarse-to-fine refinement.

    Reduces evaluations from O(N^3) to ~O(N^2) by culling cells far from the
    zero level set at coarse resolution before subdividing near-surface cells.

    Parameters
    ----------
    sdf_fn : SDFFunction
        Callable: [N, 3] float64 → [N] float64 signed distances.
    bounds : Bounds3D
        (xmin, ymin, zmin, xmax, ymax, zmax) spatial region.
    resolution : int
        Cells along the longest bounding box axis (same as uniform_grid_evaluate).
    coarse_factor : int
        Fine-to-coarse ratio. Must be a power of 2. Default 8 (3 refinement levels).
    lipschitz : float
        Culling threshold. Cells kept when min(|SDF|) < lipschitz * cell_diagonal.
    chunk_size : int | None
        Max points per sdf_fn call. None = all unique corners at once.

    Returns
    -------
    grid : np.ndarray
        SDF values with shape [Nz, Ny, Nx]. Unevaluated regions = +1.0.
    x_coords, y_coords, z_coords : np.ndarray
        1D coordinate arrays for each axis.
    """
    if coarse_factor < 1 or (coarse_factor & (coarse_factor - 1)) != 0:
        raise ValueError(f"coarse_factor must be a power of 2, got {coarse_factor}")

    bounds_min, bounds_max = validate_bounds(bounds)
    extent = bounds_max - bounds_min
    nx, ny, nz = compute_grid_dims(bounds_min, bounds_max, resolution)

    # Coarse cell counts and refinement levels
    n_levels = round(math.log2(coarse_factor)) if coarse_factor > 1 else 0
    cx, cy, cz = _compute_coarse_dims(nx, ny, nz, coarse_factor)

    # Actual fine cell counts after n_levels doublings
    scale = 2**n_levels if n_levels > 0 else 1
    fx, fy, fz = cx * scale, cy * scale, cz * scale

    # Grid node counts (points = cells + 1)
    rx, ry, rz = fx + 1, fy + 1, fz + 1
    node_counts = np.array([rx, ry, rz], dtype=np.int64)

    x_coords = np.linspace(bounds_min[0], bounds_max[0], rx)
    y_coords = np.linspace(bounds_min[1], bounds_max[1], ry)
    z_coords = np.linspace(bounds_min[2], bounds_max[2], rz)

    # Initialize: all coarse cells, coordinates in final grid node space
    step = scale
    ci, cj, ck = np.meshgrid(np.arange(cx), np.arange(cy), np.arange(cz), indexing="ij")
    active_cells = np.column_stack([ci.ravel(), cj.ravel(), ck.ravel()]).astype(np.int64) * step

    # Cross-level SDF cache: avoid re-evaluating corners seen at coarser levels.
    # Key = 1D linearized node coord, value = SDF distance.
    sdf_cache: dict[int, float] = {}

    # Coarse-to-fine refinement loop
    for level in range(n_levels):
        if len(active_cells) == 0:
            break

        corners = active_cells[:, np.newaxis, :] + _CORNER_OFFSETS[np.newaxis, :, :] * step
        flat_corners = corners.reshape(-1, 3)

        unique_coords, inverse = _deduplicate_coords(flat_corners, rx, ry, rz)

        # Check cache — only evaluate uncached corners
        keys = unique_coords[:, 0] * (ry * rz) + unique_coords[:, 1] * rz + unique_coords[:, 2]
        cached_mask = np.array([int(k) in sdf_cache for k in keys], dtype=bool)
        sdf_vals = np.empty(len(unique_coords), dtype=np.float64)

        if np.any(cached_mask):
            sdf_vals[cached_mask] = np.array([sdf_cache[int(k)] for k in keys[cached_mask]])
        uncached = ~cached_mask
        if np.any(uncached):
            new_vals = _evaluate_at_coords(
                sdf_fn,
                unique_coords[uncached],
                bounds_min,
                extent,
                node_counts,
                chunk_size,
                validate=(level == 0),
            )
            sdf_vals[uncached] = new_vals
            for k, v in zip(keys[uncached], new_vals, strict=True):
                sdf_cache[int(k)] = float(v)

        cell_sdf = sdf_vals[inverse].reshape(-1, 8)

        # Lipschitz culling
        world_step = step * extent / (node_counts - 1)
        cell_diag = float(np.linalg.norm(world_step))
        min_abs = np.min(np.abs(cell_sdf), axis=1)
        near_surface = min_abs < lipschitz * cell_diag
        active_cells = active_cells[near_surface]

        step //= 2
        if step >= 1 and len(active_cells) > 0:
            children = active_cells[:, np.newaxis, :] + _CORNER_OFFSETS[np.newaxis, :, :] * step
            all_children = children.reshape(-1, 3)
            active_cells, _ = _deduplicate_coords(all_children, fx, fy, fz)

    # Final evaluation at target resolution (step=1)
    # Determine fill value: if no surface found, use actual SDF sign at domain center
    # to avoid fabricating zero crossings (e.g., all-inside SDF filled with +1.0).
    if len(active_cells) == 0:
        center = (bounds_min + extent / 2).reshape(1, 3)
        center_val = float(sdf_fn(center)[0])
        fill_val = center_val if np.isfinite(center_val) else 1.0
        grid = np.full((rz, ry, rx), fill_val, dtype=np.float64)
        return grid, x_coords, y_coords, z_coords

    grid = np.full((rz, ry, rx), 1.0, dtype=np.float64)

    corners = active_cells[:, np.newaxis, :] + _CORNER_OFFSETS[np.newaxis, :, :]
    flat_corners = corners.reshape(-1, 3)
    unique_coords, inverse = _deduplicate_coords(flat_corners, rx, ry, rz)

    # Check cache for final level too
    keys = unique_coords[:, 0] * (ry * rz) + unique_coords[:, 1] * rz + unique_coords[:, 2]
    cached_mask = np.array([int(k) in sdf_cache for k in keys], dtype=bool)
    sdf_vals = np.empty(len(unique_coords), dtype=np.float64)

    if np.any(cached_mask):
        sdf_vals[cached_mask] = np.array([sdf_cache[int(k)] for k in keys[cached_mask]])
    uncached = ~cached_mask
    if np.any(uncached):
        sdf_vals[uncached] = _evaluate_at_coords(
            sdf_fn,
            unique_coords[uncached],
            bounds_min,
            extent,
            node_counts,
            chunk_size,
            validate=True,
        )

    # Write evaluated corners into dense grid (coords are provably in-bounds)
    grid[unique_coords[:, 2], unique_coords[:, 1], unique_coords[:, 0]] = sdf_vals

    return grid, x_coords, y_coords, z_coords
