"""SDF function evaluation on structured grids.

Evaluates arbitrary SDF callables on uniform grids, producing dense numpy
arrays compatible with the existing extraction pipeline.

Coordinate convention:
    User-facing bounds are (x, y, z) order.
    SDF functions receive points as [N, 3] in (x, y, z) order.
    Output arrays are [Nz, Ny, Nx] — matching scikit-image / existing pipeline.
"""

from __future__ import annotations

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
    """Compute grid dimensions preserving aspect ratio.

    Resolution specifies cells along the longest bounding box axis.
    Shorter axes get proportionally fewer cells (minimum 2).
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
