# Vendored from sdftoolbox (MIT License) — https://github.com/cheind/sdftoolbox
# Modifications: removed SDF node dependency, adapted DualContouringVertexStrategy
# to accept precomputed gradient arrays, replaced np.float_ with np.float64,
# removed Newton/Bisection edge strategies (not needed for array input).

from __future__ import annotations

import abc

import numpy as np

from .grid import Grid, float_dtype


class DualVertexStrategy(abc.ABC):
    """Base class for vertex placement strategies in dual methods."""

    @abc.abstractmethod
    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        gradients: np.ndarray | None,
        grid: Grid,
    ) -> np.ndarray:
        """Compute vertex locations for active voxels.

        Params:
            active_voxels: (M,) flat voxel indices in padded grid
            edge_coords: (E, 3) edge intersection coordinates in grid space
            gradients: (I, J, K, 3) precomputed gradient field, or None
            grid: the sampling grid
        """


class NaiveSurfaceNetVertexStrategy(DualVertexStrategy):
    """Place vertices at the centroid of edge intersection points per voxel."""

    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        gradients: np.ndarray | None,
        grid: Grid,
    ) -> np.ndarray:
        active_voxel_edges = grid.find_voxel_edges(active_voxels)  # (M, 12)
        e = edge_coords[active_voxel_edges]  # (M, 12, 3)
        return np.nanmean(e, 1)  # (M, 3)


class DualContouringVertexStrategy(DualVertexStrategy):
    """Place vertices using QEF (Quadric Error Function) solver.

    For each active voxel, finds the point that best agrees with all
    intersection point/normal pairs from surrounding active edges.
    Solves n^T(x - p) = 0 as a least-squares system Ax = b.

    Biases toward the naive surface net solution to keep vertices
    inside voxels.
    """

    def __init__(self, bias_strength: float = 1e-5):
        self.bias_strength = bias_strength
        self.sqrt_bias_strength = np.sqrt(self.bias_strength)

    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        gradients: np.ndarray | None,
        grid: Grid,
    ) -> np.ndarray:
        if gradients is None:
            # Fall back to surface nets if no gradients available
            return NaiveSurfaceNetVertexStrategy().find_vertex_locations(
                active_voxels, edge_coords, gradients, grid
            )

        sijk = grid.unravel_nd(active_voxels, grid.padded_shape)  # (M, 3)
        active_voxel_edges = grid.find_voxel_edges(active_voxels)  # (M, 12)
        points = edge_coords[active_voxel_edges]  # (M, 12, 3)

        # Interpolate gradients at edge intersection points
        normals = self._interpolate_gradients(points, gradients, grid)  # (M, 12, 3)

        # Compute bias vertices (naive surface net solution)
        bias_verts = NaiveSurfaceNetVertexStrategy().find_vertex_locations(
            active_voxels, edge_coords, gradients, grid
        )

        verts = []
        for off, p, n, bias in zip(sijk, points, normals, bias_verts):
            q = p - off[None, :]
            mask = np.isfinite(q).all(-1) & np.isfinite(n).all(-1)
            x = self._solve_lst(q[mask], n[mask], bias=(bias - off))
            verts.append(x + off)
        return np.array(verts, dtype=float_dtype)

    def _interpolate_gradients(
        self, points: np.ndarray, gradients: np.ndarray, grid: Grid
    ) -> np.ndarray:
        """Trilinearly interpolate gradient field at edge intersection points."""
        shape = points.shape  # (M, 12, 3)
        pts_flat = points.reshape(-1, 3)  # (M*12, 3)

        # Identify valid (non-NaN) points first
        valid = np.isfinite(pts_flat).all(-1)
        result = np.full((pts_flat.shape[0], 3), np.nan, dtype=np.float64)

        if not valid.any():
            return result.reshape(shape)

        gi = pts_flat[valid, 0]
        gj = pts_flat[valid, 1]
        gk = pts_flat[valid, 2]

        # Clamp to valid range
        I, J, K = gradients.shape[:3]
        gi = np.clip(gi, 0, I - 1.001)
        gj = np.clip(gj, 0, J - 1.001)
        gk = np.clip(gk, 0, K - 1.001)

        # Floor indices for trilinear interpolation
        i0 = np.floor(gi).astype(int)
        j0 = np.floor(gj).astype(int)
        k0 = np.floor(gk).astype(int)
        i1 = np.minimum(i0 + 1, I - 1)
        j1 = np.minimum(j0 + 1, J - 1)
        k1 = np.minimum(k0 + 1, K - 1)

        # Fractional parts
        fi = (gi - i0)[:, None]
        fj = (gj - j0)[:, None]
        fk = (gk - k0)[:, None]

        # Trilinear interpolation
        result[valid] = (
            gradients[i0, j0, k0] * (1 - fi) * (1 - fj) * (1 - fk)
            + gradients[i1, j0, k0] * fi * (1 - fj) * (1 - fk)
            + gradients[i0, j1, k0] * (1 - fi) * fj * (1 - fk)
            + gradients[i0, j0, k1] * (1 - fi) * (1 - fj) * fk
            + gradients[i1, j1, k0] * fi * fj * (1 - fk)
            + gradients[i1, j0, k1] * fi * (1 - fj) * fk
            + gradients[i0, j1, k1] * (1 - fi) * fj * fk
            + gradients[i1, j1, k1] * fi * fj * fk
        )

        return result.reshape(shape)

    def _solve_lst(self, q: np.ndarray, n: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Solve the QEF via least squares with bias toward surface net solution."""
        if len(q) == 0:
            return bias if bias is not None else np.array([0.5, 0.5, 0.5])

        A = n
        b = (q[:, None, :] @ n[..., None]).reshape(-1)

        if self.bias_strength > 0.0:
            C = np.eye(3, dtype=A.dtype) * self.sqrt_bias_strength
            d = bias * self.sqrt_bias_strength
            A = np.concatenate((A, C), 0)
            b = np.concatenate((b, d), 0)

        x, _, _, _ = np.linalg.lstsq(A.astype(float), b.astype(float), rcond=None)
        return x.astype(q.dtype)


class DualEdgeStrategy(abc.ABC):
    """Base class for edge/surface intersection strategies."""

    @abc.abstractmethod
    def find_edge_intersections(
        self,
        src_ijk: np.ndarray,
        src_sdf: np.ndarray,
        dst_ijk: np.ndarray,
        dst_sdf: np.ndarray,
        edge_dir_index: int,
        edge_dir: np.ndarray,
    ) -> np.ndarray:
        pass


class LinearEdgeStrategy(DualEdgeStrategy):
    """Determine edge intersections by linear interpolation: t = -sdf_src / (sdf_dst - sdf_src)."""

    def find_edge_intersections(
        self,
        src_ijk: np.ndarray,
        src_sdf: np.ndarray,
        dst_ijk: np.ndarray,
        dst_sdf: np.ndarray,
        edge_dir_index: int,
        edge_dir: np.ndarray,
    ) -> np.ndarray:
        return -src_sdf / (dst_sdf - src_sdf)
