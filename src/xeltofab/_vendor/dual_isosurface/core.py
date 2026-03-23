# Vendored from sdftoolbox (MIT License) — https://github.com/cheind/sdftoolbox
# Modifications: adapted for numpy array input (no SDF node dependency),
# constructs Grid internally from volume shape, computes gradients via np.gradient().

from __future__ import annotations

import numpy as np

from .grid import Grid
from .mesh_utils import triangulate_quads
from .strategies import (
    DualContouringVertexStrategy,
    LinearEdgeStrategy,
    NaiveSurfaceNetVertexStrategy,
)


def dual_isosurface(
    sdf_volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    vertex_strategy: str = "dc",
    vertex_relaxation_percent: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh from a 3D signed distance field using dual methods.

    The isosurface is extracted at the zero level-set: negative = inside,
    positive = outside (standard SDF convention).

    Params:
        sdf_volume: (I, J, K) float array. The SDF values at grid points.
            The zero level-set defines the surface.
        spacing: voxel spacing in each dimension.
        vertex_strategy: "dc" for Dual Contouring (QEF, sharp features),
            "surfnets" for Naive Surface Nets (smooth, centroid-based).
        vertex_relaxation_percent: tolerance for vertex placement outside voxel
            bounds. Higher values allow sharper features at low resolution.

    Returns:
        vertices: (N, 3) array of vertex positions in grid index coordinates.
        faces: (M, 3) array of triangle face indices.
    """
    grid = Grid.from_shape(sdf_volume.shape, spacing)
    edge_strategy = LinearEdgeStrategy()

    if vertex_strategy == "dc":
        v_strategy = DualContouringVertexStrategy()
    elif vertex_strategy == "surfnets":
        v_strategy = NaiveSurfaceNetVertexStrategy()
    else:
        raise ValueError(f"Unknown vertex_strategy: {vertex_strategy!r}")

    # Precompute gradients for DC (not needed for surface nets)
    gradients = None
    if vertex_strategy == "dc":
        gi, gj, gk = np.gradient(sdf_volume)
        gradients = np.stack([gi, gj, gk], axis=-1)  # (I, J, K, 3)

    # Pad the SDF volume to avoid boundary issues
    padded_sdf = np.pad(
        sdf_volume,
        ((0, 1), (0, 1), (0, 1)),
        mode="constant",
        constant_values=np.nan,
    )

    # Step 1: Find active edges (sign changes)
    edges_active_mask = np.zeros((grid.num_edges,), dtype=bool)
    edges_flip_mask = np.zeros((grid.num_edges,), dtype=bool)
    edges_isect_coords = np.full((grid.num_edges, 3), np.nan, dtype=np.float64)

    sijk = grid.get_all_source_vertices()
    si, sj, sk = sijk.T
    sdf_src = padded_sdf[si, sj, sk]
    src_sign = np.sign(sdf_src)

    for aidx, off in enumerate(np.eye(3, dtype=np.int32)):
        tijk = sijk + off[None, :]
        ti, tj, tk = tijk.T
        sdf_dst = padded_sdf[ti, tj, tk]

        dst_sign = np.sign(sdf_dst)
        active = (src_sign != dst_sign) & np.isfinite(sdf_dst)

        t = edge_strategy.find_edge_intersections(
            sijk[active],
            sdf_src[active],
            tijk[active],
            sdf_dst[active],
            aidx,
            off,
        )
        isect_coords = sijk[active] + off[None, :] * t[:, None]
        need_flip = (sdf_dst[active] - sdf_src[active]) < 0.0

        edges_active_mask[aidx::3] = active
        edges_flip_mask[aidx::3][active] = need_flip
        edges_isect_coords[aidx::3][active] = isect_coords

    # Step 2: Tessellation — each active edge produces a quad from 4 neighboring voxels
    active_edges = np.where(edges_active_mask)[0]
    if len(active_edges) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    active_quads, complete_mask = grid.find_voxels_sharing_edge(active_edges)
    active_edges = active_edges[complete_mask]
    active_quads = active_quads[complete_mask]

    if len(active_quads) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    # Fix winding order
    active_edges_flip = edges_flip_mask[active_edges]
    active_quads[active_edges_flip] = np.flip(active_quads[active_edges_flip], -1)

    # Deduplicate voxels — inverse gives us the face indices
    active_voxels, faces = np.unique(active_quads, return_inverse=True)

    # Step 3: Vertex placement
    grid_verts = v_strategy.find_vertex_locations(active_voxels, edges_isect_coords, gradients, grid)

    # Clip vertices to voxel bounds with relaxation tolerance
    grid_ijk = grid.unravel_nd(active_voxels, grid.padded_shape)
    grid_verts = (
        np.clip(
            grid_verts - grid_ijk,
            0.0 - vertex_relaxation_percent,
            1.0 + vertex_relaxation_percent,
        )
        + grid_ijk
    )

    # Convert to data coordinates
    verts = grid.grid_to_data(grid_verts)

    # Step 4: Triangulate quads
    faces = faces.reshape(-1, 4)
    faces = triangulate_quads(faces)

    return verts.astype(np.float64), faces.astype(np.int64)
