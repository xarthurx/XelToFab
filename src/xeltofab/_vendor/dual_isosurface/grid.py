# Vendored from sdftoolbox (MIT License) — https://github.com/cheind/sdftoolbox
# Modifications: removed SDF node dependency, replaced np.float_ with np.float64,
# added from_shape() classmethod for numpy array input.

from __future__ import annotations

import numpy as np

float_dtype = np.float64


class Grid:
    """A 3D sampling grid with topology lookup methods for dual isosurface extraction."""

    VOXEL_EDGE_OFFSETS = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [1, 0, 0, 1],
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 2],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.int32,
    ).reshape(1, 12, 4)

    EDGE_VOXEL_OFFSETS = np.array(
        [
            [  # i
                [0, 0, 0],
                [0, -1, 0],
                [0, -1, -1],
                [0, 0, -1],
            ],
            [  # j
                [0, 0, 0],
                [0, 0, -1],
                [-1, 0, -1],
                [-1, 0, 0],
            ],
            [  # k
                [0, 0, 0],
                [-1, 0, 0],
                [-1, -1, 0],
                [0, -1, 0],
            ],
        ],
        dtype=np.int32,
    )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Grid:
        """Construct a grid from a volume shape and spacing."""
        I, J, K = shape
        ranges = [
            np.linspace(0, (I - 1) * spacing[0], I, dtype=float_dtype),
            np.linspace(0, (J - 1) * spacing[1], J, dtype=float_dtype),
            np.linspace(0, (K - 1) * spacing[2], K, dtype=float_dtype),
        ]
        X, Y, Z = np.meshgrid(*ranges, indexing="ij")
        xyz = np.stack((X, Y, Z), -1)
        return cls(xyz=xyz)

    def __init__(self, xyz: np.ndarray):
        self.xyz = xyz
        self.padded_shape = (
            self.xyz.shape[0] + 1,
            self.xyz.shape[1] + 1,
            self.xyz.shape[2] + 1,
        )
        self.edge_shape = self.xyz.shape[:3] + (3,)
        self.num_edges = int(np.prod(self.edge_shape))

    @property
    def spacing(self):
        return self.xyz[1, 1, 1] - self.xyz[0, 0, 0]

    @property
    def min_corner(self):
        return self.xyz[0, 0, 0]

    @property
    def max_corner(self):
        return self.xyz[-1, -1, -1]

    @property
    def shape(self):
        return self.xyz.shape[:3]

    def ravel_nd(self, nd_indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.ravel_multi_index(list(nd_indices.T), dims=shape)

    def unravel_nd(self, indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        ur = np.unravel_index(indices, shape)
        return np.stack(ur, -1)

    def get_all_source_vertices(self) -> np.ndarray:
        I, J, K = self.edge_shape[:3]
        sijk = np.stack(
            np.meshgrid(
                np.arange(I, dtype=np.int32),
                np.arange(J, dtype=np.int32),
                np.arange(K, dtype=np.int32),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
        return sijk

    def find_voxels_sharing_edge(
        self, edges: np.ndarray, ravel: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        edges = np.asarray(edges, dtype=np.int32)
        if edges.ndim == 1:
            edges = self.unravel_nd(edges, self.edge_shape)
        voxels = edges[..., :3]
        elabels = edges[..., -1]

        neighbors = np.expand_dims(voxels, -2) + Grid.EDGE_VOXEL_OFFSETS[elabels]
        edge_mask = (neighbors >= 0) & (neighbors < np.array(self.shape) - 1)
        edge_mask = edge_mask.all(-1).all(-1)
        neighbors[~edge_mask] = 0

        if ravel:
            neighbors = self.ravel_nd(
                neighbors.reshape(-1, 3), self.padded_shape
            ).reshape(-1, 4)
        return neighbors, edge_mask

    def find_voxel_edges(self, voxels: np.ndarray, ravel: bool = True) -> np.ndarray:
        voxels = np.asarray(voxels, dtype=np.int32)
        if voxels.ndim == 1:
            voxels = self.unravel_nd(voxels, self.padded_shape)
        N = voxels.shape[0]

        voxels = np.expand_dims(
            np.concatenate((voxels, np.zeros((N, 1), dtype=np.int32)), -1), -2
        )
        edges = voxels + Grid.VOXEL_EDGE_OFFSETS
        if ravel:
            edges = self.ravel_nd(edges.reshape(-1, 4), self.edge_shape).reshape(-1, 12)
        return edges

    def grid_to_data(self, x: np.ndarray) -> np.ndarray:
        return x * self.spacing + self.min_corner

    def data_to_grid(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_corner) / self.spacing
