"""Generate a true signed distance field for the Stanford bunny.

Usage:
    uv run python scripts/generate_bunny_sdf.py

Output:
    /tmp/bunny_sdf_true.npy
"""

import numpy as np
import pyvista as pv
import trimesh

# 1. Get bunny, normalize to unit scale
print("Loading Stanford bunny...")
bunny = pv.examples.download_bunny()
verts = np.asarray(bunny.points)
faces = np.asarray(bunny.regular_faces)
center = verts.mean(axis=0)
extent = verts.max() - verts.min()
verts_norm = (verts - center) / extent

# 2. Voxelize to get grid structure
print("Voxelizing...")
mesh_tm = trimesh.Trimesh(vertices=verts_norm, faces=faces, process=True)
vox = mesh_tm.voxelized(pitch=1.0 / 120)
raw = vox.matrix
origin = vox.transform[:3, 3]
pitch = 1.0 / 120
pad = 3
shape = tuple(np.array(raw.shape) + 2 * pad)
print(f"  Grid shape: {shape}, total points: {np.prod(shape):,}")

# 3. Build query grid in mesh coordinate space
print("Building query grid...")
gi, gj, gk = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
pts = np.stack([gi, gj, gk], axis=-1).reshape(-1, 3).astype(np.float64)
pts_mesh = (pts - pad) * pitch + origin
del gi, gj, gk, pts

# 4. Compute signed distance via VTK (fast C++ implementation)
print("Computing signed distance (VTK implicit_distance)...")
query_grid = pv.PolyData(pts_mesh)
bunny_pv = pv.PolyData(verts_norm, np.column_stack([np.full(len(faces), 3), faces]))
query_grid = query_grid.compute_implicit_distance(bunny_pv)
sdf = np.asarray(query_grid["implicit_distance"]).reshape(shape)

print(f"  Shape: {sdf.shape}")
print(f"  Range: [{sdf.min():.4f}, {sdf.max():.4f}]")

# 5. Save
np.save("/tmp/bunny_sdf_true.npy", sdf)
print("Saved to /tmp/bunny_sdf_true.npy")
