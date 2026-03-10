"""Mesh smoothing operations."""

from __future__ import annotations

import numpy as np
import trimesh

from xeltofab.state import PipelineParams, PipelineState


def smooth(state: PipelineState) -> PipelineState:
    """Apply mesh smoothing to 3D mesh. No-op for 2D contours.

    Dispatches to Taubin or bilateral filtering based on params.smoothing_method.
    """
    if state.ndim == 2 or state.vertices is None or state.faces is None:
        return state

    mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces, process=False)

    if state.params.smoothing_method == "bilateral":
        vertices = _bilateral_smooth(mesh, state.params)
    else:
        trimesh.smoothing.filter_taubin(
            mesh,
            iterations=state.params.taubin_iterations,
            lamb=state.params.taubin_lambda,
        )
        vertices = mesh.vertices

    return state.model_copy(update={"smoothed_vertices": np.asarray(vertices)})


def _bilateral_smooth(mesh: trimesh.Trimesh, params: PipelineParams) -> np.ndarray:
    """Bilateral mesh filtering with normal-similarity weighting.

    For each vertex, displacement toward neighbors is weighted by both spatial
    proximity (sigma_s) and normal similarity (sigma_n). Neighbors across sharp
    edges have divergent normals and receive low weight, preserving features.

    Volume correction after each iteration counters the inherent shrinkage
    of Laplacian-family smoothers.
    """
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    # Auto-compute spatial sigma from average edge length
    sigma_s = params.bilateral_sigma_s
    if sigma_s is None:
        edges = vertices[faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)]
        edge_lengths = np.linalg.norm(edges[::2] - edges[1::2], axis=1)
        sigma_s = float(np.mean(edge_lengths))

    # Auto-compute normal sigma (in radians): controls feature sharpness threshold
    sigma_n = params.bilateral_sigma_n
    if sigma_n is None:
        sigma_n = 0.35

    inv_2sigma_s2 = -0.5 / (sigma_s * sigma_s)
    inv_2sigma_n2 = -0.5 / (sigma_n * sigma_n)

    # Precompute adjacency list (1-ring neighbors per vertex)
    adjacency = mesh.vertex_neighbors

    # Compute original volume for correction
    original_volume = _signed_volume(vertices, faces)

    for _ in range(params.bilateral_iterations):
        # Recompute vertex normals each iteration
        vertex_normals = _compute_vertex_normals(vertices, faces)
        new_vertices = np.copy(vertices)

        for i, neighbors in enumerate(adjacency):
            if len(neighbors) == 0:
                continue

            nbr_idx = np.array(neighbors)
            nbr_pos = vertices[nbr_idx]

            # Spatial weights: based on Euclidean distance
            diff = nbr_pos - vertices[i]
            dist_sq = np.sum(diff * diff, axis=1)
            w_s = np.exp(dist_sq * inv_2sigma_s2)

            # Normal weights: based on normal similarity (angle between normals)
            normal_diff = vertex_normals[nbr_idx] - vertex_normals[i]
            normal_dist_sq = np.sum(normal_diff * normal_diff, axis=1)
            w_n = np.exp(normal_dist_sq * inv_2sigma_n2)

            # Combined weight
            w = w_s * w_n
            w_sum = w.sum()
            if w_sum > 0:
                new_vertices[i] = vertices[i] + (w[:, np.newaxis] * diff).sum(axis=0) / w_sum

        vertices = new_vertices

        # Volume correction: scale uniformly to match original volume
        if abs(original_volume) > 1e-12:
            current_volume = _signed_volume(vertices, faces)
            if abs(current_volume) > 1e-12:
                centroid = vertices.mean(axis=0)
                scale = (abs(original_volume) / abs(current_volume)) ** (1.0 / 3.0)
                vertices = centroid + (vertices - centroid) * scale

    return vertices


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute signed volume of a triangle mesh via divergence theorem."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.sum(v0 * np.cross(v1, v2)) / 6.0)


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # area-weighted (not normalized)

    vertex_normals = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(vertex_normals, faces[:, j], face_normals)

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return vertex_normals / norms
