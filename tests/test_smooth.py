# tests/test_smooth.py
import numpy as np
import trimesh

from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.smooth import smooth
from xeltofab.state import PipelineParams, PipelineState


def test_smooth_3d_produces_smoothed_vertices(sphere_field: np.ndarray):
    state = extract(preprocess(PipelineState(field=sphere_field)))
    result = smooth(state)
    assert result.smoothed_vertices is not None
    assert result.smoothed_vertices.shape == result.vertices.shape


def test_smooth_3d_changes_vertices(sphere_field: np.ndarray):
    state = extract(preprocess(PipelineState(field=sphere_field)))
    result = smooth(state)
    assert not np.allclose(result.smoothed_vertices, result.vertices)


def test_smooth_3d_preserves_volume(sphere_field: np.ndarray):
    """Taubin smoothing should approximately preserve mesh volume."""
    state = extract(preprocess(PipelineState(field=sphere_field)))
    original_mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
    result = smooth(state)
    smoothed_mesh = trimesh.Trimesh(vertices=result.smoothed_vertices, faces=result.faces)
    ratio = smoothed_mesh.volume / original_mesh.volume
    assert ratio > 0.9, f"Volume ratio {ratio:.3f} is too low — smoothing destroyed the mesh"


def test_smooth_2d_is_noop(circle_field: np.ndarray):
    """Taubin smoothing only applies to 3D meshes; 2D contours pass through."""
    state = extract(preprocess(PipelineState(field=circle_field)))
    result = smooth(state)
    assert result.smoothed_vertices is None
    assert result.contours is not None


# --- Bilateral smoothing tests ---


def _bilateral_params(**overrides) -> PipelineParams:
    return PipelineParams(smoothing_method="bilateral", **overrides)


def test_bilateral_produces_smoothed_vertices(sphere_field: np.ndarray):
    state = extract(preprocess(PipelineState(field=sphere_field, params=_bilateral_params())))
    result = smooth(state)
    assert result.smoothed_vertices is not None
    assert result.smoothed_vertices.shape == result.vertices.shape


def test_bilateral_preserves_volume(sphere_field: np.ndarray):
    """Bilateral filtering causes some shrinkage on convex surfaces (expected).

    Unlike Taubin (which alternates shrink/inflate), bilateral only displaces
    along normals. On convex surfaces all neighbor heights are negative, causing
    inward drift. Volume ratio > 0.75 for 10 iterations is acceptable.
    """
    state = extract(preprocess(PipelineState(field=sphere_field, params=_bilateral_params())))
    original_mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
    result = smooth(state)
    smoothed_mesh = trimesh.Trimesh(vertices=result.smoothed_vertices, faces=result.faces)
    ratio = smoothed_mesh.volume / original_mesh.volume
    assert ratio > 0.75, f"Volume ratio {ratio:.3f} is too low — bilateral smoothing destroyed the mesh"


def test_bilateral_preserves_features():
    """Bilateral smoothing reduces displacement at sharp edges via normal weighting.

    On a subdivided cube, vertices at 90-degree edges should move less than
    vertices on flat faces, because cross-edge neighbors have divergent normals
    and receive near-zero weight.
    """
    cube = trimesh.creation.box(extents=[1, 1, 1])
    for _ in range(3):
        cube = cube.subdivide()

    # Add noise to create something to smooth
    rng = np.random.default_rng(42)
    noise_scale = 0.05 * np.mean(cube.edges_unique_length)
    noisy_verts = np.array(cube.vertices, dtype=np.float64) + rng.normal(0, noise_scale, cube.vertices.shape)

    field = np.ones((10, 10, 10))
    state = PipelineState(
        field=field,
        vertices=noisy_verts,
        faces=np.array(cube.faces),
        ndim=3,
        params=_bilateral_params(bilateral_iterations=3),
    )
    result = smooth(state)

    # Classify vertices: on a unit cube, edge vertices have coords where
    # exactly one component is not ±0.5 (interior to an edge)
    displacements = np.linalg.norm(result.smoothed_vertices - noisy_verts, axis=1)
    at_edge = np.sum(np.isclose(np.abs(cube.vertices), 0.5), axis=1) >= 2
    on_face = ~at_edge

    mean_edge_disp = np.mean(displacements[at_edge])
    mean_face_disp = np.mean(displacements[on_face])

    # Vertices at sharp edges should move less than those on flat faces
    assert mean_edge_disp < mean_face_disp, (
        f"Edge displacement {mean_edge_disp:.6f} should be less than face displacement {mean_face_disp:.6f}"
    )


def test_bilateral_2d_is_noop(circle_field: np.ndarray):
    """Bilateral smoothing only applies to 3D meshes; 2D contours pass through."""
    state = extract(preprocess(PipelineState(field=circle_field, params=_bilateral_params())))
    result = smooth(state)
    assert result.smoothed_vertices is None
    assert result.contours is not None


def test_bilateral_auto_sigma_s(sphere_field: np.ndarray):
    """When bilateral_sigma_s is None, it auto-computes from edge length."""
    params = _bilateral_params(bilateral_sigma_s=None, bilateral_iterations=1)
    state = extract(preprocess(PipelineState(field=sphere_field, params=params)))
    result = smooth(state)
    assert result.smoothed_vertices is not None
    assert not np.allclose(result.smoothed_vertices, result.vertices)
