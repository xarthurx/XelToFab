"""Tests for xeltofab.extrude."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh as _trimesh
from shapely.geometry import MultiPolygon, Polygon

import xeltofab as xtf
from xeltofab.extrude import _build_binary, _build_prism_mesh, _polygonize, _trace_contours, _triangulate_polygon

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "examples"


def test_extrude_module_importable():
    """The extrude_2d symbol is reachable via the package root."""
    assert callable(xtf.extrude_2d)


def test_rejects_1d_field():
    with pytest.raises(ValueError, match="2D"):
        xtf.extrude_2d(np.ones(10), thickness=5.0)


def test_rejects_3d_field():
    with pytest.raises(ValueError, match="2D"):
        xtf.extrude_2d(np.ones((4, 4, 4)), thickness=5.0)


def test_rejects_zero_thickness():
    with pytest.raises(ValueError, match="thickness"):
        xtf.extrude_2d(np.ones((4, 4)), thickness=0.0)


def test_rejects_negative_thickness():
    with pytest.raises(ValueError, match="thickness"):
        xtf.extrude_2d(np.ones((4, 4)), thickness=-1.0)


def test_rejects_negative_smooth_sigma():
    with pytest.raises(ValueError, match="smooth_sigma"):
        xtf.extrude_2d(np.ones((4, 4)), thickness=1.0, smooth_sigma=-0.5)


def test_rejects_negative_min_component_area():
    with pytest.raises(ValueError, match="min_component_area"):
        xtf.extrude_2d(np.ones((4, 4)), thickness=1.0, min_component_area=-1)


def test_binary_density_default_level():
    """Density field thresholds at 0.5 by default."""
    field = np.array([[0.2, 0.8], [0.6, 0.4]])
    binary = _build_binary(
        field,
        field_type="density",
        level=None,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    np.testing.assert_array_equal(binary, [[False, True], [True, False]])


def test_binary_density_custom_level():
    """Density field honors explicit level."""
    field = np.array([[0.2, 0.8], [0.6, 0.4]])
    binary = _build_binary(
        field,
        field_type="density",
        level=0.7,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    np.testing.assert_array_equal(binary, [[False, True], [False, False]])


def test_binary_sdf_inside_is_negative():
    """SDF threshold: material where value <= level (default 0.0)."""
    field = np.array([[-0.5, 0.5], [-0.1, 0.1]])
    binary = _build_binary(
        field,
        field_type="sdf",
        level=None,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    np.testing.assert_array_equal(binary, [[True, False], [True, False]])


def test_binary_empty_raises():
    """All-below-threshold input raises ValueError."""
    field = np.zeros((4, 4))
    with pytest.raises(ValueError, match="no material"):
        _build_binary(
            field,
            field_type="density",
            level=None,
            smooth_sigma=0.0,
            fill_holes=False,
            min_component_area=0,
        )


def test_binary_gaussian_smooth_bridges_gap():
    """Smoothing a single-pixel gap between two blocks bridges it before threshold."""
    field = np.zeros((5, 9), dtype=float)
    field[:, :4] = 1.0
    field[:, 5:] = 1.0
    b0 = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    assert not b0[:, 4].any()
    b1 = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=2.0,
        fill_holes=False,
        min_component_area=0,
    )
    assert b1[:, 4].any()


def test_binary_fill_holes_closes_pinhole():
    """fill_holes=True morphologically closes a 1-pixel pinhole."""
    field = np.ones((5, 5), dtype=float)
    field[2, 2] = 0.0
    b_off = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=0,
    )
    assert not b_off[2, 2]
    b_on = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=True,
        min_component_area=0,
    )
    assert b_on[2, 2]


def test_binary_min_component_area_drops_orphan():
    """min_component_area removes small disconnected islands."""
    field = np.zeros((10, 10), dtype=float)
    field[1:4, 1:4] = 1.0
    field[7:9, 7:9] = 1.0
    b = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=0.0,
        fill_holes=False,
        min_component_area=5,
    )
    assert b[1:4, 1:4].all()
    assert not b[7:9, 7:9].any()


def test_trace_contours_single_blob():
    """A single filled rectangle in the interior produces one closed contour."""
    binary = np.zeros((10, 10), dtype=bool)
    binary[3:7, 3:7] = True
    contours = _trace_contours(binary)
    assert len(contours) == 1
    c = contours[0]
    np.testing.assert_allclose(c[0], c[-1])
    assert c[:, 0].min() >= 2.4
    assert c[:, 0].max() <= 6.6


def test_trace_contours_two_disjoint_blobs():
    """Two disjoint filled regions produce two contours."""
    binary = np.zeros((10, 20), dtype=bool)
    binary[2:5, 2:5] = True
    binary[2:5, 12:15] = True
    contours = _trace_contours(binary)
    assert len(contours) == 2


def test_trace_contours_blob_on_boundary_is_closed():
    """Material touching the image edge still yields a closed contour (via the zero pad)."""
    binary = np.zeros((10, 10), dtype=bool)
    binary[0:5, 0:5] = True
    contours = _trace_contours(binary)
    assert len(contours) == 1
    c = contours[0]
    np.testing.assert_allclose(c[0], c[-1])


def _contour_square(x0, x1, y0, y1):
    """Helper: closed contour for an axis-aligned rectangle in (x, y)."""
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
        dtype=float,
    )


def test_polygonize_single_blob():
    """A single contour becomes one polygon inside a MultiPolygon."""
    contours = [_contour_square(2, 6, 2, 6)]
    mp = _polygonize(contours, height=10, width=10)
    assert isinstance(mp, MultiPolygon)
    assert len(mp.geoms) == 1
    assert mp.geoms[0].area == pytest.approx(16.0, rel=0.05)


def test_polygonize_with_interior_hole():
    """Outer CCW + inner CW rings produce a polygon with one hole."""
    outer = _contour_square(1, 9, 1, 9)
    inner = _contour_square(3, 7, 3, 7)[::-1]
    mp = _polygonize([outer, inner], height=10, width=10)
    assert len(mp.geoms) == 1
    poly = mp.geoms[0]
    assert len(poly.interiors) == 1
    assert poly.area == pytest.approx(64 - 16, rel=0.05)


def test_polygonize_snaps_to_image_rectangle():
    """When material touches the image edge, the resulting polygon has flush coords."""
    contours = [_contour_square(-0.5, 9.5, -0.5, 9.5)]
    mp = _polygonize(contours, height=10, width=10)
    assert len(mp.geoms) == 1
    minx, miny, maxx, maxy = mp.geoms[0].bounds
    assert minx == pytest.approx(0.0)
    assert miny == pytest.approx(0.0)
    assert maxx == pytest.approx(9.0)
    assert maxy == pytest.approx(9.0)


def test_polygonize_two_disjoint():
    """Two separate contours → MultiPolygon with two geoms."""
    contours = [_contour_square(1, 3, 1, 3), _contour_square(1, 3, 6, 8)]
    mp = _polygonize(contours, height=10, width=10)
    assert len(mp.geoms) == 2


def test_polygonize_corner_hugging_material():
    """Material touching two adjacent image edges snaps flush on both axes."""
    contours = [_contour_square(-0.5, 4.5, -0.5, 4.5)]
    mp = _polygonize(contours, height=10, width=10)
    assert len(mp.geoms) == 1
    minx, miny, maxx, maxy = mp.geoms[0].bounds
    assert minx == pytest.approx(0.0)
    assert miny == pytest.approx(0.0)
    assert maxx == pytest.approx(4.5)
    assert maxy == pytest.approx(4.5)


def test_triangulate_simple_square():
    """A 4-vertex square triangulates into 2 triangles, 4 vertices."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    verts, tris = _triangulate_polygon(poly)
    assert verts.shape == (4, 2)
    assert tris.shape == (2, 3)
    assert tris.max() < len(verts)
    assert tris.min() >= 0


def test_triangulate_polygon_with_hole():
    """A square with a central hole yields outer+hole vertices and enough triangles."""
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly = Polygon(outer, [hole])
    verts, tris = _triangulate_polygon(poly)
    assert verts.shape == (8, 2)
    assert len(tris) >= 8
    assert tris.max() < len(verts)


def test_triangulate_skips_closing_duplicate():
    """Shapely exteriors repeat the first point; _triangulate_polygon must drop it."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    verts, _ = _triangulate_polygon(poly)
    assert verts.shape == (4, 2)


def test_prism_mesh_from_square_is_watertight():
    poly = Polygon([(0, 0), (9, 0), (9, 9), (0, 9)])
    mp = MultiPolygon([poly])
    mesh = _build_prism_mesh(mp, thickness=5.0)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent


def test_prism_mesh_volume_matches_extrusion():
    """Volume = polygon_area * thickness."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    mp = MultiPolygon([poly])
    mesh = _build_prism_mesh(mp, thickness=3.0)
    assert mesh.volume == pytest.approx(4 * 4 * 3.0, rel=1e-6)


def test_prism_mesh_z_bounds_start_at_zero():
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    mp = MultiPolygon([poly])
    mesh = _build_prism_mesh(mp, thickness=7.0)
    zmin = mesh.vertices[:, 2].min()
    zmax = mesh.vertices[:, 2].max()
    assert zmin == pytest.approx(0.0)
    assert zmax == pytest.approx(7.0)


def test_prism_mesh_polygon_with_hole_is_watertight():
    """Annulus extrudes to a watertight genus-1 shell."""
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)][::-1]
    poly = Polygon(outer, [hole])
    mp = MultiPolygon([poly])
    mesh = _build_prism_mesh(mp, thickness=2.0)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(168.0, rel=1e-3)


def test_prism_mesh_two_disjoint_blobs():
    a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    b = Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])
    mp = MultiPolygon([a, b])
    mesh = _build_prism_mesh(mp, thickness=1.0)
    split = mesh.split(only_watertight=True)
    assert len(split) == 2
    for piece in split:
        assert piece.is_watertight


def test_extrude_2d_square_end_to_end():
    field = np.ones((10, 10), dtype=float)
    mesh = xtf.extrude_2d(field, thickness=5.0)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(9 * 9 * 5.0, rel=1e-3)


def test_extrude_2d_centered_hole():
    field = np.zeros((20, 20), dtype=float)
    field[2:18, 2:18] = 1.0
    field[8:12, 8:12] = 0.0
    mesh = xtf.extrude_2d(field, thickness=3.0)
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert mesh.volume < 20 * 20 * 3.0
    assert mesh.euler_number == 0


def test_extrude_2d_two_disjoint_blobs():
    field = np.zeros((10, 20), dtype=float)
    field[2:5, 2:5] = 1.0
    field[2:5, 12:15] = 1.0
    mesh = xtf.extrude_2d(field, thickness=2.0)
    split = mesh.split(only_watertight=True)
    assert len(split) == 2


def test_extrude_2d_boundary_flush():
    """Full-image square produces a mesh with vertices at exactly x=0 and x=W-1."""
    field = np.ones((10, 10), dtype=float)
    mesh = xtf.extrude_2d(field, thickness=1.0)
    xs = mesh.vertices[:, 0]
    ys = mesh.vertices[:, 1]
    assert np.isclose(xs.min(), 0.0)
    assert np.isclose(xs.max(), 9.0)
    assert np.isclose(ys.min(), 0.0)
    assert np.isclose(ys.max(), 9.0)


def test_extrude_2d_sdf_input():
    """SDF input with level=0.0 extrudes the inside (negative-value) region."""
    y, x = np.mgrid[-10:10, -10:10].astype(float)
    sdf = np.sqrt(x**2 + y**2) - 5.0
    mesh = xtf.extrude_2d(sdf, thickness=2.0, field_type="sdf")
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(78.5 * 2.0, rel=0.15)


def test_volume_monotone_in_thickness():
    """Same field, different thickness: volume scales linearly."""
    field = np.zeros((10, 10), dtype=float)
    field[2:8, 2:8] = 1.0
    v1 = xtf.extrude_2d(field, thickness=1.0).volume
    v5 = xtf.extrude_2d(field, thickness=5.0).volume
    v10 = xtf.extrude_2d(field, thickness=10.0).volume
    assert v5 == pytest.approx(v1 * 5.0, rel=1e-6)
    assert v10 == pytest.approx(v1 * 10.0, rel=1e-6)


def test_cap_area_tracks_binary_pixel_count():
    """Projected area tracks material pixel count within a loose discretization bound."""
    field = np.zeros((12, 12), dtype=float)
    field[3:9, 3:9] = 1.0
    mesh = xtf.extrude_2d(field, thickness=1.0)
    projected_area = mesh.volume
    assert projected_area == pytest.approx(36.0, rel=0.15)


def test_beams2d_25x50_extrusion(tmp_path):
    field = np.load(FIXTURE_DIR / "beams_2d_25x50_sample0.npy")
    assert field.ndim == 2
    mesh = xtf.extrude_2d(field, thickness=10.0)
    assert mesh.volume > 0
    stl_path = tmp_path / "beam.stl"
    mesh.export(stl_path)
    loaded = _trimesh.load(stl_path, force="mesh")
    assert len(loaded.vertices) > 0


def test_beams2d_100x200_extrusion():
    field = np.load(FIXTURE_DIR / "beams_2d_100x200_sample1.npy")
    mesh = xtf.extrude_2d(field, thickness=15.0, min_component_area=10)
    assert mesh.volume > 0
    assert len(mesh.faces) < 500_000


def test_density_preprocess_parity():
    """extrude_2d's internal binary matches preprocess() when parameters align."""
    from xeltofab.preprocess import preprocess as pipeline_preprocess
    from xeltofab.state import PipelineParams, PipelineState

    field = np.load(FIXTURE_DIR / "beams_2d_25x50_sample0.npy")
    params = PipelineParams(
        threshold=0.5,
        smooth_sigma=1.0,
        morph_radius=1,
    )
    state = pipeline_preprocess(PipelineState(field=field, params=params))
    heur_max_size = max(field.size // 200, 8) - 1
    our_binary = _build_binary(
        field,
        field_type="density",
        level=0.5,
        smooth_sigma=1.0,
        fill_holes=True,
        min_component_area=heur_max_size + 1,
    )
    np.testing.assert_array_equal(our_binary, state.binary.astype(bool))
