"""2D field → 3D extrusion for fabrication-ready output.

Turns a 2D density or SDF array into a watertight triangle mesh with
configurable extrusion thickness. Uses marching squares + shapely +
mapbox_earcut internally. Returns a trimesh.Trimesh; caller writes STL/OBJ/PLY
via mesh.export(...).
"""

from __future__ import annotations

from typing import Literal

import mapbox_earcut
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from shapely.geometry import GeometryCollection, LinearRing, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from skimage.measure import find_contours
from skimage.morphology import closing, disk, opening, remove_small_objects


def _build_binary(
    field: np.ndarray,
    *,
    field_type: Literal["density", "sdf"],
    level: float | None,
    smooth_sigma: float,
    fill_holes: bool,
    min_component_area: int,
) -> np.ndarray:
    """Clean a 2D field to a bool binary mask."""
    eff_level = level if level is not None else (0.0 if field_type == "sdf" else 0.5)
    smoothed = gaussian_filter(field, sigma=smooth_sigma) if smooth_sigma > 0.0 else field

    binary = smoothed <= eff_level if field_type == "sdf" else smoothed >= eff_level

    if fill_holes:
        selem = disk(1)
        binary = opening(binary, selem)
        binary = closing(binary, selem)

    if min_component_area > 0:
        binary = remove_small_objects(binary, max_size=min_component_area - 1)

    if not binary.any():
        raise ValueError("no material above threshold — check field values and level")

    return binary


def _trace_contours(binary: np.ndarray) -> list[np.ndarray]:
    """Trace closed contours from a bool mask in canonical (x, y) coordinates."""
    padded = np.pad(binary.astype(float), 1, constant_values=0.0)
    raw = find_contours(padded, 0.5)
    return [np.column_stack([contour[:, 1] - 1.0, contour[:, 0] - 1.0]) for contour in raw]


def _collect_polygons(geom: object) -> list[Polygon]:
    """Recursively flatten polygonal output from shapely set operations."""
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polygons: list[Polygon] = []
        for part in geom.geoms:
            polygons.extend(_collect_polygons(part))
        return polygons
    return []


def _polygonize(
    contours: list[np.ndarray],
    *,
    height: int,
    width: int,
) -> MultiPolygon:
    """Build a clean, flush-to-edge MultiPolygon from raw contours."""
    shells: list[LinearRing] = []
    holes: list[LinearRing] = []
    for contour in contours:
        if len(contour) < 4:
            continue
        ring = LinearRing(contour)
        if ring.is_ccw:
            shells.append(ring)
        else:
            holes.append(ring)

    if not shells:
        raise ValueError("no valid shell polygons after contour tracing")

    shell_polys = [Polygon(shell) for shell in shells]
    shell_holes: list[list[np.ndarray]] = [[] for _ in shell_polys]
    for hole in holes:
        hole_poly = Polygon(hole)
        containing = [
            idx for idx, shell_poly in enumerate(shell_polys) if shell_poly.contains(hole_poly.representative_point())
        ]
        if not containing:
            continue
        target = min(containing, key=lambda idx: shell_polys[idx].area)
        shell_holes[target].append(np.asarray(hole.coords))

    polygons = [
        Polygon(np.asarray(shell.coords), holes_for_shell)
        for shell, holes_for_shell in zip(shells, shell_holes, strict=True)
    ]
    merged = unary_union(polygons)
    cleaned = merged.buffer(0)
    image_rect = box(0, 0, width - 1, height - 1)
    snapped = cleaned.intersection(image_rect)

    if isinstance(snapped, Polygon):
        return MultiPolygon([snapped])
    if isinstance(snapped, MultiPolygon):
        return snapped

    polygons = _collect_polygons(snapped)
    if not polygons:
        raise ValueError("snapped geometry empty — no extrudable region")
    return MultiPolygon(polygons)


def _triangulate_polygon(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a shapely Polygon (with holes) via mapbox_earcut."""

    def _ring_coords(ring: LinearRing) -> np.ndarray:
        coords = np.asarray(ring.coords, dtype=np.float64)
        if len(coords) >= 2 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        return coords

    exterior = _ring_coords(poly.exterior)
    holes = [_ring_coords(interior) for interior in poly.interiors]

    vertices = np.concatenate([exterior, *holes], axis=0) if holes else exterior
    ring_ends = [len(exterior)]
    for hole in holes:
        ring_ends.append(ring_ends[-1] + len(hole))

    flat_triangles = mapbox_earcut.triangulate_float64(vertices, np.asarray(ring_ends, dtype=np.uint32))
    triangles = np.asarray(flat_triangles, dtype=np.int64).reshape(-1, 3)
    return vertices, triangles


def extrude_2d(
    field: np.ndarray,
    thickness: float,
    *,
    field_type: Literal["density", "sdf"] = "density",
    level: float | None = None,
    min_component_area: int = 0,
    smooth_sigma: float = 0.0,
    fill_holes: bool = False,
) -> trimesh.Trimesh:
    """Extrude a 2D field into a 3D triangle mesh."""
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape {field.shape}")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    raise NotImplementedError("extrude_2d body is filled in by later tasks")
