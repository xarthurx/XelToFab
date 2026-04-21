"""2D field → 3D extrusion for fabrication-ready output.

Turns a 2D density or SDF array into a watertight triangle mesh with
configurable extrusion thickness. Uses marching squares + shapely +
mapbox_earcut internally. Returns a trimesh.Trimesh; caller writes STL/OBJ/PLY
via mesh.export(...).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
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
