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
    raise NotImplementedError("extrude_2d body is filled in by later tasks")
