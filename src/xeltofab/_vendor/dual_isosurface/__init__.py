"""Vendored Dual Contouring / Surface Nets from sdftoolbox (MIT License).

Source: https://github.com/cheind/sdftoolbox
Modifications: adapted for numpy array input, removed SDF node dependency,
fixed np.float_ deprecation for numpy 2.x compatibility.
"""

from xeltofab._vendor.dual_isosurface.core import dual_isosurface

__all__ = ["dual_isosurface"]
