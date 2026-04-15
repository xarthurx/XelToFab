"""SDF→density conversion utilities.

Three converters translate signed-distance fields into density fields in [0, 1]
so that third-party consumers (EngiBench, density-only TO solvers) can feed an
SDF-sourced array through the density-mode pipeline.

Convention: SDF is negative inside the shape, positive outside. All converters
map ``sdf == level`` to density 0.5 so that extracting at iso=0.5 on the
returned density recovers the zero (or ``level``-offset) iso-surface.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import expit


def _validate_sdf_input(sdf) -> np.ndarray:
    """Coerce array-like input to float64 ndarray and reject non-finite values."""
    out = np.asarray(sdf, dtype=np.float64)
    if out.size > 0 and not (np.isfinite(out.min()) and np.isfinite(out.max())):
        n_bad = int(np.count_nonzero(~np.isfinite(out)))
        raise ValueError(f"sdf contains {n_bad} non-finite values (NaN or Inf)")
    return out


def _validate_finite_scalar(name: str, value: float) -> None:
    """Reject NaN/Inf scalar parameters at the API boundary."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _validate_positive_bandwidth(bandwidth: float, func_name: str) -> None:
    """Reject non-finite or non-positive bandwidth values."""
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be a finite positive number for {func_name}, got {bandwidth}")


def heaviside(sdf, *, level: float = 0.0) -> np.ndarray:
    """Hard Heaviside converter: density is {0, 0.5, 1}.

    density = 1 where sdf < level, 0.5 where sdf == level, 0 where sdf > level.

    Parameters
    ----------
    sdf : array-like
        Signed-distance field, any shape. Must be finite.
    level : float, default 0.0
        Iso-surface offset.

    Returns
    -------
    np.ndarray
        Float64 array of same shape as ``sdf`` with values in {0.0, 0.5, 1.0}.
    """
    _validate_finite_scalar("level", level)
    arr = _validate_sdf_input(sdf)
    return 1.0 - np.heaviside(arr - level, 0.5)


def linear_ramp(sdf, *, bandwidth: float = 1.0, level: float = 0.0) -> np.ndarray:
    """Clipped linear ramp: density transitions linearly across a finite band.

    density = clip(0.5 - (sdf - level) / (2 * bandwidth), 0, 1)

    Outside ``[level - bandwidth, level + bandwidth]`` the output is exactly
    0 or 1. This mirrors how marching-cubes-at-0.5 on a band-limited density
    field recovers a level set, making it the default choice.

    Parameters
    ----------
    sdf : array-like
        Signed-distance field, any shape. Must be finite.
    bandwidth : float, default 1.0
        Half-width of the transition band, in the same units as ``sdf``. > 0.
    level : float, default 0.0
        Iso-surface offset.

    Returns
    -------
    np.ndarray
        Float64 array of same shape as ``sdf`` with values in [0, 1].
    """
    _validate_positive_bandwidth(bandwidth, "linear_ramp")
    _validate_finite_scalar("level", level)
    arr = _validate_sdf_input(sdf)
    return np.clip(0.5 - (arr - level) / (2.0 * bandwidth), 0.0, 1.0)


def sigmoid(sdf, *, bandwidth: float = 1.0, level: float = 0.0) -> np.ndarray:
    """Smooth sigmoid converter: infinite-support smoothed Heaviside.

    density = 1 / (1 + exp((sdf - level) / bandwidth))

    Delegates to ``scipy.special.expit`` for a single-pass, numerically stable
    evaluation. Output is mathematically in (0, 1) but saturates to 0 or 1 at
    float64 extremes; use ``linear_ramp`` or ``heaviside`` if deterministic
    0/1 values in the far field matter.

    Parameters
    ----------
    sdf : array-like
        Signed-distance field, any shape. Must be finite.
    bandwidth : float, default 1.0
        Transition scale, in the same units as ``sdf``. > 0.
    level : float, default 0.0
        Iso-surface offset.

    Returns
    -------
    np.ndarray
        Float64 array of same shape as ``sdf`` with values in [0, 1]
        (mathematically (0, 1) but saturates at float64 extremes).
    """
    _validate_positive_bandwidth(bandwidth, "sigmoid")
    _validate_finite_scalar("level", level)
    arr = _validate_sdf_input(sdf)
    return expit((level - arr) / bandwidth)


def sdf_to_density(
    sdf,
    method: Literal["heaviside", "linear", "sigmoid"] = "linear",
    *,
    bandwidth: float = 1.0,
    level: float = 0.0,
) -> np.ndarray:
    """Convert an SDF array to a density array in [0, 1].

    Dispatcher around :func:`heaviside`, :func:`linear_ramp`, and :func:`sigmoid`.
    All three methods map ``sdf == level`` to density 0.5 so that marching
    cubes at iso=0.5 on the result recovers the input iso-surface.

    Parameters
    ----------
    sdf : array-like
        Signed-distance field, any shape. Must be finite.
    method : {"heaviside", "linear", "sigmoid"}, default "linear"
        Conversion profile. ``bandwidth`` is ignored for ``heaviside``.
    bandwidth : float, default 1.0
        Transition scale in the same units as ``sdf``. Must be > 0 for
        ``linear`` and ``sigmoid``.
    level : float, default 0.0
        Iso-surface offset.

    Returns
    -------
    np.ndarray
        Float64 density array in [0, 1] with the same shape as ``sdf``.
    """
    if method == "heaviside":
        return heaviside(sdf, level=level)
    if method == "linear":
        return linear_ramp(sdf, bandwidth=bandwidth, level=level)
    if method == "sigmoid":
        return sigmoid(sdf, bandwidth=bandwidth, level=level)
    raise ValueError(f"unknown method {method!r}; expected one of 'heaviside', 'linear', 'sigmoid'")
