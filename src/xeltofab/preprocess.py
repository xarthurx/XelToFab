"""Field preprocessing: smooth -> threshold -> morphology -> keep largest component."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import closing, opening, remove_small_objects

from xeltofab.state import PipelineState


def preprocess(state: PipelineState) -> PipelineState:
    """Preprocess field: smooth -> threshold -> morphology -> keep largest component."""
    params = state.params
    field = state.field

    # Record original volume fraction
    volume_fraction = float(np.mean(field))

    # Gaussian smooth
    smoothed = gaussian_filter(field, sigma=params.smooth_sigma)

    # Threshold to binary
    binary = (smoothed >= params.threshold).astype(np.uint8)

    # Morphological cleanup
    if params.morph_radius > 0:
        if state.ndim == 2:
            from skimage.morphology import disk

            selem = disk(params.morph_radius)
        else:
            from skimage.morphology import ball

            selem = ball(params.morph_radius)
        binary = opening(binary, selem).astype(np.uint8)
        binary = closing(binary, selem).astype(np.uint8)

    # Remove small disconnected components
    binary_bool = binary.astype(bool)
    # remove_small_objects(max_size=N) removes objects with N pixels or fewer.
    # The -1 compensates for the <= semantics: we want to keep objects >= 0.5% of total (or >= 8 px).
    max_size = max(binary.size // 200, 8) - 1
    cleaned = remove_small_objects(binary_bool, max_size=max_size)
    binary = cleaned.astype(np.uint8)

    return state.model_copy(update={"binary": binary, "volume_fraction": volume_fraction})
