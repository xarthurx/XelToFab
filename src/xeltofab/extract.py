"""Mesh/contour extraction from scalar fields."""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours, marching_cubes

from xeltofab.state import PipelineState


def extract(state: PipelineState) -> PipelineState:
    """Extract mesh (3D) or contours (2D).

    In direct mode, extracts from the continuous input field at the configured level.
    Otherwise, extracts from the preprocessed binary field at the same level.
    """
    if state.params.direct_extraction:
        field = np.asarray(state.field, dtype=np.float64)
    else:
        if state.binary is None:
            raise ValueError("binary field is None — run preprocess() first")
        field = np.asarray(state.binary, dtype=np.float64)
    level = state.params.effective_extraction_level

    if state.ndim == 2:
        return _extract_2d(state, field, level)
    return _extract_3d(state, field, level)


def _extract_2d(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract contours from 2D field using marching squares."""
    contours = find_contours(field, level=level)
    return state.model_copy(update={"contours": contours})


def _extract_3d(state: PipelineState, field: np.ndarray, level: float) -> PipelineState:
    """Extract triangle mesh from 3D field using marching cubes."""
    vertices, faces, _, _ = marching_cubes(field, level=level)
    return state.model_copy(update={"vertices": vertices, "faces": faces})
